# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings
from collections import defaultdict
from importlib import import_module
from typing import Any

import torch
from hydra.utils import instantiate
from lightning import LightningModule
from omegaconf import DictConfig, open_dict
from torch import Tensor
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import loss_parallel
from transformers import GenerationConfig

from nemo.collections.common.prompts import PromptFormatter
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.salm_dataset import SALMDataset, SALMSpkDataset, left_collate_vectors
from nemo.collections.speechlm2.models.salm import _resolve_audios_in_prompt, replace_placeholders_and_build_targets
from nemo.collections.speechlm2.parts.automodel_lora import ensure_lora_trainable, make_peft_config, maybe_install_lora
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.pretrained import (
    load_pretrained_automodel_llm,
    maybe_load_pretrained_models,
    setup_parallel_expert_encoder,
    setup_speech_encoder,
    update_perception_output_dim,
)
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, MaskType, NeuralType
from nemo.utils import logging


class SALMAutomodel(LightningModule, HFHubMixin):
    def __init__(self, cfg) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to SALMAutomodel as a Python dict to support hyperparameter serialization "
            f"in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()
        self.cfg = DictConfig(cfg)
        self.audio_locator_tag = self.cfg.audio_locator_tag

        self.tokenizer = AutoTokenizer(
            self.cfg.pretrained_llm, use_fast=True, trust_remote_code=self.cfg.get("trust_remote_code", False)
        )
        self.tokenizer.add_special_tokens({"additional_special_tokens": [self.audio_locator_tag]})
        # Native multi-speaker token setup. The LLM tokenizer is expected to already
        # contain "<spk:0>..<spk:N>" entries at fixed ids (e.g. ids 100..109 for the
        # patched Nemotron Nano v3 tokenizer). No alias rewrite or vocab growth is
        # performed; we only resolve the ids and stash them for the LSS loss.
        self.speaker_token_ids: list[int] = []
        self._init_speaker_token_ids()
        # Optional auxiliary Latent Speaker Supervision (LSS) loss. Mirrors the
        # AED Canary recipe in nemo/collections/asr/models/aed_multitask_models.py
        # (cf. EncDecMultiTaskModel.__init__ around L229-239): instantiated via
        # Hydra `_target_` from a `lss_loss:` YAML block, with `speaker_token_ids`
        # auto-injected when absent. Disabled when the YAML block is absent.
        self.lss_loss = None
        self._init_lss_loss()
        self.llm = None  # populated by configure_model
        self.perception = None  # populated by configure_model

        self._use_fsdp = False
        self._use_tp = False

        if self.cfg.get("init_configure_model", False):
            self.configure_model()

    @staticmethod
    def build_dataset(tokenizer, data_cfg=None) -> SALMDataset:
        sot_cfg = data_cfg.get("sot_cfg", None) if data_cfg is not None else None
        if sot_cfg is not None:
            return SALMSpkDataset(tokenizer=tokenizer, sot_cfg=sot_cfg)
        return SALMDataset(tokenizer=tokenizer)

    @property
    def device(self) -> torch.device:
        """Infer device from the LLM's parameters.

        ``LightningModule.device`` is set by the Trainer and defaults to CPU
        during standalone inference (no Trainer).  Override to query the actual
        parameter storage so that ``.to(self.device)`` works correctly for
        both regular and DTensor (FSDP2/distributed) parameters.
        """
        if self.llm is not None:
            p = next(self.llm.parameters(), None)
            if p is not None:
                return p._local_tensor.device if isinstance(p, DTensor) else p.device
        return super().device

    @property
    def embed_tokens(self):
        """Navigate to the LLM's embedding layer (kept inside the LLM)."""
        if self.llm is None:
            return None
        return self.llm.model.embed_tokens

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs using the LLM's embedding table.

        Uses ``F.embedding`` instead of calling the ``nn.Embedding`` module to
        avoid triggering FSDP2 pre-forward hooks (which lazily initialize the
        child before the root LLM module, causing a ``RuntimeError``).

        When the weight is a sharded ``DTensor`` (FSDP2), we ``full_tensor()``
        it first to all-gather the complete embedding table — the same operation
        FSDP2 performs inside the LLM's forward pass.
        """
        weight = self.embed_tokens.weight
        if isinstance(weight, DTensor):
            weight = weight.full_tensor()
        return torch.nn.functional.embedding(input_ids, weight)

    @property
    def text_vocab_size(self):
        """Return the size of the text tokenizer."""
        return self.embed_tokens.num_embeddings

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_pad_id(self) -> int:
        pad_id = self.tokenizer.pad
        if pad_id is None:
            pad_id = self.tokenizer.unk_id
        if pad_id is None:
            warnings.warn(
                "the text tokenizer has no <pad> or <unk> tokens available, using id 0 for padding (this may lead to silent bugs)."
            )
            pad_id = 0
        return pad_id

    @property
    def audio_locator_tag_id(self) -> int:
        return self.tokenizer.token_to_id(self.audio_locator_tag)

    def _init_speaker_token_ids(self) -> None:
        """Resolve the native ``<spk:N>`` speaker-token ids from the LLM tokenizer.

        Reads ``cfg.speaker_tokens`` with the schema::

            speaker_tokens:
              enable: true
              template: "<spk:{i}>"        # default
              max_speakers: 10
              base_token_id: 100           # expected anchor id for ``<spk:0>``

        The LLM tokenizer is expected to already contain
        ``template.format(i=0)..template.format(i=max_speakers-1)`` as fixed
        entries (e.g. produced by the upstream tokenizer-patch script that
        renames the contiguous reserved slots ``<SPECIAL_100>..<SPECIAL_109>``
        to ``<spk:0>..<spk:9>``). No alias rewrite, no
        ``resize_token_embeddings`` call, and no vocab growth happens here.

        After this call:
            * ``self.speaker_token_ids`` : ``list[int]`` of resolved ids in
              speaker order, empty when ``speaker_tokens`` is absent or
              ``enable: false``.

        Raises:
            ValueError: if any speaker token is missing from the tokenizer or
                its resolved id does not match ``base_token_id + i``, or if
                the act of resolving them grew the tokenizer's vocab size
                (which would indicate an unpatched tokenizer was passed).
        """
        cfg = self.cfg.get("speaker_tokens", None)
        if cfg is None or not bool(cfg.get("enable", True)):
            return
        template = cfg.get("template", "<spk:{i}>")
        max_speakers = int(cfg.get("max_speakers", 10))
        base_token_id = int(cfg.get("base_token_id", 100))

        before = self.tokenizer.vocab_size
        speaker_token_ids: list[int] = []
        for i in range(max_speakers):
            token = template.format(i=i)
            tid = self.tokenizer.token_to_id(token)
            expected = base_token_id + i
            if tid is None:
                raise ValueError(
                    f"Could not resolve speaker token {token!r} in the LLM tokenizer. "
                    "Ensure pretrained_llm points at the patched tokenizer dir "
                    "(e.g. '...-spk/') produced by patch_nano_v3_speaker_tokens.py."
                )
            if tid != expected:
                raise ValueError(
                    f"Speaker token {token!r} resolved to id {tid}, expected "
                    f"{expected} (= base_token_id={base_token_id} + i={i}). The "
                    "tokenizer does not match the configured speaker_tokens layout."
                )
            speaker_token_ids.append(tid)
        after = self.tokenizer.vocab_size
        if before != after:
            raise ValueError(
                f"Resolving speaker tokens grew the tokenizer ({before} -> {after}); "
                "speaker_tokens requires the tokens to already exist in the patched "
                "tokenizer (no resize_token_embeddings on this path)."
            )
        self.speaker_token_ids = speaker_token_ids

    def _init_lss_loss(self) -> None:
        """Optionally build the auxiliary Latent Speaker Supervision (LSS) loss.

        Mirrors the AED Canary recipe at
        ``nemo/collections/asr/models/aed_multitask_models.py`` (search for
        ``self.lss_loss``):

        * The loss is instantiated from ``cfg.lss_loss`` via Hydra ``_target_``,
          identical to ``EncDecMultiTaskModel.from_config_dict(self.cfg.lss_loss)``.
        * ``speaker_token_ids`` is auto-injected from
          ``self.speaker_token_ids`` (resolved by ``_init_speaker_token_ids``
          from the LLM tokenizer's native ``<spk:N>`` entries) when absent in
          YAML.
        * Enable/disable is controlled purely by the *presence* of the
          ``lss_loss:`` block (no ``enable:`` flag) — same as AED.

        SALM uses ``-100`` as the ignore index in ``target_ids`` (HF convention),
        not a tokenizer ``pad_id``. Setting ``pad_id=-100`` lets the loss build
        ``output_mask = (labels != -100)`` internally, again mirroring AED.
        """
        loss_cfg = self.cfg.get("lss_loss", None)
        if loss_cfg is None:
            return
        if loss_cfg.get("include_ce_loss", False):
            raise ValueError(
                "model.lss_loss.include_ce_loss must be False (or omitted) on the SALM "
                "automodel path: SALM already computes CE inside loss_parallel(), so a "
                "second CE term inside LSS would be double-counted."
            )
        if loss_cfg.get("is_rnnt", False):
            raise ValueError(
                "model.lss_loss.is_rnnt must be False (or omitted) on the SALM automodel "
                "path: SALM is AED-style and produces 3D (B, T, V) logits, not the 4D "
                "RNNT joint tensor expected when is_rnnt=True."
            )
        with open_dict(loss_cfg):
            loss_cfg.setdefault("pad_id", -100)
            loss_cfg.setdefault("include_ce_loss", False)
            loss_cfg.setdefault("is_rnnt", False)
            if loss_cfg.get("speaker_token_ids", None) is None:
                if not self.speaker_token_ids:
                    raise ValueError(
                        "model.lss_loss is configured but no speaker_token_ids are available. "
                        "Either set model.speaker_tokens (so ids are derived from the patched "
                        "tokenizer's native <spk:N> entries) or pass an explicit "
                        "model.lss_loss.speaker_token_ids list."
                    )
                loss_cfg.speaker_token_ids = list(self.speaker_token_ids)
        self.lss_loss = instantiate(loss_cfg)

    @property
    def token_equivalent_duration(self) -> float:
        """
        Returns the audio duration corresponding to a single frame/token at the output of ``self.perception``.
        """
        return self.perception.token_equivalent_duration

    @property
    def sampling_rate(self) -> int:
        return self.perception.preprocessor.featurizer.sample_rate

    def forward(
        self,
        input_embeds: Tensor,
        attention_mask: Tensor = None,
        cache=None,
    ) -> dict[str, Tensor]:
        """
        Implements a fully offline forward pass through the entire model.
        The flow is the following:

        |speech and text embeddings| -> |llm| -> |lm_head| -> |token ids|

        """
        # input_embeds and out: (B, T, H)
        out = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=cache is not None,
            return_dict=True,
        )
        if not isinstance(out, dict):
            # NeMo Automodel doesn't respect return_dict=True yet
            ans = {"logits": out}
        else:
            ans = {"logits": out['logits']}  # (B, T, text_vocab_size)
            if cache is not None:
                ans["cache"] = out["past_key_values"]
        return ans

    def prepare_inputs(self, batch: dict, is_inference: bool = False):
        """
        Performs additional processing on the mini-batch collected from dataloader.
        Notably:
        * Convert source audio to speech representations.
        * Convert target audio to target audio tokens.
        * Convert target text to embeddings.
        * Combine the input audio and target text embeddings.
        * Take care of any necessary slicing to align the shapes of source audio,
            target audio, and target token ids.

        ``is_inference`` controls the speaker activity fed to a
        ``ParallelExpertEncoder``. When ``False`` (training), RTTM-derived
        ``batch["spk_targets"]`` are injected as ``diar_preds``. When ``True``
        (validation / real inference), the targets are ignored so the encoder
        runs its embedded Sortformer to predict diarization, matching deployment
        where ground-truth RTTM is unavailable.
        """
        # Source audio encoding.
        # Input audio: (B, T_samples)
        # Audio embeddings: (B, T, H)
        perception_kwargs = {
            "input_signal": batch["audios"],
            "input_signal_length": batch["audio_lens"],
        }
        if not is_inference and batch.get("spk_targets", None) is not None:
            perception_kwargs["diar_preds"] = batch["spk_targets"]
        # TODO(temporary-debug): remove this print. Shows whether RTTM-derived diar_preds are
        # being threaded into the perception encoder during training/validation.
        _dp = perception_kwargs.get("diar_preds", None)
        if _dp is not None:
            print(
                f"\n[DIAR_PREDS_DEBUG] diar_preds PRESENT shape={tuple(_dp.shape)} dtype={_dp.dtype} "
                f"device={_dp.device} sum={_dp.float().sum().item():.1f} "
                f"spk_target_length={batch.get('spk_target_length', None)}",
                flush=True,
            )
        else:
            print(
                f"\n[DIAR_PREDS_DEBUG] diar_preds ABSENT (batch has spk_targets key={'spk_targets' in batch}); "
                "encoder will run embedded Sortformer / plain ASR.",
                flush=True,
            )
        ###################### END TEMPORARY DEBUG ######################
        audio_embs, audio_emb_lens = self.perception(**perception_kwargs)
        audio_embs = [emb[:emblen] for emb, emblen in zip(audio_embs, audio_emb_lens)]
        input_ids_to_embed = torch.where(batch["input_ids"] == self.audio_locator_tag_id, 0, batch["input_ids"])
        text_embs = self._embed_tokens(input_ids_to_embed)
        input_embs, target_ids, attention_mask = replace_placeholders_and_build_targets(
            input_ids=batch["input_ids"],
            embeds=text_embs,
            padding_id=self.text_pad_id,
            placeholder_id=self.audio_locator_tag_id,
            replacements=audio_embs,
            target_ids=batch["input_ids"].where(batch["loss_mask"], -100),  # CrossEntropyLoss().ignore_index
        )
        input_embs = input_embs[:, :-1]
        attention_mask = attention_mask[:, :-1]
        target_ids = target_ids[:, 1:]

        # Combine target audio and text into a single tensor to slice them together.
        # It will also help us truncate the sequence lengths to be divisible by TP world size,
        # when TP is enabled.
        # Input ids: (B, T, K+1)
        if self._use_tp:
            tp_world_size = self.device_mesh["tp"].size()
            if (remainder := (input_embs.shape[1] - 1) % tp_world_size) != 0:
                # Truncate some tokens from the end to make the sequence length shape divisible by tensor parallelism
                # world size. Otherwise, sequence parallelism will change the input shape making leading to mismatches.
                input_embs = input_embs[:, :-remainder]
                attention_mask = attention_mask[:, :-remainder]
                target_ids = target_ids[:, :-remainder]

        return {
            "input_embeds": input_embs,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
        }

    def training_step(self, batch: dict, batch_idx: int):
        self._current_batch_idx = batch_idx
        for m in (self.perception.preprocessor, self.perception.encoder, self.llm):
            if is_frozen(m):
                m.eval()

        inputs = self.prepare_inputs(batch)
        forward_outputs = self(inputs["input_embeds"], attention_mask=inputs["attention_mask"])
        num_frames = (inputs["target_ids"] != -100).long().sum()
        with loss_parallel():
            loss = (
                torch.nn.functional.cross_entropy(
                    forward_outputs["logits"].flatten(0, 1),  # (B, T, Vt) -> (*, Vt)
                    inputs["target_ids"].flatten(0, 1),
                    reduction="sum",
                    ignore_index=-100,
                )
                / num_frames
            )

        # Latent speaker supervision loss (auxiliary, optional).
        # Mirrors aed_multitask_models.py: `transf_loss = transf_loss + lss_loss(...)`.
        # Computed *outside* loss_parallel() because LSS indexes the vocab
        # dimension at specific speaker token ids — not safe on a TP-sharded
        # vocab DTensor; we materialize via .full_tensor() for that path.
        if self.lss_loss is not None and num_frames > 0:
            logits = forward_outputs["logits"]
            if isinstance(logits, DTensor):
                logits = logits.full_tensor()
            log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
            loss = loss + self.lss_loss(log_probs=log_probs, labels=inputs["target_ids"])

        B, T = inputs["input_embeds"].shape[:2]
        ans = {
            "loss": loss,
            "learning_rate": (
                torch.as_tensor(self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0)
            ),
            "batch_size": B,
            "sequence_length": T,
            "num_frames": num_frames.to(torch.float32),  # avoid warning
            "target_to_input_ratio": num_frames / (B * T),
            "padding_ratio": (batch["input_ids"] != self.text_pad_id).long().sum() / batch["input_ids"].numel(),
        }
        self.log("loss", loss, on_step=True, prog_bar=True)
        self.log_dict({k: v for k, v in ans.items() if k != "loss"}, on_step=True)
        self.maybe_log_moe_metrics(batch_idx)
        return ans

    def on_validation_epoch_start(self) -> None:
        self._partial_val_losses = defaultdict(list)
        self._partial_accuracies = defaultdict(list)
        self._partial_val_lss = defaultdict(list)

    def on_validation_epoch_end(self) -> None:
        val_losses = []
        for name, vals in self._partial_val_losses.items():
            val_loss = torch.stack(vals).mean()
            self.log(f"val_loss_{name}", val_loss, on_epoch=True, sync_dist=True)
            val_losses.append(val_loss)
        self.log("val_loss", torch.stack(val_losses).mean(), on_epoch=True, sync_dist=True)

        accuracies = []
        for name, accs in self._partial_accuracies.items():
            val_acc = torch.stack(accs).mean()
            self.log(f"val_acc_{name}", val_acc, on_epoch=True, sync_dist=True)
            accuracies.append(val_acc)
        self.log("val_acc", torch.stack(accuracies).mean(), on_epoch=True, sync_dist=True)

        if self.lss_loss is not None:
            lss_vals = []
            for name, vals in self._partial_val_lss.items():
                val_lss = torch.stack(vals).mean()
                self.log(f"val_lss_{name}", val_lss, on_epoch=True, sync_dist=True)
                lss_vals.append(val_lss)
            if lss_vals:
                self.log("val_lss", torch.stack(lss_vals).mean(), on_epoch=True, sync_dist=True)

        self._partial_val_losses.clear()
        self._partial_accuracies.clear()
        self._partial_val_lss.clear()

    def validation_step(self, batch: dict, batch_idx: int):
        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted
            # Validation mirrors real inference: ignore RTTM ground-truth targets and let the
            # ParallelExpertEncoder run its embedded Sortformer to predict speaker activity.
            inputs = self.prepare_inputs(dataset_batch, is_inference=True)
            forward_outputs = self(inputs["input_embeds"], attention_mask=inputs["attention_mask"])
            num_frames = (inputs["target_ids"] != -100).long().sum()
            with loss_parallel():
                loss = (
                    torch.nn.functional.cross_entropy(
                        forward_outputs["logits"].flatten(0, 1),
                        inputs["target_ids"].flatten(0, 1),
                        reduction="sum",
                        ignore_index=-100,
                    )
                    / num_frames
                )

            if self.lss_loss is not None and num_frames > 0:
                logits = forward_outputs["logits"]
                if isinstance(logits, DTensor):
                    logits = logits.full_tensor()
                log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
                lss_val = self.lss_loss(log_probs=log_probs, labels=inputs["target_ids"])
                self._partial_val_lss[name].append(lss_val.detach())

            preds = forward_outputs["logits"].argmax(dim=-1).view(-1)
            refs = inputs["target_ids"].reshape(-1)
            preds = preds[refs != -100]
            refs = refs[refs != -100]
            accuracy = preds.eq(refs).float().mean()

            self._partial_accuracies[name].append(accuracy)
            self._partial_val_losses[name].append(loss)

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end()

    def test_step(self, *args: Any, **kwargs: Any):
        return self.validation_step(*args, **kwargs)

    def backward(self, *args, **kwargs):
        self._setup_moe_fsdp_sync()
        with loss_parallel():
            super().backward(*args, **kwargs)

    def _setup_moe_fsdp_sync(self):
        """Configure MoE FSDP gradient sync for gradient accumulation.

        When ``accumulate_grad_batches > 1``, disables gradient all-reduce and
        resharding on intermediate backward passes and re-enables them on the
        final backward before ``optimizer.step()``.  This avoids redundant
        communication during gradient accumulation.

        Delegates to the LLM's ``MoEFSDPSyncMixin`` methods.  No-op when the
        LLM lacks the mixin or gradient accumulation is not active.
        """
        if not self._use_fsdp or not hasattr(self.llm, 'prepare_for_grad_accumulation'):
            return
        acc = self.trainer.accumulate_grad_batches if self._trainer else 1
        if acc <= 1:
            return
        batch_idx = getattr(self, '_current_batch_idx', 0)
        is_final = (batch_idx + 1) % acc == 0 or (batch_idx + 1) == self.trainer.num_training_batches
        if is_final:
            self.llm.prepare_for_final_backward()
        else:
            self.llm.prepare_for_grad_accumulation()

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm=None):
        """Override Lightning's gradient clipping to handle mixed FSDP device meshes.

        When automodel parallelizes the LLM, some parameters end up as DTensors
        on the ``(dp_replicate, dp_shard_cp)`` mesh while others may be on the
        flattened ``dp`` mesh.  PyTorch's ``clip_grad_norm_`` requires all norms
        to share the same mesh for ``torch.stack``.  We delegate to automodel's
        mesh-aware ``_clip_grad_norm_impl`` which groups parameters by
        ``(mesh_id, placements)`` and combines per-group norms as plain tensors.
        """
        if not self._use_fsdp or gradient_clip_val is None or gradient_clip_val <= 0:
            return super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)
        from nemo_automodel.components.training.utils import _clip_grad_norm_impl

        params = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
        if params:
            _clip_grad_norm_impl(params, max_norm=gradient_clip_val)

    @torch.no_grad()
    def generate(
        self,
        prompts: list[list[dict[str]]] | torch.Tensor,
        audios: torch.Tensor = None,
        audio_lens: torch.Tensor = None,
        diar_preds: torch.Tensor = None,
        generation_config: GenerationConfig = None,
        enable_thinking: bool | None = None,
        **generation_kwargs,
    ) -> torch.Tensor:
        """
        Generate LLM answers given text or mixed text+audio prompts.

        Example 1. High-level API using ``prompts`` to provide both text and audio::

            >>> answer_ids = model.generate(
            ...    prompts=[
            ...        [
            ...             {
            ...                 "role": "user",
            ...                 "content": f"Transcribe the following: {model.audio_locator_tag}",
            ...                 "audio": ["path/to/audio.wav"],
            ...             }
            ...         ]
            ...    ],
            ...    max_new_tokens=128,
            ... )

        You may also include a ``transformers.GenerationConfig`` object to customize decoding strategy::

            >>> answer_ids = model.generate(..., generation_config=GenerationConfig(do_sample=True, num_beams=5))

        Example 2. Lower-level API, using ``prompts`` for the text part,
        and pre-loaded ``audio`` and ``audio_lens`` tensors::

            >>> answer_ids = model.generate(
            ...    prompts=[
            ...        [{"role": "user", "content": f"Transcribe the following: {model.audio_locator_tag}"}],
            ...        [{"role": "user", "content": f"Transcribe the following in Polish: {model.audio_locator_tag}"}],
            ...    ],
            ...    audios=audios,  # torch.Tensor, float32, of shape (batch, time)
            ...    audio_lens=audio_lens,  # torch.Tensor, int64, of shape (batch,)
            ...    max_new_tokens=128,
            ... )

        Example 3. Lower-level API, using pre-tokenized and pre-formatted ``prompts`` for the text part,
        and pre-loaded ``audio`` and ``audio_lens`` tensors::

            >>> answer_ids = model.generate(
            ...    prompts=prompts,  # torch.Tensor, int64, of shape (batch, num_tokens)
            ...    audios=audios,  # torch.Tensor, float32, of shape (batch, time)
            ...    audio_lens=audio_lens,  # torch.Tensor, int64, of shape (batch,)
            ...    max_new_tokens=128,
            ... )

        Inputs:
            prompts: batch of prompts Tensor or as list[dict] each in the following format
                [
                  # batch example id 0
                  [{"role": "user"}, "slots": {"message": f"Transcribe the following: {model.audio_locator_tag}"}]
                  # batch example id 1
                  [{"role": "user"}, "slots": {"message": f"Transcribe the following in Polish: {model.audio_locator_tag}"}]
                ]
                "role" is LLM-specific, you can pass multiple turns as well.
                If ``prompts`` is a Tensor, we assume it was already formatted in the relevant chat template
                and tokenized with the model's tokenizer.
            audios: Optional. Time-domain audio signal zero-padded batch of shape (B, T).
                The number of audios must correspond to the number of occurrences of <audio_locator_tag> in prompts.
                Each prompt can have multiple audios.
            audio_lens: Optional. Length of each audio example.
            diar_preds: Optional ``(B, T, n_spk)`` speaker-activity tensor (e.g. oracle / RTTM-derived
                diarization) injected into the perception encoder. Only effective when the mounted
                encoder is a ``ParallelExpertEncoder`` (i.e. ``model.pe_encoder_path`` was set); it
                overrides the encoder's embedded Sortformer prediction for this call. When ``None``
                (default), the encoder runs its embedded Sortformer as usual.
            generation_config: Optional HuggingFace GenerationConfig object.
            enable_thinking: Optional prompt-formatter hint forwarded to ``encode_dialog``.
                Relevant for prompt formats that support thinking/reasoning mode.
            generation_kwargs: Keyword arguments passed directly to the underlying LLM's ``generate`` method.
        """
        # Encode prompt dicts into int token ids.
        if isinstance(prompts, torch.Tensor):
            tokens = prompts.to(self.device)
        else:
            if (
                maybe_audio := _resolve_audios_in_prompt(prompts, sampling_rate=self.sampling_rate, device=self.device)
            ) is not None:
                assert (
                    audios is None and audio_lens is None
                ), "Audios cannot be provided via ``prompts`` and ``audios``/``audio_lens`` arguments simultaneously."
                audios, audio_lens = maybe_audio
            formatter = PromptFormatter.resolve(self.cfg.prompt_format)(self.tokenizer)
            formatter_kwargs = {}
            if enable_thinking is not None:
                formatter_kwargs["enable_thinking"] = enable_thinking
            tokens = left_collate_vectors(
                [formatter.encode_dialog(turns=prompt, **formatter_kwargs)["input_ids"] for prompt in prompts],
                padding_value=self.text_pad_id,
            ).to(self.device)
        if generation_config is None:
            generation_config = GenerationConfig(
                bos_token_id=self.text_bos_id,
                eos_token_id=self.text_eos_id,
                pad_token_id=self.text_pad_id,
            )
        if audios is not None:
            # Audio + text input for generation.
            # Prepare token embeddings and audio embeddings.
            tokens_to_embed = tokens.where(tokens != self.audio_locator_tag_id, 0)
            token_embeds = self._embed_tokens(tokens_to_embed)
            # TODO: temporary workaround to perform batch_size=1 inference for audio encoder
            #   due to accuracy issues at bs>1
            perception_kwargs = {"input_signal": audios, "input_signal_length": audio_lens}
            if diar_preds is not None:
                perception_kwargs["diar_preds"] = diar_preds
            audio_embeds, audio_embed_lens = self.perception(**perception_kwargs)
            audio_embeds = [audio_embeds[i, :elen] for i, elen in enumerate(audio_embed_lens)]
            # Insert audio embeddings into relevant positions in text embeddings.
            input_embeds, _, attention_mask = replace_placeholders_and_build_targets(
                input_ids=tokens,
                embeds=token_embeds,
                padding_id=self.text_pad_id,
                placeholder_id=self.audio_locator_tag_id,
                replacements=audio_embeds,
                target_ids=None,
            )
            answer_tokens = self.llm.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                **generation_kwargs,
                generation_config=generation_config,
            )
        else:
            # Text-only generation — embed_tokens stays in LLM, HF generate uses it natively.
            attention_mask = tokens != self.text_pad_id
            answer_tokens = self.llm.generate(
                input_ids=tokens,
                attention_mask=attention_mask,
                **generation_kwargs,
                generation_config=generation_config,
            )
        return answer_tokens

    def setup_moe_options(self):
        """Apply MoE config overrides and enable load balance tracking.

        Must be called after ``self.llm`` is created.  Iterates over all Gate
        modules in the LLM and overrides their settings.  Also enables
        load balance tracking when ``moe_metrics.enabled`` is set.

        Safe no-op when the LLM has no Gate modules (non-MoE backbone).
        """
        from nemo_automodel.components.moe.layers import Gate

        aux_loss_coeff = self.cfg.get("aux_loss_coeff", 0.0)
        if aux_loss_coeff > 0:
            for module in self.llm.modules():
                if isinstance(module, Gate):
                    module.aux_loss_coeff = aux_loss_coeff

        train_gate = self.cfg.get("train_gate", False)
        if train_gate:
            for module in self.llm.modules():
                if isinstance(module, Gate):
                    module.train_gate = True
                    module.weight.requires_grad_(True)
                    if module.bias is not None:
                        module.bias.requires_grad_(True)

        moe_metrics_cfg = self.cfg.get("moe_metrics", None)
        if moe_metrics_cfg is not None and moe_metrics_cfg.get("enabled", False):
            from nemo_automodel.components.moe.load_balance_metrics import enable_load_balance_tracking

            enable_load_balance_tracking(self.llm)

    def maybe_disable_mamba_fast_kernels(self):
        """Route Nemotron Mamba blocks away from the fused mamba-ssm Triton path.

        Some container/kernel combinations hit OOM or illegal memory access in
        ``mamba_split_conv1d_scan_combined``. The Nemotron implementation checks
        a module-level ``is_fast_path_available`` flag, so disabling that flag in
        each loaded Mamba layer module forces its PyTorch fallback.
        """
        if not self.cfg.get("disable_mamba_fast_kernels", False):
            return

        disabled_modules = set()
        mamba_layers = 0
        for module in self.llm.modules():
            if hasattr(module, "cuda_kernels_forward") and hasattr(module, "torch_forward"):
                mamba_layers += 1
                layer_module = import_module(module.__class__.__module__)
                if hasattr(layer_module, "is_fast_path_available"):
                    layer_module.is_fast_path_available = False
                    disabled_modules.add(module.__class__.__module__)
                if hasattr(module, "config"):
                    try:
                        module.config.use_mamba_kernels = False
                    except AttributeError:
                        pass

        logging.info(
            "Disabled Mamba fast kernels for %s layer(s) across %s module(s).",
            mamba_layers,
            len(disabled_modules),
        )

    def maybe_log_moe_metrics(self, step: int):
        """Collect and log MoE load balance metrics.

        All ranks must call this method (the all-reduce inside
        ``collect_expert_loads`` is collective).  Metrics are logged via
        Lightning's ``self.log_dict`` which respects ``log_every_n_steps``.

        Args:
            step: Current ``batch_idx``, used to decide brief vs detailed mode.
        """
        moe_metrics_cfg = self.cfg.get("moe_metrics", None)
        if moe_metrics_cfg is None or not moe_metrics_cfg.get("enabled", False):
            return

        from nemo_automodel.components.moe.load_balance_metrics import (
            collect_expert_loads,
            compute_brief_metrics,
            compute_detailed_metrics,
        )

        dp_group = self._get_moe_dp_group()
        layer_loads = collect_expert_loads(self.llm, dp_group=dp_group)
        if not layer_loads:
            return

        mode = moe_metrics_cfg.get("mode", "brief")
        top_k = moe_metrics_cfg.get("top_k_experts", 5)

        if mode == "detailed":
            detailed_every = moe_metrics_cfg.get("detailed_every_steps", None)
            if detailed_every is not None and step % detailed_every != 0:
                metrics = compute_brief_metrics(layer_loads, top_k=top_k)
            else:
                metrics = compute_detailed_metrics(layer_loads, top_k=top_k)
        else:
            metrics = compute_brief_metrics(layer_loads, top_k=top_k)

        self.log_dict(metrics, on_step=True)

    def _get_moe_dp_group(self):
        """Return the DP process group for MoE metrics all-reduce.

        Mirrors Automodel's ``_get_dp_group(include_cp=True)`` pattern: prefers
        the ``dp_cp`` submesh (includes context parallelism) for the broadest
        reduction, falling back to ``dp``.

        Returns ``None`` when no device mesh is available (e.g. DDP training),
        causing ``collect_expert_loads`` to skip all-reduce (rank-local view).
        """
        device_mesh = getattr(self, "_device_mesh", None)
        if device_mesh is None:
            return None
        dim_names = device_mesh.mesh_dim_names
        if "dp_cp" in dim_names:
            return device_mesh["dp_cp"].get_group()
        if "dp" in dim_names:
            return device_mesh["dp"].get_group()
        return None

    def configure_optimizers(self):
        return configure_optimizers(self)

    def configure_model(
        self,
        device_mesh=None,
        distributed_config=None,
        moe_config=None,
        moe_mesh=None,
    ) -> None:
        # Use provided device_mesh, or fall back to LightningModule property
        if device_mesh is not None:
            self._device_mesh = device_mesh
        else:
            device_mesh = self.device_mesh

        # Derive dtype from trainer precision (e.g. "bf16-flash" -> bfloat16).
        dtype = torch.float32
        if self._trainer is not None:
            precision = str(self._trainer.precision)
            if "bf16" in precision:
                dtype = torch.bfloat16
            elif "16" in precision:
                dtype = torch.float16
        elif hasattr(self.cfg, 'torch_dtype') and self.cfg.torch_dtype is not None:
            td = self.cfg.torch_dtype
            dtype = getattr(torch, td) if isinstance(td, str) else td

        # Fall back to trainer.strategy for configs (Lightning training path)
        if distributed_config is None and self._trainer is not None:
            distributed_config = getattr(self._trainer.strategy, "distributed_config", None)
        if moe_mesh is None and self._trainer is not None:
            moe_mesh = getattr(self._trainer.strategy, "moe_mesh", None)
        if moe_config is None and self._trainer is not None:
            moe_config = getattr(self._trainer.strategy, "moe_config", None)

        automodel_kwargs = {}
        if device_mesh is not None:
            automodel_kwargs["device_mesh"] = device_mesh
            # automodel's instantiate_infrastructure unconditionally calls
            # .to_dict() on these configs, so we must always provide defaults.
            if distributed_config is None:
                from nemo_automodel.components.distributed.config import FSDP2Config

                distributed_config = FSDP2Config()
            if moe_config is None:
                from nemo_automodel.components.moe.config import MoEParallelizerConfig

                moe_config = MoEParallelizerConfig()
            automodel_kwargs["distributed_config"] = distributed_config
            automodel_kwargs["moe_config"] = moe_config
        if moe_mesh is not None:
            automodel_kwargs["moe_mesh"] = moe_mesh

        # When LoRA is configured and we have a device_mesh, pass peft_config
        # through automodel so LoRA is applied before FSDP2 sharding (handles
        # meta-device init correctly).
        peft_config = make_peft_config(self.cfg.lora) if "lora" in self.cfg else None
        if peft_config is not None and device_mesh is not None:
            automodel_kwargs["peft_config"] = peft_config

        # Pass compile_config through to automodel for torch.compile support.
        compile_cfg = self.cfg.get("compile", None)
        if compile_cfg is not None:
            from nemo_automodel.components.utils.compile_utils import CompileConfig

            compile_dict = dict(compile_cfg)
            automodel_kwargs["compile_config"] = CompileConfig(**compile_dict)

        pretrained_weights = self.cfg.get("pretrained_weights", True)
        pretrained_llm_weights = self.cfg.get("pretrained_llm_weights", pretrained_weights)
        pretrained_asr_weights = self.cfg.get("pretrained_asr_weights", pretrained_weights)

        self.llm = load_pretrained_automodel_llm(
            self.cfg.pretrained_llm,
            pretrained_weights=pretrained_llm_weights,
            dtype=dtype,
            trust_remote_code=self.cfg.get("trust_remote_code", False),
            **automodel_kwargs,
        )
        self.maybe_disable_mamba_fast_kernels()

        # Apply MoE options (aux_loss_coeff override, load balance tracking)
        self.setup_moe_options()

        # Create perception module (must happen after LLM so output_dim matches)
        setup_speech_encoder(self, pretrained_weights=pretrained_asr_weights)

        # Optionally replace the ASR encoder with a ParallelExpertEncoder bundle
        # before dtype casting / FSDP wrapping.
        setup_parallel_expert_encoder(self)

        # Fix projection dim for pretrained_weights=False (config output_dim may not match LLM)
        update_perception_output_dim(self)

        # Apply LoRA adapters to the LLM.
        # When device_mesh is set, LoRA was already applied inside automodel's
        # from_pretrained (before sharding).  Otherwise, apply it now.
        if peft_config is not None and device_mesh is None:
            maybe_install_lora(self)
        elif peft_config is not None:
            # LoRA was applied by automodel; still need to ensure the
            # prevent_freeze_params pattern is set for configure_optimizers.
            ensure_lora_trainable(self)

        if device_mesh is None:
            maybe_load_pretrained_models(self)
            return

        # Cast perception to training dtype BEFORE FSDP2 wrapping.
        # The LLM is already in the target dtype (loaded via torch_dtype=dtype).
        # FSDP2 requires uniform parameter dtype, so we cast all parameters.
        if dtype != torch.float32:
            self.perception.to(dtype=dtype)

        if device_mesh["tp"].size() > 1:
            self._use_tp = True

        # Use the same FSDP mesh as automodel uses for the LLM so that
        # gradient clipping can torch.stack norms from all parameters.
        dim_names = device_mesh.mesh_dim_names
        if "dp_replicate" in dim_names and "dp_shard_cp" in dim_names:
            fsdp_mesh = device_mesh["dp_replicate", "dp_shard_cp"]
        elif "dp_shard_cp" in dim_names:
            fsdp_mesh = device_mesh["dp_shard_cp"]
        else:
            fsdp_mesh = device_mesh["dp"]

        if fsdp_mesh.size() > 1:
            self._use_fsdp = True
            self.perception = fully_shard(self.perception, mesh=fsdp_mesh)

        # Enable MoE FSDP gradient accumulation optimization.
        # The MoEFSDPSyncMixin on the LLM defers gradient sync/resharding on
        # intermediate backward passes — _setup_moe_fsdp_sync() drives it.
        # TODO(pzelasko): causes issue in torch's FSDP backward, investigate later:
        # AttributeError: 'FSDPParam' object has no attribute '_unsharded_param'. Did you mean: 'unsharded_param'?
        # if self._use_fsdp and hasattr(self.llm, 'prepare_for_grad_accumulation'):
        #     self.llm.backend.enable_fsdp_optimizations = True

        # Optionally initialize weights from a previous training checkpoint
        # (fresh optimizer/scheduler). Must happen after FSDP wrapping so that
        # DCP loading can fill DTensor parameters with correct shards.
        maybe_load_pretrained_models(self)

    @property
    def oomptimizer_schema(self) -> dict:
        """
        Return a typing schema for optimal batch size calibration for various
        sequence lengths using OOMptimizer.
        """
        return {
            "cls": dict,
            "inputs": [
                {"name": "audios", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {
                    "name": "input_ids",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": self.text_vocab_size,
                },
                {"name": "loss_mask", "type": NeuralType(("B", "T"), MaskType()), "seq_length": "output"},
            ],
        }
