# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline encoder-free Speech-PED using SALMAutomodel's conversations and loss.

This is the correctness reference, not a head-packed execution implementation.
The frozen Qwen backbone retains its native Automodel grouped-GEMM MoE backend.
"""

import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
from transformers import masking_utils
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor, ConvASRDecoder
from nemo.collections.common.prompts import PromptFormatter
from nemo.collections.speechlm2.data.salm_dataset import left_collate_vectors
from nemo.collections.speechlm2.models.salm import (
    _resolve_audios_in_prompt,
    replace_placeholders_and_build_targets,
)
from nemo.collections.speechlm2.models.salm_automodel import SALMAutomodel
from nemo.collections.speechlm2.modules.parallel_expert_decoder import QwenSpeechBridge
from nemo.collections.speechlm2.modules.qwen_ped_frontend import (
    OfflineAudioSubsampler,
    PairedAudioProjection,
    QwenPEDSpec,
    SpeechTaskTokenEmbeddings,
)
from nemo.collections.speechlm2.parts.pretrained import load_pretrained_automodel_llm


def _assert_native_qwen_backend(model):
    from nemo_automodel.components.models.qwen3_5_moe.model import Qwen3_5MoeForCausalLM

    # FSDP changes __class__ to a torch-owned subclass, retaining the native
    # implementation in its MRO. A module-name check rejects valid FSDP roots.
    if not isinstance(model, Qwen3_5MoeForCausalLM):
        lineage = [f"{cls.__module__}.{cls.__name__}" for cls in type(model).__mro__]
        raise RuntimeError(
            f"Native Qwen3.5-MoE backend required; refusing a silent HF fallback. Actual model MRO: {lineage}"
        )


def _audio_spans(input_ids, lengths, pad_id, audio_id, expanded_width):
    """Locate each expanded audio prefix inside SALM's left-padded sequence."""
    if lengths.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=input_ids.device)
        return empty, empty
    valid = input_ids != pad_id
    placeholders = input_ids == audio_id
    if not torch.all(placeholders.sum(-1) == 1):
        raise ValueError("CTC requires exactly one audio prefix per example.")
    original_lengths = valid.sum(-1)
    unpadded_positions = placeholders.long().argmax(-1) - (~valid).sum(-1)
    expanded_lengths = original_lengths - 1 + lengths
    starts = expanded_width + 1 - expanded_lengths + unpadded_positions
    ends = starts + lengths
    if torch.any(starts < 0) or torch.any(ends > expanded_width):
        raise ValueError("Audio span lies outside the shifted SALM inputs.")
    return starts, ends


def _assistant_transcripts(batch):
    """Return unformatted ASR references rather than chat-formatted CE tokens."""
    transcripts = []
    for conversation in batch.get("conversations", []):
        turns = [turn.value for turn in conversation.turns if turn.role == "assistant" and hasattr(turn, "value")]
        if len(turns) != 1:
            raise ValueError("This ASR CTC recipe requires exactly one assistant transcript per conversation.")
        transcripts.append(turns[0])
    return transcripts


def _ctc_loss_sum(hidden, head, loss_fn, starts, ends, targets):
    """Return the sum of per-example length-normalized NeMo CTC losses."""
    if not targets:
        return hidden.sum() * 0.0, hidden.new_zeros((), dtype=torch.long)
    lengths = ends - starts
    minimum = [len(row) + sum(a == b for a, b in zip(row, row[1:])) for row in targets]
    if any(value == 0 for value in minimum) or any(int(frames) < needed for frames, needed in zip(lengths, minimum)):
        raise ValueError("CTC alignment impossible: audio frames must cover targets plus adjacent repeats.")
    sequences = [hidden[index, int(start) : int(end)] for index, (start, end) in enumerate(zip(starts, ends))]
    padded = pad_sequence(sequences, batch_first=True).transpose(1, 2)
    # The FSDP root owns this call. Its BF16 convolution is followed by an
    # explicit FP32 log-probability boundary for CUDA CTC stability.
    log_probs = head(encoder_output=padded).float()
    target_ids = pad_sequence(
        [torch.tensor(row, device=hidden.device, dtype=torch.long) for row in targets],
        batch_first=True,
    )
    target_lengths = torch.tensor([len(row) for row in targets], device=hidden.device, dtype=torch.long)
    losses = loss_fn(
        log_probs=log_probs,
        targets=target_ids,
        input_lengths=lengths,
        target_lengths=target_lengths,
    )
    # Match torch.nn.CTCLoss(reduction='mean'): normalize every utterance by
    # its target length, then average examples globally across the DP group.
    return (losses / target_lengths).sum(), target_lengths.new_tensor(len(targets))


class QwenPEDAutomodel(SALMAutomodel):
    """SALM input/loss conventions with two aligned, one-way-coupled decoders.

    Only BSHD, CP=TP=PP=1 is admitted initially. Do not enable THD packing until
    both DeltaNet branches have tested document-boundary reset semantics.
    """

    def __init__(self, cfg):
        if cfg.get("init_configure_model", False):
            raise ValueError("PED requires deferred configure_model after speech-token validation/distributed setup.")
        super().__init__(cfg)
        self._ped_special_token_manifest = None
        if manifest_path := self.cfg.get("speech_special_tokens_file"):
            manifest = json.loads(Path(manifest_path).read_text())
            # SALM adds <|audio|>; restore the full speech special-token list
            # without replacing the original Qwen tokenizer on disk.
            self.tokenizer.tokenizer.add_special_tokens(
                {"additional_special_tokens": manifest["preserve_special_tokens"]},
                False,
            )
            for row in manifest["tokens"]:
                if self.tokenizer.text_to_ids(row["content"]) != [row["id"]]:
                    raise ValueError(f'Speech tokenizer/manifest ID mismatch: {row["content"]}')
            self._ped_special_token_manifest = manifest

    def configure_model(self, distributed_setup=None, **kwargs):
        if self.llm is not None:
            return
        if self.cfg.get("packed_sequences") or self.cfg.get("packed_encoder_sequences"):
            raise ValueError("PED THD document-boundary isolation is not validated; use BSHD.")
        distributed_setup = distributed_setup or getattr(self, "_distributed_setup", None)
        if distributed_setup is not None:
            context = distributed_setup.mesh_context
            if any(getattr(context, name) > 1 for name in ("tp_size", "pp_size", "cp_size")):
                raise ValueError("Initial PED supports TP=PP=CP=1 only.")
            self._device_mesh = context.device_mesh
            self._moe_mesh = context.moe_mesh

        from nemo_automodel.components.models.common import BackendConfig
        from nemo_automodel.components.utils.model_utils import parse_freeze_config

        backend = BackendConfig(**OmegaConf.to_container(self.cfg.automodel_backend, resolve=True))
        if backend.experts != "gmm":
            raise ValueError("The frozen Qwen backbone must retain native grouped-GEMM experts.")
        load_kwargs = {
            "backend": backend,
            "freeze_config": parse_freeze_config({"freeze_modules": [{"glob": "*"}]}),
            # The source advertises an auxiliary MTP depth; PED does not train MTP.
            "num_nextn_predict_layers": 0,
        }
        if distributed_setup is not None:
            load_kwargs["distributed_setup"] = distributed_setup
        self.llm = load_pretrained_automodel_llm(self.cfg.pretrained_llm, dtype=torch.bfloat16, **load_kwargs)
        if self.llm.config.model_type != "qwen3_5_moe_text":
            raise ValueError("Use the verified text-only extraction of Qwen3.5-35B-A3B, not its vision wrapper.")
        _assert_native_qwen_backend(self.llm)
        self.llm.requires_grad_(False).eval()
        if getattr(self.llm, "mtp", None) is not None:
            raise RuntimeError("PED requires MTP disabled; do not initialize an untrained auxiliary text head.")
        spec = QwenPEDSpec(**dict(self.cfg.get("speech_decoder", {})))
        self.speech_branch = _SpeechBranch(self.llm.config, spec)
        if self.tokenizer.vocab_size > self.speech_branch.decoder.config.vocab_size:
            # Only the trainable speech head/table may grow. Never resize Qwen.
            self.speech_branch.decoder.resize_token_embeddings(
                self.tokenizer.vocab_size, pad_to_multiple_of=128, mean_resizing=False
            )
        if self._ped_special_token_manifest is not None:
            weight = self.llm.model.embed_tokens.weight
            if isinstance(weight, DTensor):
                weight = weight.full_tensor()
            self.task_tokens = SpeechTaskTokenEmbeddings(self._ped_special_token_manifest["tokens"], weight)
        self.perception = _EncoderFreePerception(spec)
        ctc_cfg = self.cfg.get("ctc", {})
        if bool(ctc_cfg.get("enabled", False)):
            weight = float(ctc_cfg.get("weight", 0.5))
            ce_weight = float(ctc_cfg.get("ce_weight", 0.5))
            if weight != 0.5 or ce_weight != 0.5:
                raise ValueError("The production PED recipe requires exactly 0.5 CTC + 0.5 CE.")
            ctc_vocab_size = int(self.tokenizer.vocab_size)
            self.ctc_head = ConvASRDecoder(feat_in=spec.hidden_size, num_classes=ctc_vocab_size)
            self.ctc_loss_fn = CTCLoss(num_classes=ctc_vocab_size, zero_infinity=False, reduction="none")
            self._ctc_vocab_size = ctc_vocab_size
        if distributed_setup is not None:
            mesh = self._device_mesh["dp"]
            policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
            for layer in self.speech_branch.decoder.model.layers:
                fully_shard(layer, mesh=mesh, mp_policy=policy)
            fully_shard(self.speech_branch, mesh=mesh, mp_policy=policy)
            if getattr(self, "ctc_head", None) is not None:
                fully_shard(self.ctc_head, mesh=mesh, mp_policy=policy)
            # Independent root: this runs before the speech decoder's forward,
            # so it must not initialize an FSDP child before its parent root.
            if getattr(self, "task_tokens", None) is not None:
                fully_shard(self.task_tokens, mesh=mesh, mp_policy=policy)
            # Preserve waveform precision for STFT; only learned parameters use BF16 compute.
            perception_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                cast_forward_inputs=False,
            )
            fully_shard(self.perception, mesh=mesh, mp_policy=perception_policy)
            self._use_fsdp = mesh.size() > 1
        self.assert_frozen_qwen()

    def prepare_inputs(self, batch, *, include_mtp_inputs=True):
        if getattr(self, "ctc_head", None) is None:
            return super().prepare_inputs(batch, include_mtp_inputs=include_mtp_inputs)
        observed_lengths = []

        def capture_lengths(module, args, output):
            observed_lengths.append(output[1].detach())

        handle = self.perception.register_forward_hook(capture_lengths)
        try:
            inputs = super().prepare_inputs(batch, include_mtp_inputs=include_mtp_inputs)
        finally:
            handle.remove()
        if len(observed_lengths) != 1:
            raise ValueError("Expected one whole-utterance perception call for CTC.")
        transcripts = _assistant_transcripts(batch)
        if transcripts:
            self._ctc_spans = _audio_spans(
                batch["input_ids"],
                observed_lengths[0],
                self.text_pad_id,
                self.audio_locator_tag_id,
                inputs["input_embeds"].shape[1],
            )
            self._ctc_targets = [self.tokenizer.text_to_ids(text) for text in transcripts]
            if any(any(token < 0 or token >= self._ctc_vocab_size for token in row) for row in self._ctc_targets):
                raise ValueError("A CTC target ID lies outside the configured Qwen tokenizer vocabulary.")
        else:
            empty = torch.empty(0, dtype=torch.long, device=inputs["input_embeds"].device)
            self._ctc_spans = (empty, empty)
            self._ctc_targets = []
        return inputs

    def _embed_tokens(self, input_ids):
        text_weight = self.llm.model.embed_tokens.weight
        speech_weight = self.speech_branch.decoder.model.embed_tokens.weight
        if isinstance(text_weight, DTensor):
            text_weight = text_weight.full_tensor()
        if isinstance(speech_weight, DTensor):
            from nemo_automodel.components.loss.linear_ce_base import LinearCrossEntropy

            speech_weight = LinearCrossEntropy.materialize_lm_weight(
                speech_weight, grad_reduce_group=self._get_moe_dp_group()
            )
        task_tokens = getattr(self, "task_tokens", None)
        safe_ids = task_tokens.sanitize_ids(input_ids) if task_tokens is not None else input_ids
        text = nn.functional.embedding(safe_ids, text_weight)
        if task_tokens is not None:
            text = task_tokens(input_ids, text, grad_reduce_group=self._get_moe_dp_group())
        speech = nn.functional.embedding(input_ids, speech_weight)
        return torch.cat((text, speech.to(text.dtype)), dim=-1)

    def forward(
        self,
        input_embeds,
        attention_mask=None,
        cache=None,
        logits_to_keep=0,
        **llm_kwargs,
    ):
        if cache is not None or llm_kwargs:
            raise ValueError("Initial offline PED forward does not accept caches or packed metadata.")
        text, speech = input_embeds.split((2048, 1024), dim=-1)
        text_states = {}
        layers = self.llm.model.layers
        with ExitStack() as hooks:
            for index in range(3, 40, 4):
                layer = layers[str(index)] if isinstance(layers, nn.ModuleDict) else layers[index]

                def capture(module, args, output, layer_index=index):
                    text_states[layer_index] = output[0] if isinstance(output, tuple) else output

                handle = layer.register_forward_hook(capture)
                hooks.callback(handle.remove)
            # Do not use no_grad/detach: audio-to-Qwen projections need this gradient.
            # Call the owning module, not .model: FSDP's root gather hooks live here.
            self.llm(
                inputs_embeds=text,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
            )
        states = tuple(text_states[index] for index in range(3, 40, 4))
        hidden = self.speech_branch(speech, attention_mask, states)
        if self.training and self._fused_linear_cross_entropy is not None:
            return {"hidden_states": hidden}
        weight = self.speech_branch.decoder.lm_head.weight
        if isinstance(weight, DTensor):
            weight = weight.full_tensor()
        if not isinstance(logits_to_keep, int) or logits_to_keep < 0:
            raise ValueError("logits_to_keep must be a nonnegative integer.")
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        return {"logits": nn.functional.linear(hidden, weight.to(hidden.dtype))}

    def _compute_training_cross_entropy_sum(self, forward_outputs, target_ids, dp_group, *, lm_weight=None):
        if self._fused_linear_cross_entropy is not None:
            hidden = forward_outputs["hidden_states"]
            if lm_weight is None:
                lm_weight = self._fused_linear_cross_entropy.materialize_lm_weight(
                    self.speech_branch.decoder.lm_head.weight,
                    grad_reduce_group=dp_group,
                )
            # FSDP stores FP32 master weights after resharding. CCE needs BF16
            # operands; cast only the gathered compute copy, preserving gradients.
            lm_weight = lm_weight.to(hidden.dtype)
            ce_sum = self._fused_linear_cross_entropy(
                hidden,
                target_ids,
                lm_weight,
                grad_reduce_group=dp_group,
            )
            logits = None
        else:
            ce_sum, logits = super()._compute_training_cross_entropy_sum(forward_outputs, target_ids, dp_group)
        if getattr(self, "ctc_head", None) is None:
            return ce_sum, logits

        ctc_sum, ctc_count = _ctc_loss_sum(
            forward_outputs["hidden_states"],
            self.ctc_head,
            self.ctc_loss_fn,
            *self._ctc_spans,
            self._ctc_targets,
        )
        del self._ctc_spans, self._ctc_targets
        ce_count = (target_ids != -100).long().sum()
        global_ce_count = ce_count.clone()
        global_ctc_count = ctc_count.clone()
        display_ce_sum = ce_sum.detach().float().clone()
        display_ctc_sum = ctc_sum.detach().float().clone()
        if dp_group is not None and dist.is_available() and dist.is_initialized():
            for value in (
                global_ce_count,
                global_ctc_count,
                display_ce_sum,
                display_ctc_sum,
            ):
                dist.all_reduce(value, op=dist.ReduceOp.SUM, group=dp_group)
        global_ce_count = global_ce_count.clamp_min(1)
        global_ctc_count = global_ctc_count.clamp_min(1)
        # SALM later multiplies the returned sum by dp_size/global_ce_count.
        # Rescale the CTC term here so FSDP's gradient averaging yields a true
        # global example mean, while CE remains a true global token mean.
        combined_sum = 0.5 * ce_sum + 0.5 * ctc_sum * (global_ce_count / global_ctc_count)
        ce_display = display_ce_sum / global_ce_count
        ctc_display = display_ctc_sum / global_ctc_count
        total_display = 0.5 * ce_display + 0.5 * ctc_display
        if not torch.isfinite(total_display):
            raise FloatingPointError("Non-finite CE/CTC loss; refusing a corrupted optimizer update.")
        self._training_loss_display_override = total_display
        self.log_dict(
            {
                "train/ce_loss": ce_display,
                "train/ctc_loss": ctc_display,
                "train/total_loss": total_display,
            },
            on_step=True,
            batch_size=target_ids.shape[0],
        )
        return combined_sum, logits

    def assert_frozen_qwen(self):
        if any(p.requires_grad or p.grad is not None for p in self.llm.parameters()):
            raise RuntimeError("Qwen must remain entirely frozen, including embeddings, routers and output head.")

    def train(self, mode=True):
        super().train(mode)
        # Lightning switches the whole PED back to train after validation.
        # Keep dropout/router state updates disabled in the original Qwen.
        if self.llm is not None:
            self.llm.eval()
        return self

    def configure_optimizers(self):
        result = super().configure_optimizers()
        self.assert_frozen_qwen()
        teacher_ids = {id(parameter) for parameter in self.llm.parameters()}
        optimized_ids = {id(parameter) for group in result["optimizer"].param_groups for parameter in group["params"]}
        if teacher_ids & optimized_ids:
            raise RuntimeError("The optimizer must not contain any frozen Qwen parameter.")
        return result

    def on_before_optimizer_step(self, optimizer):
        self.assert_frozen_qwen()

    def _frozen_weight_digest(self):
        """Hash local frozen shards, then combine the fixed-topology rank digests."""
        digest = hashlib.sha256()
        tensors = list(self.llm.named_parameters())
        for prefix, module in self.llm.named_modules():
            for name, buffer in module._buffers.items():
                if buffer is not None and name not in module._non_persistent_buffers_set:
                    tensors.append((f"{prefix}.{name}", buffer))
        for name, parameter in tensors:
            local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
            digest.update(name.encode())
            digest.update(str((tuple(parameter.shape), parameter.dtype)).encode())
            digest.update(local.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        shards = [digest.hexdigest()]
        if dist.is_available() and dist.is_initialized():
            shards = [None] * dist.get_world_size()
            dist.all_gather_object(shards, digest.hexdigest())
        return hashlib.sha256(json.dumps(shards).encode()).hexdigest()

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        self._expected_frozen_digest = checkpoint.get("ped_frozen_qwen_sha256")

    def on_fit_start(self):
        super().on_fit_start()
        current = self._frozen_weight_digest()
        expected = getattr(self, "_expected_frozen_digest", None)
        if expected is not None and current != expected:
            raise RuntimeError("Frozen Qwen weights differ from the resumed checkpoint invariant.")
        self._ped_frozen_digest = current

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        self.assert_frozen_qwen()
        current = self._frozen_weight_digest()
        if current != self._ped_frozen_digest:
            raise RuntimeError("Frozen Qwen tensor bytes changed during speech training.")
        checkpoint["ped_frozen_qwen_sha256"] = current

    def forward_text(self, **kwargs):
        """Unmodified original model path; no speech modules participate."""
        return self.llm(**kwargs)

    def get_text_tokenizer(self):
        """Original tokenizer for text-only requests, never the PED task tokenizer."""
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.cfg.pretrained_llm, local_files_only=True)

    def decode_speech(self, token_ids):
        """Retain speaker/event/style tags while removing chat-control tokens."""
        tokens = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else list(token_ids)
        manifest = getattr(self, "_ped_special_token_manifest", None)
        retained = {row["id"] for row in manifest["tokens"]} if manifest is not None else set()
        controls = set(self.tokenizer.tokenizer.all_special_ids) - retained
        return self.tokenizer.ids_to_text(
            [token for token in tokens if token not in controls],
            remove_special_tokens=False,
        )

    @torch.inference_mode()
    def generate(self, prompts, audios=None, audio_lens=None, generation_config=None, **kwargs):
        """Offline greedy reference decoding using the speech output head.

        Native Automodel Qwen3.5 currently has no recurrent/KV-cache API. This
        recomputes the full prefix for WER correctness, not latency measurement.
        """
        if self.training:
            raise RuntimeError("Call eval() before speech generation.")
        limit = kwargs.pop("max_new_tokens", None)
        do_sample = kwargs.pop("do_sample", False)
        if kwargs or do_sample or getattr(generation_config, "do_sample", False):
            raise ValueError("Reference PED generation supports greedy decoding and max_new_tokens only.")
        if not isinstance(prompts, torch.Tensor):
            resolved = _resolve_audios_in_prompt(prompts, sampling_rate=self.sampling_rate, device=self.device)
            if resolved is not None:
                if audios is not None or audio_lens is not None:
                    raise ValueError("Supply audio through prompts or tensors, not both.")
                audios, audio_lens = resolved
            formatter = PromptFormatter.resolve(self.cfg.prompt_format)(self.tokenizer)
            prompts = left_collate_vectors(
                [formatter.encode_dialog(turns=prompt, enable_thinking=False)["input_ids"] for prompt in prompts],
                padding_value=self.text_pad_id,
            )
        if audios is None or audio_lens is None:
            raise ValueError("Speech generation requires audio. Use the original Qwen route for text-only.")
        ids = prompts.to(self.device)
        audio, lengths = self.perception(
            input_signal=audios.to(self.device),
            input_signal_length=audio_lens.to(self.device),
        )
        replacements = [row[:length] for row, length in zip(audio, lengths)]
        embeddings = self._embed_tokens(ids.masked_fill(ids == self.audio_locator_tag_id, 0))
        prefix, _, mask = replace_placeholders_and_build_targets(
            input_ids=ids,
            embeds=embeddings,
            padding_id=self.text_pad_id,
            placeholder_id=self.audio_locator_tag_id,
            replacements=replacements,
        )
        if limit is None:
            limit = getattr(generation_config, "max_new_tokens", None) or 256
        if limit < 1:
            raise ValueError("max_new_tokens must be positive.")
        eos = getattr(generation_config, "eos_token_id", None)
        if eos is None:
            eos = self.text_eos_id
        eos_ids = torch.tensor([eos] if isinstance(eos, int) else eos, device=ids.device)
        return self._greedy_from_prefix(prefix, mask, limit, eos_ids)

    def _greedy_from_prefix(self, prefix, mask, limit, eos_ids):
        finished = torch.zeros(prefix.shape[0], dtype=torch.bool, device=prefix.device)
        generated = []
        for _ in range(limit):
            logits = self(prefix, attention_mask=mask, logits_to_keep=1)["logits"][:, -1]
            # Config vocabularies may be padded beyond the tokenizer's actual rows.
            valid_size = min(
                getattr(self.tokenizer, "vocab_size", logits.shape[-1]),
                logits.shape[-1],
            )
            logits[:, valid_size:] = -torch.inf
            if getattr(self, "audio_locator_tag", None) is not None:
                locator = self.audio_locator_tag_id
                if 0 <= locator < logits.shape[-1]:
                    logits[:, locator] = -torch.inf
            token = logits.argmax(-1)
            token = token.masked_fill(finished, self.text_pad_id)
            generated.append(token)
            prefix = torch.cat((prefix, self._embed_tokens(token[:, None])), dim=1)
            mask = torch.cat((mask, (~finished)[:, None]), dim=1)
            finished |= torch.isin(token, eos_ids)
            if finished.all():
                break
        return torch.stack(generated, dim=1)


class _SpeechBranch(nn.Module):
    def __init__(self, text_config, spec):
        super().__init__()
        self.spec = spec
        config = spec.speech_config(text_config)
        config._attn_implementation = "sdpa"
        self.decoder = Qwen3_5ForCausalLM(config)
        self.bridges = nn.ModuleList([QwenSpeechBridge(2048, spec.hidden_size) for _ in range(10)])
        for bridge in self.bridges:
            # FSDP cannot shard zero-dimensional parameters. Keep the exact
            # one-value gate computation without changing the shared 2B bridge.
            bridge.gate = nn.Parameter(bridge.gate.detach().reshape(1))

    def forward(self, hidden, attention_mask, text_states):
        backbone = self.decoder.model
        positions = torch.arange(hidden.shape[1], device=hidden.device)[None].expand(hidden.shape[0], -1)
        rotary = backbone.rotary_emb(hidden, positions[None].expand(3, -1, -1))
        mask_args = dict(
            config=backbone.config,
            inputs_embeds=hidden,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=positions,
        )
        full_mask = masking_utils.create_causal_mask(**mask_args)
        recurrent_mask_factory = getattr(masking_utils, "create_recurrent_attention_mask", None)
        linear_mask = (
            recurrent_mask_factory(**mask_args)
            if recurrent_mask_factory is not None
            else backbone._update_linear_attn_mask(attention_mask, None)
        )
        masks = [linear_mask if kind == "linear_attention" else full_mask for kind in backbone.config.layer_types]
        for index, layer in enumerate(self.decoder.model.layers):
            bridge_index = index // 4
            teacher = text_states[bridge_index] if index % 4 == 3 else None

            def run(current, text, layer=layer, mask=masks[index], bridge_index=bridge_index):
                result = layer(
                    current,
                    attention_mask=mask,
                    position_ids=positions,
                    position_embeddings=rotary,
                    use_cache=False,
                )
                return self.bridges[bridge_index](result, text) if text is not None else result

            hidden = checkpoint(run, hidden, teacher, use_reentrant=False) if self.training else run(hidden, teacher)
        return self.decoder.model.norm(hidden)


class _EncoderFreePerception(nn.Module):
    """SALM perception API with only Log-Mel, local subsampling and projections."""

    token_equivalent_duration = 0.08

    def __init__(self, spec):
        super().__init__()
        self.preprocessor = AudioToMelSpectrogramPreprocessor(
            sample_rate=16000,
            features=spec.n_mels,
            window_size=0.025,
            window_stride=0.01,
            n_fft=512,
            normalize=None,
            dither=0.0,
            pad_to=0,
        )
        self.encoder = OfflineAudioSubsampler(spec.n_mels, spec.frontend_width)
        self.proj = PairedAudioProjection(spec.frontend_width)

    def forward(self, input_signal, input_signal_length, **kwargs):
        mel, lengths = self.preprocessor(input_signal=input_signal, length=input_signal_length)
        features, lengths = self.encoder(mel.to(self.encoder.convs[0].weight.dtype), lengths)
        return self.proj(features), lengths
