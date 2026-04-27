# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Multi-speaker rich-transcription AED model.

:class:`MSEncDecMultiTaskModel` is a thin subclass of
:class:`~nemo.collections.asr.models.aed_multitask_models.EncDecMultiTaskModel`
that delegates all diarization and speaker-infusion logic to
:class:`~nemo.collections.asr.modules.parallel_expert_encoder.ParallelExpertEncoder`.
"""

from functools import partial
from typing import Dict, Optional

import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.asr.data.audio_to_sot_text_lhotse_prompted import (
    PromptedAudioToTextLhotseDataset,
    PromptedAudioToTextMiniBatch,
)
from nemo.collections.asr.models.aed_multitask_models import (
    EncDecMultiTaskModel,
    MultiTaskTranscriptionConfig,
    lens_to_mask,
)
from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import NeuralType, ProbsType
from nemo.utils import logging
from nemo.collections.asr.modules.parallel_expert_encoder import import_parallel_expert_encoder_from_nemo

__all__ = ['MSEncDecMultiTaskModel']


class MSEncDecMultiTaskModel(EncDecMultiTaskModel):
    """Multi-speaker AED multitask model.

    Thin shim around :class:`~nemo.collections.asr.modules.parallel_expert_encoder.ParallelExpertEncoder`,
    which owns all diarization + speaker-infusion logic. This class only:

    1. Swaps ``self.encoder`` for a :class:`ParallelExpertEncoder` bundle loaded
       from ``cfg.pe_encoder_path``.
    2. Routes a ``(B, T, n_spk)`` speaker-activity tensor into the encoder via
       its ``diar_preds`` kwarg, picked as follows:

       * **Training** — RTTM ground truth from ``batch.targets``.
       * **Inference** — RTTM oracle / external-diar matrices pre-loaded with
         :meth:`add_rttms_mask_mats`. When neither is available, the encoder's
         embedded Sortformer is used.
    3. Reports CpWER instead of WER.
    """

    VAL_LOSS_DATALOADER_INDICES = {0, 5}

    @staticmethod
    def _is_parallel_expert_encoder(module) -> bool:
        from nemo.collections.asr.modules.parallel_expert_encoder import ParallelExpertEncoder

        return isinstance(module, ParallelExpertEncoder)

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        self._setup_lss_loss()

        is_restoring = self._is_model_being_restored()
        has_self_contained_encoder = self._is_parallel_expert_encoder(self.encoder)

        if cfg.get('asr_model_path', None) is not None and not is_restoring:
            self._init_asr_model()
        elif cfg.get('asr_model_path', None) is not None and is_restoring:
            logging.info("Skipping `asr_model_path` warm-start during restore; checkpoint weights will be loaded next.")

        if cfg.get('pe_encoder_path', None) is not None and not has_self_contained_encoder:
            self._init_pe_encoder()

        self._rttms_mask_mats = None
        self._rttm_batch_offset = 0

        self._setup_cpwer_metric()

    def to_config_dict(self) -> DictConfig:
        """Save a deployment config that can rebuild the mounted PE encoder directly."""
        cfg = OmegaConf.create(OmegaConf.to_container(super().to_config_dict(), resolve=False))
        if self._is_parallel_expert_encoder(self.encoder):
            with open_dict(cfg):
                cfg.asr_model_path = None
                cfg.pe_encoder_path = None
                cfg.encoder = self.encoder.to_config_dict()
        return cfg

    def on_save_checkpoint(self, checkpoint):
        """Keep Lightning `.ckpt` hyperparameters aligned with the self-contained `.nemo` config."""
        checkpoint.setdefault('hyper_parameters', {})
        checkpoint['hyper_parameters']['cfg'] = self.to_config_dict()

    def to_config_file(self, path2yaml_file: str):
        """Serialize the deployment config instead of the live training bootstrap config."""
        cfg = self.to_config_dict()
        with open(path2yaml_file, 'w', encoding='utf-8') as fout:
            OmegaConf.save(config=cfg, f=fout, resolve=True)

    # ------------------------------------------------------------------
    # Optional latent-speaker-supervision (LSS) auxiliary loss
    # ------------------------------------------------------------------
    def _setup_lss_loss(self):
        """Wire up the optional LSS loss when ``cfg.lss_loss`` is provided.

        The LSS loss biases the decoder toward correct speaker tokens. It is
        mixed with the primary CE loss in :meth:`training_step` /
        :meth:`validation_pass` via ``self.lss_loss_weight``.
        """
        if self.cfg.get('lss_loss', None) is not None:
            with open_dict(self.cfg.lss_loss):
                self.cfg.lss_loss.pad_id = self.tokenizer.pad_id
                if self.cfg.lss_loss.get('speaker_token_ids', None) is None:
                    self.cfg.lss_loss.speaker_token_ids = [
                        self.tokenizer.special_tokens[f"[s{i}]"]
                        for i in range(self.cfg.get('max_num_speakers', 4))
                    ]
            self.lss_loss = MSEncDecMultiTaskModel.from_config_dict(self.cfg.lss_loss)
            self.lss_loss_weight = self.cfg.get('lss_loss_weight', 0.5)
        else:
            self.lss_loss = None
            self.lss_loss_weight = 0.0

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------
    def _init_asr_model(self):
        """Warm-start ``self.encoder`` from a pretrained ``EncDecMultiTaskModel`` checkpoint."""
        model_path = self.cfg.asr_model_path

        if model_path.endswith('.nemo'):
            pretrained_asr_model = EncDecMultiTaskModel.restore_from(model_path, map_location="cpu")
        elif model_path.endswith('.ckpt'):
            pretrained_asr_model = EncDecMultiTaskModel.load_from_checkpoint(model_path, map_location="cpu")
        else:
            logging.warning("Invalid asr_model_path %r — skipping warm-start", model_path)
            return
        logging.info("ASR Model restored locally from %s", model_path)

        self.encoder.load_state_dict(pretrained_asr_model.encoder.state_dict(), strict=True)
        self.encoder_decoder_proj.load_state_dict(
            pretrained_asr_model.encoder_decoder_proj.state_dict(), strict=True,
        )
        if self.use_transf_encoder:
            self.transf_encoder.load_state_dict(
                pretrained_asr_model.transf_encoder.state_dict(), strict=True,
            )

        if self.cfg.get('freeze_asr', False):
            self.encoder.eval()
            self.encoder_decoder_proj.eval()
            if self.use_transf_encoder:
                self.transf_encoder.eval()

    def _init_pe_encoder(self):
        """Replace ``self.encoder`` with a :class:`ParallelExpertEncoder` loaded from ``cfg.pe_encoder_path``.

        The bundle embeds both the ASR Conformer expert and the Sortformer
        diarizer, and exposes the same I/O contract as :class:`ConformerEncoder`
        (plus an optional ``diar_preds`` kwarg for RTTM injection).
        """


        model_path = self.cfg.pe_encoder_path
        if not isinstance(model_path, str) or not model_path.endswith('.nemo'):
            raise ValueError(
                f"cfg.pe_encoder_path must be a path to a .nemo bundle "
                f"(ParallelExpertEncoderPT), got {model_path!r}."
            )

        pe_encoder = import_parallel_expert_encoder_from_nemo(
            model_path, map_location='cpu', strict=True,
        )
        self.encoder = pe_encoder

        asr_hidden = self.cfg.model_defaults.asr_enc_hidden
        if int(pe_encoder.d_model) != int(asr_hidden):
            logging.warning(
                "Loaded parallel-expert encoder d_model=%s differs from "
                "cfg.model_defaults.asr_enc_hidden=%s; ensure encoder_decoder_proj matches.",
                pe_encoder.d_model, asr_hidden,
            )

        if self.cfg.get('freeze_asr', False):
            self.encoder.eval()

        logging.info("Replaced model.encoder with ParallelExpertEncoder bundle from %s", model_path)

    def _setup_cpwer_metric(self):
        """Replace the parent's WER metric with CpWER for multi-speaker evaluation."""
        from nemo.collections.asr.metrics import MultiTaskMetric

        self.metric_cfg = DictConfig(
            {
                "metrics": {
                    "cpwer": {"_target_": "nemo.collections.asr.metrics.cpwer.CpWER"},
                }
            }
        )
        self.metric = MultiTaskMetric(model=self, cfg=self.metric_cfg)

    # ------------------------------------------------------------------
    # Dataloading — inject SOT speaker-target config when configured
    # ------------------------------------------------------------------
    def _build_sot_cfg(self, config: Optional[Dict]) -> Optional[Dict]:
        """Assemble the SOT (multi-speaker) dataset config when ``cfg.max_num_speakers > 0``.

        Returns ``None`` when the model is single-speaker, in which case the dataset
        runs in its default ASR-only mode.
        """
        if self.cfg.get('max_num_speakers', 0) <= 0:
            return None
        if hasattr(self, 'encoder') and self.encoder is not None:
            subsampling_factor = getattr(self.encoder, 'subsampling_factor', 8)
        else:
            subsampling_factor = self.cfg.get('encoder', {}).get('subsampling_factor', 8)
        return {
            'num_speakers': self.cfg.get('max_num_speakers', 4),
            'sample_rate': self.cfg.get('sample_rate', 16000),
            'window_stride': self.cfg.get('preprocessor', {}).get('window_stride', 0.01),
            'subsampling_factor': subsampling_factor,
            'convert_to_wl': config.get('convert_to_wl', False),
            'no_rttm_to_ones': config.get('no_rttm_to_ones', True),
            'randomize_single_speaker_index': config.get('randomize_single_speaker_index', False),
        }

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        """Same as the parent, but threads ``sot_cfg`` into the Lhotse dataset."""
        if not config.get("use_lhotse", False):
            raise ValueError(
                "Multi-task model only supports dataloading with Lhotse. "
                "Please set config.{train,validation,test}_ds.use_lhotse=True"
            )
        global_rank = config.get("global_rank", self.global_rank)
        world_size = config.get("world_size", self.world_size)
        enable_chunking = config.get("enable_chunking", False)
        enable_chunking = enable_chunking and self.timestamps_asr_model is not None

        if enable_chunking:
            config.cut_into_windows_duration = 3600
            config.cut_into_windows_hop = 3600

        return get_lhotse_dataloader_from_config(
            config,
            global_rank=global_rank,
            world_size=world_size,
            dataset=PromptedAudioToTextLhotseDataset(
                tokenizer=self.tokenizer,
                prompt=self.prompt,
                enable_chunking=enable_chunking,
                sot_cfg=self._build_sot_cfg(config),
            ),
            tokenizer=self.tokenizer,
        )

    def _may_be_make_dict_and_fix_paths(self, json_items, manifest_path, trcfg: MultiTaskTranscriptionConfig):
        """Defer to the parent, then back-fill ``lang`` from ``target_lang`` for Lhotse/AggregateTokenizer."""
        out_json_items = super()._may_be_make_dict_and_fix_paths(json_items, manifest_path, trcfg)
        default_turn = [t for t in trcfg.prompt if t["role"] == "user"]
        default_turn = default_turn[0]["slots"] if default_turn else {}
        for entry in out_json_items:
            if 'lang' not in entry:
                entry['lang'] = entry.get('target_lang', default_turn.get('target_lang', 'en'))
        return out_json_items

    # ------------------------------------------------------------------
    # RTTM-based inference hook (used by transcribe_speech_rtmtasr.py)
    # ------------------------------------------------------------------
    def add_rttms_mask_mats(self, rttms_mask_mats, device: torch.device):
        """Stash a pre-computed ``(N, T_max, num_speakers)`` RTTM tensor for inference.

        :meth:`_transcribe_forward` slices the per-batch view out and passes it
        to :meth:`forward` as ``diar_preds`` (skipping the embedded Sortformer).
        """
        if self._rttms_mask_mats is not None:
            raise ValueError(
                f"{self._rttms_mask_mats.shape}: rttms_mask_mats already exist but new one is being added."
            )
        self._rttms_mask_mats = rttms_mask_mats.to(device)
        self._rttm_batch_offset = 0

    def _transcribe_on_begin(self, audio, trcfg):
        super()._transcribe_on_begin(audio, trcfg)
        self._rttm_batch_offset = 0

    def _pop_rttm_batch(self, batch_size: int):
        """Slice the next ``batch_size`` RTTM matrices, or return ``None`` if no oracle was registered."""
        if self._rttms_mask_mats is None:
            return None
        start = self._rttm_batch_offset
        rttm = self._rttms_mask_mats[start : start + batch_size]
        self._rttm_batch_offset = start + batch_size
        return rttm

    def _transcribe_forward(self, batch, trcfg):
        """Inject the per-batch oracle RTTM as ``diar_preds`` to :meth:`forward` for one call.

        The parent's :meth:`_transcribe_forward` calls ``self.forward(input_signal=..., input_signal_length=...)``
        without any diarization kwarg, so we bind ``diar_preds`` for the duration of that single call instead
        of stashing it on ``self`` and reading it back inside :meth:`forward`.
        """
        bs = batch.audio.shape[0] if isinstance(batch, PromptedAudioToTextMiniBatch) else batch[0].shape[0]
        diar_preds = self._pop_rttm_batch(bs)
        if diar_preds is None:
            return super()._transcribe_forward(batch, trcfg)

        original_forward = self.forward
        self.forward = partial(original_forward, diar_preds=diar_preds)
        try:
            return super()._transcribe_forward(batch, trcfg)
        finally:
            self.forward = original_forward

    # ------------------------------------------------------------------
    # Forward — delegate diarization to the parallel-expert encoder
    # ------------------------------------------------------------------
    def _select_preprocessor(self):
        """Return the preprocessor matching the active encoder.

        :class:`ParallelExpertEncoder` consumes Sortformer's native
        **un-normalised** mels and re-normalises internally for its ASR branch,
        so when the PE encoder is mounted we route mels through Sortformer's
        own preprocessor (``self.encoder.diarization_model.preprocessor``).
        Otherwise we fall back to the model's standard ASR preprocessor.
        """
        if hasattr(self.encoder, 'diarization_model'):
            return self.encoder.diarization_model.preprocessor
        return self.preprocessor

    def forward_pe_encoder(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        diar_preds=None,
    ):
        """Run preprocessor + parallel-expert encoder, optionally injecting RTTM oracle ``diar_preds``."""
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if has_input_signal == has_processed_signal:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            preprocessor = self._select_preprocessor()
            processed_signal, processed_signal_length = preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(
            audio_signal=processed_signal, length=processed_signal_length, diar_preds=diar_preds,
        )

        enc_states = encoded.permute(0, 2, 1)
        enc_states = self.encoder_decoder_proj(enc_states)
        enc_mask = lens_to_mask(encoded_len, enc_states.shape[1]).to(enc_states.dtype)
        if self.use_transf_encoder:
            enc_states = self.transf_encoder(encoder_states=enc_states, encoder_mask=enc_mask)

        return enc_states, encoded_len, enc_mask

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        """Extend the parent's input_types with optional ``diar_preds`` (B, T, n_spk)."""
        types = dict(super().input_types)
        types["diar_preds"] = NeuralType(('B', 'T', 'C'), ProbsType(), optional=True)
        return types

    @typecheck()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        transcript=None,
        transcript_length=None,
        diar_preds=None,
    ):
        """Forward pass.

        ``diar_preds`` selects the speaker activity fed to the encoder:

        * Explicit ``(B, T, n_spk)`` tensor → used as-is (RTTM ground truth at
          training time, oracle RTTM at inference time injected by
          :meth:`_transcribe_forward`).
        * ``None`` → the encoder runs its embedded Sortformer.
        """
        with torch.set_grad_enabled(not self.cfg.get('freeze_asr', False)):
            enc_states, encoded_len, enc_mask = self.forward_pe_encoder(
                input_signal=input_signal,
                input_signal_length=input_signal_length,
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
                diar_preds=diar_preds,
            )

        transf_log_probs = None
        if transcript is not None:
            dec_mask = lens_to_mask(transcript_length, transcript.shape[1]).to(transcript.dtype)
            dec_states = self.transf_decoder(
                input_ids=transcript, decoder_mask=dec_mask,
                encoder_embeddings=enc_states, encoder_mask=enc_mask,
            )
            transf_log_probs = self.log_softmax(hidden_states=dec_states)

        return transf_log_probs, encoded_len, enc_states, enc_mask

    # ------------------------------------------------------------------
    # PTL hooks — thread RTTM ground truth from batch.targets via forward kwarg
    # ------------------------------------------------------------------
    def training_step(self, batch: PromptedAudioToTextMiniBatch, batch_nb):
        if batch is None:
            return torch.tensor([0.0])

        input_ids, labels = batch.get_decoder_inputs_outputs()
        input_ids_lens = batch.prompted_transcript_lens - 1

        num_frames = batch.audio_lens.sum().float()
        num_tokens = batch.prompted_transcript_lens.sum().float()
        tot_frames = torch.as_tensor(batch.audio.numel(), device=num_frames.device, dtype=torch.float)
        tot_tokens = torch.as_tensor(batch.prompted_transcript.numel(), device=num_frames.device, dtype=torch.float)

        transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
            input_signal=batch.audio,
            input_signal_length=batch.audio_lens,
            transcript=input_ids,
            transcript_length=input_ids_lens,
            diar_preds=getattr(batch, 'targets', None),
        )

        if self.cfg.get("use_loss_mask_for_prompt", False):
            maxlen = batch.prompted_transcript.shape[1] - 1
            loss_mask = lens_to_mask(input_ids_lens, maxlen) & ~lens_to_mask(batch.prompt_lens - 1, maxlen)
        else:
            loss_mask = None
        transf_loss = self.loss(log_probs=transf_log_probs, labels=labels, output_mask=loss_mask)

        if self.lss_loss is not None:
            lss_loss = self.lss_loss(log_probs=transf_log_probs, labels=labels, output_mask=loss_mask)
            alpha = self.lss_loss_weight
            transf_loss = (1 - alpha) * transf_loss + alpha * lss_loss

        log_every_n_steps = self._trainer.log_every_n_steps if getattr(self, '_trainer', None) is not None else 1
        metric_dict = (
            self.metric.eval(
                batch=batch, predictions=enc_states, predictions_lengths=encoded_len,
                predictions_mask=enc_mask, prefix="training_batch",
            )
            if (batch_nb + 1) % log_every_n_steps == 0
            else {}
        )
        metric_dict.update(
            {
                'train_loss': transf_loss,
                'learning_rate': torch.as_tensor(self._optimizer.param_groups[0]['lr']),
                'batch_size': torch.as_tensor(batch.audio.shape[0]),
                'num_frames': num_frames,
                'num_tokens': num_tokens,
                'input_to_padding_ratio': num_frames / tot_frames,
                'output_to_padding_ratio': num_tokens / tot_tokens,
            }
        )
        return {"loss": transf_loss, "log": metric_dict}

    def validation_pass(self, batch: PromptedAudioToTextMiniBatch, batch_idx, dataloader_idx=0, eval_mode="val"):
        input_ids, labels = batch.get_decoder_inputs_outputs()
        input_ids_lens = batch.prompted_transcript_lens - 1

        transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
            input_signal=batch.audio,
            input_signal_length=batch.audio_lens,
            transcript=input_ids,
            transcript_length=batch.prompted_transcript_lens,
            diar_preds=None,
        )

        if self.cfg.get("use_loss_mask_for_prompt", False):
            maxlen = batch.prompted_transcript.shape[1] - 1
            loss_mask = lens_to_mask(input_ids_lens, maxlen) & ~lens_to_mask(batch.prompt_lens - 1, maxlen)
            num_measurements = loss_mask.long().sum()
        else:
            loss_mask = None
            num_measurements = transf_log_probs.shape[0] * transf_log_probs.shape[1]

        transf_loss = self.loss(log_probs=transf_log_probs, labels=labels, output_mask=loss_mask)

        if self.lss_loss is not None:
            lss_loss = self.lss_loss(log_probs=transf_log_probs, labels=labels, output_mask=loss_mask)
            alpha = self.lss_loss_weight
            transf_loss = (1 - alpha) * transf_loss + alpha * lss_loss

        self.val_loss(loss=transf_loss, num_measurements=num_measurements)

        metric_dict = self.metric.eval(
            batch=batch, predictions=enc_states, predictions_lengths=encoded_len,
            predictions_mask=enc_mask, prefix=eval_mode,
            return_all_metrics=True,
        )
        metric_dict[f"{eval_mode}_loss"] = transf_loss
        return metric_dict

    # ------------------------------------------------------------------
    # Epoch-end aggregation
    # ------------------------------------------------------------------
    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss = {}
        tensorboard_logs = {}

        if 'val_loss' in outputs[0] and dataloader_idx in self.VAL_LOSS_DATALOADER_INDICES:
            val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
            val_loss = {'val_loss': val_loss_mean}
            tensorboard_logs.update(val_loss)

        if "val_cpwer_num" in outputs[0]:
            cpwer_num = torch.stack([x['val_cpwer_num'] for x in outputs]).sum()
            cpwer_denom = torch.stack([x['val_cpwer_denom'] for x in outputs]).sum()
            tensorboard_logs['val_cpwer'] = cpwer_num / cpwer_denom

        return {**val_loss, 'log': tensorboard_logs}

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        test_loss = {}
        tensorboard_logs = {}

        if 'test_loss' in outputs[0]:
            test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
            test_loss = {'test_loss': test_loss_mean}
            tensorboard_logs.update(test_loss)

        if "test_cpwer_num" in outputs[0]:
            cpwer_num = torch.stack([x['test_cpwer_num'] for x in outputs]).sum()
            cpwer_denom = torch.stack([x['test_cpwer_denom'] for x in outputs]).sum()
            tensorboard_logs['test_cpwer'] = cpwer_num / cpwer_denom

        return {**test_loss, 'log': tensorboard_logs}
