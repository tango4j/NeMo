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

"""
Parallel Expert Speech Encoder.

A wrapper that runs a Sortformer speaker-diarization expert and an ASR
Conformer encoder on the same mel-spectrogram input, then fuses their
outputs (LayerNorm + sinusoidal speaker-kernel + ADD).

The wrapper expects **un-normalised** mel features (Sortformer's native
feature format). Internally, the ASR branch applies ``normalize_batch``
(e.g. ``per_feature``) so the ASR ConformerEncoder sees the normalised
features it was trained on. The Sortformer branch consumes the original
un-normalised mels directly via ``frontend_encoder`` / ``forward_infer``.

    un-normalised mel features (B, n_mels, T)
        |                                        |
        |                                        +--normalize_batch--> normalised mels
        |                                                                    |
        +--> Sortformer.frontend_encoder                                     |
                -> Sortformer.forward_infer         ASR ConformerEncoder <---+
                           |                                 |
                           v                                 v
                  diar_preds (B, T_diar, n_spk)    asr_enc_states (B, T_asr, D)
                           |                                 |
                           +---- LN + sinusoidal kernel -----+
                                          + ADD
                                            |
                                            v
                                    fused encoded (B, D, T_asr)

I/O is identical to :class:`ConformerEncoder` so this is a drop-in
replacement.

Only self-contained PE bundles are supported: their ``model_config.yaml``
must carry inline ``asr_encoder_cfg`` and ``diarization_model_cfg`` blobs.
External ``asr_model_path`` / ``diar_model_path`` source paths are *not*
consulted at load time.
"""

import math
import os
import tarfile
from collections import OrderedDict
from typing import List, Optional, Union

import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf
from torch import nn

from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
from nemo.collections.asr.parts.preprocessing.features import normalize_batch
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    BoolType,
    ChannelType,
    LengthsType,
    NeuralType,
    ProbsType,
    SpectrogramType,
)
from nemo.utils import logging

__all__ = [
    'ParallelExpertEncoder',
    'ParallelExpertEncoderPT',
    'is_parallel_expert_encoder_nemo',
    'load_parallel_expert_encoder_from_nemo',
    'save_parallel_expert_encoder_to_nemo',
]


def _clone_config(config: Optional[DictConfig]) -> Optional[DictConfig]:
    """Deep-copy a ``DictConfig`` without resolving interpolations.

    ``from_config_dict`` mutates the input cfg in place (it pops ``_target_``
    and friends), so the caller hands the sub-target builders their own copy
    to keep ``self._cfg`` immutable and reusable for save-back.
    """
    if config is None:
        return None
    return OmegaConf.create(OmegaConf.to_container(config, resolve=False))


class ParallelExpertEncoder(NeuralModule, StreamingEncoder):
    """Sortformer-diarizer + ASR Conformer encoder with I/O identical to
    :class:`ConformerEncoder`.

    Reconstructed entirely from inline configs carried in the PE bundle's
    ``model_config.yaml``. External ``.nemo`` source paths are not supported.

    Args:
        asr_encoder_cfg: Inline config for the ASR-side :class:`ConformerEncoder`.
        diarization_model_cfg: Inline config for the
            :class:`SortformerEncLabelModel`.
        asr_normalize_type: Normalization replayed on the ASR branch.
            Defaults to ``per_feature``.
        freeze_diar: Freeze the Sortformer parameters. Defaults to ``True``.
        freeze_asr: Freeze the wrapped ASR ConformerEncoder. Defaults to
            ``False``.
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
                "cache_last_channel": NeuralType(('D', 'B', 'T', 'D'), ChannelType(), optional=True),
                "cache_last_time": NeuralType(('D', 'B', 'D', 'T'), ChannelType(), optional=True),
                "cache_last_channel_len": NeuralType(tuple('B'), LengthsType(), optional=True),
                "bypass_pre_encode": NeuralType(tuple(), BoolType(), optional=True),
                "diar_preds": NeuralType(('B', 'T', 'C'), ProbsType(), optional=True),
            }
        )

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return OrderedDict(
            {
                "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
                "cache_last_channel_next": NeuralType(('D', 'B', 'T', 'D'), ChannelType(), optional=True),
                "cache_last_time_next": NeuralType(('D', 'B', 'D', 'T'), ChannelType(), optional=True),
                "cache_last_channel_next_len": NeuralType(tuple('B'), LengthsType(), optional=True),
            }
        )

    def __init__(
        self,
        asr_encoder_cfg: DictConfig,
        diarization_model_cfg: DictConfig,
        asr_normalize_type: Optional[str] = None,
        freeze_diar: bool = True,
        freeze_asr: bool = False,
    ):
        super().__init__()

        # Lazy import to break a circular dependency: this module lives
        # under nemo.collections.asr.modules, while SortformerEncLabelModel
        # itself imports from asr.modules.
        from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel

        if asr_encoder_cfg is None or diarization_model_cfg is None:
            raise ValueError(
                "ParallelExpertEncoder requires both `asr_encoder_cfg` and "
                "`diarization_model_cfg`; self-contained PE bundles supply "
                "these inline in their model_config.yaml."
            )

        self.asr_encoder = ConformerEncoder.from_config_dict(_clone_config(asr_encoder_cfg))
        if not isinstance(self.asr_encoder, ConformerEncoder):
            raise TypeError(
                f"Expected `asr_encoder_cfg._target_` to instantiate a "
                f"ConformerEncoder, got {type(self.asr_encoder).__name__} instead."
            )
        self.asr_normalize_type = asr_normalize_type or 'per_feature'
        self._feat_in = self.asr_encoder._feat_in

        self.diarization_model = SortformerEncLabelModel.from_config_dict(
            _clone_config(diarization_model_cfg)
        )

        self.freeze_diar = freeze_diar
        self.freeze_asr = freeze_asr
        self.n_spk = int(self.diarization_model.sortformer_modules.n_spk)
        self.asr_d_model = self.asr_encoder.d_model

        self.asr_norm = nn.LayerNorm(self.asr_d_model)
        self.diar_norm = nn.LayerNorm(self.n_spk)
        self.register_buffer(
            "diar_kernel",
            self._build_sinusoid_position_encoding(self.n_spk, self.asr_d_model),
            persistent=False,
        )

        if self.freeze_diar:
            self.diarization_model.eval()
            for p in self.diarization_model.parameters():
                p.requires_grad = False
        if self.freeze_asr:
            self.asr_encoder.eval()
            for p in self.asr_encoder.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    # ConformerEncoder-compatible properties (delegated). Required by
    # downstream code (e.g. SALM perception, RTASR streaming) that treats
    # this as a drop-in for ConformerEncoder.
    # ------------------------------------------------------------------
    @property
    def d_model(self) -> int:
        return self.asr_d_model

    @property
    def subsampling_factor(self) -> int:
        return self.asr_encoder.subsampling_factor

    @property
    def pre_encode(self):
        return self.asr_encoder.pre_encode

    @property
    def att_context_size(self):
        return self.asr_encoder.att_context_size

    @att_context_size.setter
    def att_context_size(self, value):
        self.asr_encoder.att_context_size = value

    @property
    def streaming_cfg(self):
        return self.asr_encoder.streaming_cfg

    def setup_streaming_params(self, *args, **kwargs):
        return self.asr_encoder.setup_streaming_params(*args, **kwargs)

    def get_initial_cache_state(self, *args, **kwargs):
        return self.asr_encoder.get_initial_cache_state(*args, **kwargs)

    def change_attention_model(self, *args, **kwargs):
        return self.asr_encoder.change_attention_model(*args, **kwargs)

    def change_subsampling_conv_chunking_factor(self, *args, **kwargs):
        return self.asr_encoder.change_subsampling_conv_chunking_factor(*args, **kwargs)

    def set_default_att_context_size(self, *args, **kwargs):
        return self.asr_encoder.set_default_att_context_size(*args, **kwargs)

    def enable_pad_mask(self, on: bool = True):
        return self.asr_encoder.enable_pad_mask(on=on)

    # ------------------------------------------------------------------
    # Fusion helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_sinusoid_position_encoding(max_position: int, embedding_dim: int) -> torch.Tensor:
        """Mirror of ``MSEncDecMultiTaskModel.get_sinusoid_position_encoding``."""
        position = torch.arange(max_position, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / embedding_dim)
        )
        pe = torch.zeros(max_position, embedding_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    @staticmethod
    def _align_diar_frames(diar_preds: torch.Tensor, target_len: int) -> torch.Tensor:
        """Pad-by-repeat or truncate ``diar_preds`` along the time axis.

        Identical semantics to ``MSEncDecMultiTaskModel._align_diar_frames``.
        """
        cur_len = diar_preds.shape[1]
        if cur_len < target_len:
            last = diar_preds[:, -1:, :]
            diar_preds = torch.cat([diar_preds, last.repeat(1, target_len - cur_len, 1)], dim=1)
        elif cur_len > target_len:
            diar_preds = diar_preds[:, :target_len, :]
        return diar_preds

    # ------------------------------------------------------------------
    # Forward — identical signature & docstring to ConformerEncoder.forward
    # ------------------------------------------------------------------
    @typecheck()
    def forward(
        self,
        audio_signal,
        length,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
        bypass_pre_encode=False,
        diar_preds=None,
    ):
        """
        Forward function for the ConformerEncoder accepting an audio signal and its corresponding length.
        The ``audio_signal`` input supports two formats depending on ``bypass_pre_encode``:

        - ``bypass_pre_encode=False`` (default): ``audio_signal`` must be a tensor
          containing audio features. Shape: ``(batch, feat_in, n_frames)``.
        - ``bypass_pre_encode=True``: ``audio_signal`` must be a tensor containing
          pre-encoded embeddings. Shape: ``(batch, n_frame, d_model)``.

        ``diar_preds`` is an optional ``(B, T, n_spk)`` speaker-activity tensor that
        overrides the internal Sortformer prediction. Use it to feed RTTM ground
        truth during training, or oracle diarization at inference. When ``None``
        (default), the wrapped Sortformer is run as usual.
        """
        if diar_preds is None and not bypass_pre_encode:
            with torch.set_grad_enabled(not self.freeze_diar):
                emb_seq, emb_seq_length = self.diarization_model.frontend_encoder(
                    processed_signal=audio_signal,
                    processed_signal_length=length,
                    bypass_pre_encode=False,
                )
                diar_preds = self.diarization_model.forward_infer(
                    emb_seq=emb_seq, emb_seq_length=emb_seq_length,
                )

        if not bypass_pre_encode and self.asr_normalize_type:
            asr_audio_signal, _, _ = normalize_batch(
                audio_signal, length, normalize_type=self.asr_normalize_type,
            )
        else:
            asr_audio_signal = audio_signal

        with torch.set_grad_enabled(not self.freeze_asr):
            asr_out = self.asr_encoder(
                audio_signal=asr_audio_signal,
                length=length,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                bypass_pre_encode=bypass_pre_encode,
            )

        if cache_last_channel is None:
            asr_encoded, asr_encoded_len = asr_out
            cache_outputs = ()
        else:
            asr_encoded, asr_encoded_len, *cache_outputs = asr_out

        if diar_preds is not None:
            asr_enc_states = asr_encoded.transpose(1, 2)  # (B, T, D)
            diar_preds = self._align_diar_frames(diar_preds, asr_enc_states.shape[1]).to(asr_enc_states.dtype)

            asr_enc_states = self.asr_norm(asr_enc_states)
            diar_preds = self.diar_norm(diar_preds)
            speaker_infusion = torch.matmul(diar_preds, self.diar_kernel.to(diar_preds.dtype))
            fused = speaker_infusion + asr_enc_states

            outputs = fused.transpose(1, 2)  # (B, D, T)
        else:
            outputs = asr_encoded

        if not cache_outputs:
            return outputs, asr_encoded_len
        return (outputs, asr_encoded_len, *cache_outputs)


class ParallelExpertEncoderPT(ModelPT):
    """Lightweight :class:`~nemo.core.classes.modelPT.ModelPT` shell so a
    :class:`ParallelExpertEncoder` can be read back with ``restore_from``
    as a ``.nemo`` archive. Bundle's ``model_config.yaml`` must carry
    inline ``asr_encoder_cfg`` + ``diarization_model_cfg``.
    """

    def __init__(self, cfg: DictConfig, trainer: Optional[Trainer] = None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.encoder = ParallelExpertEncoder(
            asr_encoder_cfg=self._cfg.get('asr_encoder_cfg', None),
            diarization_model_cfg=self._cfg.get('diarization_model_cfg', None),
            asr_normalize_type=self._cfg.get('asr_normalize_type', None),
            freeze_diar=self._cfg.get('freeze_diar', True),
            freeze_asr=self._cfg.get('freeze_asr', False),
        )

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []

    def setup_training_data(self, train_data_config: Union[DictConfig, dict]):
        pass

    def setup_validation_data(self, val_data_config: Union[DictConfig, dict]):
        pass


def is_parallel_expert_encoder_nemo(nemo_path: str) -> bool:
    """Cheaply detect whether a ``.nemo`` archive is a :class:`ParallelExpertEncoderPT` bundle.

    Reads only the archive's ``model_config.yaml`` and inspects its ``target:``
    field. Returns ``False`` for any non-existent path, non-``.nemo`` file,
    unreadable archive, missing config member, or config whose ``target:``
    does not end with ``ParallelExpertEncoderPT``.
    """
    if not (isinstance(nemo_path, str) and nemo_path.endswith('.nemo') and os.path.isfile(nemo_path)):
        return False
    try:
        with tarfile.open(nemo_path, mode='r') as tf:
            for member in tf.getmembers():
                if os.path.basename(member.name) == 'model_config.yaml':
                    fobj = tf.extractfile(member)
                    if fobj is None:
                        return False
                    cfg = OmegaConf.create(fobj.read().decode('utf-8'))
                    return str(cfg.get('target', '')).endswith('ParallelExpertEncoderPT')
    except (tarfile.TarError, OSError) as exc:
        logging.warning("[ParallelExpertEncoder] Could not inspect %s: %s", nemo_path, exc)
        return False
    return False


def load_parallel_expert_encoder_from_nemo(
    nemo_path: str,
    *,
    map_location: Union[str, torch.device] = 'cpu',
    strict: bool = True,
) -> ParallelExpertEncoder:
    """Load a self-contained :class:`ParallelExpertEncoderPT` ``.nemo`` bundle.

    The bundle's ``model_config.yaml`` must carry inline ``asr_encoder_cfg`` and
    ``diarization_model_cfg``. No external source ``.nemo`` files are consulted.
    """
    bundle = ParallelExpertEncoderPT.restore_from(
        restore_path=nemo_path, map_location=map_location, strict=strict,
    )
    return bundle.encoder


def save_parallel_expert_encoder_to_nemo(
    encoder: ParallelExpertEncoder,
    output_nemo_path: str,
    *,
    template_bundle_path: str,
) -> None:
    """Save ``encoder`` as a self-contained PE ``.nemo`` whose ``model_config.yaml``
    is reused verbatim from ``template_bundle_path``.

    Intended for the "load + train + save" round-trip: after the PE encoder has
    been mounted inside a bigger model (e.g. ``model.perception.encoder`` in
    SALM) and its weights have been updated, write them back to disk as a new
    PE bundle that any caller can reload with
    :func:`load_parallel_expert_encoder_from_nemo`.

    The template bundle's inline ``asr_encoder_cfg`` / ``diarization_model_cfg``
    must describe the same architecture as ``encoder``; this is checked on a
    few key invariants (``d_model``, ``n_spk``) and a mismatch raises
    :class:`ValueError` fail-fast at save time rather than producing an
    unreloadable bundle.

    Args:
        encoder: The :class:`ParallelExpertEncoder` whose current weights
            should be persisted.
        output_nemo_path: Destination ``.nemo`` path.
        template_bundle_path: Existing self-contained PE ``.nemo`` whose
            ``model_config.yaml`` is reused.
    """
    if not isinstance(encoder, ParallelExpertEncoder):
        raise TypeError(
            f"save_parallel_expert_encoder_to_nemo expects a ParallelExpertEncoder, "
            f"got {type(encoder).__name__}"
        )
    if not os.path.isfile(template_bundle_path):
        raise FileNotFoundError(
            f"template_bundle_path does not exist: {template_bundle_path}"
        )

    template_cfg: Optional[DictConfig] = None
    with tarfile.open(template_bundle_path, mode='r') as tf:
        for member in tf.getmembers():
            if os.path.basename(member.name) == 'model_config.yaml':
                fobj = tf.extractfile(member)
                if fobj is not None:
                    template_cfg = OmegaConf.create(fobj.read().decode('utf-8'))
                break
    if template_cfg is None:
        raise RuntimeError(
            f"Could not read 'model_config.yaml' from template bundle: {template_bundle_path}"
        )

    tmpl_asr = template_cfg.get('asr_encoder_cfg', None)
    tmpl_diar = template_cfg.get('diarization_model_cfg', None)
    if tmpl_asr in (None, {}, '') or tmpl_diar in (None, {}, ''):
        raise ValueError(
            f"Template bundle {template_bundle_path} is not self-contained "
            "(asr_encoder_cfg / diarization_model_cfg missing); it cannot be "
            "used as a save template."
        )

    tmpl_d_model = int(tmpl_asr.get('d_model', -1))
    tmpl_n_spk = int(
        tmpl_diar.get('sortformer_modules', {}).get('num_spks', -1)
    )
    enc_d_model = int(encoder.d_model)
    enc_n_spk = int(encoder.n_spk)
    if tmpl_d_model != enc_d_model:
        raise ValueError(
            f"Template asr_encoder_cfg.d_model={tmpl_d_model} does not match "
            f"encoder.d_model={enc_d_model}; the saved bundle would fail "
            "strict reload."
        )
    if tmpl_n_spk != enc_n_spk:
        raise ValueError(
            f"Template diarization_model_cfg.sortformer_modules.num_spks="
            f"{tmpl_n_spk} does not match encoder.n_spk={enc_n_spk}; the "
            "saved bundle would fail strict reload."
        )

    # Build a fresh PT shell from the template config so NeMo's save_to
    # machinery (model_config.yaml + model_weights.ckpt tarballing) is
    # inherited for free; immediately swap in the trained encoder before
    # saving. The fresh inner encoder is discarded once `shell.encoder`
    # is reassigned.
    shell = ParallelExpertEncoderPT(cfg=template_cfg, trainer=None)
    shell.encoder = encoder
    # Pin `_cfg` back to the verbatim template so any normalisation that
    # `ModelPT.__init__` may have applied to `self._cfg` does not leak into
    # the saved `model_config.yaml`. Guarantees byte-level round-trip of
    # the architecture metadata (modulo YAML key ordering).
    shell._cfg = template_cfg

    shell.save_to(output_nemo_path)
    logging.info(
        "[ParallelExpertEncoder] Saved PE bundle to %s using template config from %s",
        output_nemo_path, template_bundle_path,
    )
