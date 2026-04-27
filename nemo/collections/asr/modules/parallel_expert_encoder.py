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

The wrapper expects **un-normalised** mel features (i.e. Sortformer's
native feature format). Internally, the ASR branch applies
``normalize_batch`` (e.g. ``per_feature``) so the ASR ConformerEncoder
sees the normalised features it was trained on. The Sortformer branch
consumes the original un-normalised mels directly via
``frontend_encoder`` / ``forward_infer``.

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
"""

import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import List, Optional, Set, Union

import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf
from torch import nn

from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
from nemo.collections.asr.parts.preprocessing.features import normalize_batch
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import AccessMixin, adapter_mixins
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
    'build_parallel_expert_encoder',
    'export_parallel_expert_encoder_to_nemo',
    'import_parallel_expert_encoder_from_nemo',
]


def _load_pretrained_model(model_path: str, model_cls):
    """Restore a NeMo model from a ``.nemo`` or PyTorch Lightning ``.ckpt`` file."""
    if model_path is None:
        raise ValueError(f"model_path is None for {model_cls.__name__}")
    if model_path.endswith('.nemo'):
        model = model_cls.restore_from(model_path, map_location="cpu")
    elif model_path.endswith('.ckpt'):
        model = model_cls.load_from_checkpoint(model_path, map_location="cpu")
    else:
        raise ValueError(
            f"Unsupported checkpoint extension for {model_path!r}. Expected '.nemo' or '.ckpt'."
        )
    logging.info("[ParallelExpertEncoder] Loaded %s from %s", model_cls.__name__, model_path)
    return model


def _clone_config(config: Optional[DictConfig]) -> Optional[DictConfig]:
    """Deep-copy a config-like object without resolving interpolations.

    Hydra may partially instantiate nested `_target_` configs before they reach
    `ParallelExpertEncoder.__init__`. This helper converts any such module/model
    objects back into plain config containers recursively.
    """
    if config is None:
        return None

    if hasattr(config, '_cfg') and config._cfg is not None:
        config = config._cfg
    elif hasattr(config, 'to_config_dict'):
        config = config.to_config_dict()

    def _to_container(value):
        if OmegaConf.is_config(value):
            return _to_container(OmegaConf.to_container(value, resolve=False))
        if hasattr(value, '_cfg') and value._cfg is not None:
            return _to_container(value._cfg)
        if hasattr(value, 'to_config_dict'):
            return _to_container(value.to_config_dict())
        if isinstance(value, Mapping):
            return {k: _to_container(v) for k, v in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [_to_container(v) for v in value]
        return value

    return OmegaConf.create(_to_container(config))


class ParallelExpertEncoder(NeuralModule, StreamingEncoder, Exportable, AccessMixin):
    """Sortformer-diarizer + ASR Conformer encoder with I/O identical to
    :class:`ConformerEncoder`.

    The wrapper expects **un-normalised** mel features (Sortformer's native
    format). Internally, the ASR branch runs ``normalize_batch`` with the
    normalisation type read from the ASR model's preprocessor (e.g.
    ``per_feature`` for offline FastConformer) so the ASR encoder sees the
    normalised features it was trained on. The Sortformer branch consumes
    the original un-normalised mels directly via ``frontend_encoder`` /
    ``forward_infer``.

    Outputs are fused with the ``LayerNorm + sinusoidal-kernel + ADD``
    recipe from ``MSEncDecMultiTaskModel.forward``
    (``aed_multitask_models.py``).

    It supports two construction modes:

    * Bootstrap from external checkpoints:
      ``asr_model_path`` + ``diar_model_path``.
    * Self-contained reconstruction from inline configs:
      ``asr_encoder_cfg`` + ``diarization_model_cfg``.

    The diarizer is frozen by default to match
    ``MSEncDecMultiTaskModel._init_diar_model``.

    Args:
        asr_model_path: Path to the pretrained :class:`EncDecMultiTaskModel`
            checkpoint.
        diar_model_path: Path to the pretrained Sortformer checkpoint.
        asr_encoder_cfg: Inline config for the ASR-side :class:`ConformerEncoder`.
        diarization_model_cfg: Inline config for the :class:`SortformerEncLabelModel`.
        asr_normalize_type: Normalization replayed on the ASR branch when the
            encoder is reconstructed from ``asr_encoder_cfg``.
        freeze_diar: Freeze the Sortformer parameters. Defaults to ``True``.
        freeze_asr: Freeze the wrapped ASR ConformerEncoder parameters.
            Defaults to ``False``.
    """

    # ------------------------------------------------------------------
    # Type signatures — identical to ConformerEncoder
    # ------------------------------------------------------------------
    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        dev = next(self.parameters()).device
        if self.export_cache_support:
            window_size = max_dim
            if self.streaming_cfg is not None:
                if isinstance(self.streaming_cfg.chunk_size, list):
                    chunk_size = self.streaming_cfg.chunk_size[1]
                else:
                    chunk_size = self.streaming_cfg.chunk_size
                if isinstance(self.streaming_cfg.pre_encode_cache_size, list):
                    pre_encode_cache_size = self.streaming_cfg.pre_encode_cache_size[1]
                else:
                    pre_encode_cache_size = self.streaming_cfg.pre_encode_cache_size
                window_size = chunk_size + pre_encode_cache_size
            input_example = torch.randn(max_batch, self._feat_in, window_size, device=dev)
            input_example_length = torch.randint(
                window_size // 4, window_size, (max_batch,), device=dev, dtype=torch.int64
            )
            cache_last_channel, cache_last_time, cache_last_channel_len = self.get_initial_cache_state(
                batch_size=max_batch, device=dev, max_dim=max_dim
            )
            all_input_example = tuple(
                [
                    input_example,
                    input_example_length,
                    cache_last_channel.transpose(0, 1),
                    cache_last_time.transpose(0, 1),
                    cache_last_channel_len,
                ]
            )
        else:
            input_example = torch.randn(max_batch, self._feat_in, max_dim, device=dev)
            input_example_length = torch.randint(max_dim // 4, max_dim, (max_batch,), device=dev, dtype=torch.int64)
            all_input_example = tuple([input_example, input_example_length])

        return all_input_example

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
    def input_types_for_export(self):
        """Returns definitions of module input ports."""
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
                "cache_last_channel": NeuralType(('B', 'D', 'T', 'D'), ChannelType(), optional=True),
                "cache_last_time": NeuralType(('B', 'D', 'D', 'T'), ChannelType(), optional=True),
                "cache_last_channel_len": NeuralType(tuple('B'), LengthsType(), optional=True),
                "bypass_pre_encode": NeuralType(tuple(), BoolType(), optional=True),
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

    @property
    def output_types_for_export(self):
        """Returns definitions of module output ports."""
        return OrderedDict(
            {
                "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
                "cache_last_channel_next": NeuralType(('B', 'D', 'T', 'D'), ChannelType(), optional=True),
                "cache_last_time_next": NeuralType(('B', 'D', 'D', 'T'), ChannelType(), optional=True),
                "cache_last_channel_next_len": NeuralType(tuple('B'), LengthsType(), optional=True),
            }
        )

    @property
    def disabled_deployment_input_names(self):
        if not self.export_cache_support:
            return set(["cache_last_channel", "cache_last_time", "cache_last_channel_len"])
        else:
            return set()

    @property
    def disabled_deployment_output_names(self):
        if not self.export_cache_support:
            return set(["cache_last_channel_next", "cache_last_time_next", "cache_last_channel_next_len"])
        else:
            return set()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        asr_model_path: Optional[str],
        diar_model_path: Optional[str],
        freeze_diar: bool = True,
        freeze_asr: bool = False,
        asr_encoder_cfg: Optional[DictConfig] = None,
        diarization_model_cfg: Optional[DictConfig] = None,
        asr_normalize_type: Optional[str] = None,
    ):
        super().__init__()

        # Lazy imports to avoid a circular dependency: this module lives under
        # nemo.collections.asr.modules, but the wrapper consumes ModelPT
        # subclasses that themselves import asr.modules.
        from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel
        from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel

        if (asr_model_path is None) == (asr_encoder_cfg is None):
            raise ValueError(
                "Provide exactly one of `asr_model_path` or `asr_encoder_cfg` to ParallelExpertEncoder."
            )
        if (diar_model_path is None) == (diarization_model_cfg is None):
            raise ValueError(
                "Provide exactly one of `diar_model_path` or `diarization_model_cfg` to ParallelExpertEncoder."
            )

        # ------------- ASR Conformer expert -------------
        if asr_encoder_cfg is not None:
            if isinstance(asr_encoder_cfg, ConformerEncoder):
                self.asr_encoder = asr_encoder_cfg
            else:
                self.asr_encoder = EncDecMultiTaskModel.from_config_dict(_clone_config(asr_encoder_cfg))
            if not isinstance(self.asr_encoder, ConformerEncoder):
                raise TypeError(
                    f"Expected `asr_encoder_cfg` to instantiate a ConformerEncoder, "
                    f"got {type(self.asr_encoder)} instead."
                )
            self.asr_normalize_type = asr_normalize_type or 'per_feature'
            self._feat_in = self.asr_encoder._feat_in
        else:
            pretrained_asr_model = _load_pretrained_model(asr_model_path, EncDecMultiTaskModel)
            if not isinstance(pretrained_asr_model.encoder, ConformerEncoder):
                raise TypeError(
                    f"Expected the loaded ASR model to expose a ConformerEncoder under `.encoder`, "
                    f"got {type(pretrained_asr_model.encoder)} instead."
                )
            self.asr_encoder = pretrained_asr_model.encoder
            self.asr_normalize_type = asr_normalize_type or getattr(
                pretrained_asr_model.preprocessor.featurizer, 'normalize', 'per_feature'
            )
            self._feat_in = self.asr_encoder._feat_in
            del pretrained_asr_model

        # ------------- Sortformer diarization expert -------------
        if diarization_model_cfg is not None:
            if isinstance(diarization_model_cfg, SortformerEncLabelModel):
                self.diarization_model = diarization_model_cfg
            else:
                self.diarization_model = SortformerEncLabelModel.from_config_dict(
                    _clone_config(diarization_model_cfg)
                )
        else:
            self.diarization_model = _load_pretrained_model(diar_model_path, SortformerEncLabelModel)

        # ------------- Bookkeeping -------------
        self.freeze_diar = freeze_diar
        self.freeze_asr = freeze_asr
        self.n_spk = int(self.diarization_model.sortformer_modules.n_spk)
        self.asr_d_model = self.asr_encoder.d_model

        # ------------- Fusion layers (LN + sinusoidal kernel + ADD) -------------
        self.asr_norm = nn.LayerNorm(self.asr_d_model)
        self.diar_norm = nn.LayerNorm(self.n_spk)
        self.register_buffer(
            "diar_kernel",
            self._build_sinusoid_position_encoding(self.n_spk, self.asr_d_model),
            persistent=False,
        )

        # ------------- Freeze policy -------------
        if self.freeze_diar:
            self.diarization_model.eval()
            for p in self.diarization_model.parameters():
                p.requires_grad = False
        if self.freeze_asr:
            self.asr_encoder.eval()
            for p in self.asr_encoder.parameters():
                p.requires_grad = False

    def to_config_dict(self) -> DictConfig:
        """Return a self-contained config that can rebuild this encoder without external files."""
        return OmegaConf.create(
            {
                '_target_': 'nemo.collections.asr.modules.parallel_expert_encoder.ParallelExpertEncoder',
                '_recursive_': False,
                'asr_model_path': None,
                'diar_model_path': None,
                'asr_encoder_cfg': OmegaConf.to_container(self.asr_encoder.to_config_dict(), resolve=False),
                'diarization_model_cfg': OmegaConf.to_container(
                    self.diarization_model.to_config_dict(), resolve=False
                ),
                'asr_normalize_type': self.asr_normalize_type,
                'freeze_diar': self.freeze_diar,
                'freeze_asr': self.freeze_asr,
            }
        )

    # ------------------------------------------------------------------
    # ConformerEncoder-compatible properties (delegated)
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

    @property
    def export_cache_support(self) -> bool:
        return self.asr_encoder.export_cache_support

    @export_cache_support.setter
    def export_cache_support(self, value: bool):
        self.asr_encoder.export_cache_support = value

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
        # ---- Diarization source: external override or Sortformer expert ----
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

        # ---- Normalise for the ASR branch ----
        if not bypass_pre_encode and self.asr_normalize_type:
            asr_audio_signal, _, _ = normalize_batch(
                audio_signal, length, normalize_type=self.asr_normalize_type,
            )
        else:
            asr_audio_signal = audio_signal

        # ---- ASR Conformer expert (normalised mels) ----
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

        # ---- Fusion: LN(asr) + sinusoidal_kernel @ LN(diar) + ADD ----
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

    # ------------------------------------------------------------------
    # Adapter mixin plumbing (delegate to inner ASR encoder)
    # ------------------------------------------------------------------
    def add_adapter(self, name: str, cfg: dict):
        if isinstance(self.asr_encoder, adapter_mixins.AdapterModuleMixin):
            return self.asr_encoder.add_adapter(name, cfg)
        raise NotImplementedError("The wrapped ASR encoder does not implement AdapterModuleMixin.")

    def is_adapter_available(self) -> bool:
        if isinstance(self.asr_encoder, adapter_mixins.AdapterModuleMixin):
            return self.asr_encoder.is_adapter_available()
        return False

    def set_enabled_adapters(self, name: Optional[str] = None, enabled: bool = True):
        if isinstance(self.asr_encoder, adapter_mixins.AdapterModuleMixin):
            return self.asr_encoder.set_enabled_adapters(name=name, enabled=enabled)

    def get_enabled_adapters(self) -> List[str]:
        if isinstance(self.asr_encoder, adapter_mixins.AdapterModuleMixin):
            return self.asr_encoder.get_enabled_adapters()
        return []

    def get_accepted_adapter_types(self) -> Set[type]:
        if isinstance(self.asr_encoder, adapter_mixins.AdapterModuleMixin):
            return self.asr_encoder.get_accepted_adapter_types()
        return set()


class ParallelExpertEncoderPT(ModelPT):
    """Lightweight :class:`~nemo.core.classes.modelPT.ModelPT` shell so a
    :class:`ParallelExpertEncoder` can be written with ``save_to`` and read
    back with ``restore_from`` as a ``.nemo`` archive.

    New bundles are saved with a self-contained config (inline ASR encoder and
    Sortformer configs plus the merged state dict), so they can restore without
    touching external `.nemo` files. Older bundles that only recorded
    ``asr_model_path`` / ``diar_model_path`` remain supported for backward
    compatibility.
    """

    def __init__(self, cfg: DictConfig, trainer: Optional[Trainer] = None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.encoder = ParallelExpertEncoder(
            asr_model_path=self._cfg.get('asr_model_path', None),
            diar_model_path=self._cfg.get('diar_model_path', None),
            freeze_diar=self._cfg.get('freeze_diar', True),
            freeze_asr=self._cfg.get('freeze_asr', False),
            asr_encoder_cfg=self._cfg.get('asr_encoder_cfg', None),
            diarization_model_cfg=self._cfg.get('diarization_model_cfg', None),
            asr_normalize_type=self._cfg.get('asr_normalize_type', None),
        )

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []

    def setup_training_data(self, train_data_config: Union[DictConfig, dict]):
        pass

    def setup_validation_data(self, val_data_config: Union[DictConfig, dict]):
        pass

    def to_config_dict(self) -> DictConfig:
        """Persist a self-contained bundle config while keeping `ParallelExpertEncoderPT` as the restore class."""
        cfg = _clone_config(self.encoder.to_config_dict())
        if '_target_' in cfg:
            del cfg['_target_']
        return cfg

    def on_save_checkpoint(self, checkpoint):
        """Keep Lightning `.ckpt` hyperparameters aligned with the self-contained bundle config."""
        checkpoint.setdefault('hyper_parameters', {})
        checkpoint['hyper_parameters']['cfg'] = self.to_config_dict()

    def to_config_file(self, path2yaml_file: str):
        """Serialize the self-contained bundle config instead of the live bootstrap config."""
        cfg = self.to_config_dict()
        with open(path2yaml_file, 'w', encoding='utf-8') as fout:
            OmegaConf.save(config=cfg, f=fout, resolve=True)


def build_parallel_expert_encoder(
    asr_model_path: str,
    diar_model_path: str,
    *,
    freeze_diar: bool = True,
    freeze_asr: bool = False,
) -> ParallelExpertEncoder:
    """Construct a :class:`ParallelExpertEncoder` from two ``.nemo`` (or ``.ckpt``) paths."""
    return ParallelExpertEncoder(
        asr_model_path=asr_model_path,
        diar_model_path=diar_model_path,
        freeze_diar=freeze_diar,
        freeze_asr=freeze_asr,
    )


def export_parallel_expert_encoder_to_nemo(
    asr_model_path: str,
    diar_model_path: str,
    output_nemo_path: str,
    *,
    freeze_diar: bool = True,
    freeze_asr: bool = False,
) -> ParallelExpertEncoder:
    """Load the ASR (Fast)Conformer encoder and Sortformer diarizer, fuse them
    in a :class:`ParallelExpertEncoder`, and save a ``.nemo`` bundle.

    Args:
        asr_model_path: e.g. Canary ``EncDecMultiTaskModel`` ``.nemo`` whose
            ``encoder`` is a :class:`ConformerEncoder`.
        diar_model_path: ``SortformerEncLabelModel`` ``.nemo``.
        output_nemo_path: Destination ``.nemo`` file (parent dirs are created
            by NeMo's ``save_to`` when possible).
        freeze_diar: Passed through to :class:`ParallelExpertEncoder`.
        freeze_asr: Passed through to :class:`ParallelExpertEncoder`.

    Returns:
        The in-memory :class:`ParallelExpertEncoder` that was saved (attribute
        ``encoder`` of the internal :class:`ParallelExpertEncoderPT`).
    """
    cfg = OmegaConf.create(
        {
            'asr_model_path': asr_model_path,
            'diar_model_path': diar_model_path,
            'freeze_diar': freeze_diar,
            'freeze_asr': freeze_asr,
        }
    )
    bundle = ParallelExpertEncoderPT(cfg, trainer=None)
    bundle.save_to(output_nemo_path)
    logging.info("Saved ParallelExpertEncoder bundle to %s", output_nemo_path)
    return bundle.encoder


def import_parallel_expert_encoder_from_nemo(
    nemo_path: str,
    *,
    map_location: Union[str, torch.device] = 'cpu',
    override_config_path: Optional[Union[str, DictConfig]] = None,
    strict: bool = True,
) -> ParallelExpertEncoder:
    """Load a ``.nemo`` archive produced by :func:`export_parallel_expert_encoder_to_nemo`.

    New bundles restore from an inline self-contained config. Older bundles may
    still require the original ``asr_model_path`` / ``diar_model_path``.
    """
    bundle = ParallelExpertEncoderPT.restore_from(
        restore_path=nemo_path,
        map_location=map_location,
        override_config_path=override_config_path,
        strict=strict,
    )
    return bundle.encoder
