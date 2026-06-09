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

import contextlib
import math
import os
import tarfile
from collections import OrderedDict
from typing import List, Optional, Union

import torch
import torch.distributed as dist
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm

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


@contextlib.contextmanager
def _default_dtype(dtype: torch.dtype):
    """Temporarily set the global default floating-point dtype.

    ``SortformerModules.init_streaming_state`` allocates the speaker-cache /
    FIFO state with ``torch.zeros(..., device=device)`` and no explicit dtype,
    so those buffers default to fp32. When the diarizer runs in bf16 the fp32
    state collides with the bf16 chunk embeddings in the streaming conformer
    ("expected scalar type Float but found BFloat16"). Scoping the default dtype
    to the diarizer's dtype makes that state allocate in bf16 without touching
    shared diarization code; explicit ``dtype=torch.long`` length tensors are
    unaffected.
    """
    prev = torch.get_default_dtype()
    if dtype == prev or not dtype.is_floating_point:
        yield
        return
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


@contextlib.contextmanager
def _disable_dist_feature_sync():
    """Temporarily make ``torch.distributed`` look uninitialized.

    ``SortformerEncLabelModel.forward_streaming`` runs a cross-rank
    ``dist.all_reduce`` (guarded by ``dist.is_available() and
    dist.is_initialized()``) to pad input features to a common length across
    data-parallel ranks during DDP training/eval. In single-recording inference
    -- including inside a vLLM worker, where a single-rank NCCL group is already
    initialized for tensor parallelism -- that collective is both unnecessary
    (``max_n_frames == sig_length``) and unsafe (it dispatches on the worker's
    process group and fails on CPU tensors with "No backend type associated with
    device type cpu"). Scoping ``dist.is_initialized`` to ``False`` makes the
    Sortformer streaming loop take the non-distributed branch without touching
    shared diarization code. The original function is always restored.
    """
    if not (hasattr(dist, "is_initialized") and dist.is_initialized()):
        yield
        return
    orig_is_initialized = dist.is_initialized
    dist.is_initialized = lambda: False
    try:
        yield
    finally:
        dist.is_initialized = orig_is_initialized


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
        chunked_inference_length: Chunk length (in *encoder output frames*, i.e.
            post-subsampling 80ms frames) used for long-form inference. At the
            default ``150`` this corresponds to a 12s window (12.5 frames/s x
            12s), kept shorter than the 15-20s clips the model was trained on.
            When ``<= 0`` chunked inference is disabled and the encoder always runs
            the offline path. Chunked inference is also only engaged at *inference*
            time (``self.training is False``) and only when the input is longer than
            one chunk; shorter / training-time inputs use the offline path so
            existing behaviour is unchanged. See :meth:`_forward_chunked`.
        asr_chunk_left_context: Left context (history) attached to each ASR window
            during chunked inference, in *encoder output frames*. The window is
            encoded *with* this extra context and then trimmed back to the core,
            so the full-context-trained Conformer sees real history instead of a
            zero-padded seam. Default ``62`` (~5s). ``0`` disables left context.
        asr_chunk_right_context: Right context (lookahead) attached to each ASR
            window during chunked inference, in *encoder output frames*. Same
            overlap-and-trim mechanism as ``asr_chunk_left_context``. Default
            ``62`` (~5s). ``0`` disables lookahead. With both contexts ``0`` the
            ASR branch falls back to bare non-overlapping windows.
        diar_chunk_right_context: Sortformer streaming ``chunk_right_context``
            (future frames attached after each chunk). Defaults to the
            ``diar_streaming_sortformer_4spk-v2.1`` "very high latency" preset (40).
        diar_fifo_len: Sortformer streaming ``fifo_len`` (FIFO queue length).
            Defaults to the same preset (40).
        diar_spkcache_update_period: Sortformer streaming ``spkcache_update_period``
            (frames popped from FIFO to update the speaker cache). Defaults to 300.
        diar_spkcache_len: Sortformer streaming ``spkcache_len`` (total speaker
            cache size). Defaults to 188.
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
        chunked_inference_length: int = 150,
        asr_chunk_left_context: int = 0,
        asr_chunk_right_context: int = 0,
        diar_chunk_right_context: int = 40,
        diar_fifo_len: int = 40,
        diar_spkcache_update_period: int = 300,
        diar_spkcache_len: int = 188,
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

        # Long-form / chunked inference configuration.
        self.chunked_inference_length = int(chunked_inference_length)
        # Overlap-and-trim context for the (full-context-trained) ASR encoder,
        # in *encoder output frames* (same unit as chunked_inference_length).
        # ~62 frames ~= 5s at 12.5 frames/s.
        self.asr_chunk_left_context = max(0, int(asr_chunk_left_context))
        self.asr_chunk_right_context = max(0, int(asr_chunk_right_context))
        self.diar_chunk_right_context = int(diar_chunk_right_context)
        self.diar_fifo_len = int(diar_fifo_len)
        self.diar_spkcache_update_period = int(diar_spkcache_update_period)
        self.diar_spkcache_len = int(diar_spkcache_len)

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

    def _fuse_diar_and_asr(self, asr_encoded: torch.Tensor, diar_preds: torch.Tensor) -> torch.Tensor:
        """Fuse ASR encoder states with speaker-activity predictions.

        LayerNorm both branches, project ``diar_preds`` through the sinusoidal
        speaker kernel and ADD into the ASR states.

        Args:
            asr_encoded: ASR encoder output. Shape: ``(B, D, T_asr)``.
            diar_preds: Speaker-activity predictions. Shape: ``(B, T_diar, n_spk)``.

        Returns:
            Fused encoder output. Shape: ``(B, D, T_asr)``.
        """
        asr_enc_states = asr_encoded.transpose(1, 2)  # (B, T, D)
        diar_preds = self._align_diar_frames(diar_preds, asr_enc_states.shape[1]).to(asr_enc_states.dtype)

        asr_enc_states = self.asr_norm(asr_enc_states)
        diar_preds = self.diar_norm(diar_preds)
        speaker_infusion = torch.matmul(diar_preds, self.diar_kernel.to(diar_preds.dtype))
        fused = speaker_infusion + asr_enc_states

        return fused.transpose(1, 2)  # (B, D, T)

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

        Two execution modes are dispatched here:

        - :meth:`_forward` — the offline (non-chunked) path. The ASR encoder sees the
          whole sequence at once and Sortformer runs its offline ``forward_infer``.
        - :meth:`_forward_chunked` — long-form inference. A single lock-step loop
          walks non-overlapping windows of ``chunked_inference_length`` output
          frames, running the ASR encoder and the streaming Sortformer on the same
          window and concatenating both before a single fusion. Engaged only at
          inference time
          (``self.training is False``), when ``chunked_inference_length > 0``, when
          neither cache nor ``bypass_pre_encode`` is used, and when the input is
          actually longer than a single chunk. Otherwise the offline path runs so
          short-form / training behaviour is unchanged.
        """
        chunk_feat_len = self.chunked_inference_length * self.asr_encoder.subsampling_factor
        use_chunked = (
            self.chunked_inference_length > 0
            and not self.training
            and cache_last_channel is None
            and not bypass_pre_encode
            and audio_signal.shape[-1] > chunk_feat_len
        )
        if use_chunked:
            return self._forward_chunked(audio_signal=audio_signal, length=length, diar_preds=diar_preds)

        return self._forward(
            audio_signal=audio_signal,
            length=length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            bypass_pre_encode=bypass_pre_encode,
            diar_preds=diar_preds,
        )

    def _forward(
        self,
        audio_signal,
        length,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
        bypass_pre_encode=False,
        diar_preds=None,
    ):
        """Offline (non-chunked) forward pass. See :meth:`forward` for argument semantics."""
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
            outputs = self._fuse_diar_and_asr(asr_encoded, diar_preds)
        else:
            outputs = asr_encoded

        if not cache_outputs:
            return outputs, asr_encoded_len
        return (outputs, asr_encoded_len, *cache_outputs)

    def _forward_chunked(self, audio_signal, length, diar_preds=None):
        """Long-form inference via a single lock-step loop over fixed windows.

        One ``for`` loop walks the recording in non-overlapping windows of exactly
        ``chunked_inference_length`` encoder-output frames
        (``chunked_inference_length * subsampling_factor`` input mel frames, i.e.
        ``chunked_inference_length * 0.08 s`` of audio). The number of windows is
        estimated up-front as ``ceil(total_feat_len / chunk_feat_len)``. For each
        window, *both* experts run on the *same* slice:

        * **ASR** — the ``normalize_batch``-normalised mels are fed to the ASR
          ConformerEncoder using *overlap-and-trim*: each core window is encoded
          together with ``asr_chunk_left_context`` / ``asr_chunk_right_context``
          extra frames, then the context is trimmed off the output so only the
          seam-free core is kept (appended to ``asr_chunks``). The
          full-context-trained encoder thus always sees real history + lookahead
          instead of a zero-padded slice. Unlike the cache-aware pipeline in
          ``multispk_transcribe_utils.perform_serial_streaming_stt_spk`` (masked-
          causal encoder + threaded left-context cache), this is a regular offline
          encoder, so context is re-encoded per window rather than cached, and the
          core windows themselves remain non-overlapping after trimming.
        * **Diar** — the *un-normalised* mels (Sortformer's native input) are fed to
          ``SortformerEncLabelModel.forward_streaming_step``, which carries the
          speaker cache + FIFO state across iterations. The frames emitted for the
          current chunk are appended to ``diar_chunks``.

        Each diar chunk is aligned to its ASR chunk's frame count before buffering,
        so the two buffers stay frame-for-frame parallel. After the loop the two
        lists are concatenated along time and fused once via
        :meth:`_fuse_diar_and_asr`, exactly as if the recording had been a single
        chunk.

        Args:
            audio_signal: Un-normalised mel features. Shape: ``(B, feat_in, n_frames)``.
            length: Per-sample feature lengths. Shape: ``(B,)``.
            diar_preds: Optional ``(B, T, n_spk)`` speaker-activity override (oracle /
                RTTM). When provided, streaming diarization is skipped and only the
                ASR branch is chunked.

        Returns:
            Tuple ``(outputs, encoded_lengths)`` with ``outputs`` of shape
            ``(B, D, T_asr)``.
        """
        subsampling = self.asr_encoder.subsampling_factor
        # Window size in *input* mel frames: chunked_inference_length counts
        # post-subsampling (~80 ms) frames, so multiply by the subsampling factor.
        chunk_feat_len = self.chunked_inference_length * subsampling
        total_feat_len = min(audio_signal.shape[-1], int(length.max().item()))
        num_chunks = max(1, math.ceil(total_feat_len / chunk_feat_len))

        # ASR sees per-feature-normalised mels. Normalise the whole utterance once
        # (matching the offline path) rather than per chunk, so chunk-local
        # statistics don't drift the encoder inputs across the recording.
        if self.asr_normalize_type:
            asr_audio_signal, _, _ = normalize_batch(
                audio_signal, length, normalize_type=self.asr_normalize_type,
            )
        else:
            asr_audio_signal = audio_signal

        # Match the ASR encoder's device/dtype: mels arrive fp32 while the encoder
        # runs in bf16; feeding fp32 into the bf16 subsampling conv raises
        # "Input type (float) and bias type (c10::BFloat16) should be the same".
        asr_param = next(self.asr_encoder.parameters(), None)
        if asr_param is not None:
            asr_audio_signal = asr_audio_signal.to(device=asr_param.device, dtype=asr_param.dtype)
            length = length.to(device=asr_param.device)

        # DEBUG TOGGLE: PEE_ASR_WHOLE=1 encodes the ASR branch over the FULL
        # sequence in a single forward (no windowing), to isolate whether
        # per-window ASR encoding (boundary effects / lost context) is what
        # degrades long-form output. Diarization always streams per chunk (it is a
        # streaming system by design and is known-good). Memory permitting, this is
        # a clean A/B against the chunked ASR path.
        asr_whole = os.environ.get("PEE_ASR_WHOLE", "0") == "1"
        asr_encoded_whole = None
        if asr_whole:
            with torch.set_grad_enabled(not self.freeze_asr):
                asr_encoded_whole, asr_encoded_len_whole = self.asr_encoder(
                    audio_signal=asr_audio_signal[:, :, :total_feat_len],
                    length=length.clamp(max=total_feat_len),
                )
            logging.info(
                "[PEE] PEE_ASR_WHOLE=1: encoded full ASR in one forward -> %s (len=%s)",
                tuple(asr_encoded_whole.shape),
                asr_encoded_len_whole.tolist(),
            )

        run_streaming_diar = diar_preds is None
        if run_streaming_diar:
            streaming_state, stream_dtype, diar_audio_signal, diar_length = self._init_streaming_diar(
                audio_signal, length, batch_size=audio_signal.shape[0],
            )
            n_spk = self.diarization_model.sortformer_modules.n_spk
            total_preds = torch.zeros(
                (diar_audio_signal.shape[0], 0, n_spk),
                device=diar_audio_signal.device,
                dtype=stream_dtype,
            )

        asr_chunks: List[torch.Tensor] = []
        diar_chunks: List[torch.Tensor] = []
        asr_encoded_len = torch.zeros_like(length)

        for chunk_idx in tqdm(
            range(num_chunks),
            total=num_chunks,
            desc="PEE chunked inference",
            disable=getattr(self, '_suppress_chunk_pbar', False),
        ):
            stt = chunk_idx * chunk_feat_len
            end = min(stt + chunk_feat_len, total_feat_len)

            # --- ASR branch: overlap-and-trim windowed encoding ---
            # The ASR Conformer is full-context-trained, so a bare [stt:end] slice
            # has no left history / right lookahead and zero-pad conv edges at both
            # seams -> wrong boundary embeddings -> the LLM hits a discontinuity
            # every window. Instead, encode the window *with* left/right context and
            # then trim the context back off: the boundary artifacts land in the
            # discarded context, so each core region matches the whole-utterance
            # encode and the concatenation is seam-free.
            asr_in_frames = asr_out_frames = 0
            if not asr_whole:
                left_ctx = self.asr_chunk_left_context * subsampling
                right_ctx = self.asr_chunk_right_context * subsampling
                enc_stt = max(stt - left_ctx, 0)
                enc_end = min(end + right_ctx, total_feat_len)
                asr_chunk = asr_audio_signal[:, :, enc_stt:enc_end]
                chunk_length = (length - enc_stt).clamp(min=0, max=enc_end - enc_stt)
                with torch.set_grad_enabled(not self.freeze_asr):
                    enc_ctx, _ = self.asr_encoder(audio_signal=asr_chunk, length=chunk_length)
                # Trim the context back off in output-frame space. Core output
                # boundaries are mapped from mel frames via the subsampling stride;
                # using rounded cumulative positions guarantees the per-chunk core
                # lengths tile to the same total the whole-utterance encode produces.
                left_drop = (stt - enc_stt) // subsampling
                core_len = round(end / subsampling) - round(stt / subsampling)
                core_len = max(0, min(core_len, enc_ctx.shape[-1] - left_drop))
                enc_chunk = enc_ctx[:, :, left_drop : left_drop + core_len]
                asr_chunks.append(enc_chunk)
                asr_encoded_len = asr_encoded_len + core_len
                asr_in_frames = asr_chunk.shape[-1]
                asr_out_frames = enc_chunk.shape[-1]
                align_target = enc_chunk.shape[-1]
            else:
                # ASR encoded as a whole; the per-chunk diar preds are aligned to
                # the chunk's nominal output frame count instead.
                align_target = math.ceil((end - stt) / subsampling)

            # --- Diar branch: stream the SAME window, keep this chunk's preds ---
            diar_in_frames = diar_raw_out = diar_out_frames = 0
            if run_streaming_diar:
                prev_len = total_preds.shape[1]
                diar_chunk = diar_audio_signal[:, :, stt:end].transpose(1, 2)  # (B, t, feat_in)
                diar_chunk_length = (diar_length - stt).clamp(min=0, max=end - stt)
                with torch.set_grad_enabled(not self.freeze_diar), _disable_dist_feature_sync(), _default_dtype(
                    stream_dtype
                ):
                    streaming_state, total_preds = self.diarization_model.forward_streaming_step(
                        processed_signal=diar_chunk,
                        processed_signal_length=diar_chunk_length,
                        streaming_state=streaming_state,
                        total_preds=total_preds,
                    )
                diar_raw = total_preds[:, prev_len:]
                # Frames newly emitted for this chunk, aligned to the ASR chunk so
                # the two buffers stay frame-for-frame parallel.
                new_preds = self._align_diar_frames(diar_raw, align_target)
                diar_chunks.append(new_preds)
                diar_in_frames = diar_chunk.shape[1]
                diar_raw_out = diar_raw.shape[1]
                diar_out_frames = new_preds.shape[1]

            logging.info(
                "[PEE chunk %d/%d] ASR feed=%d mel (core+ctx) -> out=%d enc (trimmed core)%s | "
                "DIAR feed=%d mel -> raw=%d aligned=%d enc",
                chunk_idx + 1,
                num_chunks,
                asr_in_frames,
                asr_out_frames,
                " (whole-mode)" if asr_whole else "",
                diar_in_frames,
                diar_raw_out,
                diar_out_frames,
            )

        if asr_whole:
            asr_encoded = asr_encoded_whole  # (B, D, T_asr)
            asr_encoded_len = asr_encoded_len_whole
        else:
            asr_encoded = torch.cat(asr_chunks, dim=2)  # (B, D, T_asr)
        if run_streaming_diar:
            diar_preds = torch.cat(diar_chunks, dim=1)  # (B, T_asr, n_spk)

        if diar_preds is not None:
            outputs = self._fuse_diar_and_asr(asr_encoded, diar_preds)
        else:
            outputs = asr_encoded

        return outputs, asr_encoded_len

    def _init_streaming_diar(self, audio_signal: torch.Tensor, length: torch.Tensor, batch_size: int):
        """Configure the wrapped Sortformer for streaming and build its initial state.

        Sets the streaming hyper-parameters (chunk length tied to
        ``chunked_inference_length``; right-context / FIFO / speaker-cache from the
        ``diar_streaming_sortformer_4spk-v2.1`` preset), refreshes the nested
        LightningModule's device cache, and allocates the speaker-cache / FIFO
        streaming state in the diarizer's float dtype.

        Returns:
            ``(streaming_state, stream_dtype, diar_audio_signal, diar_length)`` — the
            initialised streaming state, the diarizer's float dtype, and the input
            mels / lengths cast onto the diarizer's device & dtype.
        """
        sm = self.diarization_model.sortformer_modules
        sm.chunk_len = self.chunked_inference_length
        sm.chunk_right_context = self.diar_chunk_right_context
        sm.fifo_len = self.diar_fifo_len
        sm.spkcache_update_period = self.diar_spkcache_update_period
        sm.spkcache_len = self.diar_spkcache_len
        sm._check_streaming_parameters()

        diar_param = next(self.diarization_model.parameters(), None)
        if diar_param is not None:
            # Refresh the Lightning ``_device`` cache: ``forward_streaming_step``
            # builds streaming state via ``self.device`` internally. Moving params
            # through the parent module's ``.to()`` relocates tensors but leaves
            # the nested LightningModule's ``self.device`` stale, which would make
            # state tensors land on the wrong device ("Expected all tensors to be
            # on the same device"). A bf16 diarizer also rejects fp32 mel input
            # ("mixed dtype ... expect parameter to have scalar type of Float"),
            # so cast the features to its dtype too.
            self.diarization_model.to(diar_param.device)
            diar_device, stream_dtype = diar_param.device, diar_param.dtype
        else:
            diar_device, stream_dtype = audio_signal.device, torch.get_default_dtype()

        diar_audio_signal = audio_signal.to(device=diar_device, dtype=stream_dtype)
        diar_length = length.to(device=diar_device)

        with _disable_dist_feature_sync(), _default_dtype(stream_dtype):
            streaming_state = sm.init_streaming_state(
                batch_size=batch_size,
                async_streaming=self.diarization_model.async_streaming,
                device=diar_device,
            )
        return streaming_state, stream_dtype, diar_audio_signal, diar_length


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
            chunked_inference_length=self._cfg.get('chunked_inference_length', 150),
            asr_chunk_left_context=self._cfg.get('asr_chunk_left_context', 0),
            asr_chunk_right_context=self._cfg.get('asr_chunk_right_context', 0),
            diar_chunk_right_context=self._cfg.get('diar_chunk_right_context', 40),
            diar_fifo_len=self._cfg.get('diar_fifo_len', 40),
            diar_spkcache_update_period=self._cfg.get('diar_spkcache_update_period', 300),
            diar_spkcache_len=self._cfg.get('diar_spkcache_len', 188),
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
