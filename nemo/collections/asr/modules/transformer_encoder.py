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

import math
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from nemo.collections.asr.parts.submodules.multi_head_attention import PositionalEncoding
from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling, FeatureStacking, StackingSubsampling
from nemo.collections.asr.parts.utils.regularization_utils import compute_stochastic_depth_drop_probs
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import AccessMixin
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import AcousticEncodedRepresentation, BoolType, LengthsType, NeuralType, SpectrogramType

flex_attention_compiled = torch.compile(flex_attention, dynamic=True)


@dataclass
class TransformerEncoderConfig:
    """Configuration for ``TransformerEncoder`` and its sub-blocks.

    Args:
        feat_in: Input feature dimension (e.g. number of mel bins).
        d_model: Transformer encoder state dimension, i.e. the size of the residual stream that flows
            through every block (token/frame embedding size, attention input/output size, and
            feed-forward input/output size). Also known as ``hidden_size`` in HuggingFace
            ``transformers`` configs and ``embed_dim``/``d_model`` in PyTorch's
            ``nn.TransformerEncoderLayer``.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer blocks.
        drop_rate: Dropout probability applied inside attention and feed-forward sublayers.
        qkv_bias: If True, add a learnable bias to the fused Q/K/V projection. Many modern ASR/LM
            Transformers (e.g. HuggingFace Whisper) drop the bias on the K projection because a
            constant K bias adds the same scalar to every key and is wiped out by softmax's
            shift-invariance, making it a redundant parameter. Default ``False`` matches that style.
        qk_norm: If True, apply per-head ``LayerNorm`` to Q and K before the dot product. Stabilizes
            training by preventing exponential Q/K-norm growth and "attention entropy collapse"
            (Henry et al. 2020; used in OLMo 2, Gemma 3, Qwen 3). Cheap, ~no-op for inference.
        ff_expansion: Multiplier for the per-block FFN inner hidden size:
            ``ffn_hidden_size = int(ff_expansion * d_model)``. Only widens the intermediate FFN
            projection; FFN input/output stays at ``d_model``. Typical value ``4.0``; ``float``
            allows sub-1x experts for MoE. Equivalent to ``intermediate_size / hidden_size`` in
            HuggingFace and ``dim_feedforward / d_model`` in PyTorch's ``nn.TransformerEncoderLayer``.
        pre_block_norm: If True, apply ``LayerNorm`` to embeddings before the first Transformer block
            (BERT/ViT-style). Set False to match pre-norm Transformers such as Whisper or GPT-2.
        subsampling_factor: Frame-level subsampling factor performed by the pre-encoder.
        attn_mode: Attention pattern. Currently only ``"full"`` (bidirectional) is supported.
            Future modes: ``"causal"``, ``"lookahead"``, ``"local"``, ``"sliding_window"``.
    """
    feat_in: int = 128
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 17
    drop_rate: float = 0.1
    qkv_bias: bool = False
    qk_norm: bool = False
    ff_expansion: float = 4.0
    pre_block_norm: bool = True
    subsampling_factor: int = 4
    attn_mode: str = "full"


def _make_padding_mod(lengths):
    """Mask out padding positions based on per-sample lengths."""

    def pad_mask(b, h, q_idx, kv_idx):
        return kv_idx < lengths[b]

    return pad_mask


class FeedForward(nn.Module):
    def __init__(self, cfg: TransformerEncoderConfig):
        super().__init__()
        ff_hidden = int(cfg.ff_expansion * cfg.d_model)
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, ff_hidden),
            nn.GELU(),
            nn.Dropout(cfg.drop_rate),
            nn.Linear(ff_hidden, cfg.d_model),
            nn.Dropout(cfg.drop_rate),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: TransformerEncoderConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model

        self.w_qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.qkv_bias)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)

        self.qk_norm = cfg.qk_norm
        if cfg.qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(self, x, block_mask=None):
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim

        qkv = self.w_qkv(x).view(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.qk_norm:
            q = self.q_norm(q).to(v.dtype)
            k = self.k_norm(k).to(v.dtype)

        # Use compiled FlexAttention for CUDA, fallback to unfused for CPU.
        attn_fn = flex_attention_compiled if q.is_cuda else flex_attention
        out = attn_fn(q, k, v, block_mask=block_mask)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerEncoderConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.drop = nn.Dropout(cfg.drop_rate)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x, block_mask=None):
        x = x + self.drop(self.attn(self.norm1(x), block_mask=block_mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class TransformerEncoder(NeuralModule, Exportable, AccessMixin):
    """Pre-norm Transformer encoder for ASR.

    Architecture: PreEncode -> PositionalEncoding -> LayerNorm -> N x TransformerBlock -> FinalNorm

    Uses PyTorch FlexAttention for attention computation. On CUDA, mask functions
    are compiled into fused Triton kernels with block-sparse optimization. On CPU,
    FlexAttention falls back to an unfused implementation automatically.

    Args:
        feat_in: Input feature dimension (number of mel bins).
        d_model: Transformer encoder state dimension, i.e. the size of the residual stream that flows
            through every block (token/frame embedding size, attention input/output size, and
            feed-forward input/output size). Also known as ``hidden_size`` in HuggingFace
            ``transformers`` configs and ``embed_dim``/``d_model`` in PyTorch's
            ``nn.TransformerEncoderLayer``.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer blocks.
        feat_out: Output feature dimension. Defaults to ``d_model``.
        subsampling: Subsampling method. Supports ``feature_stacking`` for the
            Transformer-native ``FeatureStacking`` module, plus Conformer-style
            ``stacking``, ``stacking_norm``, ``vggnet``, ``striding``,
            ``dw_striding``/``dw-striding``, ``striding_conv1d``, and ``dw_striding_conv1d``.
        subsampling_factor: Subsampling factor for the pre-encoder.
        subsampling_conv_chunking_factor: Optional input chunking factor for convolutional subsampling.
        subsampling_conv_channels: Hidden channels for convolutional subsampling.
        causal_downsampling: Whether convolutional subsampling should be causal.
        drop_rate: Dropout probability.
        dropout_pre_encoder: Dropout probability after positional encoding. Defaults to ``drop_rate``.
        dropout_emb: Dropout probability for positional embeddings.
        qkv_bias: If True, add a learnable bias to the fused Q/K/V projection. Many modern ASR/LM
            Transformers (e.g. HuggingFace Whisper) drop the bias on the K projection because a
            constant K bias adds the same scalar to every key and is wiped out by softmax's
            shift-invariance, making it a redundant parameter. Default ``False`` matches that style.
        qk_norm: If True, apply per-head ``LayerNorm`` to Q and K before the dot product. Stabilizes
            training by preventing exponential Q/K-norm growth and "attention entropy collapse"
            (Henry et al. 2020; used in OLMo 2, Gemma 3, Qwen 3). Cheap, ~no-op for inference.
        ff_expansion: Multiplier for the per-block FFN inner hidden size:
            ``ffn_hidden_size = int(ff_expansion * d_model)``. Only widens the intermediate FFN
            projection; FFN input/output stays at ``d_model``. Typical value ``4.0``; ``float``
            allows sub-1x experts for MoE. Equivalent to ``intermediate_size / hidden_size`` in
            HuggingFace and ``dim_feedforward / d_model`` in PyTorch's ``nn.TransformerEncoderLayer``.
        pre_block_norm: If True (default), apply LayerNorm to embeddings before the first
            Transformer block (BERT/ViT-style). Set False to match pre-norm Transformers
            such as Whisper or GPT-2 — required when loading pretrained weights from those
            checkpoints.
        pos_emb_max_len: Initial maximum length for sinusoidal positional embeddings.
        xscaling: If True, scale embeddings by ``sqrt(d_model)`` before adding positional encodings,
            following "Attention Is All You Need" article. Originally intended to balance the magnitude
            of small-variance token embeddings against unit-bounded sinusoidal positions and to keep
            tied input/pre-softmax logits well-scaled. With modern unit-variance ``nn.Linear``
            pre-encoders and the LayerNorm directly after the positional sum, this scaling is
            largely a no-op for activation magnitudes. Only meaningful when ``pre_block_norm=False``
            or when matching pretrained checkpoints that expect this scaling.
        stochastic_depth_drop_prob: Final-layer stochastic depth drop probability.
        stochastic_depth_mode: Stochastic depth schedule, ``linear`` or ``uniform``.
        stochastic_depth_start_layer: First 1-based layer index eligible for stochastic depth.
        attn_mode: Attention pattern — currently only "full" (bidirectional) is supported.
        sync_max_audio_length: When true, sync positional encoding allocation length across distributed ranks.
    """

    def input_example(self, max_batch=1, max_dim=256):
        """Generates input examples for tracing and export."""
        dev = next(self.parameters()).device
        input_example = torch.randn(max_batch, self._feat_in, max_dim, device=dev)
        input_example_length = torch.randint(max_dim // 4, max_dim, (max_batch,), device=dev, dtype=torch.int64)
        return tuple([input_example, input_example_length])

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
                "bypass_pre_encode": NeuralType(tuple(), BoolType(), optional=True),
            }
        )

    @property
    def input_types_for_export(self):
        """Returns definitions of module input ports for export."""
        return self.input_types

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return OrderedDict(
            {
                "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            }
        )

    @property
    def output_types_for_export(self):
        """Returns definitions of module output ports for export."""
        return self.output_types

    @property
    def disabled_deployment_input_names(self):
        return set()

    @property
    def disabled_deployment_output_names(self):
        return set()

    def __init__(
        self,
        feat_in: int = 128,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 17,
        feat_out: int = -1,
        causal_downsampling: bool = False,
        subsampling: str = 'feature_stacking',
        subsampling_factor: int = 4,
        subsampling_conv_chunking_factor: int = 1,
        subsampling_conv_channels: int = -1,
        drop_rate: float = 0.1,
        dropout_pre_encoder: float = None,
        dropout_emb: float = 0.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        ff_expansion: float = 4.0,
        pre_block_norm: bool = True,
        pos_emb_max_len: int = 5000,
        xscaling: bool = False,
        stochastic_depth_drop_prob: float = 0.0,
        stochastic_depth_mode: str = "linear",
        stochastic_depth_start_layer: int = 1,
        attn_mode: str = "full",
        sync_max_audio_length: bool = True,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        if attn_mode != "full":
            raise ValueError(f"attn_mode='{attn_mode}' is not yet supported. Currently only 'full' is available.")
        if dropout_pre_encoder is None:
            dropout_pre_encoder = drop_rate
        if subsampling == 'feature-stacking':
            subsampling = 'feature_stacking'
        if subsampling == 'dw-striding':
            subsampling = 'dw_striding'

        cfg = TransformerEncoderConfig(
            feat_in=feat_in,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            drop_rate=drop_rate,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            ff_expansion=ff_expansion,
            pre_block_norm=pre_block_norm,
            subsampling_factor=subsampling_factor,
            attn_mode=attn_mode,
        )
        self.d_model = d_model
        self.n_layers = n_layers
        self._feat_in = feat_in
        self.subsampling = subsampling
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor
        self.sync_max_audio_length = sync_max_audio_length

        if subsampling_conv_channels == -1:
            subsampling_conv_channels = d_model
        if subsampling == 'feature_stacking':
            self.pre_encode = FeatureStacking(subsampling_factor, feat_in, d_model)
        elif subsampling and subsampling_factor > 1:
            if subsampling in ['stacking', 'stacking_norm']:
                self.pre_encode = StackingSubsampling(
                    subsampling_factor=subsampling_factor,
                    feat_in=feat_in,
                    feat_out=d_model,
                    norm=True if subsampling == 'stacking_norm' else False,
                )
            else:
                self.pre_encode = ConvSubsampling(
                    subsampling=subsampling,
                    subsampling_factor=subsampling_factor,
                    feat_in=feat_in,
                    feat_out=d_model,
                    conv_channels=subsampling_conv_channels,
                    subsampling_conv_chunking_factor=subsampling_conv_chunking_factor,
                    activation=nn.ReLU(True),
                    is_causal=causal_downsampling,
                )
        else:
            self.pre_encode = nn.Linear(feat_in, d_model)

        self._feat_out = d_model
        if xscaling:
            self.xscale = math.sqrt(d_model)
        else:
            self.xscale = None
        self.pos_emb_max_len = pos_emb_max_len
        self.pos_enc = PositionalEncoding(
            d_model=d_model,
            dropout_rate=dropout_pre_encoder,
            max_len=pos_emb_max_len,
            xscale=self.xscale,
            dropout_rate_emb=dropout_emb,
        )
        self.embed_norm = nn.LayerNorm(d_model) if pre_block_norm else nn.Identity()
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        if feat_out > 0 and feat_out != self._feat_out:
            self.out_proj = nn.Linear(self._feat_out, feat_out)
            self._feat_out = feat_out
        else:
            self.out_proj = None

        self.set_max_audio_length(self.pos_emb_max_len)
        self.use_pad_mask = True
        self.layer_drop_probs = compute_stochastic_depth_drop_probs(
            len(self.layers), stochastic_depth_drop_prob, stochastic_depth_mode, stochastic_depth_start_layer
        )
        self.interctc_capture_at_layers = None

    def forward_for_export(self, audio_signal, length):
        """Forward function for model export. Please see ``forward()`` for details."""
        return self.forward_internal(audio_signal=audio_signal, length=length)

    @typecheck()
    def forward(self, audio_signal, length, bypass_pre_encode=False):
        """
        Args:
            audio_signal: ``(B, C, T)`` mel spectrogram when ``bypass_pre_encode=False``,
                or ``(B, T, D)`` pre-encoded embeddings when ``bypass_pre_encode=True``.
            length: (B,) — valid frame counts per sample.
            bypass_pre_encode: If true, skip the pre-encoder and consume frame-level embeddings.

        Returns:
            x: (B, D, T') — encoded representation (channels-first).
            length: (B,) — output lengths after subsampling.
        """
        if not bypass_pre_encode and audio_signal.shape[-2] != self._feat_in:
            raise ValueError(
                f"If bypass_pre_encode is False, audio_signal should have shape "
                f"(batch, {self._feat_in}, n_frame) but got last dimension {audio_signal.shape[-2]}."
            )
        if bypass_pre_encode and audio_signal.shape[-1] != self.d_model:
            raise ValueError(
                f"If bypass_pre_encode is True, audio_signal should have shape "
                f"(batch, n_frame, {self.d_model}) but got last dimension {audio_signal.shape[-1]}."
            )

        if bypass_pre_encode:
            self.update_max_seq_length(seq_length=audio_signal.size(1), device=audio_signal.device)
        else:
            self.update_max_seq_length(seq_length=audio_signal.size(2), device=audio_signal.device)
        return self.forward_internal(audio_signal, length, bypass_pre_encode=bypass_pre_encode)

    def forward_internal(self, audio_signal, length, bypass_pre_encode=False):
        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),),
                audio_signal.size(1) if bypass_pre_encode else audio_signal.size(-1),
                dtype=torch.int64,
                device=audio_signal.device,
            )

        if not bypass_pre_encode:
            if isinstance(self.pre_encode, FeatureStacking):
                x, length = self.pre_encode(audio_signal, length)
            else:
                x = torch.transpose(audio_signal, 1, 2)
            if isinstance(self.pre_encode, nn.Linear):
                x = self.pre_encode(x)
            elif not isinstance(self.pre_encode, FeatureStacking):
                x, length = self.pre_encode(x=x, lengths=length)
            length = length.to(torch.int64)
        else:
            x = audio_signal
            length = length.to(torch.int64)

        x, _ = self.pos_enc(x=x)
        x = self.embed_norm(x)

        B, T, _ = x.shape
        if self.use_pad_mask:
            block_mask = create_block_mask(_make_padding_mod(length), B=B, H=1, Q_LEN=T, KV_LEN=T, device=x.device)
        else:
            block_mask = None

        for lth, (drop_prob, layer) in enumerate(zip(self.layer_drop_probs, self.layers)):
            original_signal = x
            x = layer(x, block_mask=block_mask)

            if self.training and drop_prob > 0.0:
                should_drop = torch.rand(1, device=x.device) < drop_prob
                if should_drop:
                    x = x * 0.0 + original_signal
                else:
                    x = (x - original_signal) / (1.0 - drop_prob) + original_signal

            if self.is_access_enabled(getattr(self, "model_guid", None)):
                if self.interctc_capture_at_layers is None:
                    self.interctc_capture_at_layers = self.access_cfg.get('interctc', {}).get('capture_layers', [])
                if lth in self.interctc_capture_at_layers:
                    lth_audio_signal = x
                    if self.out_proj is not None:
                        lth_audio_signal = self.out_proj(lth_audio_signal)
                    self.register_accessible_tensor(
                        name=f'interctc/layer_output_{lth}', tensor=torch.transpose(lth_audio_signal, 1, 2)
                    )
                    self.register_accessible_tensor(name=f'interctc/layer_length_{lth}', tensor=length)

        x = self.final_norm(x)
        if self.out_proj is not None:
            x = self.out_proj(x)
        x = x.transpose(1, 2)  # (B, T, D) -> (B, D, T)
        length = length.to(dtype=torch.int64)
        return x, length

    def update_max_seq_length(self, seq_length: int, device):
        """
        Updates the maximum sequence length for positional encodings.

        Args:
            seq_length: New maximum sequence length.
            device: Device to use for computations.
        """
        if self.sync_max_audio_length and torch.distributed.is_initialized():
            global_max_len = torch.tensor([seq_length], dtype=torch.float32, device=device)
            torch.distributed.all_reduce(global_max_len, op=torch.distributed.ReduceOp.MAX)
            seq_length = global_max_len.int().item()

        if seq_length > self.max_audio_length:
            self.set_max_audio_length(seq_length)

    def set_max_audio_length(self, max_audio_length):
        """Sets maximum input length and extends positional encodings if needed."""
        self.max_audio_length = max_audio_length
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.pos_enc.extend_pe(max_audio_length, device, dtype)

    def enable_pad_mask(self, on=True):
        """Enables or disables pad masking and returns the previous state."""
        mask = self.use_pad_mask
        self.use_pad_mask = on
        return mask
