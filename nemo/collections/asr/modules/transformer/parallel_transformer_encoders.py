# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Parallel expert TransformerEncoder bank (profile-driven, MoE-style).

This module provides a true MoE-style packed implementation of multiple
NeMo ``TransformerEncoder`` experts that share the SAME ``hidden_size`` and
``num_attention_heads`` but may differ in FFN ``inner_size``.

Design (after py-spy + torch.profiler analysis on RTX 6000 Ada)
---------------------------------------------------------------

py-spy / torch.profiler showed three dominant costs in the previous
batched implementation:

    1. ``aten::bmm`` for FFN was 2.4x more compute than the serial path
       because every aux expert was zero-padded from inner=512 -> 4096.
    2. ``Q/K/V`` did three independent ``bmm`` calls per layer, each of
       size (E, M, H) @ (E, H, H) -> (E, M, H).
    3. ``LayerNorm`` was a Python list-comprehension of ``F.layer_norm``
       calls (one per expert), launching 3 LN kernels per LN site.

Each is fixed here:

    1. ``PackedFFNBank`` is now shape-bucketed: weights are packed *per
       bucket* (n_e, inner_e, H) with no padding. For our spec this gives
       one bucket of size (1, 4096, 1024) for the main expert and one of
       size (2, 512, 1024) for the two aux experts. ``forward("batched")``
       does an ``F.linear`` for the n=1 main bucket and a single ``bmm``
       for the n=2 aux bucket -> matches serial total compute.
    2. ``PackedFusedQKVBank`` packs Q, K, V into a single
       (E, 3*H, H) weight tensor and runs ONE ``bmm`` per layer that
       produces all of Q, K, V in one shot, then splits the output.
    3. ``PackedLayerNormBank`` runs ONE fused ``F.layer_norm`` over the
       (E, B, L, H) tensor with affine disabled, then applies the
       per-expert affine via a broadcast multiply-add against weights of
       shape (E, 1, 1, H). One LN kernel per LN site, regardless of E.

Both compute modes are preserved:

    * ``compute_mode="batched"``:  truly packed bmm path - the fastest on GPU.
    * ``compute_mode="per_expert"``: F.linear / F.layer_norm per expert,
      bit-exact (``torch.equal``) against the serial NeMo TransformerEncoder.

Unified state_dict
------------------

Every weight is an ``nn.Parameter`` with a leading expert dim:

    layers.l.qkv.weight                 (E, 3H, H)
    layers.l.qkv.bias                   (E, 3H)
    layers.l.out_projection.weight      (E, H, H)
    layers.l.out_projection.bias        (E, H)
    layers.l.ffn.bucket_<inner>_w_in    (n_e, inner, H)   # one per FFN bucket
    layers.l.ffn.bucket_<inner>_b_in    (n_e, inner)
    layers.l.ffn.bucket_<inner>_w_out   (n_e, H, inner)
    layers.l.ffn.bucket_<inner>_b_out   (n_e, H)
    layers.l.layer_norm_1.weight        (E, H)
    layers.l.layer_norm_1.bias          (E, H)
    layers.l.layer_norm_2.weight        (E, H)
    layers.l.layer_norm_2.bias          (E, H)
    final_layer_norm.weight             (E, H)            # if used
    final_layer_norm.bias               (E, H)

Within every bucket the leading-expert-dim layout is amenable to batched
GEMM. Across buckets the FFN is split because the inner dims differ -
keeping weights bucketed avoids the (otherwise mandatory) zero-padding
compute waste.
"""

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.collections.asr.modules.transformer.transformer_encoders import (
    TransformerEncoder,
    TransformerEncoderBlock,
)
from nemo.collections.common.parts import form_attention_mask

__all__ = [
    "PackedLinearBank",
    "PackedFusedQKVBank",
    "PackedFFNBank",
    "PackedLayerNormBank",
    "ParallelExpertTransformerEncoderBlock",
    "ParallelExpertTransformerEncoder",
]


# ---------------------------------------------------------------------------
# Packed parameter banks
# ---------------------------------------------------------------------------


class PackedLinearBank(nn.Module):
    """Packs ``E`` ``nn.Linear`` experts of identical (in, out) features.

    Weight layout matches ``nn.Linear`` (``out`` x ``in``) for direct copy.
    Forward uses one batched ``bmm`` in ``"batched"`` mode and ``F.linear``
    per expert in ``"per_expert"`` mode (bit-exact against ``nn.Linear``).
    """

    def __init__(self, num_experts: int, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features))
        if bias:
            self.bias: Optional[nn.Parameter] = nn.Parameter(torch.empty(num_experts, out_features))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linears(cls, linears: Sequence[nn.Linear]) -> "PackedLinearBank":
        if len(linears) == 0:
            raise ValueError("from_linears requires at least one nn.Linear.")
        in_features = linears[0].in_features
        out_features = linears[0].out_features
        bias_present = linears[0].bias is not None
        for idx, lin in enumerate(linears[1:], start=1):
            if lin.in_features != in_features or lin.out_features != out_features:
                raise ValueError(
                    f"PackedLinearBank requires identical (in, out) across experts; "
                    f"expert 0={(in_features, out_features)} vs expert {idx}="
                    f"{(lin.in_features, lin.out_features)}"
                )
            if (lin.bias is not None) != bias_present:
                raise ValueError("PackedLinearBank requires the bias presence to match across experts.")

        bank = cls(len(linears), in_features, out_features, bias=bias_present)
        with torch.no_grad():
            for i, lin in enumerate(linears):
                bank.weight[i].copy_(lin.weight.detach())
                if bias_present:
                    bank.bias[i].copy_(lin.bias.detach())  # type: ignore[index]
        return bank

    def forward(self, x_stacked: torch.Tensor, mode: str = "batched") -> torch.Tensor:
        if x_stacked.shape[0] != self.num_experts:
            raise ValueError(f"Expected leading expert dim={self.num_experts}, got {x_stacked.shape[0]}.")
        if x_stacked.shape[-1] != self.in_features:
            raise ValueError(f"Expected last dim={self.in_features}, got {x_stacked.shape[-1]}.")
        leading = x_stacked.shape[1:-1]

        if mode == "per_expert":
            outs = [
                F.linear(x_stacked[i], self.weight[i], self.bias[i] if self.bias is not None else None)
                for i in range(self.num_experts)
            ]
            return torch.stack(outs, dim=0)
        if mode != "batched":
            raise ValueError(f"Unknown PackedLinearBank mode: {mode!r}")

        m = 1
        for s in leading:
            m *= s
        x_flat = x_stacked.reshape(self.num_experts, m, self.in_features)
        y = torch.bmm(x_flat, self.weight.transpose(-1, -2))  # (E, M, out)
        if self.bias is not None:
            y = y + self.bias.unsqueeze(1)
        return y.reshape(self.num_experts, *leading, self.out_features)


class PackedFusedQKVBank(nn.Module):
    """Fuses ``Q``, ``K``, ``V`` projections of all experts into one packed weight.

    ``weight`` has shape ``(E, 3*H, H)`` and ``bias`` has shape ``(E, 3*H)``.
    A single ``bmm`` produces all of Q/K/V for every expert in one launch.
    Forward returns three tensors ``(Q, K, V)``, each shaped ``(E, ..., H)``.
    """

    def __init__(self, num_experts: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.empty(num_experts, 3 * hidden_size, hidden_size))
        if bias:
            self.bias: Optional[nn.Parameter] = nn.Parameter(torch.empty(num_experts, 3 * hidden_size))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_qkv_linears(
        cls,
        q_linears: Sequence[nn.Linear],
        k_linears: Sequence[nn.Linear],
        v_linears: Sequence[nn.Linear],
    ) -> "PackedFusedQKVBank":
        if not (len(q_linears) == len(k_linears) == len(v_linears)) or len(q_linears) == 0:
            raise ValueError("Q, K, V lists must have the same non-zero length.")
        H = q_linears[0].in_features
        for idx, (q, k, v) in enumerate(zip(q_linears, k_linears, v_linears)):
            for name, lin in [("q", q), ("k", k), ("v", v)]:
                if lin.in_features != H or lin.out_features != H:
                    raise ValueError(
                        f"PackedFusedQKVBank requires every Q/K/V to be (H, H); "
                        f"expert {idx} {name}=({lin.in_features},{lin.out_features}), H={H}"
                    )
        bias_present = q_linears[0].bias is not None
        bank = cls(len(q_linears), H, bias=bias_present)
        with torch.no_grad():
            for i, (q, k, v) in enumerate(zip(q_linears, k_linears, v_linears)):
                bank.weight[i, 0 * H : 1 * H, :].copy_(q.weight.detach())
                bank.weight[i, 1 * H : 2 * H, :].copy_(k.weight.detach())
                bank.weight[i, 2 * H : 3 * H, :].copy_(v.weight.detach())
                if bias_present:
                    bank.bias[i, 0 * H : 1 * H].copy_(q.bias.detach())  # type: ignore[index]
                    bank.bias[i, 1 * H : 2 * H].copy_(k.bias.detach())  # type: ignore[index]
                    bank.bias[i, 2 * H : 3 * H].copy_(v.bias.detach())  # type: ignore[index]
        return bank

    def forward(
        self, x_stacked: torch.Tensor, mode: str = "batched"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x_stacked.shape[0] != self.num_experts:
            raise ValueError(f"Expected leading expert dim={self.num_experts}, got {x_stacked.shape[0]}.")
        if x_stacked.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected last dim={self.hidden_size}, got {x_stacked.shape[-1]}.")
        H = self.hidden_size
        leading = x_stacked.shape[1:-1]

        if mode == "per_expert":
            # For bit-exactness against NeMo's serial path, run three
            # SEPARATE F.linear calls per expert (cuBLAS picks a different
            # kernel for (H, H) vs (H, 3H) gemm, which would otherwise
            # cause ULP-level reduction-order differences).
            qs, ks, vs = [], [], []
            for i in range(self.num_experts):
                w_q = self.weight[i, 0 * H : 1 * H, :]
                w_k = self.weight[i, 1 * H : 2 * H, :]
                w_v = self.weight[i, 2 * H : 3 * H, :]
                if self.bias is not None:
                    b_q = self.bias[i, 0 * H : 1 * H]
                    b_k = self.bias[i, 1 * H : 2 * H]
                    b_v = self.bias[i, 2 * H : 3 * H]
                else:
                    b_q = b_k = b_v = None
                qs.append(F.linear(x_stacked[i], w_q, b_q))
                ks.append(F.linear(x_stacked[i], w_k, b_k))
                vs.append(F.linear(x_stacked[i], w_v, b_v))
            return torch.stack(qs, 0), torch.stack(ks, 0), torch.stack(vs, 0)
        if mode != "batched":
            raise ValueError(f"Unknown PackedFusedQKVBank mode: {mode!r}")

        m = 1
        for s in leading:
            m *= s
        x_flat = x_stacked.reshape(self.num_experts, m, H)
        qkv = torch.bmm(x_flat, self.weight.transpose(-1, -2))  # (E, M, 3H)
        if self.bias is not None:
            qkv = qkv + self.bias.unsqueeze(1)
        qkv = qkv.reshape(self.num_experts, *leading, 3 * H)
        q, k, v = qkv.split(H, dim=-1)
        return q.contiguous(), k.contiguous(), v.contiguous()


class PackedFFNBank(nn.Module):
    """Shape-bucketed FFN bank for ``E`` PositionWiseFF experts.

    Experts are grouped into buckets by their FFN ``inner_size``. Each bucket
    has its own contiguous packed weight tensor of shape ``(n_e, inner, H)``,
    so there is **no zero padding** -> total compute matches the serial path.

    Forward modes:
        * ``"batched"``: for every bucket of size ``n_e``, dispatch ONE
          ``bmm`` (or ``F.linear`` if ``n_e == 1``). For our (main, aux1, aux2)
          spec this means one ``F.linear`` for the main 4096-bucket and one
          ``bmm`` for the 512-bucket of aux1+aux2.
        * ``"per_expert"``: ``F.linear`` per expert against per-bucket weight
          slices. Bit-exact against the original ``PositionWiseFF.forward``.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        true_inners: Sequence[int],
        act_fn: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        if len(true_inners) != num_experts:
            raise ValueError(
                f"true_inners must list one inner size per expert; got {len(true_inners)} for {num_experts}."
            )
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.true_inners: Tuple[int, ...] = tuple(int(x) for x in true_inners)
        self.act_fn = act_fn

        # Group experts by inner_size, preserving relative order.
        bucket_to_idxs: Dict[int, List[int]] = defaultdict(list)
        for i, inner in enumerate(self.true_inners):
            bucket_to_idxs[inner].append(i)
        # Sort buckets by descending inner so the largest bmm fires first.
        ordered = sorted(bucket_to_idxs.items(), key=lambda kv: -kv[0])
        self._bucket_inners: Tuple[int, ...] = tuple(b[0] for b in ordered)
        self._bucket_idxs: Tuple[Tuple[int, ...], ...] = tuple(tuple(b[1]) for b in ordered)
        # Whether each bucket's expert indices are contiguous in source order.
        # If all buckets ARE contiguous AND, when concatenated in bucket order,
        # they reconstruct ``[0, 1, ..., E-1]`` we can skip the scatter and use
        # ``torch.cat`` for the bucket outputs.
        self._concat_fast_path = self._compute_concat_fast_path()

        self.bucket_weights_in = nn.ParameterList()
        self.bucket_biases_in = nn.ParameterList()
        self.bucket_weights_out = nn.ParameterList()
        self.bucket_biases_out = nn.ParameterList()
        for inner_size, idxs in ordered:
            n = len(idxs)
            self.bucket_weights_in.append(nn.Parameter(torch.empty(n, inner_size, hidden_size)))
            self.bucket_biases_in.append(nn.Parameter(torch.empty(n, inner_size)))
            self.bucket_weights_out.append(nn.Parameter(torch.empty(n, hidden_size, inner_size)))
            self.bucket_biases_out.append(nn.Parameter(torch.empty(n, hidden_size)))

    def _compute_concat_fast_path(self) -> bool:
        # Each bucket must be contiguous internally and the concat of all
        # bucket idxs in bucket order must equal range(E).
        flat: List[int] = []
        for idxs in self._bucket_idxs:
            if list(idxs) != list(range(idxs[0], idxs[0] + len(idxs))):
                return False
            flat.extend(idxs)
        return flat == list(range(self.num_experts))

    @classmethod
    def from_position_wise_ffs(
        cls,
        ffns: Sequence[nn.Module],
        act_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> "PackedFFNBank":
        if len(ffns) == 0:
            raise ValueError("from_position_wise_ffs requires at least one PositionWiseFF.")
        hidden_size = ffns[0].dense_in.in_features  # type: ignore[union-attr]
        true_inners = []
        for idx, ffn in enumerate(ffns):
            d_in = ffn.dense_in  # type: ignore[union-attr]
            d_out = ffn.dense_out  # type: ignore[union-attr]
            if d_in.in_features != hidden_size or d_out.out_features != hidden_size:
                raise ValueError(
                    f"PackedFFNBank requires same hidden_size across experts; "
                    f"expert 0={hidden_size} vs expert {idx}=(dense_in.in={d_in.in_features}, "
                    f"dense_out.out={d_out.out_features})."
                )
            if d_in.out_features != d_out.in_features:
                raise ValueError(
                    f"PositionWiseFF expert {idx} has mismatched inner sizes: "
                    f"dense_in.out={d_in.out_features} vs dense_out.in={d_out.in_features}."
                )
            true_inners.append(d_in.out_features)

        bank = cls(len(ffns), hidden_size, true_inners, act_fn)
        with torch.no_grad():
            for bucket_idx, idxs in enumerate(bank._bucket_idxs):
                for j, e in enumerate(idxs):
                    src = ffns[e]
                    bank.bucket_weights_in[bucket_idx][j].copy_(src.dense_in.weight.detach())  # type: ignore[union-attr]
                    bank.bucket_biases_in[bucket_idx][j].copy_(src.dense_in.bias.detach())  # type: ignore[union-attr]
                    bank.bucket_weights_out[bucket_idx][j].copy_(src.dense_out.weight.detach())  # type: ignore[union-attr]
                    bank.bucket_biases_out[bucket_idx][j].copy_(src.dense_out.bias.detach())  # type: ignore[union-attr]
        return bank

    def _run_bucket_batched(
        self, x_bucket: torch.Tensor, b: int
    ) -> torch.Tensor:
        """Run dense_in -> act -> dense_out for bucket ``b``.

        Args:
            x_bucket: (n, M, H)
            b: bucket index
        Returns:
            (n, M, H)
        """
        w_in = self.bucket_weights_in[b]    # (n, inner, H)
        bi_in = self.bucket_biases_in[b]    # (n, inner)
        w_out = self.bucket_weights_out[b]  # (n, H, inner)
        bi_out = self.bucket_biases_out[b]  # (n, H)
        n = w_in.shape[0]

        if n == 1:
            x = x_bucket.squeeze(0)  # (M, H)
            z = F.linear(x, w_in[0], bi_in[0])
            z = self.act_fn(z)
            z = F.linear(z, w_out[0], bi_out[0])
            return z.unsqueeze(0)

        z = torch.bmm(x_bucket, w_in.transpose(-1, -2)) + bi_in.unsqueeze(1)
        z = self.act_fn(z)
        z = torch.bmm(z, w_out.transpose(-1, -2)) + bi_out.unsqueeze(1)
        return z

    def forward(self, x_stacked: torch.Tensor, mode: str = "batched") -> torch.Tensor:
        if x_stacked.shape[0] != self.num_experts:
            raise ValueError(f"Expected leading expert dim={self.num_experts}, got {x_stacked.shape[0]}.")
        if x_stacked.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected last dim={self.hidden_size}, got {x_stacked.shape[-1]}.")
        leading = x_stacked.shape[1:-1]
        m = 1
        for s in leading:
            m *= s
        x_flat = x_stacked.reshape(self.num_experts, m, self.hidden_size)

        if mode == "per_expert":
            outs: List[torch.Tensor] = [None] * self.num_experts  # type: ignore
            for b, idxs in enumerate(self._bucket_idxs):
                w_in = self.bucket_weights_in[b]
                bi_in = self.bucket_biases_in[b]
                w_out = self.bucket_weights_out[b]
                bi_out = self.bucket_biases_out[b]
                for j, e in enumerate(idxs):
                    z = F.linear(x_stacked[e], w_in[j], bi_in[j])
                    z = self.act_fn(z)
                    z = F.linear(z, w_out[j], bi_out[j])
                    outs[e] = z
            return torch.stack(outs, dim=0)

        if mode != "batched":
            raise ValueError(f"Unknown PackedFFNBank mode: {mode!r}")

        if self._concat_fast_path:
            # No scatter: concat bucket outputs in bucket order (which equals
            # source expert order). Each bucket sees a contiguous slice of
            # x_flat -> bmm input is contiguous in the leading dim.
            chunks: List[torch.Tensor] = []
            offset = 0
            for b, idxs in enumerate(self._bucket_idxs):
                n = len(idxs)
                x_bucket = x_flat[offset : offset + n]  # contiguous view
                chunks.append(self._run_bucket_batched(x_bucket, b))
                offset += n
            out = torch.cat(chunks, dim=0)
            return out.reshape(self.num_experts, *leading, self.hidden_size)

        outs2: List[torch.Tensor] = [None] * self.num_experts  # type: ignore
        for b, idxs in enumerate(self._bucket_idxs):
            x_bucket = x_flat[list(idxs)]  # gather
            z = self._run_bucket_batched(x_bucket, b)
            for j, e in enumerate(idxs):
                outs2[e] = z[j]
        return torch.stack(outs2, dim=0).reshape(self.num_experts, *leading, self.hidden_size)


class PackedLayerNormBank(nn.Module):
    """Packs ``E`` ``nn.LayerNorm`` experts of identical hidden size.

    Forward computes ONE fused ``F.layer_norm`` over the entire ``(E, ...,
    H)`` tensor with affine disabled, then applies a per-expert affine via
    a broadcast multiply-add. This is mathematically and numerically
    equivalent to running ``F.layer_norm`` per expert (the normalization is
    independent across the expert leading dim because LayerNorm reduces only
    along the last dim, ``H``), so it is bit-exact against
    ``nn.LayerNorm`` for the per-expert reduction.

    The broadcast affine is one fused multiply-add kernel; total kernels
    per LN site go from ``E * 1`` (E LN kernels) to ``2`` (1 LN + 1 affine).
    """

    def __init__(self, num_experts: int, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_experts, hidden_size))
        self.bias = nn.Parameter(torch.zeros(num_experts, hidden_size))

    @classmethod
    def from_layer_norms(cls, norms: Sequence[nn.LayerNorm]) -> "PackedLayerNormBank":
        if len(norms) == 0:
            raise ValueError("from_layer_norms requires at least one nn.LayerNorm.")
        hidden_size = norms[0].normalized_shape[0]
        eps = norms[0].eps
        for idx, ln in enumerate(norms[1:], start=1):
            if ln.normalized_shape[0] != hidden_size:
                raise ValueError(
                    f"PackedLayerNormBank requires identical hidden size; "
                    f"expert 0={hidden_size} vs expert {idx}={ln.normalized_shape[0]}."
                )
            if ln.eps != eps:
                raise ValueError(
                    f"PackedLayerNormBank requires identical eps; "
                    f"expert 0={eps} vs expert {idx}={ln.eps}."
                )
        bank = cls(len(norms), hidden_size, eps=eps)
        with torch.no_grad():
            for i, ln in enumerate(norms):
                bank.weight[i].copy_(ln.weight.detach())
                bank.bias[i].copy_(ln.bias.detach())
        return bank

    def forward(self, x_stacked: torch.Tensor, mode: str = "batched") -> torch.Tensor:
        if x_stacked.shape[0] != self.num_experts:
            raise ValueError(f"Expected leading expert dim={self.num_experts}, got {x_stacked.shape[0]}.")
        if x_stacked.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected last dim={self.hidden_size}, got {x_stacked.shape[-1]}.")

        if mode == "per_expert":
            outs = [
                F.layer_norm(
                    x_stacked[i],
                    (self.hidden_size,),
                    self.weight[i],
                    self.bias[i],
                    self.eps,
                )
                for i in range(self.num_experts)
            ]
            return torch.stack(outs, dim=0)

        # batched: one fused F.layer_norm (no affine) + broadcast affine.
        # Reshape so the leading expert dim is preserved for the broadcast.
        E = self.num_experts
        H = self.hidden_size
        leading = x_stacked.shape[1:-1]
        # F.layer_norm normalizes over the last dim only; the leading dims
        # (including E) are independent. So the result is identical to
        # running F.layer_norm per (e, b, l) row.
        x_normed = F.layer_norm(x_stacked, (H,), weight=None, bias=None, eps=self.eps)
        # Affine: broadcast (E, H) over (E, ..., H).
        affine_shape = (E,) + (1,) * len(leading) + (H,)
        return x_normed * self.weight.view(*affine_shape) + self.bias.view(*affine_shape)


# ---------------------------------------------------------------------------
# Packed encoder block / encoder
# ---------------------------------------------------------------------------


_ACT_REGISTRY: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "relu": torch.relu,
    "gelu": F.gelu,
}


def _resolve_act_fn(blocks: Sequence[TransformerEncoderBlock]) -> Callable[[torch.Tensor], torch.Tensor]:
    act = blocks[0].second_sub_layer.act_fn
    for idx, blk in enumerate(blocks[1:], start=1):
        if blk.second_sub_layer.act_fn is not act:
            raise ValueError(
                f"All packed experts must share the same activation. "
                f"Expert 0={act}, expert {idx}={blk.second_sub_layer.act_fn}."
            )
    return act


class ParallelExpertTransformerEncoderBlock(nn.Module):
    """One transformer layer packed across ``E`` experts (MoE-style)."""

    def __init__(self, blocks: Sequence[TransformerEncoderBlock], compute_mode: str = "batched"):
        super().__init__()
        if len(blocks) < 1:
            raise ValueError("blocks list must contain at least one TransformerEncoderBlock.")
        if compute_mode not in {"batched", "per_expert"}:
            raise ValueError(f"compute_mode must be 'batched' or 'per_expert'; got {compute_mode!r}.")
        self.num_experts = len(blocks)
        self.compute_mode = compute_mode

        first_attn = blocks[0].first_sub_layer
        self.hidden_size = first_attn.hidden_size
        self.num_attention_heads = first_attn.num_attention_heads
        self.attn_head_size = first_attn.attn_head_size
        self.attn_scale = first_attn.attn_scale
        self.pre_ln = blocks[0].pre_ln
        for idx, blk in enumerate(blocks[1:], start=1):
            attn = blk.first_sub_layer
            if attn.hidden_size != self.hidden_size:
                raise ValueError(
                    f"All packed experts must share hidden_size; expert 0={self.hidden_size}, "
                    f"expert {idx}={attn.hidden_size}."
                )
            if attn.num_attention_heads != self.num_attention_heads:
                raise ValueError(
                    f"All packed experts must share num_attention_heads; expert 0="
                    f"{self.num_attention_heads}, expert {idx}={attn.num_attention_heads}."
                )
            if blk.pre_ln != self.pre_ln:
                raise ValueError("All packed experts must share pre_ln mode.")

        self.attn_score_dropout_p = first_attn.attn_dropout.p
        self.attn_layer_dropout_p = first_attn.layer_dropout.p
        self.ffn_dropout_p = blocks[0].second_sub_layer.layer_dropout.p

        # Fused QKV projection (1 bmm replaces 3 bmm calls).
        self.qkv = PackedFusedQKVBank.from_qkv_linears(
            [blk.first_sub_layer.query_net for blk in blocks],
            [blk.first_sub_layer.key_net for blk in blocks],
            [blk.first_sub_layer.value_net for blk in blocks],
        )
        self.out_projection = PackedLinearBank.from_linears(
            [blk.first_sub_layer.out_projection for blk in blocks]
        )

        act_fn = _resolve_act_fn(blocks)
        self.ffn = PackedFFNBank.from_position_wise_ffs(
            [blk.second_sub_layer for blk in blocks], act_fn
        )

        self.layer_norm_1 = PackedLayerNormBank.from_layer_norms([blk.layer_norm_1 for blk in blocks])
        self.layer_norm_2 = PackedLayerNormBank.from_layer_norms([blk.layer_norm_2 for blk in blocks])

    # -- Helpers ---------------------------------------------------------

    def _dropout(self, x: torch.Tensor, p: float) -> torch.Tensor:
        return F.dropout(x, p=p, training=self.training) if p > 0.0 else x

    def _self_attention_batched(self, x_stacked: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        E, B, L, H = x_stacked.shape
        A = self.num_attention_heads
        D = self.attn_head_size

        # ONE bmm produces Q/K/V for every expert.
        q, k, v = self.qkv(x_stacked, mode="batched")  # each (E, B, L, H)

        q = q.view(E, B, L, A, D).permute(0, 1, 3, 2, 4)  # (E, B, A, L, D)
        k = k.view(E, B, L, A, D).permute(0, 1, 3, 2, 4)
        v = v.view(E, B, L, A, D).permute(0, 1, 3, 2, 4)

        # Match NeMo's scaling: pre-divide q and k by sqrt(sqrt(D)).
        q = q / self.attn_scale
        k = k / self.attn_scale

        scores = torch.matmul(q, k.transpose(-1, -2))  # (E, B, A, L, L)
        if attn_mask is not None:
            scores = scores + attn_mask.to(scores.dtype)
        probs = torch.softmax(scores, dim=-1)
        probs = self._dropout(probs, self.attn_score_dropout_p)

        context = torch.matmul(probs, v)  # (E, B, A, L, D)
        context = context.permute(0, 1, 3, 2, 4).contiguous().view(E, B, L, H)

        out = self.out_projection(context, mode="batched")
        return self._dropout(out, self.attn_layer_dropout_p)

    def _self_attention_per_expert(
        self, x_stacked: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        E, B, L, H = x_stacked.shape
        A = self.num_attention_heads
        D = self.attn_head_size

        outs = []
        for i in range(E):
            xi = x_stacked[i]
            # Three SEPARATE F.linear calls -> bit-exact against NeMo's serial
            # MultiHeadAttention (which also calls Q/K/V independently).
            w = self.qkv.weight
            b = self.qkv.bias
            qi = F.linear(xi, w[i, 0 * H : 1 * H, :], b[i, 0 * H : 1 * H] if b is not None else None)
            ki = F.linear(xi, w[i, 1 * H : 2 * H, :], b[i, 1 * H : 2 * H] if b is not None else None)
            vi = F.linear(xi, w[i, 2 * H : 3 * H, :], b[i, 2 * H : 3 * H] if b is not None else None)

            qi = qi.view(B, L, A, D).permute(0, 2, 1, 3) / self.attn_scale
            ki = ki.view(B, L, A, D).permute(0, 2, 1, 3) / self.attn_scale
            vi = vi.view(B, L, A, D).permute(0, 2, 1, 3)

            scores = torch.matmul(qi, ki.transpose(-1, -2))
            if attn_mask is not None:
                scores = scores + attn_mask[i].to(scores.dtype)
            probs = torch.softmax(scores, dim=-1)
            probs = self._dropout(probs, self.attn_score_dropout_p)

            context = torch.matmul(probs, vi)
            context = context.permute(0, 2, 1, 3).contiguous().view(B, L, H)

            out = F.linear(context, self.out_projection.weight[i], self.out_projection.bias[i])
            out = self._dropout(out, self.attn_layer_dropout_p)
            outs.append(out)
        return torch.stack(outs, dim=0)

    def _self_attention(self, x_stacked: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        if self.compute_mode == "per_expert":
            return self._self_attention_per_expert(x_stacked, attn_mask)
        return self._self_attention_batched(x_stacked, attn_mask)

    def _ffn(self, x_stacked: torch.Tensor) -> torch.Tensor:
        return self.ffn(x_stacked, mode=self.compute_mode)

    def _ln(self, ln: PackedLayerNormBank, x_stacked: torch.Tensor) -> torch.Tensor:
        return ln(x_stacked, mode=self.compute_mode)

    # -- Forward variants -----------------------------------------------

    def forward_postln(self, x_stacked: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        attn = self._self_attention(x_stacked, attn_mask)
        attn = attn + x_stacked
        attn = self._ln(self.layer_norm_1, attn)

        ffn = self._ffn(attn)
        ffn = self._dropout(ffn, self.ffn_dropout_p)
        ffn = ffn + attn
        return self._ln(self.layer_norm_2, ffn)

    def forward_preln(self, x_stacked: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        normed = self._ln(self.layer_norm_1, x_stacked)
        attn = self._self_attention(normed, attn_mask)
        attn = attn + x_stacked

        normed = self._ln(self.layer_norm_2, attn)
        ffn = self._ffn(normed)
        ffn = self._dropout(ffn, self.ffn_dropout_p)
        return ffn + attn

    def forward(self, x_stacked: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        if self.pre_ln:
            return self.forward_preln(x_stacked, attn_mask)
        return self.forward_postln(x_stacked, attn_mask)


class ParallelExpertTransformerEncoder(nn.Module):
    """A bank of ``E`` ``TransformerEncoder`` experts packed for batched compute.

    All experts must share ``hidden_size``, ``num_attention_heads``,
    ``num_layers``, ``pre_ln`` mode and activation. FFN ``inner_size`` may
    differ.
    """

    def __init__(self, encoders: Sequence[TransformerEncoder], compute_mode: str = "batched"):
        super().__init__()
        if len(encoders) < 1:
            raise ValueError("encoders list must contain at least one TransformerEncoder.")
        if compute_mode not in {"batched", "per_expert"}:
            raise ValueError(f"compute_mode must be 'batched' or 'per_expert'; got {compute_mode!r}.")

        num_layers = len(encoders[0].layers)
        hidden_size = encoders[0].d_model
        for k, enc in enumerate(encoders[1:], start=1):
            if len(enc.layers) != num_layers:
                raise ValueError(
                    f"All encoders must have the same num_layers. "
                    f"Encoder 0 has {num_layers} layers; encoder {k} has {len(enc.layers)}."
                )
            if enc.d_model != hidden_size:
                raise ValueError(
                    f"All encoders must share d_model (hidden_size). "
                    f"Encoder 0 has {hidden_size}; encoder {k} has {enc.d_model}."
                )

        self.num_experts = len(encoders)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self._compute_mode = compute_mode
        self.diags = [enc.diag for enc in encoders]

        self.layers = nn.ModuleList(
            [
                ParallelExpertTransformerEncoderBlock(
                    [enc.layers[i] for enc in encoders], compute_mode=compute_mode
                )
                for i in range(num_layers)
            ]
        )

        final_norms = [enc.final_layer_norm for enc in encoders]
        if all(norm is None for norm in final_norms):
            self.final_layer_norm: Optional[PackedLayerNormBank] = None
        elif all(norm is not None for norm in final_norms):
            self.final_layer_norm = PackedLayerNormBank.from_layer_norms(
                [n for n in final_norms if n is not None]
            )
        else:
            raise ValueError(
                "All packed encoders must either share or all omit ``final_layer_norm``."
            )

    @property
    def compute_mode(self) -> str:
        return self._compute_mode

    def set_compute_mode(self, compute_mode: str) -> None:
        if compute_mode not in {"batched", "per_expert"}:
            raise ValueError(f"compute_mode must be 'batched' or 'per_expert'; got {compute_mode!r}.")
        self._compute_mode = compute_mode
        for layer in self.layers:
            assert isinstance(layer, ParallelExpertTransformerEncoderBlock)
            layer.compute_mode = compute_mode

    @classmethod
    def from_transformer_encoders(
        cls,
        encoders: Sequence[TransformerEncoder],
        compute_mode: str = "batched",
    ) -> "ParallelExpertTransformerEncoder":
        return cls(encoders, compute_mode=compute_mode)

    # -- Forward --------------------------------------------------------

    def _stack_inputs(self, encoder_states_list: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(encoder_states_list) != self.num_experts:
            raise ValueError(
                f"Expected {self.num_experts} input tensors, got {len(encoder_states_list)}."
            )
        ref_shape = encoder_states_list[0].shape
        if ref_shape[-1] != self.hidden_size:
            raise ValueError(
                f"Inputs must have last dim={self.hidden_size}; got {ref_shape[-1]}."
            )
        for idx, t in enumerate(encoder_states_list[1:], start=1):
            if t.shape != ref_shape:
                raise ValueError(
                    f"All expert inputs must share shape; expert 0={tuple(ref_shape)}, "
                    f"expert {idx}={tuple(t.shape)}."
                )
        return torch.stack(list(encoder_states_list), dim=0)

    def _stack_attn_masks(self, encoder_mask_list: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(encoder_mask_list) != self.num_experts:
            raise ValueError(
                f"Expected {self.num_experts} masks, got {len(encoder_mask_list)}."
            )
        attn_masks = []
        for mask, diag in zip(encoder_mask_list, self.diags):
            attn_masks.append(form_attention_mask(mask, diag))
        if any(m is None for m in attn_masks):
            if not all(m is None for m in attn_masks):
                raise ValueError("All attention masks must be present or all absent.")
            return None  # type: ignore[return-value]
        return torch.stack(attn_masks, dim=0)

    def forward(
        self,
        encoder_states_list: Sequence[torch.Tensor],
        encoder_mask_list: Sequence[torch.Tensor],
    ) -> List[torch.Tensor]:
        x_stacked = self._stack_inputs(encoder_states_list)  # (E, B, L, H)
        attn_mask_stacked = self._stack_attn_masks(encoder_mask_list)
        for layer in self.layers:
            x_stacked = layer(x_stacked, attn_mask_stacked)
        if self.final_layer_norm is not None:
            x_stacked = self.final_layer_norm(x_stacked, mode=self._compute_mode)
        return [x_stacked[e] for e in range(self.num_experts)]
