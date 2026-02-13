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
Mixture-of-Experts (MoE) Conformer Encoder for ASR.

Extends the standard ConformerEncoder by optionally replacing the first and/or
second feed-forward network (FFN) in each Conformer layer with a
MoE feed-forward module (MoEFeedForward). Supports both per-layer (switch)
routing and shared (omni) routing across all MoE layers.

References:
- Hu et al., "Mixture-of-Expert Conformer for Streaming Multilingual ASR", 2023
- Gu et al., "Omni-Router: Sharing Routing Decisions in Sparse MoE for ASR", 2025
"""

from typing import List, Optional

import torch

from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.parts.submodules.moe_modules import MoEFeedForward, SwitchGate
from nemo.collections.asr.parts.utils.activations import Swish
from nemo.utils import logging

__all__ = ['MoEConformerEncoder']


class MoEConformerEncoder(ConformerEncoder):
    """Conformer Encoder with Mixture-of-Experts (MoE) feed-forward layers.

    Subclasses :class:`ConformerEncoder` and replaces the first FFN, second FFN,
    or both with :class:`MoEFeedForward` modules for configurable Conformer layers.

    Two routing strategies are supported:

    - ``'switch'``: Each MoE layer has its own independent router.
    - ``'omni'``: A single shared router is used across all MoE layers,
      encouraging coordinated expert specialization (Omni-router).

    Args:
        All standard ConformerEncoder args, plus:

        moe_num_experts (int): Number of experts per MoE layer. Defaults to 8.
        moe_top_k (int): Number of experts activated per token. Defaults to 1.
        moe_position (str): Which FFN(s) to replace with MoE.
            ``'start'`` = first FFN only, ``'end'`` = second FFN only,
            ``'both'`` = both FFNs. Defaults to ``'end'``.
        moe_router_type (str): Router strategy.
            ``'omni'`` = shared router across layers,
            ``'switch'`` = independent router per layer. Defaults to ``'omni'``.
        moe_layer_indices (list[int] or None): Indices of Conformer layers to
            apply MoE to. ``None`` means all layers. Defaults to None.
        moe_load_balance_loss_weight (float): Weight for the auxiliary
            load-balancing loss. Defaults to 0.01.
        moe_jitter_eps (float): Noise scale for the router during training.
            Defaults to 0.0.
    """

    def __init__(
        self,
        feat_in,
        n_layers,
        d_model,
        feat_out=-1,
        causal_downsampling=False,
        subsampling='striding',
        subsampling_factor=4,
        subsampling_conv_chunking_factor=1,
        subsampling_conv_channels=-1,
        reduction=None,
        reduction_position=None,
        reduction_factor=1,
        ff_expansion_factor=4,
        self_attention_model='rel_pos',
        n_heads=4,
        att_context_size=None,
        att_context_probs=None,
        att_context_style='regular',
        xscaling=True,
        untie_biases=True,
        pos_emb_max_len=5000,
        conv_kernel_size=31,
        conv_norm_type='batch_norm',
        conv_context_size=None,
        use_bias=True,
        dropout=0.1,
        dropout_pre_encoder=0.1,
        dropout_emb=0.1,
        dropout_att=0.0,
        stochastic_depth_drop_prob: float = 0.0,
        stochastic_depth_mode: str = "linear",
        stochastic_depth_start_layer: int = 1,
        global_tokens: int = 0,
        global_tokens_spacing: int = 1,
        global_attn_separate: bool = False,
        use_pytorch_sdpa: bool = False,
        use_pytorch_sdpa_backends=None,
        sync_max_audio_length: bool = True,
        # MoE parameters
        moe_num_experts: int = 8,
        moe_top_k: int = 1,
        moe_position: str = 'end',
        moe_router_type: str = 'omni',
        moe_layer_indices: Optional[List[int]] = None,
        moe_load_balance_loss_weight: float = 0.01,
        moe_jitter_eps: float = 0.0,
    ):
        # Initialize the base ConformerEncoder (creates all standard layers)
        super().__init__(
            feat_in=feat_in,
            n_layers=n_layers,
            d_model=d_model,
            feat_out=feat_out,
            causal_downsampling=causal_downsampling,
            subsampling=subsampling,
            subsampling_factor=subsampling_factor,
            subsampling_conv_chunking_factor=subsampling_conv_chunking_factor,
            subsampling_conv_channels=subsampling_conv_channels,
            reduction=reduction,
            reduction_position=reduction_position,
            reduction_factor=reduction_factor,
            ff_expansion_factor=ff_expansion_factor,
            self_attention_model=self_attention_model,
            n_heads=n_heads,
            att_context_size=att_context_size,
            att_context_probs=att_context_probs,
            att_context_style=att_context_style,
            xscaling=xscaling,
            untie_biases=untie_biases,
            pos_emb_max_len=pos_emb_max_len,
            conv_kernel_size=conv_kernel_size,
            conv_norm_type=conv_norm_type,
            conv_context_size=conv_context_size,
            use_bias=use_bias,
            dropout=dropout,
            dropout_pre_encoder=dropout_pre_encoder,
            dropout_emb=dropout_emb,
            dropout_att=dropout_att,
            stochastic_depth_drop_prob=stochastic_depth_drop_prob,
            stochastic_depth_mode=stochastic_depth_mode,
            stochastic_depth_start_layer=stochastic_depth_start_layer,
            global_tokens=global_tokens,
            global_tokens_spacing=global_tokens_spacing,
            global_attn_separate=global_attn_separate,
            use_pytorch_sdpa=use_pytorch_sdpa,
            use_pytorch_sdpa_backends=use_pytorch_sdpa_backends,
            sync_max_audio_length=sync_max_audio_length,
        )

        # Store MoE config
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_position = moe_position
        self.moe_router_type = moe_router_type
        self.moe_load_balance_loss_weight = moe_load_balance_loss_weight
        self.moe_jitter_eps = moe_jitter_eps

        # Validate moe_position
        if moe_position not in ('start', 'end', 'both'):
            raise ValueError(
                f"moe_position must be one of 'start', 'end', 'both', got '{moe_position}'"
            )

        # Validate moe_router_type
        if moe_router_type not in ('omni', 'switch'):
            raise ValueError(
                f"moe_router_type must be one of 'omni', 'switch', got '{moe_router_type}'"
            )

        # Determine which layer indices get MoE
        if moe_layer_indices is not None:
            self.moe_layer_indices = list(moe_layer_indices)
        else:
            self.moe_layer_indices = list(range(n_layers))

        # Validate layer indices
        for idx in self.moe_layer_indices:
            if idx < 0 or idx >= n_layers:
                raise ValueError(
                    f"moe_layer_indices contains invalid index {idx} for encoder with {n_layers} layers."
                )

        d_ff = d_model * ff_expansion_factor

        # Create shared routers for omni-router mode
        if moe_router_type == 'omni':
            if moe_position in ('start', 'both'):
                self.omni_router_start = SwitchGate(
                    d_model=d_model, num_experts=moe_num_experts, jitter_eps=moe_jitter_eps
                )
            else:
                self.omni_router_start = None

            if moe_position in ('end', 'both'):
                self.omni_router_end = SwitchGate(
                    d_model=d_model, num_experts=moe_num_experts, jitter_eps=moe_jitter_eps
                )
            else:
                self.omni_router_end = None
        else:
            self.omni_router_start = None
            self.omni_router_end = None

        # Replace FFN modules in selected layers with MoEFeedForward
        for layer_idx in self.moe_layer_indices:
            layer = self.layers[layer_idx]

            if moe_position in ('start', 'both'):
                router = self.omni_router_start if moe_router_type == 'omni' else None
                moe_ff1 = MoEFeedForward(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_experts=moe_num_experts,
                    top_k=moe_top_k,
                    dropout=dropout,
                    activation=Swish(),
                    use_bias=use_bias,
                    router=router,
                    jitter_eps=moe_jitter_eps,
                )
                layer.feed_forward1 = moe_ff1

            if moe_position in ('end', 'both'):
                router = self.omni_router_end if moe_router_type == 'omni' else None
                moe_ff2 = MoEFeedForward(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_experts=moe_num_experts,
                    top_k=moe_top_k,
                    dropout=dropout,
                    activation=Swish(),
                    use_bias=use_bias,
                    router=router,
                    jitter_eps=moe_jitter_eps,
                )
                layer.feed_forward2 = moe_ff2

        num_moe_layers = len(self.moe_layer_indices)
        num_moe_ffn = num_moe_layers * (2 if moe_position == 'both' else 1)
        logging.info(
            f"MoEConformerEncoder: Replaced {num_moe_ffn} FFN modules with MoE "
            f"(position='{moe_position}', experts={moe_num_experts}, top_k={moe_top_k}, "
            f"router='{moe_router_type}', layers={self.moe_layer_indices})"
        )

    def get_moe_auxiliary_loss(self) -> Optional[torch.Tensor]:
        """Collect and return the weighted sum of auxiliary load-balancing losses
        from all MoE feed-forward modules.

        Returns:
            torch.Tensor or None: Weighted scalar auxiliary loss, or None if
                no MoE layers have computed a loss yet.
        """
        total_loss = None

        for layer_idx in self.moe_layer_indices:
            layer = self.layers[layer_idx]

            for ff_attr in ('feed_forward1', 'feed_forward2'):
                ff_module = getattr(layer, ff_attr, None)
                if isinstance(ff_module, MoEFeedForward) and ff_module._aux_loss is not None:
                    if total_loss is None:
                        total_loss = ff_module._aux_loss
                    else:
                        total_loss = total_loss + ff_module._aux_loss

        if total_loss is not None:
            total_loss = self.moe_load_balance_loss_weight * total_loss

        return total_loss
