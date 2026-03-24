"""
Mixture-of-Experts (MoE) Transformer Encoder for ASR.

Extends the standard TransformerEncoder by replacing the feed-forward network
(FFN) in each Transformer layer with a MoE feed-forward module
(MoEFeedForward). Supports both per-layer (switch) routing and shared (omni)
routing across all MoE layers.

References:
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models
  with Simple and Efficient Sparsity", 2022
- Gu et al., "Omni-Router: Sharing Routing Decisions in Sparse MoE for ASR", 2025
"""

import re
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.collections.asr.modules.transformer_encoder import (
    TransformerEncoder,
    FeedForward,
)

__all__ = ['SwitchGate', 'MoEFeedForward', 'MoETransformerEncoder']


class SwitchGate(nn.Module):
    """Softmax routing gate for Mixture-of-Experts layers.

    Computes routing probabilities over N experts for each input token using
    a learned linear projection followed by softmax. Optionally adds Gaussian
    noise during training to encourage exploration (jitter).

    Can be shared across multiple MoE layers (omni-router) or used independently
    per layer (switch-style).

    Args:
        d_model (int): Input feature dimension.
        num_experts (int): Number of experts to route over.
        jitter_eps (float): Standard deviation of Gaussian noise added to logits
            during training. 0.0 disables noise. Defaults to 0.0.
    """

    def __init__(self, d_model: int, num_experts: int, jitter_eps: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.jitter_eps = jitter_eps
        self.w_gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute routing probabilities for input tokens.

        Args:
            x: Input tensor of shape (num_tokens, d_model).

        Returns:
            Routing probabilities of shape (num_tokens, num_experts).
        """
        logits = self.w_gate(x)
        if self.training and self.jitter_eps > 0.0:
            logits = logits + torch.randn_like(logits) * self.jitter_eps
        return F.softmax(logits, dim=-1)


class MoEFeedForward(nn.Module):
    """Mixture-of-Experts Feed-Forward module -- drop-in replacement for FeedForward.

    Contains N expert FFNs (each a FeedForward) and a router (SwitchGate).
    For each input token, the router selects the top-k experts and computes a
    weighted combination of their outputs.

    Computes an auxiliary load-balancing loss (GShard-style) stored in
    ``self._aux_loss``.

    Args:
        d_model (int): Input and output feature dimension.
        num_experts (int): Number of expert FFNs.
        top_k (int): Number of experts activated per token. Defaults to 1.
        router (SwitchGate, optional): External router (for omni-router sharing).
            If None, creates an internal router. Defaults to None.
        jitter_eps (float): Jitter noise for internal router (ignored if external
            router is provided). Defaults to 0.0.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 1,
        router: Optional[SwitchGate] = None,
        jitter_eps: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([FeedForward(d_model) for _ in range(num_experts)])

        if router is not None:
            self.router = router
        else:
            self.router = SwitchGate(d_model=d_model, num_experts=num_experts, jitter_eps=jitter_eps)

        self._aux_loss = None

    def _compute_load_balancing_loss(self, gate_probs: torch.Tensor, expert_mask: torch.Tensor) -> torch.Tensor:
        """GShard / Switch Transformer load-balancing auxiliary loss.

        L_load = N * sum_j(f_j * rho_j) where:
        - f_j = fraction of tokens dispatched to expert j
        - rho_j = mean router probability allocated to expert j
        """
        num_tokens = gate_probs.shape[0]
        f = expert_mask.float().sum(dim=0) / num_tokens
        rho = gate_probs.mean(dim=0)
        return self.num_experts * (f * rho).sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass -- same signature as FeedForward.forward(x).

        Args:
            x: Input tensor of shape (B, T, D).

        Returns:
            Output tensor of shape (B, T, D).
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.reshape(-1, d_model)
        num_tokens = x_flat.shape[0]

        gate_probs = self.router(x_flat)

        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)

        expert_mask = torch.zeros(num_tokens, self.num_experts, device=x.device, dtype=x.dtype)
        expert_mask.scatter_(1, top_k_indices, 1.0)

        self._aux_loss = self._compute_load_balancing_loss(gate_probs, expert_mask)

        output = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]
            expert_weights = top_k_probs[:, k]

            for i in range(self.num_experts):
                mask = expert_indices == i
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[i](expert_input)
                    output[mask] += expert_output * expert_weights[mask].unsqueeze(-1)

        return output.reshape(batch_size, seq_len, d_model)


class MoETransformerEncoder(TransformerEncoder):
    """Transformer Encoder with Mixture-of-Experts (MoE) feed-forward layers.

    Subclasses :class:`TransformerEncoder` and replaces the FFN in each
    Transformer layer with :class:`MoEFeedForward` modules for configurable
    layers.

    Two routing strategies are supported:

    - ``'switch'``: Each MoE layer has its own independent router.
    - ``'omni'``: A single shared router is used across all MoE layers,
      encouraging coordinated expert specialization (Omni-router).

    Args:
        All standard TransformerEncoder args, plus:

        moe_num_experts (int): Number of experts per MoE layer. Defaults to 8.
        moe_top_k (int): Number of experts activated per token. Defaults to 1.
        moe_router_type (str): Router strategy.
            ``'omni'`` = shared router across layers,
            ``'switch'`` = independent router per layer. Defaults to ``'omni'``.
        moe_layer_indices (list[int] or None): Indices of Transformer layers to
            apply MoE to. ``None`` means all layers. Defaults to None.
        moe_load_balance_loss_weight (float): Weight for the auxiliary
            load-balancing loss. Defaults to 0.01.
        moe_jitter_eps (float): Noise scale for the router during training.
            Defaults to 0.0.
        moe_init_from_ffn (bool): If True, initialize all expert FFN weights
            from the original (base) FFN weights created by ``super().__init__()``.
            Defaults to True.
    """

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 17,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
        causal_mask: bool = False,
        pre_encode: str = "conv",
        nan_debug: bool = True,
        qk_norm: bool = False,
        # MoE parameters
        moe_num_experts: int = 8,
        moe_top_k: int = 1,
        moe_router_type: str = 'omni',
        moe_layer_indices: Optional[List[int]] = None,
        moe_load_balance_loss_weight: float = 0.01,
        moe_jitter_eps: float = 0.0,
        moe_init_from_ffn: bool = True,
    ):
        super().__init__(
            n_mels=n_mels,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            drop_rate=drop_rate,
            qkv_bias=qkv_bias,
            causal_mask=causal_mask,
            pre_encode=pre_encode,
            nan_debug=nan_debug,
            qk_norm=qk_norm,
        )

        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_router_type = moe_router_type
        self.moe_load_balance_loss_weight = moe_load_balance_loss_weight
        self.moe_jitter_eps = moe_jitter_eps
        self.moe_init_from_ffn = moe_init_from_ffn

        if moe_router_type not in ('omni', 'switch'):
            raise ValueError(
                f"moe_router_type must be one of 'omni', 'switch', got '{moe_router_type}'"
            )

        if moe_layer_indices is not None:
            self.moe_layer_indices = list(moe_layer_indices)
        else:
            self.moe_layer_indices = list(range(n_layers))

        for idx in self.moe_layer_indices:
            if idx < 0 or idx >= n_layers:
                raise ValueError(
                    f"moe_layer_indices contains invalid index {idx} for encoder with {n_layers} layers."
                )

        # Create shared router for omni-router mode
        if moe_router_type == 'omni':
            self.omni_router = SwitchGate(
                d_model=d_model, num_experts=moe_num_experts, jitter_eps=moe_jitter_eps
            )
        else:
            self.omni_router = None

        # Replace FFN modules in selected layers with MoEFeedForward
        for layer_idx in self.moe_layer_indices:
            layer = self.layers[layer_idx]

            original_ffn_state = layer.ffn.state_dict() if moe_init_from_ffn else None

            router = self.omni_router if moe_router_type == 'omni' else None
            moe_ffn = MoEFeedForward(
                d_model=d_model,
                num_experts=moe_num_experts,
                top_k=moe_top_k,
                router=router,
                jitter_eps=moe_jitter_eps,
            )

            if original_ffn_state is not None:
                for expert in moe_ffn.experts:
                    expert.load_state_dict(original_ffn_state)

            layer.ffn = moe_ffn

        num_moe_layers = len(self.moe_layer_indices)
        print(
            f"MoETransformerEncoder: Replaced {num_moe_layers} FFN modules with MoE "
            f"(experts={moe_num_experts}, top_k={moe_top_k}, "
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
            ff_module = layer.ffn

            if isinstance(ff_module, MoEFeedForward) and ff_module._aux_loss is not None:
                if total_loss is None:
                    total_loss = ff_module._aux_loss
                else:
                    total_loss = total_loss + ff_module._aux_loss

        if total_loss is not None:
            total_loss = self.moe_load_balance_loss_weight * total_loss

        return total_loss

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Override to remap non-MoE FFN state_dict keys to MoE expert keys.

        When loading from a pretrained TransformerEncoder checkpoint, the FFN
        keys have the form::

            layers.{i}.ffn.ffn.0.weight

        but the MoE encoder expects::

            layers.{i}.ffn.experts.{j}.ffn.0.weight

        This method detects such mismatches and duplicates the pretrained FFN
        weights into all expert slots, so every expert starts from the pretrained
        FFN weights. The router weights (which have no pretrained counterpart)
        remain at their random initialization.
        """
        if not self.moe_init_from_ffn:
            return super()._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
            )

        moe_layer_set = set(self.moe_layer_indices)

        # Pattern: prefix + layers.{layer_idx}.ffn.{param_suffix}
        # where param_suffix does NOT start with "experts." or "router."
        pattern = re.compile(
            r'^(' + re.escape(prefix) + r'layers\.(\d+)\.ffn)\.((?!experts\.|router\.).+)$'
        )

        keys_to_add = {}
        keys_to_remove = []

        for key in list(state_dict.keys()):
            m = pattern.match(key)
            if m:
                layer_prefix = m.group(1)  # e.g. "layers.0.ffn"
                layer_idx = int(m.group(2))
                param_suffix = m.group(3)  # e.g. "ffn.0.weight"

                if layer_idx in moe_layer_set:
                    value = state_dict[key]
                    for expert_idx in range(self.moe_num_experts):
                        new_key = f"{layer_prefix}.experts.{expert_idx}.{param_suffix}"
                        keys_to_add[new_key] = value.clone()
                    keys_to_remove.append(key)

        for key in keys_to_remove:
            del state_dict[key]
        state_dict.update(keys_to_add)

        if keys_to_remove:
            print(
                f"MoETransformerEncoder: Remapped {len(keys_to_remove)} pretrained FFN keys "
                f"into {len(keys_to_add)} expert keys for {len(self.moe_layer_indices)} MoE layers."
            )

        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
