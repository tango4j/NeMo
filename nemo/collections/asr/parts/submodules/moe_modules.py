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
Mixture-of-Experts (MoE) modules for Conformer-based ASR encoders.

Implements:
- SwitchGate: A softmax router/gating network for MoE layers.
- MoEFeedForward: A drop-in replacement for ConformerFeedForward that routes
  tokens to multiple expert FFNs via top-k gating, with auxiliary load-balancing loss.

References:
- Hu et al., "Mixture-of-Expert Conformer for Streaming Multilingual ASR", 2023
  (https://arxiv.org/abs/2305.15663)
- Gu et al., "Omni-Router: Sharing Routing Decisions in Sparse Mixture-of-Experts
  for Speech Recognition", 2025 (https://arxiv.org/abs/2507.05724)
- Fedus et al., "Switch Transformers", 2022
- Lepikhin et al., "GShard", 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.collections.asr.parts.submodules.conformer_modules import ConformerFeedForward
from nemo.collections.asr.parts.utils.activations import Swish

__all__ = ['SwitchGate', 'MoEFeedForward']


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

    def forward(self, x: torch.Tensor):
        """Compute routing probabilities for input tokens.

        Args:
            x (torch.Tensor): Input tensor of shape (num_tokens, d_model).

        Returns:
            gate_probs (torch.Tensor): Routing probabilities of shape (num_tokens, num_experts).
                Each row sums to 1.
        """
        logits = self.w_gate(x)
        if self.training and self.jitter_eps > 0.0:
            logits = logits + torch.randn_like(logits) * self.jitter_eps
        gate_probs = F.softmax(logits, dim=-1)
        return gate_probs


class MoEFeedForward(nn.Module):
    """Mixture-of-Experts Feed-Forward module -- drop-in replacement for ConformerFeedForward.

    Contains N expert FFNs (each a ConformerFeedForward) and a router (SwitchGate).
    For each input token, the router selects the top-k experts and computes a
    weighted combination of their outputs.

    The module also computes an auxiliary load-balancing loss (GShard-style) to
    encourage even expert utilization, stored in ``self._aux_loss``.

    Args:
        d_model (int): Input and output feature dimension.
        d_ff (int): Hidden dimension of each expert FFN.
        num_experts (int): Number of expert FFNs.
        top_k (int): Number of experts activated per token. Defaults to 1.
        dropout (float): Dropout rate for expert FFNs. Defaults to 0.1.
        activation: Activation function for expert FFNs. Defaults to Swish().
        use_bias (bool): Whether to use bias in expert FFN linear layers. Defaults to True.
        router (SwitchGate, optional): External router instance (for omni-router sharing).
            If None, creates an internal router. Defaults to None.
        jitter_eps (float): Jitter noise for internal router (ignored if external router
            is provided). Defaults to 0.0.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int = 1,
        dropout: float = 0.1,
        activation=Swish(),
        use_bias: bool = True,
        router: SwitchGate = None,
        jitter_eps: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k

        # Create expert FFNs
        self.experts = nn.ModuleList(
            [
                ConformerFeedForward(
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                    use_bias=use_bias,
                )
                for _ in range(num_experts)
            ]
        )

        # Router: use external (omni-router) or create internal (switch)
        if router is not None:
            self.router = router
        else:
            self.router = SwitchGate(d_model=d_model, num_experts=num_experts, jitter_eps=jitter_eps)

        # Auxiliary load-balancing loss, set during forward
        self._aux_loss = None

    def _compute_load_balancing_loss(self, gate_probs: torch.Tensor, expert_mask: torch.Tensor) -> torch.Tensor:
        """Compute the GShard / Switch Transformer load-balancing auxiliary loss.

        L_load = N * sum_j(f_j * rho_j) where:
        - f_j = fraction of tokens dispatched to expert j
        - rho_j = mean router probability allocated to expert j

        Args:
            gate_probs (torch.Tensor): Router probabilities, shape (num_tokens, num_experts).
            expert_mask (torch.Tensor): Binary mask indicating which experts were selected
                for each token, shape (num_tokens, num_experts).

        Returns:
            torch.Tensor: Scalar load-balancing loss.
        """
        num_tokens = gate_probs.shape[0]
        # f_j: fraction of tokens routed to each expert
        f = expert_mask.float().sum(dim=0) / num_tokens
        # rho_j: mean routing probability for each expert
        rho = gate_probs.mean(dim=0)
        return self.num_experts * (f * rho).sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass -- same signature as ConformerFeedForward.forward(x).

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, D).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, D).
        """
        batch_size, seq_len, d_model = x.shape
        # Flatten to (B*T, D)
        x_flat = x.reshape(-1, d_model)
        num_tokens = x_flat.shape[0]

        # Get routing probabilities: (num_tokens, num_experts)
        gate_probs = self.router(x_flat)

        # Select top-k experts per token
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)  # (num_tokens, top_k)

        # Normalize top-k probabilities to sum to 1
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # Build expert assignment mask for load-balancing loss: (num_tokens, num_experts)
        expert_mask = torch.zeros(num_tokens, self.num_experts, device=x.device, dtype=x.dtype)
        expert_mask.scatter_(1, top_k_indices, 1.0)

        # Compute and store auxiliary loss
        self._aux_loss = self._compute_load_balancing_loss(gate_probs, expert_mask)

        # Dispatch tokens to experts and combine outputs
        output = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]  # (num_tokens,)
            expert_weights = top_k_probs[:, k]  # (num_tokens,)

            for i in range(self.num_experts):
                mask = expert_indices == i
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[i](expert_input)
                    output[mask] += expert_output * expert_weights[mask].unsqueeze(-1)

        return output.reshape(batch_size, seq_len, d_model)
