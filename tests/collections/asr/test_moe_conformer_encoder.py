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
Tests for Mixture-of-Experts (MoE) Conformer encoder components.

Tests cover:
- SwitchGate router module
- MoEFeedForward module
- MoEConformerEncoder creation and forward pass
- Omni-router (shared) vs. switch (per-layer) routing
- Configurable MoE layer indices
- Auxiliary load-balancing loss computation
"""

import pytest
import torch

from nemo.collections.asr.parts.submodules.moe_modules import MoEFeedForward, SwitchGate
from nemo.collections.asr.parts.submodules.conformer_modules import ConformerFeedForward
from nemo.collections.asr.modules.moe_conformer_encoder import MoEConformerEncoder


class TestSwitchGate:
    """Tests for the SwitchGate router module."""

    def test_output_shape(self):
        """Router output should have shape (num_tokens, num_experts) and sum to 1."""
        d_model = 16
        num_experts = 4
        num_tokens = 32

        gate = SwitchGate(d_model=d_model, num_experts=num_experts)
        x = torch.randn(num_tokens, d_model)
        probs = gate(x)

        assert probs.shape == (num_tokens, num_experts)
        # Each row should sum to ~1.0
        row_sums = probs.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(num_tokens), atol=1e-5)

    def test_output_non_negative(self):
        """Router probabilities should be non-negative (softmax output)."""
        gate = SwitchGate(d_model=8, num_experts=3)
        x = torch.randn(10, 8)
        probs = gate(x)
        assert (probs >= 0).all()

    def test_jitter_noise_during_training(self):
        """With jitter_eps > 0, training outputs should differ across calls."""
        gate = SwitchGate(d_model=8, num_experts=4, jitter_eps=1.0)
        gate.train()
        x = torch.randn(10, 8)

        probs1 = gate(x)
        probs2 = gate(x)
        # With high jitter, outputs should differ
        assert not torch.allclose(probs1, probs2, atol=1e-6)

    def test_no_jitter_during_eval(self):
        """In eval mode, outputs should be deterministic even with jitter_eps > 0."""
        gate = SwitchGate(d_model=8, num_experts=4, jitter_eps=1.0)
        gate.eval()
        x = torch.randn(10, 8)

        probs1 = gate(x)
        probs2 = gate(x)
        assert torch.allclose(probs1, probs2)


class TestMoEFeedForward:
    """Tests for the MoEFeedForward module."""

    def test_output_shape(self):
        """MoE FFN output shape should match ConformerFeedForward output shape."""
        d_model = 16
        d_ff = 64
        batch_size = 2
        seq_len = 10

        moe_ff = MoEFeedForward(d_model=d_model, d_ff=d_ff, num_experts=4, top_k=1, dropout=0.0)
        x = torch.randn(batch_size, seq_len, d_model)
        out = moe_ff(x)

        assert out.shape == (batch_size, seq_len, d_model)

    def test_output_shape_top2(self):
        """MoE FFN with top_k=2 should produce correct output shape."""
        d_model = 16
        d_ff = 64
        batch_size = 2
        seq_len = 10

        moe_ff = MoEFeedForward(d_model=d_model, d_ff=d_ff, num_experts=4, top_k=2, dropout=0.0)
        x = torch.randn(batch_size, seq_len, d_model)
        out = moe_ff(x)

        assert out.shape == (batch_size, seq_len, d_model)

    def test_aux_loss_computed(self):
        """Auxiliary load-balancing loss should be computed and stored after forward."""
        moe_ff = MoEFeedForward(d_model=16, d_ff=64, num_experts=4, top_k=1, dropout=0.0)
        x = torch.randn(2, 10, 16)

        assert moe_ff._aux_loss is None
        _ = moe_ff(x)
        assert moe_ff._aux_loss is not None
        assert moe_ff._aux_loss.ndim == 0  # scalar
        assert moe_ff._aux_loss.item() >= 0.0

    def test_external_router(self):
        """MoE FFN should accept and use an external router."""
        shared_router = SwitchGate(d_model=16, num_experts=4)
        moe_ff = MoEFeedForward(
            d_model=16, d_ff=64, num_experts=4, top_k=1, dropout=0.0, router=shared_router
        )

        assert moe_ff.router is shared_router

        x = torch.randn(2, 10, 16)
        out = moe_ff(x)
        assert out.shape == (2, 10, 16)

    def test_same_interface_as_conformer_ff(self):
        """MoEFeedForward should have the same forward interface as ConformerFeedForward."""
        d_model = 16
        d_ff = 64

        conformer_ff = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=0.0)
        moe_ff = MoEFeedForward(d_model=d_model, d_ff=d_ff, num_experts=2, top_k=1, dropout=0.0)

        x = torch.randn(2, 10, d_model)

        # Both should accept the same input and produce same shape output
        out_conf = conformer_ff(x)
        out_moe = moe_ff(x)
        assert out_conf.shape == out_moe.shape


class TestMoEConformerEncoder:
    """Tests for the MoEConformerEncoder module."""

    @staticmethod
    def _make_encoder(**kwargs):
        """Helper to create a small test encoder with overridable defaults."""
        defaults = dict(
            feat_in=80,
            n_layers=4,
            d_model=32,
            feat_out=-1,
            n_heads=4,
            conv_kernel_size=3,
            conv_norm_type='layer_norm',
            dropout=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            dropout_att=0.0,
            ff_expansion_factor=4,
            moe_num_experts=4,
            moe_top_k=1,
            moe_position='end',
            moe_router_type='omni',
            moe_layer_indices=None,
            moe_load_balance_loss_weight=0.01,
            moe_jitter_eps=0.0,
        )
        defaults.update(kwargs)
        return MoEConformerEncoder(**defaults)

    def test_encoder_creation_moe_end(self):
        """With moe_position='end', only feed_forward2 should be MoEFeedForward."""
        encoder = self._make_encoder(moe_position='end')

        for layer in encoder.layers:
            assert isinstance(layer.feed_forward2, MoEFeedForward)
            assert isinstance(layer.feed_forward1, ConformerFeedForward)

    def test_encoder_creation_moe_start(self):
        """With moe_position='start', only feed_forward1 should be MoEFeedForward."""
        encoder = self._make_encoder(moe_position='start')

        for layer in encoder.layers:
            assert isinstance(layer.feed_forward1, MoEFeedForward)
            assert isinstance(layer.feed_forward2, ConformerFeedForward)

    def test_encoder_creation_moe_both(self):
        """With moe_position='both', both FFNs should be MoEFeedForward."""
        encoder = self._make_encoder(moe_position='both')

        for layer in encoder.layers:
            assert isinstance(layer.feed_forward1, MoEFeedForward)
            assert isinstance(layer.feed_forward2, MoEFeedForward)

    def test_encoder_forward(self):
        """Encoder forward pass should produce outputs matching standard ConformerEncoder."""
        encoder = self._make_encoder(feat_in=80, d_model=32, feat_out=32)
        encoder.eval()

        batch_size = 2
        seq_len = 100
        audio_signal = torch.randn(batch_size, 80, seq_len)
        length = torch.tensor([seq_len, seq_len // 2], dtype=torch.int64)

        outputs, out_lengths = encoder(audio_signal=audio_signal, length=length)

        # Output shape: (B, D, T_subsampled)
        assert outputs.ndim == 3
        assert outputs.shape[0] == batch_size
        assert outputs.shape[1] == 32  # feat_out

    def test_omni_router_shared(self):
        """With moe_router_type='omni', all MoE layers should share the same router."""
        encoder = self._make_encoder(moe_router_type='omni', moe_position='end')

        routers = []
        for layer in encoder.layers:
            if isinstance(layer.feed_forward2, MoEFeedForward):
                routers.append(layer.feed_forward2.router)

        assert len(routers) > 1
        # All routers should be the same object
        for r in routers[1:]:
            assert r is routers[0]

    def test_omni_router_shared_both(self):
        """With moe_position='both' and omni-router, start and end routers should be different
        shared instances."""
        encoder = self._make_encoder(moe_router_type='omni', moe_position='both')

        start_routers = []
        end_routers = []
        for layer in encoder.layers:
            if isinstance(layer.feed_forward1, MoEFeedForward):
                start_routers.append(layer.feed_forward1.router)
            if isinstance(layer.feed_forward2, MoEFeedForward):
                end_routers.append(layer.feed_forward2.router)

        # All start routers should be the same object
        for r in start_routers[1:]:
            assert r is start_routers[0]

        # All end routers should be the same object
        for r in end_routers[1:]:
            assert r is end_routers[0]

        # Start and end routers should be different objects
        assert start_routers[0] is not end_routers[0]

    def test_switch_router_independent(self):
        """With moe_router_type='switch', each MoE layer should have its own router."""
        encoder = self._make_encoder(moe_router_type='switch', moe_position='end')

        routers = []
        for layer in encoder.layers:
            if isinstance(layer.feed_forward2, MoEFeedForward):
                routers.append(layer.feed_forward2.router)

        assert len(routers) > 1
        # All routers should be different objects
        for i in range(1, len(routers)):
            assert routers[i] is not routers[0]

    def test_moe_layer_indices(self):
        """Only specified layer indices should have MoE FFN modules."""
        encoder = self._make_encoder(
            n_layers=6,
            moe_layer_indices=[0, 2, 4],
            moe_position='end',
        )

        for i, layer in enumerate(encoder.layers):
            if i in [0, 2, 4]:
                assert isinstance(layer.feed_forward2, MoEFeedForward)
            else:
                assert isinstance(layer.feed_forward2, ConformerFeedForward)

    def test_auxiliary_loss(self):
        """After forward pass, get_moe_auxiliary_loss() should return a non-zero scalar."""
        encoder = self._make_encoder()

        batch_size = 2
        seq_len = 100
        audio_signal = torch.randn(batch_size, 80, seq_len)
        length = torch.tensor([seq_len, seq_len // 2], dtype=torch.int64)

        encoder.train()
        _ = encoder(audio_signal=audio_signal, length=length)

        moe_loss = encoder.get_moe_auxiliary_loss()
        assert moe_loss is not None
        assert moe_loss.ndim == 0
        assert moe_loss.item() > 0.0

    def test_auxiliary_loss_none_before_forward(self):
        """Before forward pass, get_moe_auxiliary_loss() should return None."""
        encoder = self._make_encoder()
        moe_loss = encoder.get_moe_auxiliary_loss()
        assert moe_loss is None

    def test_top_k_1(self):
        """Encoder should work correctly with top_k=1 (switch-style)."""
        encoder = self._make_encoder(moe_top_k=1)
        encoder.eval()

        audio_signal = torch.randn(2, 80, 100)
        length = torch.tensor([100, 50], dtype=torch.int64)
        outputs, out_lengths = encoder(audio_signal=audio_signal, length=length)
        assert outputs.ndim == 3

    def test_top_k_2(self):
        """Encoder should work correctly with top_k=2 (top-2 routing)."""
        encoder = self._make_encoder(moe_top_k=2)
        encoder.eval()

        audio_signal = torch.randn(2, 80, 100)
        length = torch.tensor([100, 50], dtype=torch.int64)
        outputs, out_lengths = encoder(audio_signal=audio_signal, length=length)
        assert outputs.ndim == 3

    def test_invalid_moe_position(self):
        """Invalid moe_position should raise ValueError."""
        with pytest.raises(ValueError, match="moe_position must be one of"):
            self._make_encoder(moe_position='middle')

    def test_invalid_moe_router_type(self):
        """Invalid moe_router_type should raise ValueError."""
        with pytest.raises(ValueError, match="moe_router_type must be one of"):
            self._make_encoder(moe_router_type='random')

    def test_invalid_layer_indices(self):
        """Out-of-range layer indices should raise ValueError."""
        with pytest.raises(ValueError, match="invalid index"):
            self._make_encoder(n_layers=4, moe_layer_indices=[0, 5])

    def test_gradient_flow(self):
        """Gradients should flow through MoE layers including the router."""
        encoder = self._make_encoder(moe_router_type='omni', moe_position='end')
        encoder.train()

        audio_signal = torch.randn(1, 80, 100)
        length = torch.tensor([100], dtype=torch.int64)

        outputs, _ = encoder(audio_signal=audio_signal, length=length)
        loss = outputs.sum()
        loss.backward()

        # Check that omni-router gradients exist
        assert encoder.omni_router_end.w_gate.weight.grad is not None
        assert encoder.omni_router_end.w_gate.weight.grad.abs().sum() > 0

    def test_num_experts_parameter(self):
        """Each MoE FFN should have the configured number of experts."""
        encoder = self._make_encoder(moe_num_experts=6, moe_position='end')

        for layer in encoder.layers:
            if isinstance(layer.feed_forward2, MoEFeedForward):
                assert len(layer.feed_forward2.experts) == 6
