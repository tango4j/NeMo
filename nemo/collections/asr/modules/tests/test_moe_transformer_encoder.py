"""
Tests for MoE Transformer Encoder.

Verifies:
1. Basic forward pass & output shapes
2. Expert initialization from FFN weights
3. Auxiliary load-balancing loss
4. Omni vs switch router sharing
5. Selective layer MoE application
6. State dict remapping from base TransformerEncoder
7. Parameter count comparison (MoE vs dense)
"""

import copy

import pytest
import torch

from nemo.collections.asr.modules.transformer_encoder import TransformerEncoder, FeedForward
from nemo.collections.asr.modules.moe_transformer_encoder import (
    MoETransformerEncoder,
    MoEFeedForward,
    SwitchGate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 4
N_MELS = 80
BATCH = 2
SEQ_LEN = 200  # raw audio frames (pre-subsampling)


def _make_input(batch=BATCH, seq_len=SEQ_LEN, n_mels=N_MELS, device=DEVICE):
    """Create a dummy audio input and corresponding lengths."""
    audio = torch.randn(batch, n_mels, seq_len, device=device)
    lengths = torch.full((batch,), seq_len, dtype=torch.long, device=device)
    return audio, lengths


def _make_base_encoder(**kwargs):
    defaults = dict(
        n_mels=N_MELS, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        drop_rate=0.0, qkv_bias=False, causal_mask=False, pre_encode="conv",
        nan_debug=False, qk_norm=False,
    )
    defaults.update(kwargs)
    return TransformerEncoder(**defaults).to(DEVICE)


def _make_moe_encoder(**kwargs):
    defaults = dict(
        n_mels=N_MELS, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        drop_rate=0.0, qkv_bias=False, causal_mask=False, pre_encode="conv",
        nan_debug=False, qk_norm=False,
        moe_num_experts=4, moe_top_k=1, moe_router_type='omni',
        moe_layer_indices=None, moe_load_balance_loss_weight=0.01,
        moe_jitter_eps=0.0, moe_init_from_ffn=True,
    )
    defaults.update(kwargs)
    return MoETransformerEncoder(**defaults).to(DEVICE)


# ---------------------------------------------------------------------------
# Test 1: Basic forward pass — correct output shapes
# ---------------------------------------------------------------------------

class TestForwardPass:
    def test_output_shape_matches_base(self):
        """MoE encoder should produce the same output shape as the base encoder."""
        base = _make_base_encoder()
        moe = _make_moe_encoder()
        audio, lengths = _make_input()

        with torch.no_grad():
            base_out, base_len = base(audio, lengths)
            moe_out, moe_len = moe(audio, lengths)

        assert base_out.shape == moe_out.shape, (
            f"Shape mismatch: base {base_out.shape} vs moe {moe_out.shape}"
        )
        assert torch.equal(base_len, moe_len)

    def test_output_shape_values(self):
        """Output should be (B, d_model, T') where T' is subsampled time."""
        moe = _make_moe_encoder()
        audio, lengths = _make_input()

        with torch.no_grad():
            out, out_len = moe(audio, lengths)

        assert out.shape[0] == BATCH
        assert out.shape[1] == D_MODEL
        assert out.shape[2] > 0

    def test_no_nan_in_output(self):
        """Output should not contain NaN or Inf."""
        moe = _make_moe_encoder()
        audio, lengths = _make_input()

        with torch.no_grad():
            out, _ = moe(audio, lengths)

        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_different_batch_sizes(self):
        """Should work with different batch sizes."""
        moe = _make_moe_encoder()
        for bs in [1, 3, 5]:
            audio, lengths = _make_input(batch=bs)
            with torch.no_grad():
                out, out_len = moe(audio, lengths)
            assert out.shape[0] == bs

    def test_gradients_flow(self):
        """Gradients should flow through MoE layers."""
        moe = _make_moe_encoder()
        audio, lengths = _make_input()
        out, _ = moe(audio, lengths)
        loss = out.sum()
        loss.backward()

        has_grad = False
        for name, param in moe.named_parameters():
            if 'expert' in name and param.grad is not None:
                has_grad = True
                break
        assert has_grad, "No gradients reached expert parameters"


# ---------------------------------------------------------------------------
# Test 2: Expert initialization from FFN weights
# ---------------------------------------------------------------------------

class TestExpertInitialization:
    def test_experts_initialized_from_ffn(self):
        """When moe_init_from_ffn=True, all experts should start with identical weights."""
        moe = _make_moe_encoder(moe_init_from_ffn=True, moe_num_experts=3)

        for layer_idx in moe.moe_layer_indices:
            moe_ffn = moe.layers[layer_idx].ffn
            assert isinstance(moe_ffn, MoEFeedForward)

            ref_state = moe_ffn.experts[0].state_dict()
            for expert_idx in range(1, moe_ffn.num_experts):
                expert_state = moe_ffn.experts[expert_idx].state_dict()
                for key in ref_state:
                    assert torch.equal(ref_state[key], expert_state[key]), (
                        f"Layer {layer_idx}, expert {expert_idx}, key '{key}' differs from expert 0"
                    )

    def test_experts_random_when_init_disabled(self):
        """When moe_init_from_ffn=False, experts should have different random weights."""
        moe = _make_moe_encoder(moe_init_from_ffn=False, moe_num_experts=3)

        any_different = False
        for layer_idx in moe.moe_layer_indices:
            moe_ffn = moe.layers[layer_idx].ffn
            state0 = moe_ffn.experts[0].state_dict()
            state1 = moe_ffn.experts[1].state_dict()
            for key in state0:
                if not torch.equal(state0[key], state1[key]):
                    any_different = True
                    break
            if any_different:
                break
        assert any_different, "Experts should have different random weights when init_from_ffn=False"


# ---------------------------------------------------------------------------
# Test 3: Auxiliary load-balancing loss
# ---------------------------------------------------------------------------

class TestAuxiliaryLoss:
    def test_aux_loss_is_scalar(self):
        """get_moe_auxiliary_loss() should return a scalar tensor after forward."""
        moe = _make_moe_encoder()
        audio, lengths = _make_input()

        with torch.no_grad():
            moe(audio, lengths)

        loss = moe.get_moe_auxiliary_loss()
        assert loss is not None, "Auxiliary loss should not be None after forward pass"
        assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"

    def test_aux_loss_is_positive(self):
        """Auxiliary loss should be non-negative."""
        moe = _make_moe_encoder()
        audio, lengths = _make_input()

        with torch.no_grad():
            moe(audio, lengths)

        loss = moe.get_moe_auxiliary_loss()
        assert loss >= 0, f"Auxiliary loss should be non-negative, got {loss.item()}"

    def test_aux_loss_none_before_forward(self):
        """Before any forward pass, auxiliary loss should be None."""
        moe = _make_moe_encoder()
        loss = moe.get_moe_auxiliary_loss()
        assert loss is None

    def test_aux_loss_weighted(self):
        """Auxiliary loss should scale with moe_load_balance_loss_weight."""
        moe_low = _make_moe_encoder(moe_load_balance_loss_weight=0.001)
        moe_high = _make_moe_encoder(moe_load_balance_loss_weight=1.0)

        # Share the same weights so the raw aux losses are identical
        moe_high.load_state_dict(moe_low.state_dict(), strict=False)

        audio, lengths = _make_input()
        with torch.no_grad():
            moe_low(audio, lengths)
            moe_high(audio, lengths)

        loss_low = moe_low.get_moe_auxiliary_loss()
        loss_high = moe_high.get_moe_auxiliary_loss()
        assert loss_high > loss_low, "Higher weight should produce larger auxiliary loss"

    def test_aux_loss_backprop(self):
        """Auxiliary loss should be differentiable."""
        moe = _make_moe_encoder()
        audio, lengths = _make_input()
        moe(audio, lengths)
        aux_loss = moe.get_moe_auxiliary_loss()
        aux_loss.backward()

        router_has_grad = False
        for name, param in moe.named_parameters():
            if 'router' in name or 'w_gate' in name:
                if param.grad is not None:
                    router_has_grad = True
                    break
        assert router_has_grad, "Auxiliary loss gradients should reach router parameters"


# ---------------------------------------------------------------------------
# Test 4: Omni vs switch router sharing
# ---------------------------------------------------------------------------

class TestRouterStrategies:
    def test_omni_router_shared(self):
        """In omni mode, all MoE layers should share the same router object."""
        moe = _make_moe_encoder(moe_router_type='omni')

        routers = set()
        for layer_idx in moe.moe_layer_indices:
            moe_ffn = moe.layers[layer_idx].ffn
            routers.add(id(moe_ffn.router))

        assert len(routers) == 1, f"Omni router should be shared, found {len(routers)} distinct routers"

    def test_switch_router_independent(self):
        """In switch mode, each MoE layer should have its own router."""
        moe = _make_moe_encoder(moe_router_type='switch')

        routers = set()
        for layer_idx in moe.moe_layer_indices:
            moe_ffn = moe.layers[layer_idx].ffn
            routers.add(id(moe_ffn.router))

        assert len(routers) == len(moe.moe_layer_indices), (
            f"Switch router should be independent per layer, found {len(routers)} "
            f"distinct routers for {len(moe.moe_layer_indices)} layers"
        )

    def test_omni_router_is_module_attribute(self):
        """The shared omni router should be a registered module attribute."""
        moe = _make_moe_encoder(moe_router_type='omni')
        assert moe.omni_router is not None
        assert isinstance(moe.omni_router, SwitchGate)

    def test_switch_has_no_omni_router(self):
        """In switch mode, omni_router should be None."""
        moe = _make_moe_encoder(moe_router_type='switch')
        assert moe.omni_router is None

    def test_invalid_router_type_raises(self):
        """Invalid router type should raise ValueError."""
        with pytest.raises(ValueError, match="moe_router_type"):
            _make_moe_encoder(moe_router_type='invalid')


# ---------------------------------------------------------------------------
# Test 5: Selective layer MoE application
# ---------------------------------------------------------------------------

class TestSelectiveLayers:
    def test_only_specified_layers_have_moe(self):
        """Only layers in moe_layer_indices should have MoEFeedForward."""
        moe_indices = [0, 2]
        moe = _make_moe_encoder(moe_layer_indices=moe_indices)

        for i in range(N_LAYERS):
            ffn = moe.layers[i].ffn
            if i in moe_indices:
                assert isinstance(ffn, MoEFeedForward), f"Layer {i} should be MoE"
            else:
                assert isinstance(ffn, FeedForward), f"Layer {i} should be dense FFN"

    def test_all_layers_when_none(self):
        """When moe_layer_indices=None, all layers should be MoE."""
        moe = _make_moe_encoder(moe_layer_indices=None)

        for i in range(N_LAYERS):
            assert isinstance(moe.layers[i].ffn, MoEFeedForward), (
                f"Layer {i} should be MoE when moe_layer_indices=None"
            )

    def test_single_layer_moe(self):
        """MoE on a single layer should work."""
        moe = _make_moe_encoder(moe_layer_indices=[1])
        audio, lengths = _make_input()

        with torch.no_grad():
            out, _ = moe(audio, lengths)

        assert not torch.isnan(out).any()
        assert isinstance(moe.layers[1].ffn, MoEFeedForward)
        assert isinstance(moe.layers[0].ffn, FeedForward)

    def test_invalid_layer_index_raises(self):
        """Layer index out of range should raise ValueError."""
        with pytest.raises(ValueError, match="invalid index"):
            _make_moe_encoder(moe_layer_indices=[N_LAYERS + 5])


# ---------------------------------------------------------------------------
# Test 6: State dict remapping from base TransformerEncoder
# ---------------------------------------------------------------------------

class TestStateDictRemapping:
    def test_load_base_into_moe(self):
        """Loading a base TransformerEncoder state dict into MoE encoder should work."""
        base = _make_base_encoder()
        moe = _make_moe_encoder(moe_num_experts=2)

        base_state = base.state_dict()
        moe.load_state_dict(base_state, strict=False)

        # Verify expert weights match the original FFN weights
        for layer_idx in moe.moe_layer_indices:
            base_ffn_prefix = f"layers.{layer_idx}.ffn."
            moe_ffn = moe.layers[layer_idx].ffn

            for param_name, base_param in base_state.items():
                if param_name.startswith(base_ffn_prefix):
                    suffix = param_name[len(base_ffn_prefix):]
                    # Skip if this is already an expert/router key
                    if suffix.startswith("experts.") or suffix.startswith("router."):
                        continue
                    for expert in moe_ffn.experts:
                        expert_param = dict(expert.named_parameters())
                        # The FeedForward has params like ffn.0.weight, ffn.0.bias, ffn.2.weight, ffn.2.bias
                        flat_params = {}
                        for n, p in expert.named_parameters():
                            flat_params[n] = p
                        if suffix in flat_params:
                            assert torch.allclose(flat_params[suffix], base_param.to(DEVICE)), (
                                f"Layer {layer_idx}, suffix '{suffix}' mismatch after loading"
                            )

    def test_moe_to_moe_load(self):
        """Loading a MoE state dict into another MoE encoder should work perfectly."""
        moe1 = _make_moe_encoder(moe_num_experts=2)
        moe2 = _make_moe_encoder(moe_num_experts=2)

        moe2.load_state_dict(moe1.state_dict())

        audio, lengths = _make_input()
        with torch.no_grad():
            out1, _ = moe1(audio, lengths)
            out2, _ = moe2(audio, lengths)

        assert torch.allclose(out1, out2, atol=1e-6), "Same weights should produce same output"

    def test_router_not_in_base_state(self):
        """Router keys should not exist in base encoder state dict."""
        base = _make_base_encoder()
        for key in base.state_dict():
            assert 'router' not in key
            assert 'w_gate' not in key


# ---------------------------------------------------------------------------
# Test 7: Parameter count comparison
# ---------------------------------------------------------------------------

class TestParameterCount:
    def _count_params(self, model):
        return sum(p.numel() for p in model.parameters())

    def test_moe_has_more_params(self):
        """MoE encoder should have more total parameters than base."""
        base = _make_base_encoder()
        moe = _make_moe_encoder(moe_num_experts=4)
        assert self._count_params(moe) > self._count_params(base)

    def test_single_expert_similar_to_base(self):
        """With 1 expert, MoE should have only router params extra."""
        base = _make_base_encoder()
        moe = _make_moe_encoder(moe_num_experts=1)

        base_params = self._count_params(base)
        moe_params = self._count_params(moe)

        # The only extra params should be the router: d_model * 1 (for 1 expert)
        router_params = D_MODEL * 1  # omni router: Linear(d_model, 1) has d_model params (no bias)
        assert moe_params == base_params + router_params, (
            f"Expected {base_params + router_params} params, got {moe_params}. "
            f"Diff = {moe_params - base_params}"
        )


# ---------------------------------------------------------------------------
# Test 8: SwitchGate unit tests
# ---------------------------------------------------------------------------

class TestSwitchGate:
    def test_output_shape(self):
        gate = SwitchGate(d_model=32, num_experts=4).to(DEVICE)
        x = torch.randn(10, 32, device=DEVICE)
        probs = gate(x)
        assert probs.shape == (10, 4)

    def test_probabilities_sum_to_one(self):
        gate = SwitchGate(d_model=32, num_experts=4).to(DEVICE)
        x = torch.randn(10, 32, device=DEVICE)
        probs = gate(x)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(10, device=DEVICE), atol=1e-5)

    def test_jitter_only_during_training(self):
        gate = SwitchGate(d_model=32, num_experts=4, jitter_eps=1.0).to(DEVICE)
        x = torch.randn(5, 32, device=DEVICE)

        gate.eval()
        torch.manual_seed(42)
        out1 = gate(x)
        torch.manual_seed(42)
        out2 = gate(x)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"


# ---------------------------------------------------------------------------
# Test 9: MoEFeedForward unit tests
# ---------------------------------------------------------------------------

class TestMoEFeedForward:
    def test_output_shape(self):
        moe_ff = MoEFeedForward(d_model=32, num_experts=4, top_k=1).to(DEVICE)
        x = torch.randn(2, 10, 32, device=DEVICE)
        out = moe_ff(x)
        assert out.shape == (2, 10, 32)

    def test_top_k_2(self):
        """Top-k=2 should also work correctly."""
        moe_ff = MoEFeedForward(d_model=32, num_experts=4, top_k=2).to(DEVICE)
        x = torch.randn(2, 10, 32, device=DEVICE)
        out = moe_ff(x)
        assert out.shape == (2, 10, 32)
        assert not torch.isnan(out).any()

    def test_external_router(self):
        """MoEFeedForward should accept an external router."""
        router = SwitchGate(d_model=32, num_experts=4).to(DEVICE)
        moe_ff = MoEFeedForward(d_model=32, num_experts=4, router=router).to(DEVICE)
        assert moe_ff.router is router


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
