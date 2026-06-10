# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import io
import tarfile

import pytest
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch import nn

from nemo.collections.asr.modules.parallel_expert_encoder import (
    ParallelExpertEncoder,
    ParallelExpertEncoderPT,
    _clone_config,
    _default_dtype,
    _disable_dist_feature_sync,
)

# ``@experimental`` wraps the class in a wrapt proxy, so ``__new__`` (used to build
# bare instances that skip the heavy real ``__init__``) must target the underlying
# class. Attribute access / isinstance still go through the proxy name.
_PEE = getattr(ParallelExpertEncoder, "__wrapped__", ParallelExpertEncoder)


# ----------------------------------------------------------------------------- #
# Module-level context managers / helpers
# ----------------------------------------------------------------------------- #
@pytest.mark.unit
def test_clone_config_is_deep_and_handles_none():
    cfg = OmegaConf.create({"a": {"b": 1}})
    clone = _clone_config(cfg)
    assert clone == cfg
    clone.a.b = 2
    assert cfg.a.b == 1  # original untouched
    assert _clone_config(None) is None


@pytest.mark.unit
@pytest.mark.parametrize("target_dtype", [torch.float64, torch.float16])
def test_default_dtype_sets_and_restores(target_dtype):
    prev = torch.get_default_dtype()
    with _default_dtype(target_dtype):
        assert torch.get_default_dtype() == target_dtype
    assert torch.get_default_dtype() == prev


@pytest.mark.unit
@pytest.mark.parametrize("noop_dtype", [torch.get_default_dtype(), torch.int32])
def test_default_dtype_noop_paths(noop_dtype):
    # Same-dtype and non-floating dtype are both no-ops.
    prev = torch.get_default_dtype()
    with _default_dtype(noop_dtype):
        assert torch.get_default_dtype() == prev
    assert torch.get_default_dtype() == prev


@pytest.mark.unit
def test_disable_dist_feature_sync_noop_when_uninitialized():
    assert not dist.is_initialized()
    orig = dist.is_initialized
    with _disable_dist_feature_sync():
        pass
    assert dist.is_initialized is orig  # nothing patched when dist is down


# ----------------------------------------------------------------------------- #
# Static pure helpers on ParallelExpertEncoder
# ----------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("max_pos, dim", [(4, 8), (1, 16), (10, 4)])
def test_build_sinusoid_position_encoding(max_pos, dim):
    pe = ParallelExpertEncoder._build_sinusoid_position_encoding(max_pos, dim)
    assert pe.shape == (max_pos, dim)
    # row 0: sin(0)=0 on even indices, cos(0)=1 on odd indices
    assert torch.allclose(pe[0, 0::2], torch.zeros(dim // 2))
    assert torch.allclose(pe[0, 1::2], torch.ones(dim // 2))


@pytest.mark.unit
@pytest.mark.parametrize(
    "cur_len, target_len",
    [(3, 6), (6, 3), (5, 5), (1, 4)],
)
def test_align_diar_frames_length_and_padding(cur_len, target_len):
    n_spk = 3
    diar = torch.arange(cur_len * n_spk, dtype=torch.float32).reshape(1, cur_len, n_spk)
    out = ParallelExpertEncoder._align_diar_frames(diar, target_len)
    assert out.shape == (1, target_len, n_spk)
    if target_len <= cur_len:
        # truncation keeps the leading frames unchanged
        assert torch.equal(out, diar[:, :target_len, :])
    else:
        # padding repeats the last frame
        assert torch.equal(out[:, :cur_len, :], diar)
        for t in range(cur_len, target_len):
            assert torch.equal(out[:, t, :], diar[:, -1, :])


@pytest.mark.unit
@pytest.mark.parametrize("param_dtype", [torch.float64, torch.float16])
def test_match_module_io_casts_to_param_dtype(param_dtype):
    module = nn.Linear(4, 4).to(param_dtype)
    tensor = torch.zeros(2, 4, dtype=torch.float32)
    out = ParallelExpertEncoder._match_module_io(tensor, module)
    assert out.dtype == param_dtype


@pytest.mark.unit
def test_match_module_io_paramless_module_unchanged():
    module = nn.Identity()  # no parameters
    tensor = torch.zeros(2, 4, dtype=torch.float32)
    out = ParallelExpertEncoder._match_module_io(tensor, module)
    assert out.dtype == torch.float32
    assert out is tensor


# ----------------------------------------------------------------------------- #
# forward() offline/online dispatch
# ----------------------------------------------------------------------------- #
def _dispatch_stub(online_inference_length, chunk_feat_len, training):
    """Build a bare ParallelExpertEncoder with stubbed branch methods."""
    enc = _PEE.__new__(_PEE)
    nn.Module.__init__(enc)
    enc.online_inference_length = online_inference_length
    enc.chunk_feat_len = chunk_feat_len
    enc.training = training
    enc._forward = lambda **kw: "offline"
    enc._forward_online = lambda **kw: "online"
    return enc


@pytest.mark.unit
@pytest.mark.parametrize(
    "online_len, chunk_feat_len, training, n_frames, expected",
    [
        (500, 100, False, 200, "online"),  # eval + long enough -> online
        (500, 100, False, 50, "offline"),  # eval but shorter than one window
        (500, 100, True, 200, "offline"),  # training always offline
        (0, 100, False, 200, "offline"),  # online disabled
        (500, 100, False, 100, "offline"),  # exactly one window (not strictly greater)
    ],
)
def test_forward_dispatch(online_len, chunk_feat_len, training, n_frames, expected):
    enc = _dispatch_stub(online_len, chunk_feat_len, training)
    audio = torch.zeros(1, 8, n_frames)
    length = torch.tensor([n_frames])
    assert enc.forward(audio, length) == expected


# ----------------------------------------------------------------------------- #
# _forward_online orchestration (stubbed ASR encoder, provided diar_preds)
# ----------------------------------------------------------------------------- #
class _FakeASR(nn.Module):
    """Minimal stand-in for the wrapped ConformerEncoder."""

    def __init__(self, d_model: int, sf: int):
        super().__init__()
        self.subsampling_factor = sf
        self.d_model = d_model
        self._p = nn.Parameter(torch.zeros(1))

    def forward(self, audio_signal, length):
        b, _, t = audio_signal.shape
        # generous frame count so the trim logic never clamps
        t_out = (t + self.subsampling_factor - 1) // self.subsampling_factor + 8
        out = torch.randn(b, self.d_model, t_out)
        return out, length // self.subsampling_factor


def _online_stub(d_model, n_spk, sf, win, lc, rc):
    enc = _PEE.__new__(_PEE)
    nn.Module.__init__(enc)
    enc.asr_encoder = _FakeASR(d_model, sf)
    enc.asr_normalize_type = None
    enc.online_inference_length = win
    enc.chunk_left_context = lc
    enc.chunk_right_context = rc
    enc.chunk_feat_len = win * sf
    enc.left_ctx_feat_len = lc * sf
    enc.right_ctx_feat_len = rc * sf
    enc.freeze_asr = True
    enc.asr_norm = nn.LayerNorm(d_model)
    enc.diar_norm = nn.LayerNorm(n_spk)
    enc.register_buffer("diar_kernel", torch.randn(n_spk, d_model))
    enc._suppress_online_pbar = True
    enc.eval()
    return enc


@pytest.mark.unit
@pytest.mark.parametrize(
    "sf, win, lc, rc, n_frames",
    [
        (8, 10, 2, 2, 240),  # 3 full chunks
        (8, 10, 0, 0, 200),  # partial last chunk, no context
        (4, 5, 1, 1, 64),  # 4 chunks, small subsampling
        (8, 50, 5, 5, 160),  # single chunk (n_frames < window)
    ],
)
def test_forward_online_output_length_telescopes(sf, win, lc, rc, n_frames):
    d_model, n_spk, b = 16, 4, 2
    enc = _online_stub(d_model, n_spk, sf, win, lc, rc)

    mels = torch.randn(b, 80, n_frames)
    length = torch.tensor([n_frames] * b)
    diar_preds = torch.rand(b, 5, n_spk)  # arbitrary; aligned internally

    outputs, encoded_len = enc._forward_online(audio_signal=mels, length=length, diar_preds=diar_preds)

    expected_t = round(n_frames / sf)
    assert outputs.shape == (b, d_model, expected_t)
    assert encoded_len.tolist() == [expected_t] * b


# ----------------------------------------------------------------------------- #
# ParallelExpertEncoderPT.is_pe_nemo
# ----------------------------------------------------------------------------- #
def _write_nemo(path, *, target=None, include_cfg=True):
    with tarfile.open(path, "w") as tf:
        if include_cfg:
            data = (f"target: {target}\n" if target is not None else "foo: bar\n").encode()
            info = tarfile.TarInfo(name="model_config.yaml")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        else:
            data = b"not a config"
            info = tarfile.TarInfo(name="weights.ckpt")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


@pytest.mark.unit
@pytest.mark.parametrize(
    "target, expected",
    [
        ("nemo.collections.asr.modules.parallel_expert_encoder.ParallelExpertEncoderPT", True),
        ("ParallelExpertEncoderPT", True),
        ("nemo.collections.asr.models.SomethingElse", False),
        (None, False),  # model_config.yaml present but no `target`
    ],
)
def test_is_pe_nemo_by_target(tmp_path, target, expected):
    nemo_path = str(tmp_path / "bundle.nemo")
    _write_nemo(nemo_path, target=target)
    assert ParallelExpertEncoderPT.is_pe_nemo(nemo_path) is expected


@pytest.mark.unit
def test_is_pe_nemo_without_model_config(tmp_path):
    nemo_path = str(tmp_path / "no_cfg.nemo")
    _write_nemo(nemo_path, include_cfg=False)
    assert ParallelExpertEncoderPT.is_pe_nemo(nemo_path) is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_path",
    [None, 123, "missing.nemo", "not_a_nemo.txt"],
)
def test_is_pe_nemo_rejects_bad_paths(tmp_path, bad_path):
    # a real-but-non-.nemo file to exercise the suffix check
    if bad_path == "not_a_nemo.txt":
        p = tmp_path / "not_a_nemo.txt"
        p.write_text("hello")
        bad_path = str(p)
    assert ParallelExpertEncoderPT.is_pe_nemo(bad_path) is False


# ----------------------------------------------------------------------------- #
# ParallelExpertEncoderPT.save_to_nemo guard rails
# ----------------------------------------------------------------------------- #
@pytest.mark.unit
def test_save_to_nemo_rejects_non_encoder(tmp_path):
    with pytest.raises(TypeError):
        ParallelExpertEncoderPT.save_to_nemo(
            nn.Linear(2, 2), str(tmp_path / "out.nemo"), template_bundle_path=str(tmp_path / "tpl.nemo")
        )


@pytest.mark.unit
def test_save_to_nemo_missing_template(tmp_path):
    # __new__ produces a real ParallelExpertEncoder instance (passes isinstance)
    # without running the heavy __init__, so we reach the template existence check.
    fake_encoder = _PEE.__new__(_PEE)
    with pytest.raises(FileNotFoundError):
        ParallelExpertEncoderPT.save_to_nemo(
            fake_encoder,
            str(tmp_path / "out.nemo"),
            template_bundle_path=str(tmp_path / "does_not_exist.nemo"),
        )
