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

import contextlib
import io
import tarfile

import pytest
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch import nn

from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.modules import parallel_expert_encoder as pee_module
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
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
def dispatch_stub(online_inference_length, chunk_feat_len, training):
    """Build a bare ParallelExpertEncoder with stubbed branch methods."""
    enc = _PEE.__new__(_PEE)
    nn.Module.__init__(enc)
    enc.online_inference_length = online_inference_length
    enc.online_inference_enabled = False
    enc.chunk_feat_len = chunk_feat_len
    enc.training = training
    enc._forward = lambda **kw: "offline"
    enc._forward_online = lambda **kw: "online"
    return enc


@pytest.mark.unit
@pytest.mark.parametrize(
    "online_len, chunk_feat_len, training, n_frames, generating, expected",
    [
        # Training and validation are offline whatever the audio length and whichever mode
        # the module is in: a length-dependent choice would let ranks run different numbers
        # of encoder calls in the same step.
        (500, 100, True, 200, False, "offline"),
        (500, 100, True, 50, False, "offline"),
        (500, 100, False, 200, False, "offline"),  # validation: eval mode, still offline
        (500, 100, False, 50, False, "offline"),
        # Generation is online whatever the audio length, including below one window.
        (500, 100, False, 200, True, "online"),
        (500, 100, False, 50, True, "online"),
        (500, 100, False, 100, True, "online"),
        # Master switch off.
        (0, 100, False, 200, True, "offline"),
    ],
)
def test_forward_dispatch(online_len, chunk_feat_len, training, n_frames, generating, expected):
    enc = dispatch_stub(online_len, chunk_feat_len, training)
    audio = torch.zeros(1, 8, n_frames)
    length = torch.tensor([n_frames])
    with enc.online_inference() if generating else contextlib.nullcontext():
        assert enc.forward(audio, length) == expected


@pytest.mark.unit
def test_online_inference_scope_restores_previous_state():
    enc = dispatch_stub(500, 100, training=False)
    assert enc.online_inference_enabled is False
    with enc.online_inference():
        assert enc.online_inference_enabled is True
        with enc.online_inference(False):
            assert enc.online_inference_enabled is False
        assert enc.online_inference_enabled is True
    assert enc.online_inference_enabled is False


@pytest.mark.unit
def test_online_inference_scope_restores_state_on_exception():
    enc = dispatch_stub(500, 100, training=False)
    with pytest.raises(RuntimeError):
        with enc.online_inference():
            raise RuntimeError("boom")
    assert enc.online_inference_enabled is False


# ----------------------------------------------------------------------------- #
# _forward_online orchestration (stubbed ASR encoder, provided spk_targets)
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


def online_stub(d_model, n_spk, sf, win, lc, rc):
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
    enc.freeze_diar = False  # The stub has no `diarization_model`, so `freeze_diar` must be False to keep
    enc.asr_norm = nn.LayerNorm(d_model)
    enc.diar_norm = nn.LayerNorm(n_spk)
    enc.register_buffer("diar_kernel", torch.randn(n_spk, d_model))
    enc.missing_rttm_target = -1.0
    enc.speaker_activity_threshold = 0.5
    enc.spk_kernel_scale = 1.0
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
    enc = online_stub(d_model, n_spk, sf, win, lc, rc)

    mels = torch.randn(b, 80, n_frames)
    length = torch.tensor([n_frames] * b)
    spk_targets = torch.rand(b, 5, n_spk)  # arbitrary; aligned internally

    outputs, encoded_len = enc._forward_online(audio_signal=mels, length=length, spk_targets=spk_targets)

    expected_t = round(n_frames / sf)
    assert outputs.shape == (b, d_model, expected_t)
    assert encoded_len.tolist() == [expected_t] * b


# ----------------------------------------------------------------------------- #
# ParallelExpertEncoderPT.is_pe_nemo
# ----------------------------------------------------------------------------- #
def write_nemo(path, *, target=None, include_cfg=True):
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
    write_nemo(nemo_path, target=target)
    assert ParallelExpertEncoderPT.is_pe_nemo(nemo_path) is expected


@pytest.mark.unit
def test_is_pe_nemo_without_model_config(tmp_path):
    nemo_path = str(tmp_path / "no_cfg.nemo")
    write_nemo(nemo_path, include_cfg=False)
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


# ----------------------------------------------------------------------------- #
# End-to-end fusion with real toy encoders
#
# ParallelExpertEncoder loads two real sub-encoders and fuses them:
#   * an ASR ConformerEncoder (cf. tests/collections/asr/test_conformer_encoder.py)
#   * a Sortformer diarizer    (cf. tests/collections/speaker_tasks/test_diar_sortformer_models.py)
# These tests build tiny-but-real instances of both and run the wrapper end to end.
# ----------------------------------------------------------------------------- #
_MEL_FEATURES = 128
_ASR_D_MODEL = 32
_DIAR_FC_D_MODEL = 32
_DIAR_TF_D_MODEL = 16
_N_SPK = 4
_SUBSAMPLING_FACTOR = 8


def toy_asr_encoder_cfg() -> DictConfig:
    """Tiny ConformerEncoder config the PE encoder mounts as its ASR branch."""
    return DictConfig(
        {
            '_target_': 'nemo.collections.asr.modules.ConformerEncoder',
            'feat_in': _MEL_FEATURES,
            'feat_out': -1,
            'n_layers': 1,
            'd_model': _ASR_D_MODEL,
            'subsampling': 'dw_striding',
            'subsampling_factor': _SUBSAMPLING_FACTOR,
            'subsampling_conv_channels': 16,
            'ff_expansion_factor': 4,
            'self_attention_model': 'rel_pos',
            'n_heads': 4,
            'att_context_size': [-1, -1],
            'conv_kernel_size': 9,
            'dropout': 0.0,
            'dropout_pre_encoder': 0.0,
            'dropout_emb': 0.0,
            'dropout_att': 0.0,
        }
    )


def toy_diarization_model_cfg() -> DictConfig:
    """Tiny SortformerEncLabelModel config the PE encoder mounts as its diar branch."""
    model_defaults = {'fc_d_model': _DIAR_FC_D_MODEL, 'tf_d_model': _DIAR_TF_D_MODEL}
    return DictConfig(
        {
            'target': 'nemo.collections.asr.models.sortformer_diar_models.SortformerEncLabelModel',
            'sample_rate': 16000,
            'pil_weight': 0.5,
            'ats_weight': 0.5,
            'max_num_of_spks': _N_SPK,
            'streaming_mode': False,
            'async_streaming': False,
            'model_defaults': DictConfig(model_defaults),
            'preprocessor': DictConfig(
                {
                    '_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
                    'normalize': 'per_feature',
                    'window_size': 0.025,
                    'sample_rate': 16000,
                    'window_stride': 0.01,
                    'window': 'hann',
                    'features': _MEL_FEATURES,
                    'n_fft': 512,
                    'frame_splicing': 1,
                    'dither': 0.00001,
                }
            ),
            'encoder': DictConfig(
                {
                    '_target_': 'nemo.collections.asr.modules.ConformerEncoder',
                    'feat_in': _MEL_FEATURES,
                    'feat_out': -1,
                    'n_layers': 1,
                    'd_model': _DIAR_FC_D_MODEL,
                    'subsampling': 'dw_striding',
                    'subsampling_factor': _SUBSAMPLING_FACTOR,
                    'subsampling_conv_channels': 16,
                    'causal_downsampling': False,
                    'ff_expansion_factor': 4,
                    'self_attention_model': 'rel_pos',
                    'n_heads': 4,
                    'att_context_size': [-1, -1],
                    'conv_kernel_size': 9,
                    'conv_norm_type': 'batch_norm',
                    'dropout': 0.0,
                    'dropout_pre_encoder': 0.0,
                    'dropout_emb': 0.0,
                    'dropout_att': 0.0,
                }
            ),
            'transformer_encoder': DictConfig(
                {
                    '_target_': 'nemo.collections.asr.modules.transformer.transformer_encoders.TransformerEncoder',
                    'num_layers': 1,
                    'hidden_size': _DIAR_TF_D_MODEL,
                    'inner_size': 32,
                    'num_attention_heads': 4,
                    'attn_score_dropout': 0.0,
                    'attn_layer_dropout': 0.0,
                    'ffn_dropout': 0.0,
                    'hidden_act': 'relu',
                    'pre_ln': False,
                    'pre_ln_final_layer_norm': True,
                }
            ),
            'sortformer_modules': DictConfig(
                {
                    '_target_': 'nemo.collections.asr.modules.sortformer_modules.SortformerModules',
                    'num_spks': _N_SPK,
                    'dropout_rate': 0.0,
                    'fc_d_model': _DIAR_FC_D_MODEL,
                    'tf_d_model': _DIAR_TF_D_MODEL,
                }
            ),
            'loss': DictConfig(
                {
                    '_target_': 'nemo.collections.asr.losses.bce_loss.BCELoss',
                    'weight': None,
                    'reduction': 'mean',
                }
            ),
        }
    )


def build_toy_pe_encoder(**overrides) -> ParallelExpertEncoder:
    """Construct a real ParallelExpertEncoder from the tiny ASR + diar configs."""
    kwargs = dict(
        asr_encoder_cfg=toy_asr_encoder_cfg(),
        diarization_model_cfg=toy_diarization_model_cfg(),
        asr_normalize_type='per_feature',
        # Keep the input far below one window so forward() stays on the offline path.
        online_inference_length=500,
    )
    kwargs.update(overrides)
    return ParallelExpertEncoder(**kwargs)


@pytest.mark.unit
def test_pe_encoder_builds_and_wires_both_real_encoders():
    enc = build_toy_pe_encoder()
    # The two fused sub-encoders are the real classes, not stubs.
    assert isinstance(enc.asr_encoder, ConformerEncoder)
    assert isinstance(enc.diarization_model, SortformerEncLabelModel)
    # ConformerEncoder-compatible drop-in properties come from the ASR branch.
    assert enc.d_model == _ASR_D_MODEL
    assert enc.subsampling_factor == _SUBSAMPLING_FACTOR
    # Speaker count + fusion kernel come from the diar branch.
    assert enc.n_spk == _N_SPK
    assert enc.diar_kernel.shape == (_N_SPK, _ASR_D_MODEL)
    assert enc.missing_rttm_target == -1.0
    assert enc.speaker_activity_threshold == 0.5
    assert enc.spk_kernel_scale == 1.0
    # freeze_diar defaults to True -> diar params are frozen, ASR params remain trainable.
    assert all(not p.requires_grad for p in enc.diarization_model.parameters())
    assert any(p.requires_grad for p in enc.asr_encoder.parameters())


# ----------------------------------------------------------------------------- #
# Rank-invariant collective schedule
#
# Both sub-encoders wrap a ConformerEncoder whose `update_max_seq_length` all-reduces
# on the default process group, and the diarization branch used to be entered per batch
# content. Together those let ranks emit different numbers of default-PG collectives on
# the same step, which NCCL matches positionally -> permanent deadlock. These tests pin
# both halves of the fix without needing a distributed process group.
# ----------------------------------------------------------------------------- #
def _conformer_submodules(enc: ParallelExpertEncoder) -> list[ConformerEncoder]:
    return [m for m in enc.modules() if isinstance(m, ConformerEncoder)]


@pytest.mark.unit
def test_pe_encoder_disables_conformer_max_length_sync_by_default():
    enc = build_toy_pe_encoder()
    conformers = _conformer_submodules(enc)
    # Both the ASR branch and the Sortformer frontend must be covered.
    assert len(conformers) == 2
    assert all(not c.sync_max_audio_length for c in conformers)
    assert not enc.sync_max_audio_length


@pytest.mark.unit
def test_pe_encoder_max_length_sync_is_opt_in():
    enc = build_toy_pe_encoder(sync_max_audio_length=True)
    assert enc.sync_max_audio_length
    # Left at the ConformerEncoder default; only safe when every rank runs both encoders.
    assert all(c.sync_max_audio_length for c in _conformer_submodules(enc))


@pytest.mark.unit
@pytest.mark.parametrize("missing_rows", [[], [0], [1], [0, 1]])
def test_pe_encoder_diarization_branch_does_not_depend_on_batch_content(missing_rows):
    """The diarization frontend must run the same number of times regardless of how many
    rows carry the missing-RTTM sentinel, so peer ranks stay on the same collective."""
    enc = build_toy_pe_encoder().train()
    batch_size, n_frames = 2, 160
    mels = torch.randn(batch_size, _MEL_FEATURES, n_frames)
    length = torch.full((batch_size,), n_frames, dtype=torch.long)
    spk_targets = torch.rand(batch_size, 7, _N_SPK)
    for row in missing_rows:
        spk_targets[row].fill_(enc.missing_rttm_target)

    calls = []
    real_frontend = enc.diarization_model.frontend_encoder

    def counting_frontend(**kwargs):
        calls.append(1)
        return real_frontend(**kwargs)

    enc.diarization_model.frontend_encoder = counting_frontend
    with torch.no_grad():
        outputs, _ = enc(mels, length, spk_targets=spk_targets)

    assert len(calls) == 1
    assert torch.isfinite(outputs).all()


@contextlib.contextmanager
def _forbid_host_sync():
    """Make every tensor -> Python-scalar conversion raise.

    Those conversions block the host until the compute stream drains. With
    ``activation_checkpointing_perception`` the encoder forward re-runs inside the backward
    pass, so a sync there stalls the host mid-backward and reorders the surrounding FSDP
    gradient reduce-scatters against peer ranks.
    """
    original = {name: getattr(torch.Tensor, name) for name in ("__bool__", "item", "tolist")}

    def _blocked(name):
        def _raise(self, *args, **kwargs):
            raise AssertionError(f"device-to-host sync via Tensor.{name}()")

        return _raise

    for name in original:
        setattr(torch.Tensor, name, _blocked(name))
    try:
        yield
    finally:
        for name, fn in original.items():
            setattr(torch.Tensor, name, fn)


@pytest.mark.unit
def test_forbid_host_sync_helper_actually_catches_a_sync():
    """Guard the guard: if patching Tensor.__bool__ ever stops working, the sync tests
    below would pass vacuously."""
    with pytest.raises(AssertionError, match="device-to-host sync"):
        with _forbid_host_sync():
            bool(torch.zeros(3).any())


@pytest.mark.unit
@pytest.mark.parametrize("missing_rows", [[], [0], [1], [0, 1]])
def test_pe_encoder_training_forward_never_syncs_host(missing_rows):
    """The training path must reach a fused output without reading any tensor on the host,
    no matter how many rows carry the missing-RTTM sentinel. With activation checkpointing
    this code re-runs inside the backward pass, where a host stall reorders the surrounding
    FSDP gradient reduce-scatters against peer ranks."""
    enc = build_toy_pe_encoder().train()
    batch_size, n_frames = 2, 160
    mels = torch.randn(batch_size, _MEL_FEATURES, n_frames)
    length = torch.full((batch_size,), n_frames, dtype=torch.long)
    spk_targets = torch.rand(batch_size, 7, _N_SPK)
    for row in missing_rows:
        spk_targets[row].fill_(enc.missing_rttm_target)

    with torch.no_grad(), _forbid_host_sync():
        outputs, _ = enc(mels, length, spk_targets=spk_targets)

    assert torch.isfinite(outputs).all()


@pytest.mark.unit
def test_pe_encoder_training_forward_never_syncs_host_without_spk_targets():
    enc = build_toy_pe_encoder().train()
    mels = torch.randn(1, _MEL_FEATURES, 160)
    length = torch.full((1,), 160, dtype=torch.long)

    with torch.no_grad(), _forbid_host_sync():
        outputs, _ = enc(mels, length)

    assert torch.isfinite(outputs).all()


def _build_online_capable_encoder():
    enc = build_toy_pe_encoder(
        online_inference_length=10,
        chunk_left_context=2,
        chunk_right_context=2,
        diar_fifo_len=10,
        diar_spkcache_update_period=20,
        diar_spkcache_len=20,
    )
    enc._suppress_online_pbar = True
    return enc


def _record_dispatch(enc):
    """Record which of the two forward paths `enc.forward` selects."""
    taken = []
    real_online, real_offline = enc._forward_online, enc._forward
    enc._forward_online = lambda **kw: (taken.append("online"), real_online(**kw))[1]
    enc._forward = lambda **kw: (taken.append("offline"), real_offline(**kw))[1]
    return taken


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.parametrize("spk_targets_given", [False, True])
@pytest.mark.parametrize("n_frames", [80, 320])
def test_pe_encoder_training_and_validation_never_use_online_path(mode, spk_targets_given, n_frames):
    """Training and validation are offline unconditionally. Neither audio length, nor the
    presence of speaker targets, nor eval mode may steer it, or ranks in the same step
    would run different numbers of encoder calls."""
    enc = _build_online_capable_encoder()
    enc.train() if mode == "train" else enc.eval()
    mels = torch.randn(1, _MEL_FEATURES, n_frames)
    length = torch.full((1,), n_frames, dtype=torch.long)
    spk_targets = torch.rand(1, 7, _N_SPK) if spk_targets_given else None

    taken = _record_dispatch(enc)
    with torch.no_grad():
        outputs, _ = enc(mels, length, spk_targets=spk_targets)

    assert taken == ["offline"]
    assert torch.isfinite(outputs).all()


@pytest.mark.unit
@pytest.mark.parametrize("spk_targets_given", [False, True])
@pytest.mark.parametrize("n_frames", [80, 320])
def test_pe_encoder_generation_always_uses_online_path(spk_targets_given, n_frames):
    """Generation is online unconditionally, including for audio shorter than one window
    and for batches that supply their own speaker targets."""
    enc = _build_online_capable_encoder().eval()
    mels = torch.randn(1, _MEL_FEATURES, n_frames)
    length = torch.full((1,), n_frames, dtype=torch.long)
    spk_targets = torch.rand(1, 7, _N_SPK) if spk_targets_given else None

    taken = _record_dispatch(enc)
    with torch.no_grad(), enc.online_inference():
        outputs, _ = enc(mels, length, spk_targets=spk_targets)

    assert taken == ["online"]
    assert torch.isfinite(outputs).all()


@pytest.mark.unit
def test_pe_encoder_online_path_predicts_for_sentinel_rows():
    """A `-1` row has no RTTM, so the online path must fill it from the streaming
    Sortformer instead of feeding the sentinel into the speaker kernel."""
    enc = _build_online_capable_encoder().eval()
    batch_size, n_frames = 2, 320
    mels = torch.randn(batch_size, _MEL_FEATURES, n_frames)
    length = torch.full((batch_size,), n_frames, dtype=torch.long)

    spk_targets = torch.rand(batch_size, 7, _N_SPK)
    sentinel_batch = spk_targets.clone()
    sentinel_batch[1].fill_(enc.missing_rttm_target)

    with torch.no_grad(), enc.online_inference():
        kept, _ = enc(mels, length, spk_targets=spk_targets)
        replaced, _ = enc(mels, length, spk_targets=sentinel_batch)

    # Row 0 keeps its RTTM targets, so it is untouched by row 1's sentinel.
    torch.testing.assert_close(kept[0], replaced[0])
    # Row 1 was overridden by a prediction rather than fused from `-1`.
    assert not torch.allclose(kept[1], replaced[1])
    assert torch.isfinite(replaced).all()


@pytest.mark.unit
def test_pe_encoder_sentinel_free_batch_output_unaffected_by_always_run_diarization():
    """Running the frontend unconditionally costs compute but must not change outputs
    for a batch where no row requests predicted diarization."""
    batch_size, n_frames = 2, 160
    mels = torch.randn(batch_size, _MEL_FEATURES, n_frames)
    length = torch.full((batch_size,), n_frames, dtype=torch.long)
    spk_targets = torch.rand(batch_size, 7, _N_SPK)

    with torch.no_grad():
        eager = build_toy_pe_encoder(always_run_diarization=True).eval()
        lazy = build_toy_pe_encoder(always_run_diarization=False).eval()
        lazy.load_state_dict(eager.state_dict())
        out_eager, len_eager = eager(mels, length, spk_targets=spk_targets)
        out_lazy, len_lazy = lazy(mels, length, spk_targets=spk_targets)

    assert torch.equal(len_eager, len_lazy)
    torch.testing.assert_close(out_eager, out_lazy, rtol=0, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize("batch_size, n_frames", [(1, 160), (2, 200)])
def test_pe_encoder_offline_forward_runs_internal_diarizer(batch_size, n_frames):
    enc = build_toy_pe_encoder().eval()
    mels = torch.randn(batch_size, _MEL_FEATURES, n_frames)
    length = torch.full((batch_size,), n_frames, dtype=torch.long)

    with torch.no_grad():
        outputs, encoded_len = enc(mels, length)  # spk_targets=None -> Sortformer runs internally

    expected_t = int(encoded_len[0].item())
    assert outputs.shape == (batch_size, _ASR_D_MODEL, expected_t)
    assert expected_t > 0
    assert torch.isfinite(outputs).all()
    assert encoded_len.tolist() == [expected_t] * batch_size


@pytest.mark.unit
def test_pe_encoder_offline_forward_accepts_diar_override_and_fuses_it():
    enc = build_toy_pe_encoder().eval()
    batch_size, n_frames = 2, 160
    mels = torch.randn(batch_size, _MEL_FEATURES, n_frames)
    length = torch.full((batch_size,), n_frames, dtype=torch.long)

    # Arbitrary diar frame count: PE aligns it to the ASR frame count internally.
    dp1 = torch.rand(batch_size, 7, _N_SPK)
    dp2 = torch.rand(batch_size, 7, _N_SPK)

    with torch.no_grad():
        out1, len1 = enc(mels, length, spk_targets=dp1)
        out2, len2 = enc(mels, length, spk_targets=dp2)
        out1_binary, len1_binary = enc(mels, length, spk_targets=(dp1 > 0.5).to(dp1.dtype))

    expected_t = int(len1[0].item())
    assert out1.shape == (batch_size, _ASR_D_MODEL, expected_t)
    assert torch.equal(len1, len2)
    assert torch.equal(len1, len1_binary)
    torch.testing.assert_close(out1, out1_binary, rtol=0, atol=0)
    assert torch.isfinite(out1).all()
    # Same audio + same (dropout-free, eval) ASR branch, but different speaker
    # predictions must change the fused output -> proves the diar branch is fused in.
    assert not torch.allclose(out1, out2)

    enc.spk_kernel_scale = 0.0
    with torch.no_grad():
        unscaled_out1, _ = enc(mels, length, spk_targets=dp1)
        unscaled_out2, _ = enc(mels, length, spk_targets=dp2)
    torch.testing.assert_close(unscaled_out1, unscaled_out2, rtol=0, atol=0)


@pytest.mark.unit
def test_pe_encoder_mixed_rttm_and_missing_rows_use_their_requested_sources():
    enc = build_toy_pe_encoder().eval()
    batch_size, n_frames = 2, 160
    mels = torch.randn(batch_size, _MEL_FEATURES, n_frames)
    length = torch.full((batch_size,), n_frames, dtype=torch.long)
    rttm_targets = torch.rand(batch_size, 7, _N_SPK)
    mixed_targets = rttm_targets.clone()
    mixed_targets[1].fill_(-1.0)

    with torch.no_grad():
        mixed_out, mixed_len = enc(mels, length, spk_targets=mixed_targets)
        rttm_out, rttm_len = enc(mels, length, spk_targets=rttm_targets)
        sortformer_out, sortformer_len = enc(mels, length)

    assert torch.equal(mixed_len, rttm_len)
    assert torch.equal(mixed_len, sortformer_len)
    torch.testing.assert_close(mixed_out[0], rttm_out[0], rtol=0, atol=0)
    torch.testing.assert_close(mixed_out[1], sortformer_out[1], rtol=0, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize("missing_row", [0, 1])
def test_pe_encoder_mixed_batch_replaces_only_missing_rows_with_mocked_diarization(missing_row):
    class _MockDiarizationModel(nn.Module):
        def __init__(self, predictions):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)
            self.register_buffer("predictions", predictions)
            self.forward_infer_calls = 0

        def frontend_encoder(self, processed_signal, processed_signal_length, bypass_pre_encode):
            batch_size, _, num_frames = processed_signal.shape
            embeddings = processed_signal.new_zeros(batch_size, num_frames, 2)
            return embeddings, processed_signal_length

        def forward_infer(self, emb_seq, emb_seq_length):
            self.forward_infer_calls += 1
            return self.predictions.to(device=emb_seq.device, dtype=emb_seq.dtype)

    class _MockASREncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)

        def forward(self, audio_signal, length):
            batch_size, _, num_frames = audio_signal.shape
            return audio_signal.new_zeros(batch_size, 2, num_frames), length

    rttm_targets = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
        ]
    )
    diarization_predictions = torch.tensor(
        [
            [[0.1, 0.9], [0.8, 0.2], [0.2, 0.7], [0.6, 0.4]],
            [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]],
        ]
    )
    mixed_targets = rttm_targets.clone()
    mixed_targets[missing_row].fill_(-1.0)

    enc = _PEE.__new__(_PEE)
    nn.Module.__init__(enc)
    enc.diarization_model = _MockDiarizationModel(diarization_predictions)
    enc.asr_encoder = _MockASREncoder()
    enc.asr_normalize_type = None
    enc.freeze_diar = True
    enc.freeze_asr = True
    enc.missing_rttm_target = -1.0
    enc.speaker_activity_threshold = 0.5
    enc.spk_kernel_scale = 1.0
    enc.always_run_diarization = True
    enc.asr_norm = nn.Identity()
    enc.diar_norm = nn.Identity()
    enc.register_buffer("diar_kernel", torch.eye(2))

    audio_signal = torch.randn(2, 3, 4)
    length = torch.tensor([4, 4], dtype=torch.long)
    outputs, output_lengths = enc._forward(audio_signal, length, spk_targets=mixed_targets)

    expected_targets = rttm_targets.clone()
    expected_targets[missing_row] = (diarization_predictions[missing_row] > 0.5).to(rttm_targets.dtype)
    assert enc.diarization_model.forward_infer_calls == 1
    assert torch.equal(output_lengths, length)
    assert torch.equal(outputs.transpose(1, 2), expected_targets)


@pytest.mark.unit
def test_pe_encoder_online_forward_matches_conformer_io_with_real_encoders():
    # Small window so a modest input crosses onto the long-form online path.
    enc = build_toy_pe_encoder(
        online_inference_length=10,
        chunk_left_context=2,
        chunk_right_context=2,
        diar_fifo_len=10,
        diar_spkcache_update_period=20,
        diar_spkcache_len=20,
    ).eval()
    enc._suppress_online_pbar = True

    batch_size, n_frames = 1, 320  # > online_inference_length * subsampling_factor (=80)
    mels = torch.randn(batch_size, _MEL_FEATURES, n_frames)
    length = torch.full((batch_size,), n_frames, dtype=torch.long)

    with torch.no_grad():
        outputs, encoded_len = enc(mels, length)

    expected_t = int(encoded_len[0].item())
    assert outputs.shape == (batch_size, _ASR_D_MODEL, expected_t)
    assert expected_t > 0
    assert torch.isfinite(outputs).all()


# ----------------------------------------------------------------------------- #
# GPU end-to-end fusion with real toy encoders
#
# These mirror the CPU end-to-end tests but run on CUDA. They additionally
# exercise the device/dtype-bridging machinery the wrapper exists for: fp32 mels
# fed into (optionally) bf16 experts on the GPU, handled by `_match_module_io`
# (offline) and `_default_dtype` / `_disable_dist_feature_sync` (online).
# ----------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.run_only_on('GPU')
@pytest.mark.skipif(not torch.cuda.is_available(), reason="PEE GPU test requires CUDA")
@pytest.mark.parametrize("batch_size, n_frames", [(1, 160), (2, 200)])
def test_pe_encoder_offline_forward_on_gpu(batch_size, n_frames):
    enc = build_toy_pe_encoder().eval().cuda()
    # Mels arrive un-normalised in fp32 (the SALM perception contract).
    mels = torch.randn(batch_size, _MEL_FEATURES, n_frames, device="cuda", dtype=torch.float32)
    length = torch.full((batch_size,), n_frames, dtype=torch.long, device="cuda")

    with torch.no_grad():
        outputs, encoded_len = enc(mels, length)  # spk_targets=None -> Sortformer runs internally

    expected_t = int(encoded_len[0].item())
    assert outputs.is_cuda
    assert outputs.shape == (batch_size, _ASR_D_MODEL, expected_t)
    assert expected_t > 0
    assert torch.isfinite(outputs).all()
    assert encoded_len.tolist() == [expected_t] * batch_size


@pytest.mark.unit
@pytest.mark.run_only_on('GPU')
@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    reason="PEE bf16 GPU test requires CUDA with bf16 support",
)
def test_pe_encoder_offline_forward_bf16_experts_on_gpu():
    # Experts run in bf16 while mels stay fp32 -> exercises `_match_module_io`
    # device/dtype bridging on both branches before their conv subsampling.
    enc = build_toy_pe_encoder().eval().cuda().to(torch.bfloat16)
    batch_size, n_frames = 2, 200
    mels = torch.randn(batch_size, _MEL_FEATURES, n_frames, device="cuda", dtype=torch.float32)
    length = torch.full((batch_size,), n_frames, dtype=torch.long, device="cuda")

    with torch.no_grad():
        outputs, encoded_len = enc(mels, length)

    expected_t = int(encoded_len[0].item())
    assert outputs.is_cuda
    assert outputs.dtype == torch.bfloat16
    assert outputs.shape == (batch_size, _ASR_D_MODEL, expected_t)
    assert torch.isfinite(outputs).all()


@pytest.mark.unit
@pytest.mark.run_only_on('GPU')
@pytest.mark.skipif(not torch.cuda.is_available(), reason="PEE GPU test requires CUDA")
def test_pe_encoder_offline_forward_accepts_diar_override_on_gpu():
    enc = build_toy_pe_encoder().eval().cuda()
    batch_size, n_frames = 2, 160
    mels = torch.randn(batch_size, _MEL_FEATURES, n_frames, device="cuda", dtype=torch.float32)
    length = torch.full((batch_size,), n_frames, dtype=torch.long, device="cuda")

    dp1 = torch.rand(batch_size, 7, _N_SPK, device="cuda")
    dp2 = torch.rand(batch_size, 7, _N_SPK, device="cuda")

    with torch.no_grad():
        out1, len1 = enc(mels, length, spk_targets=dp1)
        out2, len2 = enc(mels, length, spk_targets=dp2)

    expected_t = int(len1[0].item())
    assert out1.is_cuda
    assert out1.shape == (batch_size, _ASR_D_MODEL, expected_t)
    assert torch.equal(len1, len2)
    assert torch.isfinite(out1).all()
    # Different speaker predictions must change the fused output.
    assert not torch.allclose(out1, out2)


@pytest.mark.unit
@pytest.mark.run_only_on('GPU')
@pytest.mark.skipif(not torch.cuda.is_available(), reason="PEE GPU test requires CUDA")
def test_pe_encoder_online_forward_on_gpu():
    enc = (
        build_toy_pe_encoder(
            online_inference_length=10,
            chunk_left_context=2,
            chunk_right_context=2,
            diar_fifo_len=10,
            diar_spkcache_update_period=20,
            diar_spkcache_len=20,
        )
        .eval()
        .cuda()
    )
    enc._suppress_online_pbar = True

    batch_size, n_frames = 1, 320  # > online_inference_length * subsampling_factor (=80)
    mels = torch.randn(batch_size, _MEL_FEATURES, n_frames, device="cuda", dtype=torch.float32)
    length = torch.full((batch_size,), n_frames, dtype=torch.long, device="cuda")

    with torch.no_grad():
        outputs, encoded_len = enc(mels, length)

    expected_t = int(encoded_len[0].item())
    assert outputs.is_cuda
    assert outputs.shape == (batch_size, _ASR_D_MODEL, expected_t)
    assert expected_t > 0
    assert torch.isfinite(outputs).all()
