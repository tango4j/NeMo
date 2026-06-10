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

import types

import pytest
import torch
from omegaconf import OmegaConf

import nemo.collections.speechlm2.parts.pretrained as pretrained

PE_D_MODEL = 512


@pytest.fixture
def patched_pe(monkeypatch):
    """Stub out the heavy .nemo bundle loading so only the mount logic is exercised."""
    fake_pe = types.SimpleNamespace(d_model=PE_D_MODEL, n_spk=4, freeze_diar=True, freeze_asr=False)
    monkeypatch.setattr(pretrained.ParallelExpertEncoderPT, "is_pe_nemo", lambda path: True)
    monkeypatch.setattr(pretrained.ParallelExpertEncoderPT, "load_from_nemo", lambda *a, **k: fake_pe)
    return fake_pe


def _build_model(
    *,
    pe_encoder_path="bundle.nemo",
    with_perception=True,
    with_encoder=True,
    encoder_d_model=None,
    adapter_d_model=None,
    proj=None,
):
    cfg_dict = {"pe_encoder_path": pe_encoder_path, "perception": {"preprocessor": {"normalize": "per_feature"}}}
    if adapter_d_model is not None:
        cfg_dict["perception"]["modality_adapter"] = {"d_model": adapter_d_model}
    cfg = OmegaConf.create(cfg_dict)

    model = types.SimpleNamespace(cfg=cfg)
    if with_perception:
        perception = types.SimpleNamespace(
            preprocessor=types.SimpleNamespace(featurizer=types.SimpleNamespace(normalize="per_feature"))
        )
        if with_encoder:
            encoder = types.SimpleNamespace()
            if encoder_d_model is not None:
                encoder.d_model = encoder_d_model
            perception.encoder = encoder
        if proj is not None:
            perception.proj = proj
        model.perception = perception
    return model


@pytest.mark.unit
@pytest.mark.parametrize("pe_encoder_path", [None, "", False])
def test_noop_when_path_unset(monkeypatch, pe_encoder_path):
    # is_pe_nemo / load_from_nemo must never be called on the no-op path.
    monkeypatch.setattr(
        pretrained.ParallelExpertEncoderPT,
        "is_pe_nemo",
        lambda path: pytest.fail("is_pe_nemo should not be called"),
    )
    model = _build_model(pe_encoder_path=pe_encoder_path)
    original_encoder = model.perception.encoder
    assert pretrained.setup_parallel_expert_encoder(model) is None
    assert model.perception.encoder is original_encoder


@pytest.mark.unit
def test_raises_when_no_perception(patched_pe):
    model = _build_model(with_perception=False)
    with pytest.raises(RuntimeError):
        pretrained.setup_parallel_expert_encoder(model)


@pytest.mark.unit
@pytest.mark.parametrize("bad_path", [123, "model.bin"])
def test_raises_on_non_nemo_path(patched_pe, bad_path):
    model = _build_model(pe_encoder_path=bad_path)
    with pytest.raises(ValueError):
        pretrained.setup_parallel_expert_encoder(model)


@pytest.mark.unit
def test_raises_when_perception_has_no_encoder(patched_pe):
    model = _build_model(with_encoder=False)
    with pytest.raises(RuntimeError):
        pretrained.setup_parallel_expert_encoder(model)


@pytest.mark.unit
def test_raises_when_not_pe_bundle(monkeypatch):
    monkeypatch.setattr(pretrained.ParallelExpertEncoderPT, "is_pe_nemo", lambda path: False)
    model = _build_model()
    with pytest.raises(ValueError):
        pretrained.setup_parallel_expert_encoder(model)


@pytest.mark.unit
def test_raises_on_encoder_d_model_mismatch(patched_pe):
    model = _build_model(encoder_d_model=PE_D_MODEL // 2)
    with pytest.raises(ValueError):
        pretrained.setup_parallel_expert_encoder(model)


@pytest.mark.unit
def test_raises_on_adapter_d_model_mismatch(patched_pe):
    model = _build_model(adapter_d_model=PE_D_MODEL + 1)  # encoder has no d_model -> adapter check fires
    with pytest.raises(ValueError):
        pretrained.setup_parallel_expert_encoder(model)


@pytest.mark.unit
def test_raises_on_proj_in_features_mismatch(patched_pe):
    model = _build_model(proj=torch.nn.Linear(PE_D_MODEL + 7, 10))
    with pytest.raises(ValueError):
        pretrained.setup_parallel_expert_encoder(model)


@pytest.mark.unit
def test_happy_path_mounts_and_disables_normalization(patched_pe):
    model = _build_model(
        encoder_d_model=PE_D_MODEL,
        adapter_d_model=PE_D_MODEL,
        proj=torch.nn.Linear(PE_D_MODEL, 10),
    )
    pretrained.setup_parallel_expert_encoder(model)

    assert model.perception.encoder is patched_pe
    assert model.perception.preprocessor.featurizer.normalize is None
    assert model.cfg.perception.preprocessor.normalize is None
