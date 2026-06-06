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

"""vLLM plugin registration for NeMo Speech LM models.

Registers NeMoSpeechLMConfig and NeMoSpeechLMForConditionalGeneration
into vLLM's model and config registries via the ``vllm.general_plugins``
entry point.
"""

_PKG = "nemo.collections.speechlm2.vllm.nemotron_v3"


def register():
    """Register the NeMo Speech LM model and config with vLLM.

    Called automatically by vLLM when ``VLLM_PLUGINS=nemo_speechlm``
    is set, via the ``vllm.general_plugins`` entry point in
    ``pyproject.toml``.
    """
    from transformers import AutoConfig

    from nemo.collections.speechlm2.vllm.nemotron_v3.config import NeMoSpeechLMConfig

    AutoConfig.register("nemo_speechlm", NeMoSpeechLMConfig)

    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    _CONFIG_REGISTRY["nemo_speechlm"] = NeMoSpeechLMConfig

    from vllm.model_executor.models.registry import ModelRegistry

    ModelRegistry.register_model(
        "NeMoSpeechLMForConditionalGeneration",
        f"{_PKG}.model:NeMoSpeechLMForConditionalGeneration",
    )

    _apply_backend_patches()


def _apply_backend_patches():
    """Apply patches for LLM backends that need them.

    NemotronH's HF config uses ``layer_norm_epsilon`` but vLLM expects
    ``rms_norm_eps``.  This patches the config class at runtime.
    """
    try:
        from transformers import AutoConfig as _AC

        _nhc = _AC.from_pretrained(
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            trust_remote_code=True,
        )
        NHConfigCls = type(_nhc)
        _orig_getattr = getattr(NHConfigCls, "__getattr__", None)

        def _patched_getattr(self, name):
            if name == "rms_norm_eps":
                return getattr(self, "layer_norm_epsilon", 1e-5)
            if _orig_getattr:
                return _orig_getattr(self, name)
            raise AttributeError(name)

        NHConfigCls.__getattr__ = _patched_getattr
    except Exception:
        pass
