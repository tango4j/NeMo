# Restore TODO for `/home/taejinp/projects/canary-dev/canary-dev/speechlm-2026h1/NeMo/nemo/collections/speechlm2/vllm/nemotron_v3/config.py`

`2 patches` were recorded against this file in transcript
`87524b29-66f4-4136-a40f-b331b1d6705e`.  Apply each one in order
after the file body has been restored from the disassembly or rsync'd
from cluster.

## Patch (transcript line 345)
- `old_string` length: 676 chars
- `new_string` length: 1105 chars
- replace_all: False

### OLD (what the file should contain before this patch)
```python
    def __init__(
        self,
        perception: dict | None = None,
        pretrained_llm: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        pretrained_asr: str = "nvidia/canary-1b-v2",
        audio_locator_tag: str = "<|audio|>",
        prompt_format: str = "nemotron-nano-v3",
        pretrained_weights: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.perception = perception or {}
        self.pretrained_llm = pretrained_llm
        self.pretrained_asr = pretrained_asr
        self.audio_locator_tag = audio_locator_tag
        self.prompt_format = prompt_format
        self.pretrained_weights = pretrained_weights
```

### NEW (what to replace it with)
```python
    def __init__(
        self,
        perception: dict | None = None,
        pretrained_llm: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        pretrained_asr: str = "nvidia/canary-1b-v2",
        audio_locator_tag: str = "<|audio|>",
        prompt_format: str = "nemotron-nano-v3",
        pretrained_weights: bool = True,
        # phPEE (Parallel Expert Encoder) bundle path. When set, the plugin
        # swaps the single-encoder AudioPerceptionModule.encoder for a
        # ParallelExpertEncoder loaded from this .nemo file, matching the
        # training-time setup_parallel_expert_encoder() in
        # nemo/collections/speechlm2/parts/pretrained.py.
        pe_encoder_path: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.perception = perception or {}
        self.pretrained_llm = pretrained_llm
        self.pretrained_asr = pretrained_asr
        self.audio_locator_tag = audio_locator_tag
        self.prompt_format = prompt_format
        self.pretrained_weights = pretrained_weights
        self.pe_encoder_path = pe_encoder_path
```

## Patch (transcript line 346)
- `old_string` length: 365 chars
- `new_string` length: 396 chars
- replace_all: False

### OLD (what the file should contain before this patch)
```python
    def __getattr__(self, name):
        if name.startswith("_") or name in (
            "perception",
            "pretrained_llm",
            "pretrained_asr",
            "audio_locator_tag",
            "prompt_format",
            "pretrained_weights",
            "text_config",
            "_ATTR_ALIASES",
        ):
            raise AttributeError(name)
```

### NEW (what to replace it with)
```python
    def __getattr__(self, name):
        if name.startswith("_") or name in (
            "perception",
            "pretrained_llm",
            "pretrained_asr",
            "audio_locator_tag",
            "prompt_format",
            "pretrained_weights",
            "pe_encoder_path",
            "text_config",
            "_ATTR_ALIASES",
        ):
            raise AttributeError(name)
```
