# Restore TODO for `/home/taejinp/projects/canary-dev/canary-dev/speechlm-2026h1/NeMo/nemo/collections/speechlm2/vllm/nemotron_v3/model.py`

`5 patches` were recorded against this file in transcript
`87524b29-66f4-4136-a40f-b331b1d6705e`.  Apply each one in order
after the file body has been restored from the disassembly or rsync'd
from cluster.

## Patch (transcript line 314)
- `old_string` length: 164 chars
- `new_string` length: 288 chars
- replace_all: False

### OLD (what the file should contain before this patch)
```python
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
```

### NEW (what to replace it with)
```python
from vllm.multimodal import MULTIMODAL_REGISTRY
# MultiModalDataDict moved to vllm.inputs in vllm 0.19.x; the rest still live in vllm.multimodal.inputs.
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
```

## Patch (transcript line 348)
- `old_string` length: 574 chars
- `new_string` length: 2715 chars
- replace_all: False

### OLD (what the file should contain before this patch)
```python
def _load_nemo_perception(perception_cfg: dict, output_dim: int) -> nn.Module:
    try:
        from omegaconf import DictConfig

        from nemo.collections.speechlm2.modules import AudioPerceptionModule
    except ImportError as e:
        raise ImportError(
            "NeMo is required for the audio encoder. " "Install with: pip install nemo_toolkit[asr]"
        ) from e

    cfg = DictConfig(perception_cfg)
    if "output_dim" not in cfg:
        cfg.output_dim = output_dim
    perception = AudioPerceptionModule(cfg)
    perception.eval()
    return perception
```

### NEW (what to replace it with)
```python
def _mount_parallel_expert_encoder(perception: nn.Module, pe_encoder_path: str) -> None:
    """Replace ``perception.encoder`` with a ParallelExpertEncoder loaded from a
    self-contained PE ``.nemo`` bundle.

    Mirrors training-time ``setup_parallel_expert_encoder`` in
    nemo/collections/speechlm2/parts/pretrained.py:
      1. Validate the bundle is a ParallelExpertEncoderPT.
      2. Load the ParallelExpertEncoder (asr_encoder + diarization_model +
         asr_norm + diar_norm + diar_kernel buffer).
      3. Sanity-check d_model matches the existing single-encoder slot the
         AudioPerceptionModule built (so perception.proj stays compatible).
      4. Disable preprocessor featurizer normalization -- PE expects
         un-normalised mels and replays ASR normalisation internally.
    """
    from nemo.collections.asr.modules.parallel_expert_encoder import (
        is_parallel_expert_encoder_nemo,
        load_parallel_expert_encoder_from_nemo,
    )

    if not is_parallel_expert_encoder_nemo(pe_encoder_path):
        raise ValueError(
            f"pe_encoder_path={pe_encoder_path!r} is not a ParallelExpertEncoderPT .nemo bundle."
        )

    pe_encoder = load_parallel_expert_encoder_from_nemo(
        pe_encoder_path, map_location="cpu", strict=True,
    )

    existing_encoder = getattr(perception, "encoder", None)
    if existing_encoder is None:
        raise RuntimeError(
            "perception module has no `encoder` attribute to replace with the PE bundle."
        )
    existing_d_model = int(getattr(existing_encoder, "d_model", -1))
    if existing_d_model > 0 and int(pe_encoder.d_model) != existing_d_model:
        raise ValueError(
            f"ParallelExpertEncoder d_model={pe_encoder.d_model} does not match the "
            f"existing perception encoder d_model={existing_d_model}."
        )

    try:
        perception.preprocessor.featurizer.normalize = None
    except AttributeError:
        pass

    perception.encoder = pe_encoder


def _load_nemo_perception(
    perception_cfg: dict, output_dim: int, pe_encoder_path: str | None = None
) -> nn.Module:
    try:
        from omegaconf import DictConfig

        from nemo.collections.speechlm2.modules import AudioPerceptionModule
    except ImportError as e:
        raise ImportError(
            "NeMo is required for the audio encoder. " "Install with: pip install nemo_toolkit[asr]"
        ) from e

    cfg = DictConfig(perception_cfg)
    if "output_dim" not in cfg:
        cfg.output_dim = output_dim
    perception = AudioPerceptionModule(cfg)

    if pe_encoder_path:
        _mount_parallel_expert_encoder(perception, pe_encoder_path)

    perception.eval()
    return perception
```

## Patch (transcript line 349)
- `old_string` length: 218 chars
- `new_string` length: 339 chars
- replace_all: False

### OLD (what the file should contain before this patch)
```python
        with self._mark_tower_model(vllm_config, {"audio"}):
            self.perception = _load_nemo_perception(config.perception, output_dim=llm_hidden)
            self.perception = self.perception.to(torch.float32)
```

### NEW (what to replace it with)
```python
        with self._mark_tower_model(vllm_config, {"audio"}):
            self.perception = _load_nemo_perception(
                config.perception,
                output_dim=llm_hidden,
                pe_encoder_path=getattr(config, "pe_encoder_path", None),
            )
            self.perception = self.perception.to(torch.float32)
```

## Patch (transcript line 361)
- `old_string` length: 286 chars
- `new_string` length: 1092 chars
- replace_all: False

### OLD (what the file should contain before this patch)
```python
    cfg = DictConfig(perception_cfg)
    if "output_dim" not in cfg:
        cfg.output_dim = output_dim
    perception = AudioPerceptionModule(cfg)

    if pe_encoder_path:
        _mount_parallel_expert_encoder(perception, pe_encoder_path)

    perception.eval()
    return perception
```

### NEW (what to replace it with)
```python
    cfg = DictConfig(perception_cfg)
    if "output_dim" not in cfg:
        cfg.output_dim = output_dim
    perception = AudioPerceptionModule(cfg)

    if pe_encoder_path:
        _mount_parallel_expert_encoder(perception, pe_encoder_path)

    # ConformerEncoders default to sync_max_audio_length=True, which fires an
    # NCCL all-reduce inside .forward() whenever torch.distributed is
    # initialised. vLLM initialises a single-rank NCCL group for its own KV
    # cache profile_run, and the all-reduce target tensor is built fresh on a
    # CPU-resolved device path -- crashes with "No backend type associated
    # with device type cpu". For single-rank inference this sync is a no-op
    # anyway; disable it on every Conformer under the perception module.
    try:
        from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder

        for module in perception.modules():
            if isinstance(module, ConformerEncoder):
                module.sync_max_audio_length = False
    except Exception:
        pass

    perception.eval()
    return perception
```

## Patch (transcript line 416)
- `old_string` length: 865 chars
- `new_string` length: 1430 chars
- replace_all: False

### OLD (what the file should contain before this patch)
```python
    def _process_audio(self, audio_input: NeMoSpeechLMAudioInputs) -> tuple[torch.Tensor, ...]:
        device = next(self.perception.parameters()).device
        self.perception = self.perception.to(device)

        audio_signal = audio_input.audio_signal
        if isinstance(audio_signal, list):
            audio_signal = torch.stack(audio_signal, dim=0)
        audio_signal = audio_signal.to(device=device, dtype=torch.float32)
        audio_lengths = audio_input.audio_signal_length.to(device=device)

        with torch.no_grad():
            audio_embeds, audio_embed_lens = self.perception(
                input_signal=audio_signal,
                input_signal_length=audio_lengths,
            )

        audio_embeds = audio_embeds.to(torch.bfloat16)

        return tuple(audio_embeds[i, : audio_embed_lens[i]] for i in range(audio_embeds.shape[0]))
```

### NEW (what to replace it with)
```python
    def _process_audio(self, audio_input: NeMoSpeechLMAudioInputs) -> tuple[torch.Tensor, ...]:
        # Pin perception to the language model's device. vLLM moves
        # language_model to GPU automatically but treats self.perception
        # (tower_model) as opaque, so on the very first call we still find
        # the encoder + PE bundle sitting on CPU -- which then leaks CPU
        # tensors into the LLM's masked_scatter on inputs_embeds (cuda).
        try:
            device = next(self.language_model.parameters()).device
        except StopIteration:
            device = next(self.perception.parameters()).device
        if next(self.perception.parameters()).device != device:
            self.perception = self.perception.to(device)

        audio_signal = audio_input.audio_signal
        if isinstance(audio_signal, list):
            audio_signal = torch.stack(audio_signal, dim=0)
        audio_signal = audio_signal.to(device=device, dtype=torch.float32)
        audio_lengths = audio_input.audio_signal_length.to(device=device)

        with torch.no_grad():
            audio_embeds, audio_embed_lens = self.perception(
                input_signal=audio_signal,
                input_signal_length=audio_lengths,
            )

        audio_embeds = audio_embeds.to(device=device, dtype=torch.bfloat16)

        return tuple(audio_embeds[i, : audio_embed_lens[i]] for i in range(audio_embeds.shape[0]))
```
