# ============================================================================
# DECOMPILED FROM /home/taejinp/projects/canary-dev/canary-dev/speechlm-2026h1/NeMo/nemo/collections/speechlm2/vllm/nemotron_v3/__pycache__/model.cpython-312.pyc
# ----------------------------------------------------------------------------
# This file was destroyed and only the compiled bytecode (.pyc) survived.
# pycdc (https://github.com/zrax/pycdc) was used to decompile it; the output
# below is INCOMPLETE because Python 3.12 introduced opcodes (MAKE_CELL,
# LOAD_FAST_AND_CLEAR, LOAD_FAST_CHECK) that pycdc does not fully support.
#
# Sources of truth for finishing the restoration:
#   * full pycdas disassembly:  /home/taejinp/projects/canary-dev/canary-dev/speechlm-2026h1/NeMo/nemo/collections/speechlm2/vllm/nemotron_v3/model.py.pyc_disasm.txt
#   * pycdc warnings:           /home/taejinp/projects/canary-dev/canary-dev/speechlm-2026h1/NeMo/nemo/collections/speechlm2/vllm/nemotron_v3/model.py.pyc_decompile_warn.txt
#   * recorded StrReplace ops:  /home/taejinp/projects/canary-dev/canary-dev/speechlm-2026h1/NeMo/nemo/collections/speechlm2/vllm/nemotron_v3/model.py.RESTORE_TODO.md
#     (these were applied on top of the original file during the destroyed
#      session; apply them again after restoring the body)
#
# Until this file is fully restored, importing
#   nemo.collections.speechlm2.vllm.nemotron_v3
# may fail.  The original is intact on draco-oci-iad at
#   /lustre/fs12/portfolios/llmservice/users/taejinp/projects/canary-dev/canary-dev/
#       speechlm-2026h1/NeMo/nemo/collections/speechlm2/vllm/nemotron_v3/
# so an rsync from there is the cleanest recovery.
# ============================================================================

'''Inference-only NeMo Speech LM model for vLLM.

Architecture: NeMo speech encoder (e.g. FastConformer) + projection + LLM backbone
(e.g. NemotronH).  Requires NeMo toolkit for the audio encoder:
``pip install nemo_toolkit[asr]``
'''
import re
from collections.abc import Iterable, Mapping
from typing import Annotated, Literal
import torch
from torch import nn
from transformers import BatchFeature
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.models.interfaces import IsHybrid, MultiModalEmbeddings, SupportsMambaPrefixCaching, SupportsMultiModal, SupportsPP
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import BaseMultiModalProcessor, BaseProcessingInfo, PromptReplacement, PromptUpdate, PromptUpdateDetails
from vllm.multimodal.processing.dummy_inputs import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape
_AUDIO_PLACEHOLDER = '<|audio|>'
_SAMPLING_RATE = 16000
_MAX_AUDIO_DURATION_S = 40

def _ensure_special_tokens(tokenizer):
    special = [
        _AUDIO_PLACEHOLDER]
    existing = set(tokenizer.get_vocab().keys())
# WARNING: Decompyle incomplete


def _mount_parallel_expert_encoder(perception = None, pe_encoder_path = None):
    '''Replace ``perception.encoder`` with a ParallelExpertEncoder loaded from a
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
    '''
    is_parallel_expert_encoder_nemo = is_parallel_expert_encoder_nemo
    load_parallel_expert_encoder_from_nemo = load_parallel_expert_encoder_from_nemo
    import nemo.collections.asr.modules.parallel_expert_encoder
    if not is_parallel_expert_encoder_nemo(pe_encoder_path):
        raise ValueError(f'''pe_encoder_path={pe_encoder_path!r} is not a ParallelExpertEncoderPT .nemo bundle.''')
    pe_encoder = load_parallel_expert_encoder_from_nemo(pe_encoder_path, map_location = 'cpu', strict = True)
    existing_encoder = getattr(perception, 'encoder', None)
# WARNING: Decompyle incomplete


def _load_nemo_perception(perception_cfg = None, output_dim = None, pe_encoder_path = None):
    
    try:
        DictConfig = DictConfig
        import omegaconf
        AudioPerceptionModule = AudioPerceptionModule
        import nemo.collections.speechlm2.modules
        cfg = DictConfig(perception_cfg)
        if 'output_dim' not in cfg:
            cfg.output_dim = output_dim
        perception = AudioPerceptionModule(cfg)
        if pe_encoder_path:
            _mount_parallel_expert_encoder(perception, pe_encoder_path)
        
        try:
            ConformerEncoder = ConformerEncoder
            import nemo.collections.asr.modules.conformer_encoder
            for module in perception.modules():
                if not isinstance(module, ConformerEncoder):
                    continue
                    
                    try:
                        module.sync_max_audio_length = False
                        continue
                        perception.eval()
                        return perception
                        except ImportError:
                            e = None
                            raise ImportError('NeMo is required for the audio encoder. Install with: pip install nemo_toolkit[asr]'), e
                            e = None
                            del e
                    except Exception:
                        continue





class NeMoSpeechLMAudioInputs(TensorSchema):
    """Typed schema for audio inputs passed through vLLM's multimodal pipeline."""
    audio_signal_length: Annotated[(torch.Tensor, TensorShape('b'))] = 'audio_features'


class NeMoSpeechLMProcessingInfo(BaseProcessingInfo):
    '''Processing info for NeMo Speech LM: audio token estimation and limits.'''
    
    def get_data_parser(self = None):
        return MultiModalDataParser(target_sr = _SAMPLING_RATE, expected_hidden_size = self._get_expected_hidden_size())

    
    def get_supported_mm_limits(self = None):
        return {
            'audio': 1 }

    
    def get_max_audio_tokens(self = None):
        return self._estimate_audio_tokens(self.get_max_audio_len())

    
    def get_max_audio_len(self = None):
        return int(_MAX_AUDIO_DURATION_S * _SAMPLING_RATE)

    _estimate_audio_tokens = (lambda audio_length_samples = None: n_fft = 512hop_length = 160stft_pad = n_fft // 2fbank_len = (audio_length_samples + 2 * stft_pad - n_fft) // hop_length(kernel, stride, repeat) = (3, 2, 3)add_pad = 2 - kernellength = float(fbank_len)for _ in range(repeat):
length = (length + add_pad) / stride + 1max(1, int(length)))()


def NeMoSpeechLMMultiModalProcessor():
    '''NeMoSpeechLMMultiModalProcessor'''
    __doc__ = 'Multimodal processor that handles audio tokenization and prompt expansion.'
    
    def _get_mm_fields_config(self = None, hf_inputs = None, hf_processor_mm_kwargs = None):
        return dict(audio_signal = MultiModalFieldConfig.batched('audio'), audio_signal_length = MultiModalFieldConfig.batched('audio'))

    
    def _hf_processor_applies_updates(self, prompt_text = None, mm_items = None, hf_processor_mm_kwargs = None, tokenization_kwargs = ('prompt_text', str, 'mm_items', MultiModalDataItems, 'hf_processor_mm_kwargs', Mapping[(str, object)], 'tokenization_kwargs', Mapping[(str, object)], 'return', bool)):
        return False

    
    def _get_prompt_updates(self = None, mm_items = None, hf_processor_mm_kwargs = None, out_mm_kwargs = ('mm_items', MultiModalDataItems, 'hf_processor_mm_kwargs', Mapping[(str, object)], 'out_mm_kwargs', MultiModalKwargsItems, 'return', list[PromptUpdate])):
        pass
    # WARNING: Decompyle incomplete

    
    def _call_hf_processor(self, prompt = None, mm_data = None, mm_kwargs = None, tok_kwargs = ('prompt', str, 'mm_data', Mapping[(str, object)], 'mm_kwargs', Mapping[(str, object)], 'tok_kwargs', Mapping[(str, object)], 'return', BatchFeature)):
        tokenizer = self.info.get_tokenizer()
        _ensure_special_tokens(tokenizer)
        mm_data = dict(mm_data)
        audios = mm_data.pop('audios', [])
        if audios:
            audio_list = []
            audio_lengths = []
            parts = re.split(f'''({re.escape(_AUDIO_PLACEHOLDER)})''', prompt)
            audio_idx = 0
            for i, part in enumerate(parts):
                if not part == _AUDIO_PLACEHOLDER:
                    continue
                if not audio_idx < len(audios):
                    continue
                audio = audios[audio_idx]
                audio_tensor = audio if isinstance(audio, torch.Tensor) else torch.as_tensor(audio, dtype = torch.float32)
                if audio_tensor.dim() > 1:
                    audio_tensor = audio_tensor.squeeze()
                n_tokens = self.info._estimate_audio_tokens(audio_tensor.shape[-1])
                parts[i] = _AUDIO_PLACEHOLDER * n_tokens
                audio_list.append(audio_tensor)
                audio_lengths.append(audio_tensor.shape[-1])
                audio_idx += 1
            prompt = ''.join(parts)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens = True)
        result = BatchFeature(dict(input_ids = [
            prompt_ids]), tensor_type = 'pt')
    # WARNING: Decompyle incomplete


NeMoSpeechLMMultiModalProcessor = <NODE:27>(NeMoSpeechLMMultiModalProcessor, 'NeMoSpeechLMMultiModalProcessor', BaseMultiModalProcessor[NeMoSpeechLMProcessingInfo])

def NeMoSpeechLMDummyInputsBuilder():
    '''NeMoSpeechLMDummyInputsBuilder'''
    __doc__ = "Builds dummy audio inputs for vLLM's model profiling and warmup."
    
    def get_dummy_mm_data(self = None, seq_len = None, mm_counts = None, mm_options = ('seq_len', int, 'mm_counts', Mapping[(str, int)], 'mm_options', Mapping[(str, BaseDummyOptions)], 'return', MultiModalDataDict)):
        num_audios = mm_counts.get('audio', 0)
        return {
            'audio': self._get_dummy_audios(length = self.info.get_max_audio_len(), num_audios = num_audios) }

    
    def get_dummy_text(self = None, mm_counts = None):
        num_audios = mm_counts.get('audio', 0)
        return 'Transcribe the following: ' + _AUDIO_PLACEHOLDER * num_audios


NeMoSpeechLMDummyInputsBuilder = <NODE:27>(NeMoSpeechLMDummyInputsBuilder, 'NeMoSpeechLMDummyInputsBuilder', BaseDummyInputsBuilder[NeMoSpeechLMProcessingInfo])
NeMoSpeechLMForConditionalGeneration = <NODE:12>()
