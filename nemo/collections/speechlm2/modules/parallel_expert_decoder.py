# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense speech decoder conditioned by an immutable Qwen3.5 text backbone.

The text route calls the original causal LM directly. Speech uses the same token
timeline in two independent hybrid decoders, with six one-way residual bridges.
This correctness-first execution uses eager per-branch kernels; it does not claim
the encoder GGEMM implementation supports Qwen's SwiGLU or DeltaNet operators.
"""

import hashlib
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM


@dataclass
class SpeechDecoderSpec:
    """Speech body dimensions; the vocabulary comes from the frozen checkpoint."""

    hidden_size: int = 1024
    intermediate_size: int = 3072
    linear_heads: int = 8
    num_attention_heads: int = 4
    num_key_value_heads: int = 1
    head_dim: int = 256
    linear_head_dim: int = 128
    frontend_width: int = 256
    n_mels: int = 128
    bridge_every: int = 4
    checkpoint_layers: bool = True


class CausalAudioPerception(nn.Module):
    """Three causal stride-two convolutions, with no utterance-wide statistics."""

    def __init__(self, n_mels: int, width: int):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(n_mels if i == 0 else width, width, 5, stride=2) for i in range(3)])
        self.norm = nn.LayerNorm(width)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # End-align the three stride-two stages: output token i sees frames
        # through 8*i+7. Pad only the incomplete final group of eight frames.
        mel = F.pad(mel, (0, (-mel.shape[-1]) % 8))
        for convolution in self.convs:
            mel = F.silu(convolution(F.pad(mel, (3, 0))))
        return self.norm(mel.transpose(1, 2))


class QwenSpeechBridge(nn.Module):
    """A trainable one-way bridge; initialization admits adapter gradients."""

    def __init__(self, text_width: int, speech_width: int):
        super().__init__()
        self.norm = nn.LayerNorm(text_width)
        self.projection = nn.Linear(text_width, speech_width, bias=False)
        self.gate = nn.Parameter(torch.tensor(0.01))

    def forward(self, speech: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        return speech + self.gate.tanh() * self.projection(self.norm(text))


class ParallelExpertDecoder(nn.Module):
    """Frozen Qwen plus a dense speech decoder on a shared causal timeline."""

    def __init__(self, qwen: Qwen3_5ForCausalLM, spec: SpeechDecoderSpec):
        super().__init__()
        self.qwen = qwen.requires_grad_(False).eval()
        self.spec = spec
        text_config = qwen.config
        if text_config.model_type != 'qwen3_5_text':
            raise ValueError('PED requires the Qwen3.5 text causal-LM backbone.')
        if text_config.num_hidden_layers % spec.bridge_every:
            raise ValueError('The bridge interval must divide the decoder layer count.')
        config_dict = text_config.to_dict()
        config_dict.update(
            hidden_size=spec.hidden_size,
            intermediate_size=spec.intermediate_size,
            num_attention_heads=spec.num_attention_heads,
            num_key_value_heads=spec.num_key_value_heads,
            head_dim=spec.head_dim,
            linear_num_key_heads=spec.linear_heads,
            linear_num_value_heads=spec.linear_heads,
            linear_key_head_dim=spec.linear_head_dim,
            linear_value_head_dim=spec.linear_head_dim,
            tie_word_embeddings=True,
            use_cache=False,
        )
        speech_config = Qwen3_5TextConfig(**config_dict)
        speech_config._attn_implementation = text_config._attn_implementation
        self.speech = Qwen3_5ForCausalLM(speech_config)
        self.perception = CausalAudioPerception(spec.n_mels, spec.frontend_width)
        self.audio_to_speech = nn.Linear(spec.frontend_width, spec.hidden_size)
        self.audio_to_qwen = nn.Linear(spec.frontend_width, text_config.hidden_size)
        self.bridges = nn.ModuleList(
            QwenSpeechBridge(text_config.hidden_size, spec.hidden_size)
            for _ in range(text_config.num_hidden_layers // spec.bridge_every)
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.qwen.eval()
        return self

    def forward_text(self, **kwargs):
        """Bypass every speech module and preserve the original text API."""
        return self.qwen(**kwargs)

    def generate_text(self, **kwargs):
        """Delegate sampling, caches, and generation to the original Qwen model."""
        return self.qwen.generate(**kwargs)

    def forward_speech(
        self,
        input_ids: torch.Tensor,
        audio_indices: torch.Tensor,
        mel: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: dict | None = None,
        audio_features: torch.Tensor | None = None,
    ) -> dict:
        """Compute speech predictions; ``labels`` already contain next-token targets.

        ``audio_indices`` is -1 at text positions, otherwise an index into the
        utterance's subsampled audio. Both streams receive identical positions.
        No Qwen activation is detached: gradients reach its audio input adapter.
        """
        if use_cache and self.training:
            raise ValueError('Mutable decoder caches are only supported in evaluation mode.')
        caches = past_key_values
        if use_cache and caches is None:
            caches = {'speech': DynamicCache(config=self.speech.config), 'qwen': DynamicCache(config=self.qwen.config)}
        speech_cache = caches['speech'] if use_cache else None
        text_cache = caches['qwen'] if use_cache else None
        audio = self.perception(mel.float()) if audio_features is None else audio_features
        speech = self._splice_audio(
            self.speech.model.embed_tokens(input_ids), audio, audio_indices, self.audio_to_speech
        )
        text = self._splice_audio(self.qwen.model.embed_tokens(input_ids), audio, audio_indices, self.audio_to_qwen)
        speech_context = self._layer_context(self.speech.model, speech, attention_mask, speech_cache)
        text_context = self._layer_context(self.qwen.model, text, attention_mask, text_cache)
        for index, (text_layer, speech_layer) in enumerate(zip(self.qwen.model.layers, self.speech.model.layers)):
            text_mask, text_position, text_rotary = text_context
            speech_mask, speech_position, speech_rotary = speech_context
            text = self._run_layer(text_layer, text, text_mask[index], text_position, text_rotary, text_cache)
            speech = self._run_layer(
                speech_layer, speech, speech_mask[index], speech_position, speech_rotary, speech_cache
            )
            if (index + 1) % self.spec.bridge_every == 0:
                speech = self.bridges[index // self.spec.bridge_every](speech, text)
        speech = self.speech.model.norm(speech)
        if labels is None:
            return {'logits': self.speech.lm_head(speech[:, -1:]), 'past_key_values': caches}
        selected = labels != -100
        if not selected.any():
            raise ValueError('A training example must have at least one supervised text target.')
        # Avoid allocating a vocabulary-sized logit tensor at acoustic positions.
        logits = self.speech.lm_head(speech[selected])
        loss = F.cross_entropy(logits.float(), labels[selected])
        return {'loss': loss, 'tokens': selected.sum(), 'logits': logits}

    def trainable_state_dict(self) -> dict:
        """Checkpoint only speech/perception/bridges; frozen Qwen is referenced."""
        return {name: value for name, value in self.state_dict().items() if not name.startswith('qwen.')}

    def assert_frozen(self) -> None:
        if any(parameter.requires_grad or parameter.grad is not None for parameter in self.qwen.parameters()):
            raise RuntimeError('Frozen Qwen received a trainable parameter or a parameter gradient.')

    def qwen_digest(self) -> str:
        """Hash all frozen tensor bytes to verify preservation across training."""
        digest = hashlib.sha256()
        for name, parameter in self.qwen.state_dict().items():
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def _run_layer(self, layer, hidden, mask, positions, rotary, cache=None):
        def run(current):
            return layer(
                current,
                attention_mask=mask,
                position_ids=positions,
                position_embeddings=rotary,
                past_key_values=cache,
                use_cache=cache is not None,
            )

        if self.training and self.spec.checkpoint_layers and torch.is_grad_enabled():
            return checkpoint(run, hidden, use_reentrant=False)
        return run(hidden)

    @staticmethod
    def _splice_audio(embedding, audio, indices, projection):
        projected = projection(audio).to(embedding.dtype)
        gathered = projected.gather(1, indices.clamp(min=0).unsqueeze(-1).expand(-1, -1, projected.shape[-1]))
        return torch.where((indices >= 0).unsqueeze(-1), gathered, embedding)

    @staticmethod
    def _layer_context(backbone, hidden, mask, cache=None):
        batch, length, _ = hidden.shape
        start = cache.get_seq_length() if cache is not None else 0
        positions = torch.arange(start, start + length, device=hidden.device).unsqueeze(0).expand(batch, -1)
        rotary = backbone.rotary_emb(hidden, positions.unsqueeze(0).expand(3, -1, -1))
        causal = create_causal_mask(
            config=backbone.config,
            inputs_embeds=hidden,
            attention_mask=mask,
            past_key_values=cache,
            position_ids=positions,
        )
        linear = backbone._update_linear_attn_mask(mask, cache)
        masks = [linear if kind == 'linear_attention' else causal for kind in backbone.config.layer_types]
        return masks, positions, rotary
