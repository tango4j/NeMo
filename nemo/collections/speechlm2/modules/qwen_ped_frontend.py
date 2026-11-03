# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Encoder-free, offline audio input and dimension contracts for Qwen Speech-PED."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class QwenPEDSpec:
    """The agreed dense ~850M speech branch; head widths are not residual widths."""

    hidden_size: int = 1024
    intermediate_size: int = 2048
    num_hidden_layers: int = 40
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 256
    linear_num_key_heads: int = 8
    linear_num_value_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    bridge_every: int = 4
    frontend_width: int = 256
    n_mels: int = 128

    def validate_backbone(self, config):
        """Reject a checkpoint whose operators cannot use the planned head packing."""
        if config.hidden_size != 2048 or config.num_hidden_layers != self.num_hidden_layers:
            raise ValueError('Expected the 40-layer, width-2048 Qwen3.5-35B-A3B text backbone.')
        expected = ['linear_attention'] * 3 + ['full_attention']
        if list(config.layer_types) != expected * (self.num_hidden_layers // 4):
            raise ValueError('Qwen and speech must have identical hybrid layer schedules.')
        for name in ('head_dim', 'linear_key_head_dim', 'linear_value_head_dim'):
            if getattr(config, name) != getattr(self, name):
                raise ValueError(f'Incompatible head dimension: {name}')
        if (
            config.num_attention_heads * self.num_key_value_heads
            != self.num_attention_heads * config.num_key_value_heads
        ):
            raise ValueError('Full-attention GQA ratios must match.')
        if (
            config.linear_num_value_heads * self.linear_num_key_heads
            != self.linear_num_value_heads * config.linear_num_key_heads
        ):
            raise ValueError('DeltaNet value-to-key head ratios must match.')

    def speech_config(self, text_config):
        """Construct a dense speech decoder, not a copy of the 256-expert MoE."""
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

        self.validate_backbone(text_config)
        return Qwen3_5TextConfig(
            vocab_size=text_config.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            linear_num_key_heads=self.linear_num_key_heads,
            linear_num_value_heads=self.linear_num_value_heads,
            linear_key_head_dim=self.linear_key_head_dim,
            linear_value_head_dim=self.linear_value_head_dim,
            linear_conv_kernel_dim=text_config.linear_conv_kernel_dim,
            layer_types=list(text_config.layer_types),
            rope_parameters=dict(text_config.rope_parameters),
            rms_norm_eps=text_config.rms_norm_eps,
            tie_word_embeddings=True,
            use_cache=False,
        )


class OfflineAudioSubsampler(nn.Module):
    """Local convolutions only: no speech encoder, attention, or recurrent context.

    Three stride-two convolutions give ceil(mel_frames / 8) tokens. Explicit
    frame masking makes valid outputs invariant to other examples' padding.
    """

    subsampling_factor = 8

    def __init__(self, n_mels=128, width=256):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(n_mels if i == 0 else width, width, 5, stride=2) for i in range(3)])
        self.norm = nn.LayerNorm(width)

    def forward(self, mel, lengths):
        features = mel
        for conv in self.convs:
            mask = torch.arange(features.shape[-1], device=features.device)[None] < lengths[:, None]
            features = features.masked_fill(~mask[:, None], 0)
            features = F.silu(conv(F.pad(features, (2, 2))))
            lengths = (lengths + 1) // 2
        features = self.norm(features.transpose(1, 2))
        mask = torch.arange(features.shape[1], device=features.device)[None] < lengths[:, None]
        return features.masked_fill(~mask[..., None], 0), lengths


class PairedAudioProjection(nn.Module):
    """Project the same audio frames into aligned, independently learned streams."""

    def __init__(self, width=256, text_width=2048, speech_width=1024):
        super().__init__()
        self.to_text = nn.Linear(width, text_width)
        self.to_speech = nn.Linear(width, speech_width)

    def forward(self, features):
        return torch.cat((self.to_text(features), self.to_speech(features)), dim=-1)


class SpeechTaskTokenEmbeddings(nn.Module):
    """Speech-only teacher-width embeddings; original Qwen rows stay untouched.

    This small sidecar has an independent FSDP root because it runs before the
    speech decoder. A directly DTensor-sharded caller also gets gradient-correct
    materialization, just like the speech decoder's tied input/output table.
    """

    def __init__(self, records, frozen_weight):
        super().__init__()
        records = sorted(records, key=lambda row: row['id'])
        identifiers = [row['id'] for row in records]
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError('Speech task token IDs must be nonempty and unique.')
        self.register_buffer('token_ids', torch.tensor(identifiers, dtype=torch.long))
        self.embedding = nn.Embedding(len(records), frozen_weight.shape[1], dtype=torch.float32)
        with torch.no_grad():
            for index, record in enumerate(records):
                decomposition = record['text_init_ids']
                if not decomposition or min(decomposition) < 0 or max(decomposition) >= frozen_weight.shape[0]:
                    raise ValueError('Invalid original-Qwen token decomposition for speech task embedding.')
                value = frozen_weight[decomposition].detach().float().mean(0)
                self.embedding.weight[index].copy_(value)

    def matching_rows(self, input_ids):
        identifiers = self.token_ids.to(input_ids.device)
        rows = torch.searchsorted(identifiers, input_ids.contiguous()).clamp_max(identifiers.numel() - 1)
        return rows, identifiers[rows].eq(input_ids)

    def sanitize_ids(self, input_ids):
        _, matched = self.matching_rows(input_ids)
        return input_ids.masked_fill(matched, 0)

    def forward(self, input_ids, frozen_embeddings, grad_reduce_group=None):
        from nemo_automodel.components.loss.linear_ce_base import LinearCrossEntropy

        weight = LinearCrossEntropy.materialize_lm_weight(self.embedding.weight, grad_reduce_group=grad_reduce_group)
        rows, matched = self.matching_rows(input_ids)
        values = F.embedding(rows, weight).to(frozen_embeddings.dtype)
        return torch.where(matched[..., None], values, frozen_embeddings)
