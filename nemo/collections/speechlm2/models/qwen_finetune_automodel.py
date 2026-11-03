# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-decoder Qwen3.5 full fine-tuning using SALM Automodel's native pipeline.

Unlike qwen_ped_automodel, there is no speech decoder, gate, bridge, or LoRA.
Only a necessary 8-frame acoustic input projection precedes the original Qwen.
"""

from collections import defaultdict

import jiwer
import torch
from torch import nn

from nemo.collections.asr.parts.submodules.subsampling import FeatureStacking
from nemo.collections.speechlm2.models.salm_automodel import SALMAutomodel
from nemo.collections.speechlm2.models.salm import replace_placeholders_and_build_targets
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers_exclude_norm_from_wd


class QwenFinetuneAutomodel(SALMAutomodel):
    """Retain SALM preprocessing, prompt packing, Lhotse loader and training loss."""

    def __init__(self, cfg):
        if cfg.get('lora') is not None or cfg.get('freeze_params') or cfg.get('automodel_freeze_config'):
            raise ValueError('The full-finetuning baseline forbids LoRA and frozen decoder parameters.')
        super().__init__(cfg)

    def configure_model(self, **kwargs):
        if self.llm is not None:
            return
        super().configure_model(**kwargs)
        if self.llm.config.model_type != 'qwen3_5_text':
            raise ValueError('Use the dense Qwen3.5-2B text decoder, not a MoE or vision wrapper.')
        self.llm.requires_grad_(True)
        if self.cfg.get('fp32_master_weights', True):
            self.llm.float()
            self.perception.float()
        dropout = float(self.cfg.get('residual_dropout', 0.1))
        layers = self.llm.model.layers
        for layer in layers.values() if isinstance(layers, nn.ModuleDict) else layers:
            layer.register_forward_hook(_residual_dropout_hook(dropout))
        if self.cfg.get('gradient_checkpointing', True):
            if hasattr(self.llm, 'gradient_checkpointing_enable'):
                self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            else:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

                for key in list(layers.keys()) if isinstance(layers, nn.ModuleDict) else range(len(layers)):
                    layers[key] = checkpoint_wrapper(layers[key], preserve_rng_state=True)
        if any(not parameter.requires_grad for parameter in self.llm.parameters()):
            raise RuntimeError('Every original Qwen decoder parameter must be trainable.')

    def configure_optimizers(self):
        return configure_optimizers_exclude_norm_from_wd(self)

    def _compute_training_cross_entropy_sum(self, forward_outputs, target_ids, dp_group, *, lm_weight=None):
        logits = forward_outputs['logits']
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1),
            ignore_index=-100, reduction='sum', label_smoothing=float(self.cfg.get('label_smoothing', 0.1)),
        )
        return loss, logits

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._wer_totals = defaultdict(lambda: [0, 0, 0, 0, 0])

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        super().validation_step(batch, batch_idx)
        for name, example in batch.items():
            if example is None:
                continue
            if example['input_ids'].shape[0] != 1:
                raise ValueError('Autoregressive validation uses batch_size=1 to avoid prompt leakage.')
            totals = self._wer_totals[name]
            if totals[4] >= int(self.cfg.get('validation_wer_examples', 64)):
                continue
            mask = example['loss_mask'][0].bool()
            first_target = int(mask.nonzero()[0])
            reference = self.tokenizer.tokenizer.decode(
                example['input_ids'][0][mask].tolist(), skip_special_tokens=True
            )
            answer = self._decode_audio_prefix(
                example['input_ids'][:, :first_target], example['audios'], example['audio_lens']
            )
            hypothesis = self.tokenizer.tokenizer.decode(answer[0].tolist(), skip_special_tokens=True)
            normalizer = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                                       jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
            score = jiwer.process_words(normalizer(reference), normalizer(hypothesis))
            totals[0] += score.substitutions
            totals[1] += score.deletions
            totals[2] += score.insertions
            totals[3] += score.hits + score.substitutions + score.deletions
            totals[4] += 1
            if totals[4] % 8 == 0:
                print(f'Autoregressive validation {name}: {totals[4]} utterances', flush=True)

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        combined = [0, 0]
        for name, (sub, delete, insert, words, utterances) in self._wer_totals.items():
            if words == 0:
                continue
            wer = (sub + delete + insert) / words
            self.log(f'val_wer_{name}', wer, sync_dist=True)
            self.log(f'val_wer_percent_{name}', 100 * wer, sync_dist=True)
            self.log(f'val_wer_utterances_{name}', float(utterances), sync_dist=True)
            self.log(f'val_wer_deletions_{name}', float(delete), sync_dist=True)
            self.log(f'val_wer_insertions_{name}', float(insert), sync_dist=True)
            self.log(f'val_wer_substitutions_{name}', float(sub), sync_dist=True)
            combined[0] += sub + delete + insert
            combined[1] += words
        if combined[1]:
            self.log('val_wer', combined[0] / combined[1], prog_bar=True, sync_dist=True)

    @torch.no_grad()
    def _decode_audio_prefix(self, tokens, audios, audio_lens):
        """Native Automodel dense Qwen has no generation/cache API yet.

        Recompute the causal prefix for exact greedy inference; never replace
        native training with an HF fallback or report teacher-forced WER.
        """
        audio_embeds, lengths = self.perception(input_signal=audios, input_signal_length=audio_lens)
        embeddings = self._embed_tokens(tokens.where(tokens != self.audio_locator_tag_id, 0))
        embeddings, _, mask = replace_placeholders_and_build_targets(
            input_ids=tokens, embeds=embeddings,
            replacements=[row[:int(length)] for row, length in zip(audio_embeds, lengths)],
            placeholder_id=self.audio_locator_tag_id,
            padding_id=self.text_pad_id, target_ids=None,
        )
        return _greedy_native(
            self.llm, self._embed_tokens, embeddings, mask,
            int(self.cfg.get('validation_max_new_tokens', 256)), self.text_eos_id,
        )


class FrameStackingInput(nn.Module):
    """Only native NeMo frame stacking/projection, not an acoustic encoder stack."""

    def __init__(self, feat_in=128, d_model=2048, subsampling_factor=8, dropout=0.1):
        super().__init__()
        self.subsampling_factor = subsampling_factor
        self.stacking = FeatureStacking(subsampling_factor, feat_in, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, audio_signal, length):
        mask = torch.arange(audio_signal.shape[-1], device=audio_signal.device)[None] < length[:, None]
        audio_signal = audio_signal.masked_fill(~mask[:, None], 0)
        features, lengths = self.stacking(audio_signal, length)
        return self.dropout(features).transpose(1, 2), lengths


def _residual_dropout_hook(probability):
    if not 0 <= probability < 1:
        raise ValueError('Residual dropout must be in [0, 1).')

    def hook(module, args, output):
        if isinstance(output, tuple):
            return (nn.functional.dropout(output[0], probability, module.training), *output[1:])
        return nn.functional.dropout(output, probability, module.training)

    return hook


@torch.no_grad()
def _greedy_native(llm, embed_tokens, embeddings, mask, limit, eos_id):
    generated = []
    for _ in range(limit):
        result = llm(inputs_embeds=embeddings, attention_mask=mask, use_cache=False, logits_to_keep=1)
        token = result['logits'][:, -1].argmax(-1, keepdim=True)
        generated.append(token)
        if int(token[0, 0]) == eos_id:
            break
        embeddings = torch.cat((embeddings, embed_tokens(token)), dim=1)
        mask = torch.cat((mask, torch.ones_like(mask[:, :1])), dim=1)
    return torch.cat(generated, dim=1)
