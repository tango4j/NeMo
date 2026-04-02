# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import re
from typing import List, Optional, Union

import torch
from torchmetrics import Metric

from nemo.collections.asr.metrics.der import calculate_session_cpWER
from nemo.collections.asr.metrics.wer import move_dimension_to_the_front
from nemo.collections.asr.parts.submodules.ctc_decoding import AbstractCTCDecoding
from nemo.collections.asr.parts.submodules.multitask_decoding import AbstractMultiTaskDecoding
from nemo.collections.asr.parts.submodules.rnnt_decoding import AbstractRNNTDecoding
from nemo.utils import logging

__all__ = ['CpWER', 'split_text_by_speaker_tags']

DEFAULT_SPEAKER_TAG_PATTERN = re.compile(r'\[s\d+\]')


def split_text_by_speaker_tags(text: str, speaker_tag_pattern: re.Pattern = None) -> List[str]:
    """
    Split SOT-formatted text by bracket speaker tags ([s0], [s1], ...) and
    return a list of per-speaker transcripts.

    Segments belonging to the same speaker are concatenated in order of
    appearance. Speakers are sorted by tag index so the list position is
    deterministic.

    If the text contains no speaker tags the entire string is returned as a
    single-element list.

    Example:
        >>> split_text_by_speaker_tags("[s0] hello world [s1] good morning [s0] how are you")
        ['hello world how are you', 'good morning']
    """
    if speaker_tag_pattern is None:
        speaker_tag_pattern = DEFAULT_SPEAKER_TAG_PATTERN
    tag_matches = list(speaker_tag_pattern.finditer(text))
    if not tag_matches:
        stripped_text = text.strip()
        return [stripped_text] if stripped_text else []

    speaker_segments: dict[str, list[str]] = {}
    for idx, tag_match in enumerate(tag_matches):
        speaker_tag = tag_match.group()
        seg_start = tag_match.end()
        seg_end = tag_matches[idx + 1].start() if idx + 1 < len(tag_matches) else len(text)
        segment_text = text[seg_start:seg_end].strip()
        if speaker_tag not in speaker_segments:
            speaker_segments[speaker_tag] = []
        if segment_text:
            speaker_segments[speaker_tag].append(segment_text)

    return [' '.join(speaker_segments[tag]) for tag in sorted(speaker_segments)]


class CpWER(Metric):
    """
    Concatenated minimum-permutation Word Error Rate (cpWER) metric for
    multi-speaker ASR evaluation.

    Uses ``[s0]``, ``[s1]``, ... bracket speaker tags in both hypothesis and
    reference to create per-speaker transcripts, then calls
    ``calculate_session_cpWER`` (Hungarian-algorithm optimal assignment) to
    compute the session-level cpWER.

    Accumulates total edit-distance errors and total reference words across
    all samples so that ``compute()`` returns a micro-averaged cpWER that is
    correct under DDP (states are summed across workers).

    The ``compute()`` return signature ``(rate, numerator, denominator)``
    matches ``WER.compute()`` so the existing ``MultiTaskMetric`` aggregation
    pipeline works consistently.
    """

    full_state_update: bool = True

    def __init__(
        self,
        decoding: Union[AbstractCTCDecoding, AbstractRNNTDecoding, AbstractMultiTaskDecoding],
        log_prediction=True,
        batch_dim_index=0,
        dist_sync_on_step=False,
        sync_on_compute=True,
        speaker_tag_pattern: re.Pattern = None,
        **kwargs,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step, sync_on_compute=sync_on_compute)

        self.decoding = decoding
        self.log_prediction = log_prediction
        self.batch_dim_index = batch_dim_index
        self.speaker_tag_pattern = speaker_tag_pattern if speaker_tag_pattern is not None else DEFAULT_SPEAKER_TAG_PATTERN

        self.decode = None
        if isinstance(self.decoding, AbstractRNNTDecoding):
            self.decode = lambda predictions, predictions_lengths, predictions_mask, input_ids: (
                self.decoding.rnnt_decoder_predictions_tensor(
                    encoder_output=predictions, encoded_lengths=predictions_lengths, return_hypotheses=False
                )
            )
        elif isinstance(self.decoding, AbstractCTCDecoding):
            self.decode = lambda predictions, predictions_lengths, predictions_mask, input_ids: (
                self.decoding.ctc_decoder_predictions_tensor(
                    decoder_outputs=predictions,
                    decoder_lengths=predictions_lengths,
                    fold_consecutive=True,
                    return_hypotheses=False,
                )
            )
        elif isinstance(self.decoding, AbstractMultiTaskDecoding):
            self.decode = lambda predictions, prediction_lengths, predictions_mask, input_ids: (
                self.decoding.decode_predictions_tensor(
                    encoder_hidden_states=predictions,
                    encoder_input_mask=predictions_mask,
                    decoder_input_ids=input_ids,
                    return_hypotheses=False,
                )
            )
        else:
            raise TypeError(f"CpWER metric does not support decoding of type {type(self.decoding)}")

        self.add_state("total_edit_distance", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)
        self.add_state("total_ref_word_count", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)

    def update(
        self,
        predictions: torch.Tensor,
        predictions_lengths: torch.Tensor,
        targets: torch.Tensor,
        targets_lengths: torch.Tensor,
        predictions_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        references = []
        with torch.no_grad():
            target_lengths_cpu = targets_lengths.long().cpu()
            targets_cpu = targets.long().cpu()
            if self.batch_dim_index != 0:
                targets_cpu = move_dimension_to_the_front(targets_cpu, self.batch_dim_index)
            for sample_idx in range(targets_cpu.shape[0]):
                target_len = target_lengths_cpu[sample_idx].item()
                target_token_ids = targets_cpu[sample_idx][:target_len].numpy().tolist()
                reference_text = self.decoding.decode_ids_to_str(target_token_ids)
                references.append(reference_text)
            hypotheses = (
                self.decode(predictions, predictions_lengths, predictions_mask, input_ids)
                if predictions.numel() > 0
                else []
            )

        batch_edit_distance = 0
        batch_ref_word_count = 0

        for hypothesis, reference_text in zip(hypotheses, references):
            if isinstance(hypothesis, list):
                hypothesis = hypothesis[0]
            hyp_text = hypothesis.text if hasattr(hypothesis, 'text') else str(hypothesis)

            spk_hyp_transcripts = split_text_by_speaker_tags(hyp_text, self.speaker_tag_pattern)
            spk_ref_transcripts = split_text_by_speaker_tags(reference_text, self.speaker_tag_pattern)

            if not spk_ref_transcripts:
                continue

            ref_word_count = sum(len(transcript.split()) for transcript in spk_ref_transcripts)
            if ref_word_count == 0:
                continue

            cpwer_rate, _, _ = calculate_session_cpWER(spk_hyp_transcripts, spk_ref_transcripts)
            batch_edit_distance += round(cpwer_rate * ref_word_count)
            batch_ref_word_count += ref_word_count

        if self.log_prediction and hypotheses and references:
            first_hyp = hypotheses[0]
            first_hyp_text = first_hyp.text if hasattr(first_hyp, 'text') else str(first_hyp)
            logging.info("\n")
            logging.info(f"cpWER reference : {references[0]}")
            logging.info(f"cpWER predicted : {first_hyp_text}")

        self.total_edit_distance = torch.tensor(
            batch_edit_distance, device=self.total_edit_distance.device, dtype=self.total_edit_distance.dtype
        )
        self.total_ref_word_count = torch.tensor(
            batch_ref_word_count, device=self.total_ref_word_count.device, dtype=self.total_ref_word_count.dtype
        )

    def compute(self):
        edit_distance = self.total_edit_distance.detach().float()
        ref_word_count = self.total_ref_word_count.detach().float()
        cpwer_rate = edit_distance / ref_word_count if ref_word_count > 0 else torch.tensor(float('inf'))
        return cpwer_rate, edit_distance, ref_word_count
