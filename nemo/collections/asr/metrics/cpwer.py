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

from __future__ import annotations

import re
from itertools import permutations
from typing import List, Tuple, Union

import numpy as np
import torch
from kaldialign import edit_distance
from scipy.optimize import linear_sum_assignment as scipy_linear_sum_assignment
from torchmetrics import Metric

from nemo.utils import logging

__all__ = [
    'CpWER',
    'split_text_by_speaker_tags',
    'calculate_session_cpWER',
    'calculate_session_cpWER_bruteforce',
    'concat_perm_word_error_rate',
]

DEFAULT_SPEAKER_TAG_PATTERN = re.compile(r'<spk:\d+>')


def calculate_session_cpWER_bruteforce(spk_hypothesis: List[str], spk_reference: List[str]) -> Tuple[float, str, str]:
    """
    Calculate cpWER with brute-force permutation search. Matches MeetEval's cpWER algorithm:
    each (ref_speaker, hyp_speaker) pair is scored independently via edit distance, then
    cpWER = sum(errors) / sum(ref_word_counts).

    Args:
        spk_hypothesis (list):
            List containing the hypothesis transcript for each speaker.

            Example:
            >>> spk_hypothesis = ["hey how are you we that's nice", "i'm good yes hi is your sister"]

        spk_reference (list):
            List containing the reference transcript for each speaker.

            Example:
            >>> spk_reference = ["hi how are you well that's nice", "i'm good yeah how is your sister"]

    Returns:
        cpWER (float):
            cpWER value for the given session.
        min_perm_hyp_trans (str):
            Hypothesis transcript containing the permutation that minimizes WER. Words are separated by spaces.
        ref_trans (str):
            Reference transcript in an arbitrary permutation. Words are separated by spaces.
    """
    num_hyp = len(spk_hypothesis)
    num_ref = len(spk_reference)
    num_speakers_padded = max(num_hyp, num_ref)

    ref_word_lists = [
        spk_reference[ref_idx].split() if ref_idx < num_ref else [] for ref_idx in range(num_speakers_padded)
    ]
    hyp_word_lists = [
        spk_hypothesis[hyp_idx].split() if hyp_idx < num_hyp else [] for hyp_idx in range(num_speakers_padded)
    ]

    best_total_errors = float('inf')
    best_hyp_trans = ""
    total_ref_length = sum(len(word_list) for word_list in ref_word_lists)

    for perm in permutations(range(num_speakers_padded)):
        total_errors = 0
        hyp_texts = []
        for ref_idx, hyp_idx in enumerate(perm):
            total_errors += edit_distance(ref_word_lists[ref_idx], hyp_word_lists[hyp_idx])['total']
            hyp_texts.append(spk_hypothesis[hyp_idx] if hyp_idx < num_hyp else "")
        if total_errors < best_total_errors:
            best_total_errors = total_errors
            best_hyp_trans = " ".join(hyp_texts)

    cpWER = best_total_errors / total_ref_length if total_ref_length > 0 else float('inf')
    ref_trans = " ".join(spk_reference)
    return cpWER, best_hyp_trans, ref_trans


def calculate_session_cpWER(spk_hypothesis: List[str], spk_reference: List[str]) -> Tuple[float, str, str]:
    """
    Calculate a session-level concatenated minimum-permutation word error rate (cpWER) value,
    matching MeetEval's cpWER algorithm (https://github.com/fgnt/meeteval).

    Algorithm (identical to MeetEval):
        1. Build a square cost matrix of size max(num_hyp, num_ref) using raw edit distance
           counts between every (ref_speaker, hyp_speaker) pair. Missing speakers are padded
           with empty word lists.
        2. Use the Hungarian algorithm (scipy.optimize.linear_sum_assignment) to find the
           speaker assignment that minimizes total edit distance.
        3. Compute per-pair edit distance independently for the optimal assignment.
        4. cpWER = sum(errors_per_pair) / sum(ref_word_counts_per_pair).

    Args:
        spk_hypothesis (list):
            List containing the hypothesis transcript for each speaker.

            Example:
            >>> spk_hypothesis = ["hey how are you we that's nice", "i'm good yes hi is your sister"]

        spk_reference (list):
            List containing the reference transcript for each speaker.

            Example:
            >>> spk_reference = ["hi how are you well that's nice", "i'm good yeah how is your sister"]

    Returns:
        cpWER (float):
            cpWER value for the given session.
        min_perm_hyp_trans (str):
            Hypothesis transcript containing the permutation that minimizes WER. Words are separated by spaces.
        ref_trans (str):
            Reference transcript in an arbitrary permutation. Words are separated by spaces.
    """
    num_hyp = len(spk_hypothesis)
    num_ref = len(spk_reference)

    if num_hyp == 0 and num_ref == 0:
        return 0.0, "", ""

    num_speakers_padded = max(num_hyp, num_ref)

    ref_word_lists = [
        spk_reference[ref_idx].split() if ref_idx < num_ref else [] for ref_idx in range(num_speakers_padded)
    ]
    hyp_word_lists = [
        spk_hypothesis[hyp_idx].split() if hyp_idx < num_hyp else [] for hyp_idx in range(num_speakers_padded)
    ]

    cost_matrix = np.zeros((num_speakers_padded, num_speakers_padded), dtype=np.float64)
    for ref_idx in range(num_speakers_padded):
        for hyp_idx in range(num_speakers_padded):
            cost_matrix[ref_idx, hyp_idx] = edit_distance(ref_word_lists[ref_idx], hyp_word_lists[hyp_idx])['total']

    row_ind, col_ind = scipy_linear_sum_assignment(cost_matrix)

    total_errors = 0
    total_ref_length = 0
    hyp_texts = []
    for ref_idx, hyp_idx in zip(row_ind, col_ind):
        total_errors += int(cost_matrix[ref_idx, hyp_idx])
        total_ref_length += len(ref_word_lists[ref_idx])
        hyp_texts.append(spk_hypothesis[hyp_idx] if hyp_idx < num_hyp else "")

    cpWER = total_errors / total_ref_length if total_ref_length > 0 else float('inf')

    min_perm_hyp_trans = " ".join(hyp_texts)
    ref_trans = " ".join(spk_reference)

    return cpWER, min_perm_hyp_trans, ref_trans


def concat_perm_word_error_rate(
    spk_hypotheses: List[List[str]], spk_references: List[List[str]]
) -> Tuple[List[float], List[str], List[str]]:
    """
    Launcher function for `calculate_session_cpWER`. Calculate session-level cpWER and average cpWER.
    For detailed information about cpWER, see docstrings of `calculate_session_cpWER` function.

    As opposed to `cpWER`, `WER` is the regular WER value where the hypothesis transcript contains
    words in temporal order regardless of the speakers. `WER` value can be different from cpWER value,
    depending on the speaker diarization results.

    Args:
        spk_hypotheses (list):
            List containing the lists of speaker-separated hypothesis transcripts.
        spk_references (list):
            List containing the lists of speaker-separated reference transcripts.

    Returns:
        cpWER (float):
            List containing cpWER values for each session
        min_perm_hyp_trans (list):
            List containing transcripts that lead to the minimum WER in string format
        ref_trans (list):
            List containing concatenated reference transcripts
    """
    if len(spk_hypotheses) != len(spk_references):
        raise ValueError(
            "In concatenated-minimum permutation word error rate calculation, "
            "hypotheses and reference lists must have the same number of elements. But got arguments:"
            f"{len(spk_hypotheses)} and {len(spk_references)} correspondingly"
        )
    cpWER_values, hyps_spk, refs_spk = [], [], []
    for spk_hypothesis, spk_reference in zip(spk_hypotheses, spk_references):
        cpWER, min_hypothesis, concat_reference = calculate_session_cpWER(spk_hypothesis, spk_reference)
        cpWER_values.append(cpWER)
        hyps_spk.append(min_hypothesis)
        refs_spk.append(concat_reference)
    return cpWER_values, hyps_spk, refs_spk


def split_text_by_speaker_tags(text: str, speaker_tag_pattern: re.Pattern = None) -> List[str]:
    """
    Split SOT-formatted text by speaker tags (<spk:0>, <spk:1>, ...) and
    return a list of per-speaker transcripts.

    Segments belonging to the same speaker are concatenated in order of
    appearance. Speakers are sorted by tag index so the list position is
    deterministic.

    If the text contains no speaker tags the entire string is returned as a
    single-element list.

    Example:
        >>> split_text_by_speaker_tags("<spk:0> hello world <spk:1> good morning <spk:0> how are you")
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

    Operates purely on **decoded text**: callers pass hypothesis/reference
    transcripts (already detokenized) and the metric splits them on ``<spk:0>``,
    ``<spk:1>``, ... speaker tags, then calls ``calculate_session_cpWER``
    (Hungarian-algorithm optimal assignment) for the session-level cpWER. It is
    decoding-agnostic; turning model outputs into text is the caller's job.

    Accumulates total edit-distance errors and total reference words so
    ``compute()`` returns a micro-averaged cpWER, summed across workers under DDP.
    The ``compute()`` return signature ``(rate, numerator, denominator)`` matches
    ``WER.compute()``.

    Args:
        log_prediction (bool): If True, log the first hyp/ref pair on each update.
        dist_sync_on_step (bool): torchmetrics DDP sync-on-step flag.
        sync_on_compute (bool): torchmetrics DDP sync-on-compute flag.
        speaker_tag_pattern (re.Pattern): Regex for speaker tags; defaults to ``<spk:N>``.
    """

    full_state_update: bool = True

    def __init__(
        self,
        log_prediction: bool = True,
        dist_sync_on_step: bool = False,
        sync_on_compute: bool = True,
        speaker_tag_pattern: re.Pattern = None,
        **kwargs,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step, sync_on_compute=sync_on_compute)

        self.log_prediction = log_prediction
        self.speaker_tag_pattern = (
            speaker_tag_pattern if speaker_tag_pattern is not None else DEFAULT_SPEAKER_TAG_PATTERN
        )

        self.add_state("total_edit_distance", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)
        self.add_state("total_ref_word_count", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)

    def _as_speaker_list(self, transcript: Union[str, List[str]]) -> List[str]:
        """Normalize one sample's transcript into a per-speaker text list.

        Args:
            transcript (str | List[str]): SOT-tagged text, or an already-split per-speaker list.

        Returns:
            List[str]: Per-speaker transcripts.
        """
        if isinstance(transcript, str):
            return split_text_by_speaker_tags(transcript, self.speaker_tag_pattern)
        return [t for t in transcript if t]

    def update(
        self,
        hypotheses: List[Union[str, List[str]]],
        references: List[Union[str, List[str]]],
        **kwargs,
    ):
        """Accumulate cpWER edit distance and reference word counts from decoded text.

        Args:
            hypotheses (List[str | List[str]]): Per-sample hypothesis text; each item is a
                SOT-tagged string (split on speaker tags) or a pre-split per-speaker list.
            references (List[str | List[str]]): Per-sample reference text, same format as hypotheses.
        """
        batch_edit_distance = 0
        batch_ref_word_count = 0

        for hypothesis, reference in zip(hypotheses, references):
            spk_hyp_transcripts = self._as_speaker_list(hypothesis)
            spk_ref_transcripts = self._as_speaker_list(reference)

            if not spk_ref_transcripts:
                continue

            ref_word_count = sum(len(transcript.split()) for transcript in spk_ref_transcripts)
            if ref_word_count == 0:
                continue

            cpwer_rate, _, _ = calculate_session_cpWER(spk_hyp_transcripts, spk_ref_transcripts)
            batch_edit_distance += round(cpwer_rate * ref_word_count)
            batch_ref_word_count += ref_word_count

        if self.log_prediction and hypotheses and references:
            logging.info("\n")
            logging.info(f"cpWER reference : {references[0]}")
            logging.info(f"cpWER predicted : {hypotheses[0]}")

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
