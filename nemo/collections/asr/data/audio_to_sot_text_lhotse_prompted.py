# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import os
import random
import re
from dataclasses import dataclass
from itertools import permutations
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.data
from lhotse import CutSet
from lhotse.cut import MixedCut
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_matrices, collate_vectors

from nemo.collections.common.data import apply_prompt_format_fn
from nemo.collections.common.prompts import PromptFormatter
from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.utils import logging


@dataclass
class PromptedAudioToTextMiniBatch:
    audio: torch.Tensor
    audio_lens: torch.Tensor
    transcript: torch.Tensor
    transcript_lens: torch.Tensor
    prompt: torch.Tensor
    prompt_lens: torch.Tensor
    prompted_transcript: torch.Tensor
    prompted_transcript_lens: torch.Tensor
    targets: Optional[torch.Tensor] = None
    target_length: Optional[torch.Tensor] = None
    cuts: Optional[CutSet] = None

    def get_decoder_inputs_outputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the inputs and outputs of transformer decoder for training.
        The input is ``prompted_transcript`` (minus last token),
        and the output is ``prompted_transcript`` (minus first token).
        """
        return self.prompted_transcript[:, :-1], self.prompted_transcript[:, 1:]


class PromptedAudioToTextLhotseDataset(torch.utils.data.Dataset):
    """
    This dataset is based on :class:`~nemo.collections.asr.data.audio_to_text_lhotse.LhotseSpeechToTextBpeDataset`.
    It is a Lhotse-style dataset that converts a mini-batch of Cuts into tensors.
    The main difference from ``LhotseSpeechToTextBpeDataset`` is that we introduce
    a special prompt format for multitask encoder-decoder models.

    To perform the prompt formatting, we accept a ``prompt_format_fn``.
    It's expected to accept:
    * a ``CutSet`` which it will internally iterate over for utterances, and
    * a ``TokenizerWrapper`` object that will be internally used to tokenize the utterances

    Tokenized utterances will be extended with special prompt tokens according to ``prompt_format_fn`` logic.
    We support cuts with multiple supervision segments -- their tokenized texts will be concatenated before we add the prompt tokens.
    This is useful, for example, in code-switched scenarios where each segment is spoken in a different language.

    When ``sot_cfg`` is provided, the dataset additionally computes frame-level
    speaker activity targets from RTTM annotations and aligns the RTTM speaker
    columns with the SOT ``[sN]`` speaker tokens in the transcript using
    DTW-based permutation search.

    Chunking:
    If `enable_chunking` is True, each audio sample is split into optimally sized chunks
    (see `find_optimal_chunk_size` and `chunk_waveform`). This is useful for long audio inputs,
    allowing the model to process them in manageable segments.

    NOTE:
    If the environment variable `USE_AIS_GET_BATCH` is set to `true` (case-insensitive),
    then batch audio loading from AIStore will be enabled for this dataset. This will use the
    AISBatchLoader to load the audio from AIStore. This can improve data loading efficiency in some setups.
    """

    def __init__(
        self,
        tokenizer: TokenizerSpec,
        prompt: PromptFormatter,
        enable_chunking: bool = False,
        sot_cfg: Optional[Dict] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.use_ais_get_batch = os.environ.get("USE_AIS_GET_BATCH", "False").lower() == "true"

        # Try to use use_batch_loader if available (Lhotse >= 1.32.0)
        try:
            self.load_audio = AudioSamples(fault_tolerant=True, use_batch_loader=self.use_ais_get_batch)
        except TypeError:
            # Lhotse < 1.32.0 doesn't support use_batch_loader
            if self.use_ais_get_batch:
                logging.warning(
                    "AIS batch loading requested but not supported by this Lhotse version. "
                    "Please upgrade to Lhotse >= 1.32.0"
                )
            self.load_audio = AudioSamples(fault_tolerant=True)

        self.padding_value = self.tokenizer.pad_id
        self.prompt = prompt
        self.enable_chunking = enable_chunking

        # SOT speaker-activity configuration
        self.sot_enabled = sot_cfg is not None
        if self.sot_enabled:
            self.num_speakers = sot_cfg.get('num_speakers', 4)
            self.randomize_single_speaker_index = sot_cfg.get('randomize_single_speaker_index', False)
            self.no_rttm_to_ones = sot_cfg.get('no_rttm_to_ones', True)
            self.num_sample_per_mel_frame = int(
                sot_cfg.get('window_stride', 0.01) * sot_cfg.get('sample_rate', 16000)
            )
            self.num_mel_frame_per_target_frame = int(sot_cfg.get('subsampling_factor', 8))
            # self.convert_to_wl = sot_cfg.get('convert_to_wl', False)

    # ── SOT text parsing utilities ─────────────────────────────────────

    @staticmethod
    def _sl_to_wl(text: str) -> str:
        """Convert segment-level (SL) SOT text to word-level (WL) SOT text.

        SL has a speaker token only at turn boundaries:
            ``[s0] hello how are you [s1] i am fine``

        WL repeats the active speaker token before every word:
            ``[s0] hello [s0] how [s0] are [s0] you [s1] i [s1] am [s1] fine``
        """
        parts = re.split(r'(\[s\d+\])', text)
        result = []
        current_token = None
        for part in parts:
            if re.fullmatch(r'\[s\d+\]', part):
                current_token = part
                continue
            words = part.split()
            if current_token is None:
                result.extend(words)
                continue
            for w in words:
                result.append(current_token)
                result.append(w)
        return ' '.join(result)

    @staticmethod
    def _parse_speaker_tokens(text: str) -> List[int]:
        """Extract the sequence of speaker indices from SOT text.

        Works with both SL and WL formats.
        Each word inherits the most recent speaker token (forward-fill).
        Returns one speaker index per word.
        """
        parts = re.split(r'(\[s\d+\])', text)
        spk_seq: List[int] = []
        current_spk = -1
        for part in parts:
            match = re.fullmatch(r'\[s(\d+)\]', part)
            if match:
                current_spk = int(match.group(1))
                continue
            if current_spk < 0:
                continue
            for _ in part.split():
                spk_seq.append(current_spk)
        return spk_seq

    @staticmethod
    def _get_text_speaker_char_counts(text: str, num_spk: int) -> np.ndarray:
        """Estimate speaking time per speaker from char counts of words.

        Returns ``(num_spk,)`` array of normalised char counts.
        """
        parts = re.split(r'(\[s\d+\])', text)
        char_counts = np.zeros(num_spk, dtype=np.float32)
        current_spk = -1
        for part in parts:
            match = re.fullmatch(r'\[s(\d+)\]', part)
            if match:
                current_spk = int(match.group(1))
                continue
            if current_spk < 0 or current_spk >= num_spk:
                continue
            for word in part.split():
                char_counts[current_spk] += len(word)
        total = char_counts.sum()
        if total > 0:
            char_counts /= total
        return char_counts

    # ── DTW / speaker-frequency cost functions ─────────────────────────

    @staticmethod
    def _dtw_cost_batch(
        activity: np.ndarray,
        spk_seq_arr: np.ndarray,
        perm_batch: np.ndarray,
        num_spk: int,
        token_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute DTW costs for a batch of permutations in one vectorized pass.

        Cost is one-hot based: each speaker token is a one-hot vector;
        cost at ``(k, t) = 1 - dot(text_one_hot, activity_permuted_at_t)``.
        """
        K = spk_seq_arr.shape[0]
        T = activity.shape[0]
        P = perm_batch.shape[0]
        if K == 0 or T == 0:
            return np.full(P, np.float32(np.inf))

        valid = spk_seq_arr < num_spk
        activity_permuted = activity[:, perm_batch].transpose(1, 0, 2)  # (P, T, num_spk)
        activity_sum = np.maximum(activity.sum(axis=1), 1.0).astype(np.float32)  # (T,)
        cols = np.where(valid, spk_seq_arr, 0)
        local = 1.0 - activity_permuted[:, :, cols].transpose(0, 2, 1) / activity_sum
        local[:, ~valid, :] = 1.0

        if token_weights is not None:
            local = local * token_weights[np.newaxis, :, np.newaxis]

        INF = np.float32(np.inf)
        prev_row = np.cumsum(local[:, 0, :], axis=1).astype(np.float32)

        for k in range(1, K):
            cur_row = np.full((P, T), INF, dtype=np.float32)
            cur_row[:, 0] = prev_row[:, 0] + local[:, k, 0]
            for t in range(1, T):
                cur_row[:, t] = (
                    np.minimum(
                        np.minimum(prev_row[:, t], prev_row[:, t - 1]),
                        cur_row[:, t - 1],
                    )
                    + local[:, k, t]
                )
            prev_row = cur_row

        return prev_row[:, T - 1] / (K + T)

    @staticmethod
    def _speaker_freq_cost_batch(
        text_freq: np.ndarray,
        rttm_freq: np.ndarray,
        perm_batch: np.ndarray,
    ) -> np.ndarray:
        """L1 mismatch between text and RTTM speaker frequency under each permutation."""
        rttm_freq_perm = rttm_freq[perm_batch]  # (P, num_spk)
        return np.abs(text_freq - rttm_freq_perm).sum(axis=1).astype(np.float32)

    @staticmethod
    def _dtw_cost(
        activity: np.ndarray,
        spk_seq_arr: np.ndarray,
        perm: List[int],
        num_spk: int,
        token_weights: Optional[np.ndarray] = None,
    ) -> float:
        """Compute DTW cost for a single permutation (delegates to batch)."""
        perm_batch = np.array([perm], dtype=np.intp)
        costs = PromptedAudioToTextLhotseDataset._dtw_cost_batch(
            activity, spk_seq_arr, perm_batch, num_spk, token_weights
        )
        return float(costs[0])

    # ── RTTM ↔ SOT speaker alignment ──────────────────────────────────

    def _fix_speaker_activity(self, cut, speaker_activity: torch.Tensor) -> torch.Tensor:
        """Align speaker_activity columns (from RTTM) with the SOT speaker
        token ordering in the transcript using DP-based DTW alignment.

        The SOT text contains a sequence of speaker tokens (one per word in WL
        mode, or one per turn in SL mode).  The RTTM-derived speaker_activity
        matrix ``(T, num_spk)`` has columns in an arbitrary order.  We need to
        find the column permutation that best aligns the text speaker sequence
        with the frame-level activity.

        Algorithm:
          1. Parse the SOT text into a word-level speaker token sequence of
             length K (forward-filling speaker labels for SL format).
          2. For each candidate permutation of the distinct speakers present in
             the text, run DTW on the ``K x T`` grid to find the minimum-cost
             monotonic alignment.
          3. Pick the permutation with the lowest DTW cost.
          4. Reorder columns accordingly and zero out columns for speakers
             absent from the text.

        Falls back to identity mapping when the number of distinct speakers
        exceeds ``num_speakers + 1`` (to keep permutation count tractable).
        """
        text = cut.text or ''
        if not text:
            return speaker_activity

        T, num_spk = speaker_activity.shape

        active_frames = speaker_activity.sum(dim=0)
        num_active = int((active_frames > 0).sum().item())
        num_active = min(num_active, num_spk)

        spk_seq = self._parse_speaker_tokens(text)
        if not spk_seq:
            return speaker_activity

        speakers_in_text = sorted(set(spk_seq))
        spk_seq_arr = np.array(spk_seq, dtype=np.intp)
        K = len(spk_seq_arr)

        activity_np = speaker_activity.detach().cpu().numpy().astype(np.float32)

        token_counts = np.bincount(spk_seq_arr, minlength=num_spk).astype(np.float32)
        token_counts = np.maximum(token_counts, 1.0)
        token_weights = (K / token_counts)[spk_seq_arr]

        text_freq = self._get_text_speaker_char_counts(text, num_spk)
        rttm_freq = activity_np.sum(axis=0).astype(np.float32)
        rttm_total = rttm_freq.sum()
        if rttm_total > 0:
            rttm_freq /= rttm_total

        identity_perm = list(range(num_spk))

        max_permutable = self.num_speakers + 1
        if num_active > 0 and num_active <= max_permutable:
            perm_active = np.array(list(permutations(range(num_active))), dtype=np.intp)
            perm_batch = np.zeros((perm_active.shape[0], num_spk), dtype=np.intp)
            perm_batch[:, :num_active] = perm_active
            perm_batch[:, num_active:] = np.arange(num_active, num_spk)

            dtw_costs = self._dtw_cost_batch(
                activity_np, spk_seq_arr, perm_batch, num_spk, token_weights
            )
            freq_costs = self._speaker_freq_cost_batch(text_freq, rttm_freq, perm_batch)
            total_costs = dtw_costs + freq_costs

            best_idx = int(np.argmin(total_costs))
            best_perm = perm_batch[best_idx].tolist()
        else:
            best_perm = identity_perm

        fixed = speaker_activity[:, best_perm].clone()

        speakers_set = set(speakers_in_text)
        cols_to_zero = [c for c in range(num_spk) if c not in speakers_set]
        if cols_to_zero:
            fixed[:, cols_to_zero] = 0.0

        if best_perm != identity_perm:
            identity_cost = float(
                self._dtw_cost(activity_np, spk_seq_arr, identity_perm, num_spk, token_weights)
                + self._speaker_freq_cost_batch(text_freq, rttm_freq, np.array([identity_perm], dtype=np.intp))[0]
            )
            best_cost = float(total_costs[best_idx])
            logging.info(
                "fix_speaker_activity [%s]: perm %s → %s | cost %.4f → %.4f (Δ=%.4f)",
                cut.id, identity_perm[:num_active], best_perm[:num_active],
                identity_cost, best_cost, identity_cost - best_cost,
            )

        return fixed

    def __getitem__(self, cuts: CutSet) -> PromptedAudioToTextMiniBatch:
        # Lazy import to avoid circular import at module load time
        if self.sot_enabled:
            from nemo.collections.asr.parts.utils.asr_multispeaker_utils import (
                get_hidden_length_from_sample_length,
                speaker_to_target,
            )

        # # ── SOT: optionally apply SL→WL conversion in-place before tokenization
        # if self.sot_enabled and self.convert_to_wl:
        #     for cut in cuts:
        #         for sup in cut.supervisions:
        #             text = sup.text or ""
        #             if text:
        #                 text = self._sl_to_wl(text)
        #                 sup.text = text
        #                 cut.text = text

        # ── SOT: compute speaker activity targets from RTTM before audio loading
        speaker_activities = []
        if self.sot_enabled:
            mono_cuts = []
            for cut in cuts:
                if cut.num_channels is not None and cut.num_channels > 1:
                    logging.warning(
                        "Multiple channels detected in cut '%s' (%d channels). "
                        "Only the first channel will be used; remaining channels are ignored.",
                        cut.id,
                        cut.num_channels,
                    )
                mono_cut = cut.with_channels(channels=[0])

                speaker_activity = speaker_to_target(
                    a_cut=mono_cut,
                    num_speakers=self.num_speakers,
                    num_sample_per_mel_frame=self.num_sample_per_mel_frame,
                    num_mel_frame_per_asr_frame=self.num_mel_frame_per_target_frame,
                    boundary_segments=True,
                    no_rttm_to_ones=self.no_rttm_to_ones,
                )
                
                # If there is no "[s*]" token in the text, that is single speaker training data.
                if not re.search(r'\[s\d+\]', mono_cut.text):
                    spk_idx = random.randint(0, self.num_speakers - 1) if self.randomize_single_speaker_index else 0
                    new_text = f"[s{spk_idx}] {cut.text}"
                    mono_cut.text = new_text
                    for sup in mono_cut.supervisions:
                        sup.text = new_text
                    for cut_sup in cut.supervisions:
                        cut_sup.text = new_text
                    if spk_idx != 0:
                        speaker_activity[:, [0, spk_idx]] = speaker_activity[:, [spk_idx, 0]]

                mono_cuts.append(mono_cut)

                speaker_activity = self._fix_speaker_activity(cut, speaker_activity)

                if speaker_activity.shape[1] > self.num_speakers:
                    logging.warning(
                        "Number of speakers in the target %s is greater than "
                        "the maximum number of speakers %s. Truncating extra speakers.",
                        speaker_activity.shape[1],
                        self.num_speakers,
                    )
                    speaker_activity = speaker_activity[:, :self.num_speakers]
                speaker_activities.append(speaker_activity)

            cuts = type(cuts).from_cuts(mono_cuts)

        # Load audio
        audio, audio_lens, cuts = self.load_audio(cuts)

        # Will work if batch_size is set to 1.
        if self.enable_chunking:
            new_audio = []
            new_audio_lens = []
            for i in range(audio.shape[0]):
                waveform = audio[i, : audio_lens[i]]
                chunks, chunk_lens = self._chunk_waveform(waveform)
                new_audio.extend(chunks)
                new_audio_lens.extend(chunk_lens)
            audio = torch.stack(new_audio)
            audio_lens = torch.tensor(new_audio_lens, dtype=torch.long)
            if cuts[0].start != 0:
                cuts[0].id = cuts[0].id + '_cut_segmented'

        # Fast-path: the tokenization and prompt formatting was already done before sampling.
        attrs = ("input_ids", "context_ids", "answer_ids")
        pre_formatted = all(hasattr(c, a) for c in cuts for a in attrs)
        if pre_formatted:
            prompts_with_answers, prompts, answers = zip(*((c.input_ids, c.context_ids, c.answer_ids) for c in cuts))
        else:
            formatted = [apply_prompt_format_fn(cut, self.prompt) for cut in cuts]
            prompts_with_answers = [ex["input_ids"] for ex in formatted]
            prompts = [ex["context_ids"] for ex in formatted]
            answers = [ex["answer_ids"] for ex in formatted]

        transcript, transcript_lens = self._collate_tokens(answers)
        prompts_with_answers, prompts_with_answers_lens = self._collate_tokens(prompts_with_answers)
        prompts, prompt_lens = self._collate_tokens(prompts)

        # ── SOT: collate speaker activity targets ──────────────────────
        targets = None
        target_length = None
        if self.sot_enabled and speaker_activities:
            targets = collate_matrices(speaker_activities).to(audio.dtype)  # (B, T, N)
            if targets.shape[2] > self.num_speakers:
                targets = targets[:, :, :self.num_speakers]
            elif targets.shape[2] < self.num_speakers:
                targets = torch.nn.functional.pad(
                    targets, (0, self.num_speakers - targets.shape[2]), mode='constant', value=0
                )
            target_length = torch.tensor(
                [
                    get_hidden_length_from_sample_length(
                        al, self.num_sample_per_mel_frame, self.num_mel_frame_per_target_frame
                    )
                    for al in audio_lens
                ]
            )

        return PromptedAudioToTextMiniBatch(
            audio=audio,
            audio_lens=audio_lens,
            transcript=transcript,
            transcript_lens=transcript_lens,
            prompt=prompts,
            prompt_lens=prompt_lens,
            prompted_transcript=prompts_with_answers,
            prompted_transcript_lens=prompts_with_answers_lens,
            targets=targets,
            target_length=target_length,
            cuts=_drop_in_memory_data(cuts),
        )

    def _collate_tokens(self, tokens: list[Union[list[int], torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = [torch.as_tensor(t) for t in tokens]
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=self.padding_value)
        return tokens, token_lens

    def _find_optimal_chunk_size(
        self, total_len: int, min_sec: int = 30, max_sec: int = 40, sample_rate: int = 16000, overlap_sec: float = 1.0
    ) -> int:
        """
        Find the optimal chunk size for audio processing that minimizes paddings to the last chunk.

        Args:
            total_len (int): Total length of the audio waveform in samples
            min_sec (int, optional): Minimum chunk size in seconds. Defaults to 30.
            max_sec (int, optional): Maximum chunk size in seconds. Defaults to 40.
            sample_rate (int, optional): Audio sample rate in Hz. Defaults to 16000.
            overlap_sec (float, optional): Overlap duration between consecutive chunks in seconds.
                                         Defaults to 1.0.

        Returns:
            int: Optimal chunk size in samples that maximizes the last chunk length
        """
        best_chunk_size = min_sec * sample_rate
        best_last_chunk_len = 0
        if total_len < max_sec * sample_rate:
            return total_len
        # Try each possible chunk duration in the range
        for sec in range(min_sec, max_sec + 1):
            chunk_size = sec * sample_rate
            overlap_size = int(overlap_sec * sample_rate)
            step_size = chunk_size - overlap_size

            if step_size <= 0:  # Invalid overlap
                continue
            if chunk_size > total_len:
                continue

            # Calculate how many chunks we'd need and the last chunk's length
            n_chunks = (total_len + step_size - 1) // step_size
            last_chunk_len = total_len - step_size * (n_chunks - 1)

            if last_chunk_len > best_last_chunk_len:
                best_last_chunk_len = last_chunk_len
                best_chunk_size = chunk_size

        return best_chunk_size

    def _chunk_waveform(
        self, waveform: torch.Tensor, chunk_size: int = None, overlap_sec: float = 1.0, sample_rate: int = 16000
    ) -> tuple[list[torch.Tensor], list[int]]:
        """
        Split a waveform tensor into overlapping chunks.

        Args:
            waveform (torch.Tensor): Input audio waveform tensor of shape (time_samples,)
            chunk_size (int, optional): Size of each chunk in samples. If None, automatically
                                       determines optimal chunk size using find_optimal_chunk_size().
                                       Defaults to None.
            sample_rate (int, optional): Audio sample rate in Hz. Defaults to 16000.
            overlap_sec (float, optional): Overlap duration between consecutive chunks in seconds.
                                          Used to calculate step size. Defaults to 2.

        Returns:
            tuple[list[torch.Tensor], list[int]]: A tuple containing:
                - List of chunk tensors, each of shape (chunk_size,)
                - List of original lengths for each chunk before padding (useful for masking
                  padded regions during processing.
        """
        # If chunk_size is None, find the optimal chunk size for this waveform
        total_len = waveform.shape[0]
        if chunk_size is None:
            chunk_size = self._find_optimal_chunk_size(total_len, overlap_sec=overlap_sec)
        if chunk_size >= total_len:
            return [waveform], [total_len]
        overlap_size = int(overlap_sec * sample_rate)
        step_size = chunk_size - overlap_size
        chunks = []
        chunk_lens = []
        start = 0
        while start + overlap_size < total_len:
            end = min(start + chunk_size, total_len)
            chunk = waveform[start:end]
            length = chunk.shape[0]
            if length < chunk_size:
                pad = torch.zeros(chunk_size - length, dtype=chunk.dtype, device=chunk.device)
                chunk = torch.cat([chunk, pad], dim=0)
            chunks.append(chunk)
            chunk_lens.append(length)
            start += step_size

        return chunks, chunk_lens


class ProbablyIncorrectLanguageKeyError(RuntimeError):
    pass


def _drop_in_memory_data(
    cuts: CutSet,
    _fields=frozenset(MixedCut.__dataclass_fields__.keys()),
) -> CutSet:
    """Workaround for an edge case in cuts.drop_in_memory_data() on MixedCut with Lhotse<1.29.0"""
    ans = []
    for c in cuts:
        # Not a mixed cut or a mixed cut that wasn't assigned any extra attributes.
        if not isinstance(c, MixedCut) or _fields.issuperset(c.__dict__.keys()):
            ans.append(c.drop_in_memory_data())
        else:
            extra_attrs = {k: v for k, v in c.__dict__.items() if k not in _fields}
            for k in extra_attrs:
                delattr(c, k)
            ans.append(c.drop_in_memory_data())
            for k, v in extra_attrs.items():
                setattr(ans[-1], k, v)
                setattr(c, k, v)
    return CutSet(ans)
