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

import re
from itertools import permutations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_matrices, collate_vectors

from nemo.collections.asr.parts.utils.asr_multispeaker_utils import (
    get_hidden_length_from_sample_length,
    speaker_to_target,
)
from nemo.collections.common.tokenizers.aggregate_tokenizer import TokenizerWrapper
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging
from nemo.collections.asr.parts.utils.optimization_utils import linear_sum_assignment

class LhotseSpeechToTextBpeDataset(torch.utils.data.Dataset):
    """
    Lhotse dataset for SOT multi-talker ASR training with ground-truth
    diarization supervision.

    Returns BPE-tokenized SOT transcripts **and** frame-level speaker
    activity targets derived from RTTM annotations in the CutSet.
    Using ground-truth speaker targets (instead of running the diar model
    on short training segments) eliminates the noise from inaccurate
    diarization predictions and lets the ASR model learn to trust the
    speaker signal without confusion.

    Unlike native NeMo datasets, Lhotse dataset defines only the mapping from
    a CutSet (meta-data) to a mini-batch with PyTorch tensors.
    Specifically, it performs tokenization, I/O, augmentation, and feature
    extraction (if any).  Managing data, sampling, de-duplication across
    workers/nodes etc. is all handled by Lhotse samplers instead.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'targets': NeuralType(('B', 'T', 'N'), LabelsType()),
            'target_length': NeuralType(tuple('B'), LengthsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(self, tokenizer: TokenizerSpec, cfg, return_cuts: bool = False):
        super().__init__()
        self.tokenizer = TokenizerWrapper(tokenizer)
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.return_cuts = return_cuts
        self.cfg = cfg
        self.num_speakers = self.cfg.get('num_speakers', 4)
        self.num_sample_per_mel_frame = int(
            self.cfg.get('window_stride', 0.01) * self.cfg.get('sample_rate', 16000)
        )
        self.num_mel_frame_per_target_frame = int(self.cfg.get('subsampling_factor', 8))
        self.convert_to_wl = self.cfg.get('convert_to_wl', False)

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

        Works with both SL and WL formats:
          SL: ``[s0] hi there [s1] bye``  →  ``[0, 0, 0, 1, 1]``  (one per word)
          WL: ``[s0] hi [s0] there [s1] bye``  →  ``[0, 0, 1]``   (one per word)

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
    def _dtw_cost(
        cost_matrix: np.ndarray,
        spk_seq_arr: np.ndarray,
        perm: List[int],
        num_spk: int,
    ) -> float:
        """Compute the minimum-cost DTW alignment on a ``K × T`` grid.

        Args:
            cost_matrix: ``(num_spk, T)`` precomputed mismatch costs where
                ``cost_matrix[col, t] = 1 - activity[t, col]``.
            spk_seq_arr: ``(K,)`` int array of speaker indices per word.
            perm: column permutation — ``perm[text_spk]`` → RTTM column.
            num_spk: number of speaker columns.

        Returns:
            Total DTW path cost normalised by ``K + T``.
        """
        K = spk_seq_arr.shape[0]
        T = cost_matrix.shape[1]
        if K == 0 or T == 0:
            return float('inf')

        # Map each word to its RTTM column under this permutation
        mapped = np.array([perm[s] if s < num_spk else -1 for s in spk_seq_arr], dtype=np.intp)

        # Build the K × T local-cost matrix by indexing into precomputed costs
        # For invalid mappings (col == -1), cost is 1.0 everywhere.
        local = np.ones((K, T), dtype=np.float32)
        valid = mapped >= 0
        local[valid] = cost_matrix[mapped[valid]]

        INF = np.float32(np.inf)
        prev_row = np.full(T, INF, dtype=np.float32)
        prev_row[0] = local[0, 0]
        np.cumsum(local[0], out=prev_row)

        for k in range(1, K):
            cur_row = np.full(T, INF, dtype=np.float32)
            cur_row[0] = prev_row[0] + local[k, 0]
            lk = local[k]
            for t in range(1, T):
                cur_row[t] = min(prev_row[t], prev_row[t - 1], cur_row[t - 1]) + lk[t]
            prev_row = cur_row

        return float(prev_row[T - 1]) / (K + T)

    def _fix_speaker_activity_hungarian(self, cut, speaker_activity: torch.Tensor) -> torch.Tensor:
        text = cut.text or ''
        if not text:
            return speaker_activity

        T, num_spk = speaker_activity.shape

        # 1. Parse SOT text into (speaker_idx, word_text) segments
        parts = re.split(r'<\|spltoken(\d+)\|>', text)
        segments = []
        for i in range(1, len(parts), 2):
            spk_idx = int(parts[i])
            word_text = parts[i + 1].strip() if i + 1 < len(parts) else ''
            if word_text:
                segments.append((spk_idx, word_text))

        if not segments:
            return speaker_activity

        total_chars = sum(len(w) for _, w in segments)
        if total_chars == 0:
            return speaker_activity

        speakers_in_text = set(spk for spk, _ in segments)

        # 2. Build pseudo speaker-activity from text (proportional to char length)
        pseudo = torch.zeros(T, num_spk, dtype=speaker_activity.dtype,
                            device=speaker_activity.device)
        cum = 0
        for spk_idx, word_text in segments:
            start_frame = int(cum / total_chars * T)
            cum += len(word_text)
            end_frame = min(int(cum / total_chars * T), T)
            if spk_idx < num_spk:
                pseudo[start_frame:end_frame, spk_idx] = 1.0

        # 3. Cost matrix: negative dot-product (maximize overlap)
        cost = torch.zeros(num_spk, num_spk, dtype=speaker_activity.dtype,
                        device=speaker_activity.device)
        for i in range(num_spk):
            for j in range(num_spk):
                cost[i, j] = -torch.dot(pseudo[:, i], speaker_activity[:, j])

        # 4. Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost)

        # 5. Reorder columns
        fixed = torch.zeros_like(speaker_activity)
        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            fixed[:, r] = speaker_activity[:, c]

        # 6. Zero out columns for speakers absent from the text
        for col in range(num_spk):
            if col not in speakers_in_text:
                fixed[:, col] = 0.0

        return fixed

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
             the text, run DTW on the ``K × T`` grid to find the minimum-cost
             monotonic alignment.  The cell cost is
             ``1 - activity[t, perm[spk]]`` (mismatch with ground truth).
          3. Pick the permutation with the lowest DTW cost.
          4. Reorder columns accordingly and zero out columns for speakers
             absent from the text.

        Falls back to identity mapping when the number of distinct speakers
        exceeds ``num_speakers + 1`` (to keep permutation count tractable).

        Args:
            cut: A Lhotse Cut whose ``.text`` contains the SOT transcript
                with ``[sN]`` speaker tokens.
            speaker_activity: ``(T, num_spk)`` binary matrix from RTTM.

        Returns:
            ``(T, num_spk)`` matrix with columns reordered to match SOT
            speaker indices, absent-speaker columns zeroed.
        """
        text = cut.text or ''
        if not text:
            return speaker_activity

        T, num_spk = speaker_activity.shape

        spk_seq = self._parse_speaker_tokens(text)
        if not spk_seq:
            return speaker_activity

        speakers_in_text = sorted(set(spk_seq))
        num_distinct = len(speakers_in_text)
        spk_seq_arr = np.array(spk_seq, dtype=np.intp)

        # Precompute mismatch cost: cost_matrix[col, t] = 1 - activity[t, col]
        activity_np = speaker_activity.detach().cpu().numpy().astype(np.float32)
        cost_matrix = 1.0 - activity_np.T  # (num_spk, T)

        identity_perm = list(range(num_spk))

        max_permutable = self.num_speakers + 1
        if num_spk <= max_permutable:
            best_cost = float('inf')
            best_perm = identity_perm

            for full_perm in permutations(range(num_spk)):
                perm = list(full_perm)
                cost = self._dtw_cost(cost_matrix, spk_seq_arr, perm, num_spk)
                if cost < best_cost:
                    best_cost = cost
                    best_perm = perm
        else:
            best_perm = identity_perm

        # ── Reorder columns using best permutation ─────────────────────
        fixed = torch.zeros_like(speaker_activity)
        for text_col, rttm_col in enumerate(best_perm):
            if text_col < num_spk and rttm_col < num_spk:
                fixed[:, text_col] = speaker_activity[:, rttm_col]

        # ── Zero out columns for speakers absent from the text ─────────
        speakers_set = set(speakers_in_text)
        for col in range(num_spk):
            if col not in speakers_set:
                fixed[:, col] = 0.0

        if best_perm != identity_perm:
            identity_cost = self._dtw_cost(cost_matrix, spk_seq_arr, identity_perm, num_spk)
            logging.info("SPK_FIX %s | cost %.4f→%.4f | perm %s→%s", cut.id.split('/')[-1], identity_cost, best_cost, identity_perm, best_perm)
        return fixed

    def _tokenize_cuts(self, cuts) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize SOT transcripts from cuts, optionally converting SL→WL.

        When ``self.convert_to_wl`` is True, the supervision text on each cut
        is mutated **in-place** to the WL form so that downstream code
        (e.g. ``_fix_speaker_activity``) sees the converted text via
        ``cut.text``.
        """
        all_tokens = []
        for cut in cuts:
            sup_tokens = []
            for sup in cut.supervisions:
                text = sup.text or ""
                if self.convert_to_wl and text:
                    text = self._sl_to_wl(text)
                    sup.text = text
                    cut.text = text
                    cut.supervisions[0].text = text
                sup_tokens.append(torch.as_tensor(self.tokenizer(text, sup.language)))
            all_tokens.append(torch.cat(sup_tokens, dim=0))
        token_lens = torch.tensor([token.size(0) for token in all_tokens], dtype=torch.long)
        tokens = collate_vectors(all_tokens, padding_value=0)
        return tokens, token_lens

    def __getitem__(self, cuts) -> Tuple[torch.Tensor, ...]:
        # ── Tokenize SOT transcripts (before any cuts mutation) ────────
        # Also applies SL→WL conversion in-place on sup.text so that
        # cut.text reflects WL for _fix_speaker_activity and WER logging.
        tokens, token_lens = self._tokenize_cuts(cuts)

        # ── Convert multi-channel cuts to mono and compute speaker activity targets
        mono_cuts = []
        speaker_activities = []
        for cut in cuts:
            if cut.num_channels is not None and cut.num_channels > 1:
                logging.warning(
                    "Multiple channels detected in cut '%s' (%d channels). "
                    "Only the first channel will be used; remaining channels are ignored.",
                    cut.id,
                    cut.num_channels,
                )
            mono_cut = cut.with_channels(channels=[0])
            mono_cuts.append(mono_cut)

            speaker_activity = speaker_to_target(
                a_cut=mono_cut,
                num_speakers=self.num_speakers,
                num_sample_per_mel_frame=self.num_sample_per_mel_frame,
                num_mel_frame_per_asr_frame=self.num_mel_frame_per_target_frame,
                boundary_segments=True,
            )
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
        audio, audio_lens, cuts = self.load_audio(cuts)

        # ── Collate speaker activity targets from RTTM ────────────────
        targets = collate_matrices(speaker_activities).to(audio.dtype)  # (B, T, N)
        if targets.shape[2] > self.num_speakers:
            targets = targets[:, :, :self.num_speakers]
        elif targets.shape[2] < self.num_speakers:
            targets = torch.nn.functional.pad(
                targets, (0, self.num_speakers - targets.shape[2]), mode='constant', value=0
            )

        target_lens = torch.tensor(
            [
                get_hidden_length_from_sample_length(
                    al, self.num_sample_per_mel_frame, self.num_mel_frame_per_target_frame
                )
                for al in audio_lens
            ]
        )

        if self.return_cuts:
            return audio, audio_lens, tokens, token_lens, targets, target_lens, cuts.drop_in_memory_data()
        
        return audio, audio_lens, tokens, token_lens, targets, target_lens
