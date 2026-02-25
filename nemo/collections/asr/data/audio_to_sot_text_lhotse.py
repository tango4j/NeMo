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
    def _get_text_speaker_char_counts(text: str, num_spk: int) -> np.ndarray:
        """Estimate speaking time per speaker from char counts of words.

        Returns (num_spk,) array of char counts per text speaker, normalised.
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

    @staticmethod
    def _dtw_cost_batch(
        activity: np.ndarray,
        spk_seq_arr: np.ndarray,
        perm_batch: np.ndarray,
        num_spk: int,
        token_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute DTW costs for a batch of permutations in one vectorized pass.

        Cost is one-hot based: each speaker token is a one-hot vector [0,0,1,0];
        cost at (k,t) = 1 - dot(text_one_hot, activity_permuted_at_t).
        Optional token_weights give less frequent speakers equal importance.

        Args:
            activity: ``(T, num_spk)`` RTTM speaker activity (0/1 per frame).
            spk_seq_arr: ``(K,)`` int array of speaker indices per word.
            perm_batch: ``(P, num_spk)`` array of permutations; each row maps
                text_spk → RTTM column.
            num_spk: number of speaker columns.
            token_weights: ``(K,)`` optional; weight[k] = K / count(spk_seq[k]).

        Returns:
            ``(P,)`` array of total DTW path costs, each normalised by ``K + T``.
        """
        K = spk_seq_arr.shape[0]
        T = activity.shape[0]
        P = perm_batch.shape[0]
        if K == 0 or T == 0:
            return np.full(P, np.float32(np.inf))

        # One-hot based cost: text token k has speaker s = spk_seq[k] (one-hot at s).
        # Activity at t in text space: activity[t, perm[i]] for each text speaker i.
        # Cost = 1 - activity[t, perm[s]] / total_activity[t]; normalized so overlap
        # (multiple speakers active) yields higher cost than single-speaker match.
        valid = spk_seq_arr < num_spk
        # activity_permuted[p, t, i] = activity[t, perm_batch[p, i]]
        activity_permuted = activity[:, perm_batch].transpose(1, 0, 2)  # (P, T, num_spk)
        activity_sum = np.maximum(activity.sum(axis=1), 1.0).astype(np.float32)  # (T,)
        # local[p, k, t] = 1 - activity_permuted[p, t, spk_seq[k]] / activity_sum[t]
        cols = np.where(valid, spk_seq_arr, 0)  # placeholder for invalid
        local = 1.0 - activity_permuted[:, :, cols].transpose(0, 2, 1) / activity_sum
        local[:, ~valid, :] = 1.0  # invalid speakers: max cost

        # Speaker token count normalization: less frequent speakers get higher weight
        if token_weights is not None:
            local = local * token_weights[np.newaxis, :, np.newaxis]

        INF = np.float32(np.inf)
        # First row: cumsum along T
        prev_row = np.cumsum(local[:, 0, :], axis=1).astype(np.float32)

        # DP: O(K * T) iterations, vectorized over P
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
        """L1 mismatch between text and RTTM speaker frequency under each permutation.

        text_freq[i], rttm_freq[j]: normalised speaking-time ratios.
        Under perm p: text speaker i maps to RTTM column perm[p,i].
        Cost = sum_i |text_freq[i] - rttm_freq[perm[p,i]]|.

        Returns:
            ``(P,)`` array of L1 costs per permutation.
        """
        # rttm_freq_perm[p, i] = rttm_freq[perm_batch[p, i]]
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
        costs = LhotseSpeechToTextBpeDataset._dtw_cost_batch(
            activity, spk_seq_arr, perm_batch, num_spk, token_weights
        )
        return float(costs[0])

    def _fix_speaker_activity_hungarian(
        self, cut, speaker_activity: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[List[int]]]:
        """Align speaker_activity columns with SOT speaker token ordering
        using speaking-time ratio matching and the Hungarian algorithm.

        Instead of building a pseudo frame-level activity matrix, we compare
        compact speaking-time ratio vectors from both the text and the RTTM
        activity, then run ``linear_sum_assignment`` on the L1-distance cost
        matrix to find the best column permutation.

        We return the top-N permutation candidates (N = number of distinct
        speakers in the text), ranked from best (lowest cost) to worst.

        Args:
            cut: A Lhotse Cut whose ``.text`` contains SOT transcript with
                ``[sN]`` speaker tokens.
            speaker_activity: ``(T, num_spk)`` binary matrix from RTTM.

        Returns:
            fixed: ``(T, num_spk)`` matrix with columns reordered by the best
                permutation, absent-speaker columns zeroed.
            perm_candidates: List of N permutations (each a list of length
                ``num_spk``), sorted best-to-worst by cost.  ``perm[i]``
                gives the RTTM column that maps to text speaker ``i``.
        """
        text = cut.text or ''
        if not text:
            return speaker_activity, [list(range(speaker_activity.shape[1]))]

        T, num_spk = speaker_activity.shape
        identity_perm = list(range(num_spk))

        # ── 1. Parse SOT text to get per-word speaker sequence ────────────
        # Uses [s0], [s1], ... convention via _parse_speaker_tokens.
        spk_seq = self._parse_speaker_tokens(text)
        if not spk_seq:
            return speaker_activity, [identity_perm]

        speakers_in_text = sorted(set(spk_seq))
        num_distinct = len(speakers_in_text)

        # ── 2. Speaking-time ratio from text (token count ratio) ──────────
        # Count how many word tokens each speaker has, then normalise.
        spk_arr = np.array(spk_seq, dtype=np.intp)
        token_counts = np.bincount(spk_arr[spk_arr < num_spk], minlength=num_spk).astype(np.float64)
        total_tokens = token_counts.sum()
        if total_tokens == 0:
            return speaker_activity, [identity_perm]
        text_ratio = torch.from_numpy(
            (token_counts / total_tokens).astype(np.float32)
        ).to(dtype=speaker_activity.dtype, device=speaker_activity.device)

        # ── 3. Speaking-time ratio from RTTM activity ─────────────────────
        # Sum active frames per column, then normalise.
        active_frames = speaker_activity.sum(dim=0)  # (num_spk,)
        total_active = active_frames.sum()
        if total_active == 0:
            return speaker_activity, [identity_perm]
        rttm_ratio = active_frames / total_active

        # ── 4. Cost matrix: L1 distance between text_ratio[row] and rttm_ratio[col]
        cost = torch.abs(text_ratio.unsqueeze(1) - rttm_ratio.unsqueeze(0))

        # ── 5. Hungarian algorithm for best permutation ───────────────────
        row_ind, col_ind = linear_sum_assignment(cost)
        best_perm = identity_perm[:]
        for row, col in zip(row_ind.tolist(), col_ind.tolist()):
            best_perm[row] = col
        best_cost = cost[range(num_spk), best_perm].sum().item()

        # ── 6. Collect top-N permutation candidates from cost matrix ──────
        # Re-run linear_sum_assignment with each assignment pair masked
        # (cost set to large value) to discover next-best permutations.
        candidates: List[Tuple[float, List[int]]] = [(best_cost, best_perm)]
        seen_perms = {tuple(best_perm)}

        for mask_row, mask_col in zip(row_ind.tolist(), col_ind.tolist()):
            masked_cost = cost.clone()
            masked_cost[mask_row, mask_col] = 1e9
            alt_row_ind, alt_col_ind = linear_sum_assignment(masked_cost)
            alt_perm = identity_perm[:]
            for row, col in zip(alt_row_ind.tolist(), alt_col_ind.tolist()):
                alt_perm[row] = col
            perm_key = tuple(alt_perm)
            if perm_key not in seen_perms:
                seen_perms.add(perm_key)
                alt_cost = cost[range(num_spk), alt_perm].sum().item()
                candidates.append((alt_cost, alt_perm))

        candidates.sort(key=lambda x: x[0])
        perm_candidates = [perm for _, perm in candidates[:num_distinct]]

        # ── 7. Apply best permutation ─────────────────────────────────────
        fixed = speaker_activity[:, best_perm].clone()

        # ── 8. Zero out columns for speakers absent from the text ─────────
        speakers_set = set(speakers_in_text)
        cols_to_zero = [c for c in range(num_spk) if c not in speakers_set]
        if cols_to_zero:
            fixed[:, cols_to_zero] = 0.0

        return fixed, perm_candidates

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

        # Count non-zero speaker columns to limit permutation space
        active_frames = speaker_activity.sum(dim=0)
        num_active = int((active_frames > 0).sum().item())
        num_active = min(num_active, num_spk)

        spk_seq = self._parse_speaker_tokens(text)
        if not spk_seq:
            return speaker_activity

        speakers_in_text = sorted(set(spk_seq))
        spk_seq_arr = np.array(spk_seq, dtype=np.intp)
        K = len(spk_seq_arr)

        activity_np = speaker_activity.detach().cpu().numpy().astype(np.float32)  # (T, num_spk)

        # (1) Speaker token count weight: less frequent speakers get equal importance
        token_counts = np.bincount(spk_seq_arr, minlength=num_spk).astype(np.float32)
        token_counts = np.maximum(token_counts, 1.0)  # avoid div by zero
        token_weights = (K / token_counts)[spk_seq_arr]  # (K,)

        # (2) Speaker frequency: text (char-based) and RTTM (frame-based), both normalised
        text_freq = self._get_text_speaker_char_counts(text, num_spk)  # (num_spk,)
        rttm_freq = activity_np.sum(axis=0).astype(np.float32)
        rttm_total = rttm_freq.sum()
        if rttm_total > 0:
            rttm_freq /= rttm_total

        identity_perm = list(range(num_spk))

        # Only permute active columns; inactive columns stay identity-mapped
        max_permutable = self.num_speakers + 1
        if num_active > 0 and num_active <= max_permutable:
            perm_active = np.array(list(permutations(range(num_active))), dtype=np.intp)
            # Extend to full (num_spk,) perms: active cols permuted, rest identity
            perm_batch = np.zeros((perm_active.shape[0], num_spk), dtype=np.intp)
            perm_batch[:, :num_active] = perm_active
            perm_batch[:, num_active:] = np.arange(num_active, num_spk)

            # (3) Combined cost: DTW (with token weights) + speaker frequency mismatch
            dtw_costs = self._dtw_cost_batch(
                activity_np, spk_seq_arr, perm_batch, num_spk, token_weights
            )
            freq_costs = self._speaker_freq_cost_batch(text_freq, rttm_freq, perm_batch)
            total_costs = dtw_costs + freq_costs

            best_idx = int(np.argmin(total_costs))
            best_perm = perm_batch[best_idx].tolist()
            best_cost = float(total_costs[best_idx])
        else:
            best_perm = identity_perm
            best_cost = float('inf')

        # ── Reorder columns using best permutation ─────────────────────
        fixed = speaker_activity[:, best_perm].clone()

        # ── Zero out columns for speakers absent from the text ─────────
        speakers_set = set(speakers_in_text)
        cols_to_zero = [c for c in range(num_spk) if c not in speakers_set]
        if cols_to_zero:
            fixed[:, cols_to_zero] = 0.0

        if False and best_perm != identity_perm:
            identity_dtw = self._dtw_cost(
                activity_np, spk_seq_arr, identity_perm, num_spk, token_weights
            )
            identity_freq = self._speaker_freq_cost_batch(
                text_freq, rttm_freq, np.array([identity_perm], dtype=np.intp)
            )[0]
            identity_cost = float(identity_dtw) + float(identity_freq)
            logging.info("SPK_FIX %s | cost %.4f→%.4f | perm %s→%s", cut.id.split('/')[-1], identity_cost, best_cost, identity_perm, best_perm)
            # if "fe" in cut.id:
            #     import os; _id = cut.id.split('/')[-1].replace(' ', '_'); _dir = os.path.expanduser(f'~/projects/sot_mt_asr_rt/SOT_DP_fix_example/{_id}'); os.makedirs(_dir, exist_ok=True)
            #     np.save(f'{_dir}/activity_np.npy', activity_np)
            #     np.save(f'{_dir}/spk_seq_arr.npy', spk_seq_arr)
            #     np.save(f'{_dir}/token_weights.npy', token_weights)
            #     np.save(f'{_dir}/text_freq.npy', text_freq)
            #     np.save(f'{_dir}/rttm_freq.npy', rttm_freq)
            #     np.save(f'{_dir}/perm_batch.npy', perm_batch)
            #     np.save(f'{_dir}/dtw_costs.npy', dtw_costs)
            #     np.save(f'{_dir}/freq_costs.npy', freq_costs)
            #     np.save(f'{_dir}/total_costs.npy', total_costs)
            #     np.save(f'{_dir}/best_perm.npy', np.array(best_perm))
            #     np.save(f'{_dir}/identity_perm.npy', np.array(identity_perm))
            #     np.save(f'{_dir}/fixed.npy', fixed.detach().cpu().numpy())
            #     np.save(f'{_dir}/speaker_activity_orig.npy', speaker_activity.detach().cpu().numpy())
            #     with open(f'{_dir}/text.txt', 'w') as _f: _f.write(text)
            #     with open(f'{_dir}/meta.txt', 'w') as _f: _f.write(f'cut_id: {cut.id}\nT: {T}\nK: {K}\nnum_spk: {num_spk}\nnum_active: {num_active}\nbest_perm: {best_perm}\nidentity_perm: {identity_perm}\nbest_cost: {best_cost}\nidentity_cost: {identity_cost}\nspeakers_in_text: {speakers_in_text}\n')
            #     print(f'Saved to {_dir}/')
            #     import ipdb; ipdb.set_trace()

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
