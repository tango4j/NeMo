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

from typing import Dict, Optional, Tuple

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

    def __getitem__(self, cuts) -> Tuple[torch.Tensor, ...]:
        # Convert multi-channel cuts to mono and compute speaker activity targets
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

        # ── Tokenize SOT transcripts ──────────────────────────────────
        tokens = [
            torch.cat(
                [
                    torch.as_tensor(
                        s.tokens if hasattr(s, "tokens") else self.tokenizer(s.text or "", s.language)
                    )
                    for s in c.supervisions
                ],
                dim=0,
            )
            for c in cuts
        ]
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=0)

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
