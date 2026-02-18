# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

"""
Cache-aware streaming inference script for SOT (Serialized Output Training)
multi-talker ASR models (MSEncDecRNNTBPEModel).

This script simulates cache-aware streaming for ASR models. The ASR model to be
used with this script needs to be trained in streaming mode. Currently only
Conformer-based models support this streaming mode.

It works both on a manifest of audio files or a single audio file. It can perform
streaming for a single stream (audio) or perform the evaluation in multi-stream
mode (batch_size > 1).

The manifest file must conform to standard ASR definition - containing
`audio_filepath` and `text` as the ground truth.

# Usage

## To evaluate a model in cache-aware streaming mode on a single audio file:

python speech_to_text_cache_aware_streaming_sot_mtasr_infer.py \
    asr_model=asr_model.nemo \
    audio_file=audio_file.wav \
    use_amp=true \
    debug_mode=true

## To evaluate a model in cache-aware streaming mode on a manifest file:

python speech_to_text_cache_aware_streaming_sot_mtasr_infer.py \
    asr_model=asr_model.nemo \
    manifest_file=manifest_file.json \
    batch_size=16 \
    use_amp=true \
    debug_mode=true

## Multi-lookahead models

For models which support multiple lookaheads, the default is the first one in the
list of model.encoder.att_context_size. To change it, you may use att_context_size,
for example att_context_size=[70,1].

## Evaluate a model trained with full context for offline mode

You may try the cache-aware streaming with a model trained with full context in
offline mode, but the accuracy would not be very good with small chunks.
To use a model trained with full context, you need to pass the chunk_size and
shift_size arguments.

python speech_to_text_cache_aware_streaming_sot_mtasr_infer.py \
    pretrained_name=stt_en_conformer_ctc_large \
    chunk_size=100 \
    shift_size=50 \
    left_chunks=2 \
    online_normalization=true \
    manifest_file=manifest_file.json \
    batch_size=16 \
    debug_mode=true

"""


import glob
import json
import os
import re
import time
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import List, Optional, Union

import librosa
import lightning.pytorch as pl
import soundfile as sf
import torch
from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models.rnnt_bpe_models import MSEncDecRNNTBPEModel
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from nemo.collections.asr.parts.utils.transcribe_utils import get_inference_device, get_inference_dtype, setup_model
from nemo.core.config import hydra_runner
from nemo.utils import logging


@dataclass
class TranscriptionConfig:
    """
    Configuration for cache-aware streaming inference with SOT multi-talker ASR.

    Variable naming follows the convention in MultitalkerTranscriptionConfig
    (tutorials/asr/speech_to_text_multitalker_streaming_infer.py) so the two
    scripts can be merged in the future.
    """

    # ── ASR model configs ──────────────────────────────────────────────
    asr_model: Optional[str] = None  # Path to a .nemo file
    pretrained_name: Optional[str] = None  # Name of a pretrained model (used by setup_model fallback)

    # ── Diarization model configs ──────────────────────────────────────
    diar_model: Optional[str] = None  # Path to a diarization .nemo file (null to disable)
    diar_pretrained_name: Optional[str] = None  # Name of a pretrained diarization model
    max_num_of_spks: Optional[int] = 4  # Maximum number of speakers
    parallel_speaker_strategy: bool = True  # Whether to use parallel speaker strategy
    masked_asr: bool = True  # Whether to use masked ASR
    mask_preencode: bool = False  # Whether to mask preencode or mask features
    cache_gating: bool = True  # Whether to use cache gating
    cache_gating_buffer_size: int = 2  # Buffer size for cache gating
    single_speaker_mode: bool = False  # Whether to use single speaker mode

    # ── Input / output configs ─────────────────────────────────────────
    audio_dir: Optional[str] = None  # Path to a directory which contains audio files
    audio_type: str = "wav"  # Type of audio file if audio_dir passed
    audio_file: Optional[str] = None  # Path to an audio file to perform streaming
    manifest_file: Optional[str] = None  # Path to dataset's JSON manifest
    output_path: Optional[str] = None  # Path to output file when manifest is used as input
    output_seglst_file: Optional[str] = None  # Directory for SegLST output; auto-named <manifest>_sot_mtasr.seglst.json

    # ── General configs ────────────────────────────────────────────────
    session_len_sec: float = -1  # End-to-end diarization session length in seconds
    batch_size: int = 32
    num_workers: int = 8
    random_seed: Optional[int] = None  # Seed number going to be used in seed_everything()
    log: bool = True  # If True, log will be printed
    debug_mode: bool = False  # Whether to print more detail in the output

    # ── Streaming / chunked configs ────────────────────────────────────
    streaming_mode: bool = True  # If True, streaming diarization will be used
    chunk_size: int = -1  # The chunk_size for models trained with full context / offline models
    shift_size: int = -1  # The shift_size for models trained with full context / offline models
    left_chunks: int = 2  # Number of left chunks to be used as left context via caching
    online_normalization: bool = False  # Perform normalization on the run per chunk
    pad_and_drop_preencoded: bool = False
    att_context_size: Optional[list] = None  # Sets the att_context_size for multi-lookahead models
    last_channel_cache_size: int = -1  # Override encoder cache size (frames). -1 = use model default (att_context_size[0])

    # ── Streaming diarization configs ──────────────────────────────────
    spkcache_len: int = 188
    spkcache_refresh_rate: int = 0
    fifo_len: int = 188
    chunk_len: int = 0
    chunk_left_context: int = 0
    chunk_right_context: int = 0

    # ── Device / precision configs ─────────────────────────────────────
    device: str = 'cuda'
    cuda: Optional[int] = None
    allow_mps: bool = False  # Allow to select MPS device (Apple Silicon M-series GPU)
    use_amp: bool = False
    amp_dtype: str = "float16"  # Can be "float16" or "bfloat16" when using amp
    compute_dtype: Optional[str] = (
        "float32"  # "float32" (default), "bfloat16" or "float16"; if None: bfloat16 if available else float32
    )
    matmul_precision: str = "highest"  # Literal["highest", "high", "medium"]

    # ── Decoding configs ───────────────────────────────────────────────
    ctc_decoding: CTCDecodingConfig = field(default_factory=CTCDecodingConfig)
    rnnt_decoding: RNNTDecodingConfig = field(default_factory=lambda: RNNTDecodingConfig(fused_batch_size=-1))
    set_decoder: Optional[str] = None  # Literal["ctc", "rnnt"]

    # ── Real-time / display configs (for future merge) ─────────────────
    generate_realtime_scripts: bool = True  # Whether to generate real-time demo scripts


def extract_transcriptions(hyps):
    """
    The transcribed_texts returned by CTC and RNNT models are different.
    This method would extract and return the text section of the hypothesis.
    """
    if isinstance(hyps[0], Hypothesis):
        transcriptions = []
        for hyp in hyps:
            transcriptions.append(hyp.text)
    else:
        transcriptions = hyps
    return transcriptions


def calc_drop_extra_pre_encoded(asr_model, step_num, pad_and_drop_preencoded):
    # For the first step there is no need to drop any tokens after the
    # downsampling as no caching is being used.
    if step_num == 0 and not pad_and_drop_preencoded:
        return 0
    else:
        return asr_model.encoder.streaming_cfg.drop_extra_pre_encoded


def get_samples_with_offset(audio_file: str, offset: float = 0.0, duration: float = 0.0,
                            target_sr: int = 16000):
    """
    Read audio samples from *audio_file*, honouring *offset* and *duration*
    fields that come from a NeMo manifest.

    Args:
        audio_file: Path to the audio file.
        offset: Start time in seconds (0 = beginning).
        duration: Length in seconds to read (0 or negative = read to end).
        target_sr: Resample to this rate if the file has a different one.

    Returns:
        numpy array of samples (shape ``[channels]`` or ``[samples]``).
    """
    with sf.SoundFile(audio_file, 'r') as f:
        orig_sr = f.samplerate
        if offset > 0:
            f.seek(int(offset * orig_sr))
        if duration > 0:
            samples = f.read(frames=int(duration * orig_sr), dtype='float32')
        else:
            samples = f.read(dtype='float32')
        if orig_sr != target_sr:
            samples = librosa.core.resample(samples, orig_sr=orig_sr, target_sr=target_sr)
        samples = samples.transpose()
    return samples


def get_audio_duration_sec(audio_file: str, offset: float = 0.0, duration: float = 0.0) -> float:
    """
    Return the effective duration (in seconds) of an audio segment.

    If *duration* > 0 it is returned directly (manifest already specifies it).
    Otherwise the file is inspected and the duration from *offset* to end is
    returned.
    """
    if duration > 0:
        return duration
    with sf.SoundFile(audio_file, 'r') as f:
        total_dur = f.frames / f.samplerate
    return total_dur - offset


def sot_text_to_seglst(transcription: str, session_id: str, session_duration_sec: float = -1.0) -> list:
    """
    Convert a SOT (Serialized Output Training) transcription string with
    speaker tokens into a list of SegLST-format dictionaries.

    Speaker tokens ``<|spltoken0|>``, ``<|spltoken1|>``, ... delimit speaker
    turns.  Each contiguous run of text after a speaker token becomes one
    SegLST entry attributed to that speaker.

    When *session_duration_sec* > 0, start and end times are **estimated** by
    distributing the session duration proportionally across segments based on
    their character length.  SOT serialises turns roughly in chronological
    order, so the proportional assignment gives a reasonable approximation
    that is sufficient for evaluation metrics that tolerate moderate timing
    errors (e.g. cpWER with a collar).  When *session_duration_sec* <= 0 the
    timestamps fall back to ``-1``.

    Args:
        transcription: SOT transcription, e.g.
            ``"<|spltoken0|> right <|spltoken1|> and i got ..."``
        session_id: Session identifier for this audio file.
        session_duration_sec: Total duration of the audio session in seconds.
            Used to proportionally estimate segment timestamps.  Pass ``-1``
            (default) to disable timestamp estimation.

    Returns:
        List[dict] with keys ``session_id``, ``words``, ``speaker``,
        ``start_time``, ``end_time``.

    Example output (with duration=10.0)::

        [
            {"session_id": "en_0638", "words": "right",
             "speaker": "speaker_0", "start_time": 0.0, "end_time": 1.515},
            {"session_id": "en_0638", "words": "and i got another apartment",
             "speaker": "speaker_1", "start_time": 1.515, "end_time": 9.697},
            ...
        ]
    """
    seglst_entries = []
    # Matches <|spltoken0|>, <|spltoken1|>, etc. and captures the digit(s)
    spk_token_pattern = re.compile(r'<\|spltoken(\d+)\|>')

    # re.split with a capturing group returns interleaved:
    #   [text_before, spk_id, text, spk_id, text, ...]
    parts = spk_token_pattern.split(transcription)

    current_speaker = None
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # Odd indices are captured speaker ids
            current_speaker = int(part)
        else:
            # Even indices are text segments
            text = part.strip()
            if text and current_speaker is not None:
                seglst_entries.append(
                    {
                        "session_id": session_id,
                        "words": text,
                        "speaker": f"speaker_{current_speaker}",
                    }
                )

    # ── Estimate proportional timestamps from character lengths ────────
    total_chars = sum(len(entry["words"]) for entry in seglst_entries)
    if session_duration_sec > 0 and total_chars > 0:
        cursor = 0.0
        for entry in seglst_entries:
            proportion = len(entry["words"]) / total_chars
            seg_dur = proportion * session_duration_sec
            entry["start_time"] = round(cursor, 3)
            entry["end_time"] = round(cursor + seg_dur, 3)
            cursor += seg_dur
    else:
        for entry in seglst_entries:
            entry["start_time"] = -1
            entry["end_time"] = -1

    return seglst_entries


def write_seglst_file(seglst_list: list, output_path: str):
    """Write a list of SegLST dicts to a JSON file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(seglst_list, f, indent=4)
    logging.info(f"Wrote SegLST file with {len(seglst_list)} entries to {output_path}")


def get_session_id(sample: dict) -> str:
    """Derive a session_id from a manifest sample."""
    if "session_id" in sample:
        return sample["session_id"]
    # Fall back to audio filename stem
    return os.path.splitext(os.path.basename(sample["audio_filepath"]))[0]


def perform_streaming(
    asr_model,
    streaming_buffer,
    compute_dtype: torch.dtype,
    debug_mode=False,
    pad_and_drop_preencoded=False,
):
    """
    Perform cache-aware streaming inference with diarization fusion.

    This function uses conformer_stream_step_with_diarization() to fuse
    streaming diarization predictions with the ASR encoder states at every
    chunk.  This adds speaker_infusion_asr to the encoder output, which is
    critical for multi-talker ASR in streaming mode where cache context is
    limited (~5 s).
    """
    batch_size = len(streaming_buffer.streams_length)
    use_pre_encode_diar_fusion = getattr(asr_model, 'use_pre_encode_diar_fusion', False)

    logging.info("Streaming with diarization fusion enabled")
    if use_pre_encode_diar_fusion:
        logging.info("Pre-encode diar fusion enabled (sinusoidal + merge)")

    cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(
        batch_size=batch_size
    )

    # ── Initialize diar streaming state ────────────────────────────────
    model_device = next(asr_model.parameters()).device
    diar_streaming_state = asr_model.diarization_model.sortformer_modules.init_streaming_state(
        batch_size=batch_size, device=model_device,
    )
    diar_pred_out_stream = torch.zeros(
        (batch_size, 0, asr_model.max_num_speakers),
        device=model_device,
    )

    previous_hypotheses = None
    streaming_buffer_iter = iter(streaming_buffer)
    pred_out_stream = None
    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
        with torch.inference_mode():
            chunk_audio = chunk_audio.to(compute_dtype)
            drop_extra = calc_drop_extra_pre_encoded(asr_model, step_num, pad_and_drop_preencoded)

            # ── Diarization streaming step ─────────────────────────────
            with torch.no_grad():
                prev_diar_len = diar_pred_out_stream.shape[1]
                diar_streaming_state, diar_pred_out_stream = (
                    asr_model.diarization_model.forward_streaming_step(
                        processed_signal=chunk_audio.transpose(1, 2),
                        processed_signal_length=chunk_lengths,
                        streaming_state=diar_streaming_state,
                        total_preds=diar_pred_out_stream,
                        drop_extra_pre_encoded=drop_extra,
                    )
                )
                # Per-chunk diar predictions (matches training flow)
                chunk_diar_preds = diar_pred_out_stream[:, prev_diar_len:, :]

            # ── ASR streaming step (with diar fusion) ──────────────────
            with torch.no_grad():
                if use_pre_encode_diar_fusion:
                    # Pre-encode level fusion: run ASR pre_encode,
                    # fuse speaker info, then feed to encoder with
                    # bypass_pre_encode=True — exactly matching the
                    # training flow in forward_simulated_streaming().
                    fused_pre_encode, fused_lengths = asr_model._apply_pre_encode_diar_fusion(
                        chunk_audio, chunk_lengths, chunk_diar_preds, drop_extra,
                    )
                    (
                        pred_out_stream,
                        transcribed_texts,
                        cache_last_channel,
                        cache_last_time,
                        cache_last_channel_len,
                        previous_hypotheses,
                    ) = asr_model.conformer_stream_step_with_diarization(
                        processed_signal=fused_pre_encode,
                        processed_signal_length=fused_lengths,
                        cache_last_channel=cache_last_channel,
                        cache_last_time=cache_last_time,
                        cache_last_channel_len=cache_last_channel_len,
                        keep_all_outputs=streaming_buffer.is_buffer_empty(),
                        previous_hypotheses=previous_hypotheses,
                        previous_pred_out=pred_out_stream,
                        drop_extra_pre_encoded=0,
                        return_transcription=True,
                        bypass_pre_encode=True,
                        diar_preds=chunk_diar_preds,
                    )
                else:
                    (
                        pred_out_stream,
                        transcribed_texts,
                        cache_last_channel,
                        cache_last_time,
                        cache_last_channel_len,
                        previous_hypotheses,
                    ) = asr_model.conformer_stream_step_with_diarization(
                        processed_signal=chunk_audio,
                        processed_signal_length=chunk_lengths,
                        cache_last_channel=cache_last_channel,
                        cache_last_time=cache_last_time,
                        cache_last_channel_len=cache_last_channel_len,
                        keep_all_outputs=streaming_buffer.is_buffer_empty(),
                        previous_hypotheses=previous_hypotheses,
                        previous_pred_out=pred_out_stream,
                        drop_extra_pre_encoded=drop_extra,
                        return_transcription=True,
                        diar_preds=chunk_diar_preds,
                    )

        if debug_mode:
            logging.info(f"Streaming transcriptions: {extract_transcriptions(transcribed_texts)}")

    final_streaming_tran = extract_transcriptions(transcribed_texts)
    logging.info(f"Final streaming transcriptions: {final_streaming_tran}")

    return final_streaming_tran


@hydra_runner(config_name="TranscriptionConfig", schema=TranscriptionConfig)
def main(cfg: TranscriptionConfig) -> Union[TranscriptionConfig]:
    # Convert 'None' strings to actual None (Hydra passes null as the string "None")
    for key in cfg:
        cfg[key] = None if cfg[key] == 'None' else cfg[key]

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision(cfg.matmul_precision)

    if cfg.random_seed:
        pl.seed_everything(cfg.random_seed)

    # ── Device setup ───────────────────────────────────────────────────
    device = get_inference_device(cuda=cfg.cuda, allow_mps=cfg.allow_mps)

    if (cfg.compute_dtype is not None and cfg.compute_dtype != "float32") and cfg.use_amp:
        raise ValueError("use_amp=true is mutually exclusive with a compute_dtype other than float32")

    amp_dtype = torch.float16 if cfg.amp_dtype == "float16" else torch.bfloat16

    compute_dtype: torch.dtype
    if cfg.use_amp:
        # With amp, model weights are required to be in float32
        compute_dtype = torch.float32
    else:
        compute_dtype = get_inference_dtype(compute_dtype=cfg.compute_dtype, device=device)

    if compute_dtype != torch.float32:
        raise NotImplementedError(
            f"Compute dtype {compute_dtype} is not yet supported for cache-aware models, use float32 instead"
        )

    # ── Input validation ───────────────────────────────────────────────
    if sum((cfg.audio_file is not None, cfg.manifest_file is not None, cfg.audio_dir is not None)) != 1:
        raise ValueError("Exactly one of `audio_file`, `manifest_file` or `audio_dir` should be non-empty!")

    # ── Load ASR model ─────────────────────────────────────────────────
    # setup_model reads cfg.model_path, so map asr_model → model_path
    with open_dict(cfg):
        cfg.model_path = cfg.asr_model

    asr_model, model_name = setup_model(cfg=cfg, map_location=device)

    logging.info(f"Loaded model class: {type(asr_model).__name__}")
    if isinstance(asr_model, MSEncDecRNNTBPEModel):
        logging.info(
            f"MSEncDecRNNTBPEModel detected. diar={getattr(asr_model, 'diar', False)}, "
            f"max_num_speakers={getattr(asr_model, 'max_num_speakers', 'N/A')}"
        )

        # ── Configure diar model streaming params (follows tutorial convention) ──
        if asr_model.diar:
            diar_model = asr_model.diarization_model
            diar_model.streaming_mode = cfg.streaming_mode
            diar_model.sortformer_modules.chunk_len = cfg.chunk_len
            diar_model.sortformer_modules.spkcache_len = cfg.spkcache_len
            diar_model.sortformer_modules.chunk_left_context = cfg.chunk_left_context
            diar_model.sortformer_modules.chunk_right_context = cfg.chunk_right_context
            diar_model.sortformer_modules.fifo_len = cfg.fifo_len
            diar_model.sortformer_modules.log = cfg.log
            diar_model.sortformer_modules.spkcache_refresh_rate = cfg.spkcache_refresh_rate
            logging.info(
                f"Diar streaming params: chunk_len={cfg.chunk_len}, spkcache_len={cfg.spkcache_len}, "
                f"fifo_len={cfg.fifo_len}, spkcache_refresh_rate={cfg.spkcache_refresh_rate}"
            )

    logging.info(asr_model.encoder.streaming_cfg)

    if cfg.att_context_size is not None:
        if hasattr(asr_model.encoder, "set_default_att_context_size"):
            asr_model.encoder.set_default_att_context_size(att_context_size=cfg.att_context_size)
        else:
            raise ValueError("Model does not support multiple lookaheads.")

    # ── Setup decoding strategy ────────────────────────────────────────
    if hasattr(asr_model, 'change_decoding_strategy') and hasattr(asr_model, 'decoding'):
        if cfg.set_decoder is not None:
            decoding_cfg = cfg.rnnt_decoding if cfg.set_decoder == 'rnnt' else cfg.ctc_decoding

            if hasattr(asr_model, 'cur_decoder'):
                asr_model.change_decoding_strategy(decoding_cfg, decoder_type=cfg.set_decoder)
            else:
                asr_model.change_decoding_strategy(decoding_cfg)

        # Check if ctc or rnnt model
        elif hasattr(asr_model, 'joint'):  # RNNT model
            cfg.rnnt_decoding.fused_batch_size = -1
            if hasattr(asr_model, 'cur_decoder'):
                asr_model.change_decoding_strategy(cfg.rnnt_decoding, decoder_type=cfg.set_decoder)
            else:
                asr_model.change_decoding_strategy(cfg.rnnt_decoding)
        else:
            asr_model.change_decoding_strategy(cfg.ctc_decoding)

    asr_model = asr_model.to(device=device, dtype=compute_dtype)
    asr_model.eval()

    # ── Streaming params setup ─────────────────────────────────────────
    # chunk_size is set automatically for models trained for streaming.
    # For models trained for offline mode with full context, we need to pass
    # the chunk_size explicitly.
    if cfg.chunk_size > 0:
        if cfg.shift_size < 0:
            shift_size = cfg.chunk_size
        else:
            shift_size = cfg.shift_size
        asr_model.encoder.setup_streaming_params(
            chunk_size=cfg.chunk_size, left_chunks=cfg.left_chunks, shift_size=shift_size
        )

    # Ensure streaming_cfg is initialized (lazy init if model hasn't set it yet)
    if asr_model.encoder.streaming_cfg is None:
        asr_model.encoder.setup_streaming_params()

    # ── Override last_channel_cache_size if requested ──────────────────
    if cfg.last_channel_cache_size > 0:
        original = asr_model.encoder.streaming_cfg.last_channel_cache_size
        asr_model.encoder.streaming_cfg.last_channel_cache_size = cfg.last_channel_cache_size
        logging.info(
            f"Overriding last_channel_cache_size: {original} -> {cfg.last_channel_cache_size} "
            f"(~{cfg.last_channel_cache_size * 0.08:.1f}s at 80ms/frame)"
        )

    logging.info(f"Final streaming_cfg.last_channel_cache_size = "
                 f"{asr_model.encoder.streaming_cfg.last_channel_cache_size}")

    # In streaming, offline normalization is not feasible as we don't have
    # access to the whole audio at the beginning.  When online_normalization
    # is enabled, the normalization of the input features (mel-spectrograms)
    # are done per step.  It is suggested to train the streaming models
    # without any normalization in the input features.
    if cfg.online_normalization:
        if asr_model.cfg.preprocessor.normalize not in ["per_feature", "all_feature"]:
            logging.warning(
                "online_normalization is enabled but the model has "
                "no normalization in the feature extraction part, so it is ignored."
            )
            online_normalization = False
        else:
            online_normalization = True
    else:
        online_normalization = False

    # ── Streaming buffer ───────────────────────────────────────────────
    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=asr_model,
        online_normalization=online_normalization,
        pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
    )

    # ── Collect all SegLST entries across sessions ──────────────────────
    all_seglst_entries = []

    with torch.amp.autocast('cuda' if device.type == "cuda" else "cpu", dtype=amp_dtype, enabled=cfg.use_amp):
        if cfg.audio_file is not None:
            # ── Stream a single audio file ─────────────────────────────
            _ = streaming_buffer.append_audio_file(cfg.audio_file, stream_id=-1)
            streaming_tran = perform_streaming(
                asr_model=asr_model,
                streaming_buffer=streaming_buffer,
                compute_dtype=compute_dtype,
                pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
            )
            # Convert SOT transcription to SegLST
            session_id = os.path.splitext(os.path.basename(cfg.audio_file))[0]
            audio_dur = get_audio_duration_sec(cfg.audio_file)
            for tran in streaming_tran:
                all_seglst_entries.extend(sot_text_to_seglst(tran, session_id, session_duration_sec=audio_dur))

            dataset_title = session_id
        else:
            # ── Stream audio files in a manifest / audio_dir ───────────
            all_streaming_tran = []
            all_refs_text = []
            batch_size = cfg.batch_size

            if cfg.manifest_file is not None:
                manifest_dir = Path(cfg.manifest_file).parent
                samples = read_manifest(cfg.manifest_file)
                # Fix relative paths
                for item in samples:
                    audio_filepath = Path(item["audio_filepath"])
                    if not audio_filepath.is_absolute():
                        item["audio_filepath"] = str(manifest_dir / audio_filepath)

                logging.info(f"Loaded {len(samples)} from the manifest at {cfg.manifest_file}.")
                dataset_title = os.path.splitext(os.path.basename(cfg.manifest_file))[0]
            else:
                assert cfg.audio_dir is not None
                samples = [
                    {"audio_filepath": audio_filepath}
                    for audio_filepath in (
                        glob.glob(os.path.join(cfg.audio_dir, f"**/*.{cfg.audio_type}"), recursive=True)
                    )
                ]
                dataset_title = os.path.basename(cfg.audio_dir)

            start_time = time.time()
            # Track which samples belong to the current batch for SegLST
            batch_start_idx = 0
            for sample_idx, sample in enumerate(samples):
                offset = float(sample.get("offset", 0))
                duration = float(sample.get("duration", 0))
                if offset > 0 or duration > 0:
                    audio = get_samples_with_offset(
                        sample['audio_filepath'], offset=offset, duration=duration,
                    )
                    _ = streaming_buffer.append_audio(audio, stream_id=-1)
                else:
                    _ = streaming_buffer.append_audio_file(sample['audio_filepath'], stream_id=-1)
                if "text" in sample:
                    all_refs_text.append(sample["text"])
                logging.info(
                    f'Added this sample to the buffer: {sample["audio_filepath"]}'
                    f' (offset={offset}, duration={duration})'
                )

                if (sample_idx + 1) % batch_size == 0 or sample_idx == len(samples) - 1:
                    logging.info(
                        f"Starting to stream samples {sample_idx - len(streaming_buffer) + 1} to {sample_idx}..."
                    )
                    streaming_tran = perform_streaming(
                        asr_model=asr_model,
                        streaming_buffer=streaming_buffer,
                        compute_dtype=compute_dtype,
                        debug_mode=cfg.debug_mode,
                        pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
                    )
                    all_streaming_tran.extend(streaming_tran)

                    # Convert each session's SOT transcription to SegLST
                    batch_samples = samples[batch_start_idx : sample_idx + 1]
                    for tran, samp in zip(streaming_tran, batch_samples):
                        sid = get_session_id(samp)
                        audio_dur = get_audio_duration_sec(
                            samp['audio_filepath'],
                            offset=float(samp.get("offset", 0)),
                            duration=float(samp.get("duration", 0)),
                        )
                        all_seglst_entries.extend(sot_text_to_seglst(tran, sid, session_duration_sec=audio_dur))
                    batch_start_idx = sample_idx + 1

                    streaming_buffer.reset_buffer()

        if len(all_refs_text) == len(all_streaming_tran):
            streaming_wer = word_error_rate(hypotheses=all_streaming_tran, references=all_refs_text)
            logging.info(f"WER% of streaming mode: {round(streaming_wer * 100, 2)}")

        end_time = time.time()
        logging.info(f"The whole streaming process took: {round(end_time - start_time, 2)}s")

        # Store results including the transcriptions of streaming inference
        if cfg.output_path is not None and len(all_refs_text) == len(all_streaming_tran):
            fname = "streaming_out_" + os.path.splitext(os.path.basename(model_name))[0] + f"_{dataset_title}.json"

            hyp_json = os.path.join(cfg.output_path, fname)
            os.makedirs(cfg.output_path, exist_ok=True)
            with open(hyp_json, "w") as out_f:
                for i, hyp in enumerate(all_streaming_tran):
                    record = {
                        "pred_text": hyp,
                        "text": all_refs_text[i],
                        "wer": round(word_error_rate(hypotheses=[hyp], references=[all_refs_text[i]]) * 100, 2),
                    }
                    out_f.write(json.dumps(record) + '\n')

    # ── Write SegLST output file ───────────────────────────────────────
    if all_seglst_entries:
        # Determine seglst output path
        seglst_fname = f"{dataset_title}_sot_mtasr.seglst.json"
        if cfg.output_seglst_file is not None:
            # User provided an explicit directory
            seglst_path = os.path.join(cfg.output_seglst_file, seglst_fname)
        elif cfg.output_path is not None:
            seglst_path = os.path.join(cfg.output_path, seglst_fname)
        else:
            seglst_path = seglst_fname  # Write to current directory

        write_seglst_file(all_seglst_entries, seglst_path)


if __name__ == '__main__':
    main()
