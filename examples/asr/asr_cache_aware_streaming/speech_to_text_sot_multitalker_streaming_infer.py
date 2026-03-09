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
Single-speaker ASR evaluation for WL-SOT (Word-Level Serialized Output Training)
multi-talker streaming models (MSEncDecRNNTBPEModel).

This script loads the WL-SOT model and its embedded diarization model, runs
cache-aware streaming inference with diarization fusion, then strips all
speaker tokens ``[s0]``, ``[s1]``, ... from hypothesis text and applies
Open ASR Leaderboard text normalization (EnglishTextNormalizer) to both
hypothesis and reference text before computing WER.

The output format matches the standard NeMo cache-aware streaming evaluation
(speech_to_text_cache_aware_streaming_infer.py).

# Usage

python speech_to_text_sot_multitalker_streaming_infer.py \\
    asr_model=wl_sot_model.nemo \\
    manifest_file=manifest.json \\
    batch_size=100 \\
    att_context_size="[-1,13]"
"""

import glob
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Optional, Union

import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

sys.path.append('/gpfs/fs1/projects/ent_aiapps/datasets/data/diarization_e2e/open_asr_leaderboard')
from normalizer import EnglishTextNormalizer

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

SPK_TOKEN_RE = re.compile(r'\[s\d+\]\s*')
_text_normalizer = EnglishTextNormalizer()


def strip_speaker_tokens(text: str) -> str:
    """Remove all [s0], [s1], ... speaker tokens and collapse whitespace."""
    return SPK_TOKEN_RE.sub('', text).strip()


def normalize_text(text: str) -> str:
    """Apply Open ASR Leaderboard EnglishTextNormalizer."""
    return _text_normalizer(text)


@dataclass
class TranscriptionConfig:
    """Configuration for single-speaker eval of WL-SOT streaming model."""

    # ASR model
    asr_model: Optional[str] = None
    pretrained_name: Optional[str] = None

    # Input / output
    audio_dir: Optional[str] = None
    audio_type: str = "wav"
    audio_file: Optional[str] = None
    manifest_file: Optional[str] = None
    output_path: Optional[str] = None

    # General
    batch_size: int = 32
    num_workers: int = 8
    random_seed: Optional[int] = None
    log: bool = True
    debug_mode: bool = False
    single_speaker_mode: bool = True

    # Streaming / chunked
    streaming_mode: bool = True
    chunk_size: int = -1
    shift_size: int = -1
    left_chunks: int = 2
    online_normalization: bool = False
    pad_and_drop_preencoded: bool = False
    att_context_size: Optional[list] = None
    last_channel_cache_size: int = -1

    # Diar streaming params (applied to embedded diar model)
    spkcache_len: int = 188
    spkcache_refresh_rate: int = 0
    fifo_len: int = 188
    chunk_len: int = 0
    chunk_left_context: int = 0
    chunk_right_context: int = 0

    # Speaker config
    max_num_of_spks: Optional[int] = 4
    spk_supervision: str = "diar"
    masked_asr: bool = False

    # Device / precision
    device: str = 'cuda'
    cuda: Optional[int] = None
    allow_mps: bool = False
    use_amp: bool = False
    amp_dtype: str = "float16"
    compute_dtype: Optional[str] = "float32"
    matmul_precision: str = "highest"
    strict_restore: bool = True

    # Decoding
    ctc_decoding: CTCDecodingConfig = field(default_factory=CTCDecodingConfig)
    rnnt_decoding: RNNTDecodingConfig = field(default_factory=lambda: RNNTDecodingConfig(fused_batch_size=-1))
    set_decoder: Optional[str] = None

    # Display
    generate_realtime_scripts: bool = False


def extract_transcriptions(hyps):
    if isinstance(hyps[0], Hypothesis):
        return [hyp.text for hyp in hyps]
    return hyps


def calc_drop_extra_pre_encoded(asr_model, step_num, pad_and_drop_preencoded):
    if step_num == 0 and not pad_and_drop_preencoded:
        return 0
    return asr_model.encoder.streaming_cfg.drop_extra_pre_encoded


def _estimate_streaming_steps(streaming_buffer):
    buffer_len = streaming_buffer.buffer.size(-1)
    cfg = streaming_buffer.streaming_cfg

    if isinstance(cfg.shift_size, list):
        first_shift = cfg.shift_size[1] if streaming_buffer.pad_and_drop_preencoded else cfg.shift_size[0]
        subsequent_shift = cfg.shift_size[1]
    else:
        first_shift = cfg.shift_size
        subsequent_shift = cfg.shift_size

    if buffer_len <= 0:
        return 0
    if buffer_len <= first_shift:
        return 1
    return 1 + math.ceil((buffer_len - first_shift) / subsequent_shift)


def perform_streaming(
    asr_model,
    streaming_buffer,
    compute_dtype: torch.dtype,
    debug_mode=False,
    pad_and_drop_preencoded=False,
):
    """
    Cache-aware streaming inference with diarization fusion (live diar model).
    Adapted from speech_to_text_cache_aware_streaming_sot_mtasr_infer.py
    but always uses live diar model (spk_supervision="diar").
    """
    batch_size = len(streaming_buffer.streams_length)
    use_pre_encode_diar_fusion = getattr(asr_model, 'use_pre_encode_diar_fusion', False)

    cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(
        batch_size=batch_size
    )

    model_device = next(asr_model.parameters()).device
    num_speakers = asr_model.max_num_speakers
    nframes_per_chunk = asr_model.encoder.streaming_cfg.valid_out_len

    previous_hypotheses = None
    streaming_buffer_iter = iter(streaming_buffer)
    total_steps = _estimate_streaming_steps(streaming_buffer)
    pred_out_stream = None

    for step_num, (chunk_audio, chunk_lengths) in enumerate(
        tqdm(streaming_buffer_iter, total=total_steps, desc="Streaming")
    ):
        with torch.inference_mode():
            chunk_audio = chunk_audio.to(compute_dtype)
            drop_extra = calc_drop_extra_pre_encoded(asr_model, step_num, pad_and_drop_preencoded)

            # Dummy diar preds: speaker 0 active, all others silent
            chunk_diar_preds = torch.zeros(
                (batch_size, nframes_per_chunk, num_speakers), device=model_device,
            )
            chunk_diar_preds[:, :, 0] = 1.0

            with torch.no_grad():
                if use_pre_encode_diar_fusion:
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
    if final_streaming_tran:
        logging.info(f"Final streaming transcription (raw): {final_streaming_tran[0][:200]}")

    return final_streaming_tran


@hydra_runner(config_name="TranscriptionConfig", schema=TranscriptionConfig)
def main(cfg: TranscriptionConfig) -> Union[TranscriptionConfig]:
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

    # ── Load ASR model (MSEncDecRNNTBPEModel with embedded diar) ───────
    with open_dict(cfg):
        cfg.model_path = cfg.asr_model

    asr_model, model_name = setup_model(cfg=cfg, map_location=device)

    logging.info(f"Loaded model class: {type(asr_model).__name__}")
    if isinstance(asr_model, MSEncDecRNNTBPEModel):
        logging.info(
            f"MSEncDecRNNTBPEModel: diar={getattr(asr_model, 'diar', False)}, "
            f"max_num_speakers={getattr(asr_model, 'max_num_speakers', 'N/A')}"
        )

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
    else:
        logging.warning(
            f"Model is {type(asr_model).__name__}, not MSEncDecRNNTBPEModel. "
            f"Diarization fusion will not be used."
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
        elif hasattr(asr_model, 'joint'):
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
    if cfg.chunk_size > 0:
        shift_size = cfg.chunk_size if cfg.shift_size < 0 else cfg.shift_size
        asr_model.encoder.setup_streaming_params(
            chunk_size=cfg.chunk_size, left_chunks=cfg.left_chunks, shift_size=shift_size
        )

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

    with torch.amp.autocast('cuda' if device.type == "cuda" else "cpu", dtype=amp_dtype, enabled=cfg.use_amp):
        if cfg.audio_file is not None:
            _ = streaming_buffer.append_audio_file(cfg.audio_file, stream_id=-1)
            streaming_tran = perform_streaming(
                asr_model=asr_model,
                streaming_buffer=streaming_buffer,
                compute_dtype=compute_dtype,
                pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
            )
            cleaned = [normalize_text(strip_speaker_tokens(t)) for t in streaming_tran]
            logging.info(f"Cleaned transcription: {cleaned[0]}")
        else:
            all_streaming_tran = []
            all_refs_text = []
            batch_size = cfg.batch_size

            if cfg.manifest_file is not None:
                manifest_dir = Path(cfg.manifest_file).parent
                samples = read_manifest(cfg.manifest_file)
                for item in samples:
                    audio_filepath = Path(item["audio_filepath"])
                    if not audio_filepath.is_absolute():
                        item["audio_filepath"] = str(manifest_dir / audio_filepath)
                logging.info(f"Loaded {len(samples)} from the manifest at {cfg.manifest_file}.")
                dataset_title = os.path.splitext(os.path.basename(cfg.manifest_file))[0]
            else:
                assert cfg.audio_dir is not None
                samples = [
                    {"audio_filepath": af}
                    for af in glob.glob(os.path.join(cfg.audio_dir, f"**/*.{cfg.audio_type}"), recursive=True)
                ]
                dataset_title = os.path.basename(cfg.audio_dir)

            start_time = time.time()
            for sample_idx, sample in enumerate(samples):
                _ = streaming_buffer.append_audio_file(sample['audio_filepath'], stream_id=-1)
                if "text" in sample:
                    all_refs_text.append(normalize_text(sample["text"]))
                logging.info(f'Added this sample to the buffer: {sample["audio_filepath"]}')

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
                    cleaned = [normalize_text(strip_speaker_tokens(t)) for t in streaming_tran]
                    all_streaming_tran.extend(cleaned)
                    streaming_buffer.reset_buffer()

            if len(all_refs_text) == len(all_streaming_tran):
                streaming_wer = word_error_rate(hypotheses=all_streaming_tran, references=all_refs_text)
                logging.info(f"WER% of streaming mode: {round(streaming_wer * 100, 2)}")
            else:
                logging.warning(
                    f"Reference count ({len(all_refs_text)}) != hypothesis count ({len(all_streaming_tran)}). "
                    f"Skipping WER computation."
                )

            end_time = time.time()
            logging.info(f"The whole streaming process took: {round(end_time - start_time, 2)}s")

            if cfg.output_path is not None and len(all_refs_text) == len(all_streaming_tran):
                fname = f"streaming_out_{os.path.splitext(os.path.basename(model_name))[0]}_{dataset_title}.json"
                hyp_json = os.path.join(cfg.output_path, fname)
                os.makedirs(cfg.output_path, exist_ok=True)
                with open(hyp_json, "w", encoding="utf-8") as out_f:
                    for i, hyp in enumerate(all_streaming_tran):
                        record = {
                            "pred_text": hyp,
                            "text": all_refs_text[i],
                            "wer": round(
                                word_error_rate(hypotheses=[hyp], references=[all_refs_text[i]]) * 100, 2
                            ),
                        }
                        out_f.write(json.dumps(record) + '\n')
                logging.info(f"Wrote per-sample results to {hyp_json}")


if __name__ == '__main__':
    main()
