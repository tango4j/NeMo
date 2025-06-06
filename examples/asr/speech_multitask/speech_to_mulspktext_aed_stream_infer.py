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

"""
This script chunks long audios into non-overlapping segments of `chunk_len_in_secs` seconds and performs inference on each 
segment individually. The results are then concatenated to form the final output.

Below is an example of how to run this script with the Canary-1b model.
It's recommended to use manifest input, otherwise the model will perform English ASR with punctuations and capitalizations. 
An example manifest line:
{
    "audio_filepath": "/path/to/audio.wav",  # path to the audio file
    "duration": 10000.0,  # duration of the audio
    "taskname": "asr",  # use "s2t_translation" for AST
    "source_lang": "en",  # Set `source_lang`==`target_lang` for ASR, choices=['en','de','es','fr']
    "target_lang": "de",  # choices=['en','de','es','fr']
    "pnc": "yes",  # whether to have PnC output, choices=['yes', 'no'] 
}

Example Usage:
python speech_to_text_aed_chunked_infer.py \
    model_path=null \
    pretrained_name="nvidia/canary-1b" \
    audio_dir="<(optional) path to folder of audio files>" \
    dataset_manifest="<(optional) path to manifest>" \
    output_filename="<(optional) specify output filename>" \
    chunk_len_in_secs=40.0 \
    batch_size=16 \
    decoding.beam.beam_size=5
    
"""

import contextlib
import copy
import glob
import os
from dataclasses import dataclass, is_dataclass
from typing import Optional

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.parts.submodules.multitask_decoding import MultiTaskDecodingConfig
from nemo.collections.asr.parts.utils.eval_utils import cal_write_wer
from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchMultiTaskMultiStreamSpkAED
from nemo.collections.asr.parts.utils.transcribe_utils import (
    compute_output_filename,
    get_buffered_pred_feat_multitaskAED,
    get_buffered_pred_sample_streaming_multitaskAED,
    setup_model,
    write_transcription,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
import re


def merge_consecutive_tokens(text, pattern= r'<\|spltoken\d+\|>',  pattern_left="<|spltoken"):
    tokens = text.split()
    merged_tokens = []
    str_pattern = pattern.replace("\\", '')
    left_str, right_str = str_pattern.split('d+')[0], str_pattern.split('d+')[1]

    # Initialize a variable to track the last added token
    last_token = None

    for token in tokens:
        # Check if current token is a special token and if it's the same as the last token
        if len(re.findall(pattern, token)) > 0:
            if token == last_token:
                continue  # Skip adding the token if it's a duplicate special token
            else:
                if len(re.findall(pattern, token)) > 0:
                    spk_token_int = int(token.replace(left_str,'').replace(right_str, ''))
                    new_token = f"<|spltoken{spk_token_int}|>"
                else:
                    new_token = token
                merged_tokens.append(new_token)
                last_token = token  # Update the last token to the current one
        else:
            merged_tokens.append(token)
            # last_token = token  # Update the last token to the current one
    merged_text = ' '.join(merged_tokens)
    merged_text = merged_text.replace(f"{pattern_left}", f"\n{pattern_left}")
    return merged_text


@dataclass
class TranscriptionConfig:
    # Required configs
    model_path: Optional[str] = None  # Path to a .nemo file
    diar_pred_model_path = None
    pretrained_name: Optional[str] = None  # Name of a pretrained model
    audio_dir: Optional[str] = None  # Path to a directory which contains audio files
    dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest

    # General configs
    output_filename: Optional[str] = None  # if None, output will be stored in the same directory as the input
    batch_size: int = 8  # number of chunks to process in parallel.
    append_pred: bool = False  # Sets mode of work, if True it will add new field transcriptions.
    pred_name_postfix: Optional[str] = None  # If you need to use another model name, rather than standard one.
    random_seed: Optional[int] = None  # seed number going to be used in seed_everything()

    # Set to True to output greedy timestamp information (only supported models)
    compute_timestamps: bool = False

    # Set to True to output language ID information
    compute_langs: bool = False

    # Chunked configs
    chunk_len_in_secs: float = 5.0  # Chunk length in seconds
    step_len_in_secs: float = 0.5
    model_stride: int = 8  # Model downsampling factor, 8 for Citrinet and FasConformer models and 4 for Conformer models.

    # Decoding strategy for MultitaskAED models
    decoding: MultiTaskDecodingConfig = MultiTaskDecodingConfig()

    # Set `cuda` to int to define CUDA device. If 'None', will look for CUDA
    # device anyway, and do inference on CPU only if CUDA device is not found.
    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    amp: bool = False
    amp_dtype: str = "float16"  # can be set to "float16" or "bfloat16" when using amp
    matmul_precision: str = "highest"  # Literal["highest", "high", "medium"]
    audio_type: str = "wav"

    # Recompute model transcription, even if the output folder exists with scores.
    overwrite_transcripts: bool = True

    # Config for word / character error rate calculation
    calculate_wer: bool = True
    clean_groundtruth_text: bool = False
    langid: str = "en"  # specify this for convert_num_to_words step in groundtruth cleaning
    use_cer: bool = False


@hydra_runner(config_name="TranscriptionConfig", schema=TranscriptionConfig)
def main(cfg: TranscriptionConfig) -> TranscriptionConfig:
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    torch.set_grad_enabled(False)

    for key in cfg:
        cfg[key] = None if cfg[key] == 'None' else cfg[key]

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if cfg.random_seed:
        pl.seed_everything(cfg.random_seed)

    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both cfg.model_path and cfg.pretrained_name cannot be None!")
    if cfg.audio_dir is None and cfg.dataset_manifest is None:
        raise ValueError("Both cfg.audio_dir and cfg.dataset_manifest cannot be None!")

    filepaths = None
    manifest = cfg.dataset_manifest
    if cfg.audio_dir is not None:
        filepaths = list(glob.glob(os.path.join(cfg.audio_dir, f"**/*.{cfg.audio_type}"), recursive=True))
        manifest = None  # ignore dataset_manifest if audio_dir and dataset_manifest both presents

    # setup GPU
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    if cfg.cuda is None:
        if torch.cuda.is_available():
            device = [0]  # use 0th CUDA device
            accelerator = 'gpu'
        else:
            device = 1
            accelerator = 'cpu'
    else:
        device = [cfg.cuda]
        accelerator = 'gpu'
    map_location = torch.device('cuda:{}'.format(device[0]) if accelerator == 'gpu' else 'cpu')
    logging.info(f"Inference will be done on device : {device}")

    asr_model, model_name = setup_model(cfg, map_location)
    
    if cfg.get('diar_pred_model_path', None) is not None:
        asr_model.setup_diar_pred_model(cfg.diar_pred_model_path, map_location)
        
    model_cfg = copy.deepcopy(asr_model._cfg)
    OmegaConf.set_struct(model_cfg.preprocessor, False)
    # some changes for streaming scenario
    model_cfg.preprocessor.dither = 0.0
    model_cfg.preprocessor.pad_to = 0

    if model_cfg.preprocessor.normalize != "per_feature":
        logging.error(
            "Only EncDecMultiTaskModel models trained with per_feature normalization are supported currently"
        )

    # Disable config overwriting
    OmegaConf.set_struct(model_cfg.preprocessor, True)

    # setup AMP (optional)
    if cfg.amp and torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
        logging.info("AMP enabled!\n")
        autocast = torch.cuda.amp.autocast
    else:

        @contextlib.contextmanager
        def autocast(*args, **kwargs):
            yield

    # Compute output filename
    cfg = compute_output_filename(cfg, model_name)

    # if transcripts should not be overwritten, and already exists, skip re-transcription step and return
    if not cfg.overwrite_transcripts and os.path.exists(cfg.output_filename):
        logging.info(
            f"Previous transcripts found at {cfg.output_filename}, and flag `overwrite_transcripts`"
            f"is {cfg.overwrite_transcripts}. Returning without re-transcribing text."
        )
        return cfg

    asr_model.change_decoding_strategy(cfg.decoding)

    asr_model.eval()
    asr_model = asr_model.to(asr_model.device)

    feature_stride = model_cfg.preprocessor['window_stride']
    model_stride_in_secs = feature_stride * cfg.model_stride

    frame_asr = FrameBatchMultiTaskMultiStreamSpkAED(
        asr_model=asr_model,
        frame_len=cfg.step_len_in_secs,
        total_buffer=cfg.chunk_len_in_secs,
        batch_size=cfg.batch_size,
        step_len_in_secs=cfg.step_len_in_secs,
    )
    amp_dtype = torch.float16 if cfg.amp_dtype == "float16" else torch.bfloat16
    print("Sleeping for 4 seconds to load the model"); import time; time.sleep(4.0)
    with autocast(dtype=amp_dtype):
        with torch.no_grad():
            hyps = get_buffered_pred_sample_streaming_multitaskAED(
                frame_asr, model_cfg.preprocessor, model_stride_in_secs, asr_model.device, manifest, filepaths,
            )

    # output_filename, pred_text_attr_name = write_transcription(
    #     hyps, cfg, model_name, filepaths=filepaths, compute_langs=False, compute_timestamps=False
    # )
    # logging.info(f"Finished writing predictions to {output_filename}!")

    # if cfg.calculate_wer:
    #     output_manifest_w_wer, total_res, _ = cal_write_wer(
    #         pred_manifest=output_filename,
    #         pred_text_attr_name=pred_text_attr_name,
    #         clean_groundtruth_text=cfg.clean_groundtruth_text,
    #         langid=cfg.langid,
    #         use_cer=cfg.use_cer,
    #         output_filename=None,
    #     )
    #     if output_manifest_w_wer:
    #         logging.info(f"Writing prediction and error rate of each sample to {output_manifest_w_wer}!")
    #         logging.info(f"{total_res}")

    # return cfg


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
