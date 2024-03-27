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

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

import contextlib
import glob
import json
import os
from dataclasses import dataclass, is_dataclass
from tempfile import NamedTemporaryFile
from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.models import EncDecCTCModel, EncDecHybridRNNTCTCModel
from nemo.collections.asr.modules.conformer_encoder import ConformerChangeConfig
# from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
# from nemo.collections.asr.parts.submodules.multitask_decoding import MultiTaskDecoding, MultiTaskDecodingConfig
# from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
# from nemo.collections.asr.parts.utils.eval_utils import cal_write_wer
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.transcribe_utils import (
    compute_output_filename,
    prepare_audio_data,
    # read_and_maybe_sort_manifest,
    # restore_transcription_order,
    setup_model,
    transcribe_partial_audio,
    write_transcription,
)
from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.core.config import hydra_runner
from nemo.utils import logging
"""
Example of end-to-end diarization inference 
"""

seed_everything(42)

@dataclass
class ModelChangeConfig:

    # Sub-config for changes specific to the Conformer Encoder
    conformer: ConformerChangeConfig = ConformerChangeConfig()


@dataclass
class TranscriptionConfig:
    # Required configs
    model_path: Optional[str] = None  # Path to a .nemo file
    pretrained_name: Optional[str] = None  # Name of a pretrained model
    audio_dir: Optional[str] = None  # Path to a directory which contains audio files
    tensor_image_dir: Optional[str] = None  # Path to a directory which contains tensor images
    dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest
    channel_selector: Optional[
        Union[int, str]
    ] = None  # Used to select a single channel from multichannel audio, or use average across channels
    audio_key: str = 'audio_filepath'  # Used to override the default audio key in dataset_manifest
    eval_config_yaml: Optional[str] = None  # Path to a yaml file of config of evaluation
    presort_manifest: bool = True  # Significant inference speedup on short-form data due to padding reduction

    # General configs
    output_filename: Optional[str] = None
    batch_size: int = 4
    num_workers: int = 0
    append_pred: bool = False  # Sets mode of work, if True it will add new field transcriptions.
    pred_name_postfix: Optional[str] = None  # If you need to use another model name, rather than standard one.
    random_seed: Optional[int] = None  # seed number going to be used in seed_everything()

    # Set to True to output greedy timestamp information (only supported models)
    compute_timestamps: bool = False
    # set to True if need to return full alignment information
    preserve_alignment: bool = False

    # Set to True to output language ID information
    compute_langs: bool = False

    # Set `cuda` to int to define CUDA device. If 'None', will look for CUDA
    # device anyway, and do inference on CPU only if CUDA device is not found.
    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    allow_mps: bool = False  # allow to select MPS device (Apple Silicon M-series GPU)
    amp: bool = False
    amp_dtype: str = "float16"  # can be set to "float16" or "bfloat16" when using amp
    matmul_precision: str = "highest"  # Literal["highest", "high", "medium"]
    audio_type: str = "wav"

    # Recompute model transcription, even if the output folder exists with scores.
    overwrite_transcripts: bool = True


    # decoder type: ctc or rnnt, can be used to switch between CTC and RNNT decoder for Hybrid RNNT/CTC models
    decoder_type: Optional[str] = None
    # att_context_size can be set for cache-aware streaming models with multiple look-aheads
    att_context_size: Optional[list] = None

    # Config for word / character error rate calculation
    calculate_wer: bool = True
    clean_groundtruth_text: bool = False
    langid: str = "en"  # specify this for convert_num_to_words step in groundtruth cleaning
    use_cer: bool = False

    # can be set to True to return list of transcriptions instead of the config
    # if True, will also skip writing anything to the output file
    return_transcriptions: bool = False

    # Set to False to return text instead of hypotheses from the transcribe function, so as to save memory
    return_hypotheses: bool = True

    # key for groundtruth text in manifest
    gt_text_attr_name: str = "text"
    gt_lang_attr_name: str = "lang"

    # Use model's transcribe() function instead of transcribe_partial_audio() by default
    # Only use transcribe_partial_audio() when the audio is too long to fit in memory
    # Your manifest input should have `offset` field to use transcribe_partial_audio()
    allow_partial_transcribe: bool = False


@hydra_runner(config_name="TranscriptionConfig", schema=TranscriptionConfig)
def main(cfg: TranscriptionConfig) -> Union[TranscriptionConfig]:
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

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

    # Load augmentor from exteranl yaml file which contains eval info, could be extend to other feature such VAD, P&C
    augmentor = None
    if cfg.eval_config_yaml:
        eval_config = OmegaConf.load(cfg.eval_config_yaml)
        augmentor = eval_config.test_ds.get("augmentor")
        logging.info(f"Will apply on-the-fly augmentation on samples during transcription: {augmentor} ")

    # setup GPU
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    if cfg.cuda is None:
        if torch.cuda.is_available():
            device = [0]  # use 0th CUDA device
            accelerator = 'gpu'
            map_location = torch.device('cuda:0')
        elif cfg.allow_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logging.warning(
                "MPS device (Apple Silicon M-series GPU) support is experimental."
                " Env variable `PYTORCH_ENABLE_MPS_FALLBACK=1` should be set in most cases to avoid failures."
            )
            device = [0]
            accelerator = 'mps'
            map_location = torch.device('mps')
        else:
            device = 1
            accelerator = 'cpu'
            map_location = torch.device('cpu')
    else:
        device = [cfg.cuda]
        accelerator = 'gpu'
        map_location = torch.device(f'cuda:{cfg.cuda}')

    logging.info(f"Inference will be done on device: {map_location}")

    # diar_model, model_name = setup_model(cfg, map_location)
    # diar_model = SortformerEncLabelModel.restore_from(restore_path=cfg.model_path)
    diar_model = SortformerEncLabelModel.load_from_checkpoint(checkpoint_path=cfg.model_path, map_location=map_location)
    # diar_model.cfg_e2e_diarizer_model.tensor_image_dir 
    diar_model._cfg.diarizer.out_dir = cfg.tensor_image_dir
    trainer = pl.Trainer(devices=device, accelerator=accelerator)
    diar_model.set_trainer(trainer)
    diar_model = diar_model.eval()
    diar_model._cfg.test_ds.manifest_filepath = cfg.dataset_manifest
    diar_model._cfg.test_ds.batch_size = cfg.batch_size
    diar_model.setup_test_data(test_data_config=diar_model._cfg.test_ds)    
    trainer.test(diar_model)

if __name__ == '__main__':
    
    main()
