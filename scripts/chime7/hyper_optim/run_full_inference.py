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

"""Optimize the Neural Diarizer hyper-parameters onto your dev set using Optuna."""

import gc
import glob
import argparse
import logging
import os
import shutil
import tempfile
import time
from typing import List
from multiprocessing import Process
import torch
from pathlib import Path
import numpy as np
import optuna
import wget
from omegaconf import OmegaConf

from nemo.collections.asr.models.msdd_v2_models import NeuralDiarizer
from nemo.utils import logging as nemo_logger


ESPNET_ROOT = "/home/heh/github/espnet/egs2/chime7_task1/asr1"
CHIME7_ROOT = "/media/data2/chime7-challenge/datasets/chime7_official_cleaned_v2"
NEMO_CHIME7_ROOT = "/media/data2/chime7-challenge/nemo-gitlab-chime7/scripts/chime7"

ASR_MODEL_PATH = "/media/data2/chime7-challenge/nemo_asr_chime6_finetuned_rnnt/checkpoints/rno_chime7_chime6_ft_ptDataSetasrset3_frontend_nemoGSSv1_prec32_layers24_heads8_conv5_d1024_dlayers2_dsize640_bs128_adamw_CosineAnnealing_lr0.0001_wd1e-2_spunigram1024.nemo"
MSDD_MODEL_PATH = "/home/heh/nemo_asr_eval/chime7/checkpoints/msdd_v2_PALO_bs6_a003_version6_e53.ckpt"
VAD_MODEL_PATH = "/home/heh/nemo_asr_eval/chime7/checkpoints/frame_vad_chime7_acrobat.nemo"


###### Default ######
MANIFEST_PATTERN = "*-dev.json"
SCENARIOS = "chime6 dipco mixer6" 
DIAR_CONFIG="system_B_V05_D03"
DIAR_PARAM="T"
ASR_TAG="asr"
####################


def get_gss_command(gpu_id, diar_config, diar_base_dir, output_dir, mc_mask_min_db, mc_postmask_min_db,
                    bss_iterations, dereverb_filter_length):
    command = f"MAX_SEGMENT_LENGTH=200 MAX_BATCH_DURATION=200 BSS_ITERATION={bss_iterations} MC_MASK_MIN_DB={mc_mask_min_db} MC_POSTMASK_MIN_DB={mc_postmask_min_db} DEREVERB_FILTER_LENGTH={dereverb_filter_length} " \
              f" {NEMO_CHIME7_ROOT}/process/run_processing.sh '{SCENARIOS}' " \
              f" {gpu_id} {diar_config} {DIAR_PARAM} {diar_base_dir} {output_dir} " \
              f" {ESPNET_ROOT} {CHIME7_ROOT} {NEMO_CHIME7_ROOT}"
              
    return command


def get_asr_eval_command(gpu_id, diar_config, normalize_db, output_dir, asr_model_path=ASR_MODEL_PATH):
    command = f"EVAL_CHIME=True {NEMO_CHIME7_ROOT}/evaluation/run_asr.sh '{SCENARIOS}' dev " \
              f"{diar_config}-{DIAR_PARAM} {output_dir}/processed {output_dir} {normalize_db} {asr_model_path} 1 4 {CHIME7_ROOT} {NEMO_CHIME7_ROOT} {gpu_id} {ASR_TAG}"

    return command


def run_gss_asr(
    gpu_id: int,
    output_dir: str,
    diar_config: str,
    diar_base_dir: str,
    mc_mask_min_db: float = -60.0,
    mc_postmask_min_db: float = -9.0,
    bss_iterations: int = 5,
    dereverb_filter_length: int = 5,
    normalize_db: float = -25,
    asr_model_path: str = ASR_MODEL_PATH,
):
    start_time = time.time()
    command_gss = get_gss_command(gpu_id, diar_config, diar_base_dir, output_dir, mc_mask_min_db, mc_postmask_min_db,  bss_iterations, dereverb_filter_length)
    code = os.system(command_gss)
    if code != 0:
        raise RuntimeError(f"command failed: {command_gss}")
    command_asr = get_asr_eval_command(gpu_id, diar_config, normalize_db, output_dir, asr_model_path)
    code = os.system(command_asr)
    if code != 0:
        raise RuntimeError(f"command failed: {command_asr}")
    eval_results = os.path.join(output_dir, f"eval_results_{diar_config}-{DIAR_PARAM}_{ASR_TAG}_ln{normalize_db}/macro_wer.txt")
    with open(eval_results, "r") as f:
        wer = float(f.read().strip())
    print(f"Time taken for GSS-ASR: {time.time() - start_time:.2f}s")

    return wer


def move_diar_results(diar_out_dir, system_name, scenario="chime6"):
    curr_diar_out_dir = Path(diar_out_dir, scenario, system_name, f"pred_jsons_{DIAR_PARAM}")
    new_diar_out_dir = Path(diar_out_dir, system_name, scenario, f"pred_jsons_{DIAR_PARAM}")
    new_diar_out_dir.mkdir(parents=True, exist_ok=True)
    for json_file in curr_diar_out_dir.glob("*.json"):
        os.rename(json_file, Path(new_diar_out_dir, json_file.name))
    return diar_out_dir


def scale_weights(r, K):
    return [r - kvar * (r - 1) / (K - 1) for kvar in range(K)]

def run_chime7_mcmsasr(
    config,
    diarizer_manifest_path: str,
    msdd_model_path: str,
    vad_model_path: str,
    asr_model_path: str,
    output_dir: str,
    gpu_id: int,
    mc_mask_min_db: float = -60.0,
    mc_postmask_min_db: float = -9.0,
    bss_iterations: int = 5,
    dereverb_filter_length: int = 5,
    normalize_db: float = -25,
    keep_speaker_output: bool = True,
    manifest_pattern: str = MANIFEST_PATTERN,
):
    """
    [Note] Diarizaiton out `outputs`
    
    outputs = (metric, mapping_dict, itemized_erros)
    itemized_errors = (DER, CER, FA, MISS)
    """
    speaker_output_dir = os.path.join(output_dir, "speaker_outputs")
    Path(speaker_output_dir).mkdir(parents=True, exist_ok=True)
    
    for manifest_json in Path(diarizer_manifest_path).glob(manifest_pattern):
        scenario = manifest_json.stem.split("-")[0]
        print(f"Start Diarization on {manifest_json}")
        curr_output_dir = os.path.join(output_dir, scenario)
        Path(curr_output_dir).mkdir(parents=True, exist_ok=True)

        # Step:1-1 Configure Diarization
        config.diarizer.vad.model_path = vad_model_path
        config.diarizer.msdd_model.model_path = msdd_model_path
        config.diarizer.oracle_vad = False
        config.diarizer.msdd_model.parameters.diar_eval_settings = [
            (config.diarizer.collar, config.diarizer.ignore_overlap),
        ]
        config.diarizer.manifest_filepath = diarizer_manifest_path
        config.diarizer.out_dir = curr_output_dir  # Directory to store intermediate files and prediction outputs
        config.diarizer.speaker_out_dir = speaker_output_dir if speaker_output_dir else output_dir # Directory to store speaker embeddings
        config.prepared_manifest_vad_input = os.path.join(output_dir, 'manifest_vad.json')
        config.diarizer.clustering.parameters.oracle_num_speakers = False
        r_value = config.diarizer.speaker_embeddings.parameters.r_value
        scale_n = len(config.diarizer.speaker_embeddings.parameters.multiscale_weights)
        config.diarizer.speaker_embeddings.parameters.multiscale_weights = scale_weights(r_value, scale_n)

        # Step:1-2 Run Diarization 
        diarizer_model = NeuralDiarizer(cfg=config).to(f"cuda:{gpu_id}")
        diarizer_model.diarize(verbose=False)
        
        move_diar_results(output_dir, config.diarizer.msdd_model.parameters.system_name, scenario=scenario)
        
        del diarizer_model
        if not keep_speaker_output:
            shutil.rmtree(speaker_output_dir)
    
    print("Start GSS-ASR")
    WER = run_gss_asr(
        gpu_id=gpu_id,
        output_dir=output_dir,
        diar_base_dir=output_dir,
        diar_config=config.diarizer.msdd_model.parameters.system_name,
        mc_mask_min_db=mc_mask_min_db,
        mc_postmask_min_db=mc_postmask_min_db,
        bss_iterations=bss_iterations,
        dereverb_filter_length=dereverb_filter_length,
        normalize_db=normalize_db,
        asr_model_path=asr_model_path,
    )

    return WER


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
   
    parser.add_argument("--manifest_path", help="path to the manifest dir", type=str)
    parser.add_argument(
        "--config_url",
        help="path to the config yaml file to use",
        type=str,
        default="./chime7_full_infer.yaml",
    )
    parser.add_argument(
        "--vad_model_path",
        help="path to the VAD model",
        type=str,
        default=VAD_MODEL_PATH,
    )
    parser.add_argument(
        "--msdd_model_path",
        help="path to the Neural Diarizer model",
        type=str,
        default=MSDD_MODEL_PATH,
    )
    parser.add_argument(
        "--asr_model_path",
        help="path to the Neural ASR model",
        type=str,
        default=ASR_MODEL_PATH,
    )
    parser.add_argument("--output_dir", help="path to store output files", type=str, default="./outputs")
    parser.add_argument("--batch_size", help="Batch size for mc-embedings and MSDD", type=int, default=8)
    parser.add_argument("--gpu", help="GPU ID to use", type=int, default=0)
    parser.add_arguement("--pattern", help="manfiest patterns for glob", type=str, default=MANIFEST_PATTERN)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    nemo_logger.setLevel(logging.ERROR)
    model_config = args.config_url
    config = OmegaConf.load(model_config)
    config.batch_size = args.batch_size
    
    # func = lambda trial, gpu_id: objective_gss_asr(
    wer = run_chime7_mcmsasr(
        config=config,
        gpu_id=args.gpu,
        temp_dir=args.temp_dir,
        diarizer_manifest_path=args.manifest_path,
        msdd_model_path=args.msdd_model_path,
        vad_model_path=args.vad_model_path,
        asr_model_path=args.asr_model_path,
        speaker_output_dir=args.output_dir,
        mc_mask_min_db=config.gss.mc_mask_min_db,
        mc_postmask_min_db=config.gss.mc_postmask_min_db,
        bss_iterations=config.gss.bss_iterations,
        dereverb_filter_length=config.gss.dereverb_filter_length,
        normalize_db=config.diarizer.asr.parameters.normalize_db,
        keep_speaker_output=config.keep_speaker_output,
        manifest_pattern=args.pattern,
    )

    logging.info(f"SA-WER: {wer}")
