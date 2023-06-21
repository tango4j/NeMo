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

from pathlib import Path
import glob
import argparse
import logging
import os
import tempfile
import time
from typing import List
from multiprocessing import Process
import torch

import numpy as np
import optuna
import wget
from omegaconf import OmegaConf

from nemo.collections.asr.models.msdd_v2_models import NeuralDiarizer
from nemo.utils import logging as nemo_logger
from optimize_diar import diar_config_setup


NGC_WS_MOUNT="/ws"
ESPNET_ROOT="/workspace/espnet/egs2/chime7_task1/asr1"
NEMO_CHIME7_ROOT=f"{NGC_WS_MOUNT}/chime7"
CHIME7_ROOT=f"{NGC_WS_MOUNT}/chime7_official_cleaned_v2"
ASR_MODEL_PATH=f"{NGC_WS_MOUNT}/model_checkpoints/rno_chime7_chime6_ft_ptDataSetasrset3_frontend_nemoGSSv1_prec32_layers24_heads8_conv5_d1024_dlayers2_dsize640_bs128_adamw_CosineAnnealing_lr0.0001_wd1e-2_spunigram1024.nemo"

SCENARIOS = "chime6 dipco mixer6" # chime6 dipco mixer6


def get_gss_command(gpu_id, diar_config, diar_param, diar_base_dir, output_dir, mc_mask_min_db, mc_postmask_min_db,
                    bss_iterations, dereverb_filter_length):
    command = f"MAX_SEGMENT_LENGTH=40 MAX_BATCH_DURATION=40 BSS_ITERATION={bss_iterations} MC_MASK_MIN_DB={mc_mask_min_db} MC_POSTMASK_MIN_DB={mc_postmask_min_db} DEREVERB_FILTER_LENGTH={dereverb_filter_length} " \
              f" {NEMO_CHIME7_ROOT}/process/run_processing.sh '{SCENARIOS}' " \
              f" {gpu_id} {diar_config} {diar_param} {diar_base_dir} {output_dir} " \
              f" {ESPNET_ROOT} {CHIME7_ROOT} {NEMO_CHIME7_ROOT}"
              
    return command


def get_asr_eval_command(gpu_id, diar_config, diar_param, normalize_db, output_dir):
    command = f"EVAL_CHIME=True {NEMO_CHIME7_ROOT}/evaluation/run_asr.sh '{SCENARIOS}' dev " \
              f"{diar_config}-{diar_param} {output_dir}/processed {output_dir} {normalize_db} {ASR_MODEL_PATH} 1 4 {CHIME7_ROOT} {NEMO_CHIME7_ROOT} {gpu_id}"

    return command


def objective_gss_asr(
        trial: optuna.Trial,
        gpu_id: int,
        output_dir: str,
        diar_config: str,
        diar_base_dir: str,
        diar_param: str = "T",
):
    mc_mask_min_db = trial.suggest_int("mc_mask_min_db", -60, -5, 10)
    mc_postmask_min_db = trial.suggest_int("mc_postmask_min_db", -9, 0, 3)
    bss_iterations = trial.suggest_int("bss_iterations", 5, 30, 5)
    dereverb_filter_length = trial.suggest_int("dereverb_filter_length", 5, 20, 5)
    normalize_db = trial.suggest_int("normalize_db", -35, -20, 5)
    
    command_gss = get_gss_command(gpu_id, diar_config, diar_param, diar_base_dir, output_dir, mc_mask_min_db, mc_postmask_min_db,  bss_iterations, dereverb_filter_length)
    code = os.system(command_gss)
    if code != 0:
        raise RuntimeError(f"command failed: {command_gss}")
    command_asr = get_asr_eval_command(gpu_id, diar_config, diar_param, normalize_db, output_dir)
    code = os.system(command_asr)
    if code != 0:
        raise RuntimeError(f"command failed: {command_asr}")
    eval_results = os.path.join(output_dir, f"eval_results_{diar_config}-{diar_param}_chime6_ft_rnnt_ln{normalize_db}/macro_wer.txt")
    with open(eval_results, "r") as f:
        wer = float(f.read().strip())
    
    logging.info(f"-------------WER={wer:.4f}--------------------")
    logging.info(f"Trial: {trial.number}")
    logging.info(f"mc_mask_min_db: {mc_mask_min_db}, mc_postmask_min_db: {mc_postmask_min_db}, bss_iterations: {bss_iterations}, dereverb_filter_length: {dereverb_filter_length}, normalize_db: {normalize_db}")
    logging.info("-----------------------------------------------")

    return wer

def move_diar_results(diar_out_dir, system_name, scenario="chime6"):
    curr_diar_out_dir = Path(diar_out_dir, scenario, system_name, "pred_jsons_T")
    new_diar_out_dir = Path(diar_out_dir, system_name, scenario, "pred_jsons_T")
    new_diar_out_dir.mkdir(parents=True, exist_ok=True)
    for json_file in curr_diar_out_dir.glob("*.json"):
        os.rename(json_file, Path(new_diar_out_dir, json_file.name))
    return diar_out_dir

def objective_chime7_mcmsasr(
    trial: optuna.Trial,
    config,
    diarizer_manifest_path: str,
    msdd_model_path: str,
    vad_model_path: str,
    speaker_output_dir: str,
    gpu_id: int,
    temp_dir: str,
    keep_mixer6: bool = False,
):
    """
    [Note] Diarizaiton out `outputs`
    
    outputs = (metric, mapping_dict, itemized_erros)
    itemized_errors = (DER, CER, FA, MISS)
    """
    start_time = time.time()
    with tempfile.TemporaryDirectory(dir=temp_dir, prefix=str(trial.number)) as output_dir:
        with tempfile.TemporaryDirectory(dir="/workspace", prefix=str(trial.number)) as local_output_dir:
            for manifest_json in Path(diarizer_manifest_path).glob("*-dev.json"):
                logging.info(f"Start Diarization on {manifest_json}")
                scenario = manifest_json.stem.split("-")[0]
                curr_output_dir = os.path.join(output_dir, scenario)
                Path(curr_output_dir).mkdir(parents=True, exist_ok=True)

                if "mixer6" in scenario and not keep_mixer6:  # don't save speaker outputs for mixer6
                    curr_speaker_output_dir = os.path.join(local_output_dir, "speaker_outputs")
                else:
                    curr_speaker_output_dir = speaker_output_dir

                config.device = f"cuda:{gpu_id}"
                # Step:1-1 Configure Diarization
                config = diar_config_setup(
                    trial, 
                    config,
                    str(manifest_json),
                    msdd_model_path,
                    vad_model_path,
                    output_dir=curr_output_dir,
                    speaker_output_dir=curr_speaker_output_dir,
                    tune_vad=True,
                )
                # Step:1-2 Run Diarization 
                diarizer_model = NeuralDiarizer(cfg=config).to(f"cuda:{gpu_id}")
                outputs = diarizer_model.diarize(verbose=False)
                move_diar_results(output_dir, config.diarizer.msdd_model.parameters.system_name, scenario=scenario)

                metric = outputs[0][0]
                DER = abs(metric)
                logging.info(f"[optuna] Diarization DER: {DER}")
                del diarizer_model

        WER = objective_gss_asr(
            trial,
            gpu_id,
            output_dir,
            diar_base_dir=output_dir,
            diar_config=config.diarizer.msdd_model.parameters.system_name,
        )
    logging.info(f"Time taken for trial {trial.number}: {(time.time() - start_time)/60:.2f} mins")    
    return WER

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_name", help="Name of study.", type=str, default="optuna_chime7")
    parser.add_argument("--storage", help="Shared storage (i.e sqlite:///testDB.db).", type=str, default="sqlite:///optuna-msdd-gss-asr.db")
    parser.add_argument("--manifest_path", help="path to the manifest file", type=str)
    parser.add_argument(
        "--config_url",
        help="path to the config yaml file to use",
        type=str,
        default="https://raw.githubusercontent.com/NVIDIA/NeMo/msdd_scripts_docs/"
        "examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml",
    )
    parser.add_argument(
        "--vad_model_path",
        help="path to the VAD model",
        type=str,
        default="vad_multilingual_marblenet",
    )
    parser.add_argument(
        "--msdd_model_path",
        help="path to the Neural Diarizer model",
        type=str,
        default="diar_msdd_telephonic",
    )
    parser.add_argument("--temp_dir", help="path to store temporary files", type=str, default="temp/")
    parser.add_argument("--output_dir", help="path to store temporary files", type=str, default="speaker_outputs/")
    parser.add_argument("--output_log", help="Where to store optuna output log", type=str, default="output.log")
    parser.add_argument("--n_trials", help="Number of trials to run optuna", type=int, default=100)
    parser.add_argument("--n_jobs", help="Number of parallel jobs to run, set -1 to use all GPUs", type=int, default=-1)
    parser.add_argument("--batch_size", help="Batch size for mc-embedings and MSDD", type=int, default=8)
    parser.add_argument("--keep_mixer6", help="Keep mixer6 in the evaluation", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.temp_dir, exist_ok=True)

    nemo_logger.setLevel(logging.ERROR)
    model_config = args.config_url
    config = OmegaConf.load(model_config)
    config.batch_size = args.batch_size
    
    # func = lambda trial, gpu_id: objective_gss_asr(
    func = lambda trial, gpu_id: objective_chime7_mcmsasr(
        trial=trial,
        config=config,
        gpu_id=gpu_id,
        temp_dir=args.temp_dir,
        diarizer_manifest_path=args.manifest_path,
        msdd_model_path=args.msdd_model_path,
        vad_model_path=args.vad_model_path,
        speaker_output_dir=args.output_dir,
        keep_mixer6=args.keep_mixer6,
    )

    def optimize(gpu_id=0):
        worker_func = lambda trial: func(trial, gpu_id)

        study = optuna.create_study(
            direction="minimize", study_name=args.study_name, storage=args.storage, load_if_exists=True
        )
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # Setup the root logger.
        logger.addHandler(logging.FileHandler(args.output_log, mode="a"))
        logger.addHandler(logging.StreamHandler())
        optuna.logging.enable_propagation()  # Propagate logs to the root logger.
        study.optimize(worker_func, n_trials=args.n_trials, show_progress_bar=True)

    processes = []
    if args.n_jobs == -1:
        args.n_jobs = torch.cuda.device_count()
    n_jobs = min(args.n_jobs, torch.cuda.device_count())
    logging.info(f"Running {args.n_trials} trials on {n_jobs} GPUs")

    for i in range(0, n_jobs):
        p = Process(target=optimize, args=(i,))
        processes.append(p)
        p.start()

    for t in processes:
        t.join()

    study = optuna.load_study(
        study_name=args.study_name,
        storage=args.storage,
    )
    logging.info(f"Best SA-WER {study.best_value}")
    logging.info(f"Best Parameter Set: {study.best_params}")
