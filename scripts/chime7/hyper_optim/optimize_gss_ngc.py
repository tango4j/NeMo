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

import argparse
import logging
import os
import tempfile
import time
from multiprocessing import Process
import torch

import optuna

from nemo.utils import logging as nemo_logger


# NGC workspace: nemo_asr_eval
NGC_WS_MOUNT="/ws"
ESPNET_ROOT="/workspace/espnet/egs2/chime7_task1/asr1"
NEMO_CHIME7_ROOT=f"{NGC_WS_MOUNT}/nemo-gitlab-chime7/scripts/chime7"
CHIME7_ROOT=f"{NGC_WS_MOUNT}/chime7_official_cleaned_v2"
ASR_MODEL_PATH=f"{NGC_WS_MOUNT}/model_checkpoints/rno_chime7_chime6_ft_ptDataSetasrset3_frontend_nemoGSSv1_prec32_layers24_heads8_conv5_d1024_dlayers2_dsize640_bs128_adamw_CosineAnnealing_lr0.0001_wd1e-2_spunigram1024.nemo"
# DIAR_BASE_DIR=f"{NGC_WS_MOUNT}/chime7_diar_results"
DIAR_BASE_DIR=f"{NGC_WS_MOUNT}/chime7_outputs/optuna-msdd-gss-asr2-trial315"

# DIAR_CONFIG="system_B_V05_D03"
DIAR_CONFIG="sys-B-V07"
DIAR_PARAM="T"

SCENARIOS = "chime6 dipco mixer6" # chime6 dipco mixer6
SUBSETS = "dev"

def scale_weights(r, K):
    return [r - kvar * (r - 1) / (K - 1) for kvar in range(K)]

def get_gss_command(gpu_id, diar_config, diar_param, diar_base_dir, output_dir, mc_mask_min_db, mc_postmask_min_db,
                    bss_iterations, dereverb_filter_length, top_k, scenarios=SCENARIOS, subsets=SUBSETS, 
                    dereverb_prediction_delay=2, dereverb_num_iterations=3, mc_filter_type="pmwf", mc_filter_postfilter="ban"):
    command = f"MAX_SEGMENT_LENGTH=1000 MAX_BATCH_DURATION=20 BSS_ITERATION={bss_iterations} MC_MASK_MIN_DB={mc_mask_min_db} MC_POSTMASK_MIN_DB={mc_postmask_min_db} DEREVERB_FILTER_LENGTH={dereverb_filter_length} TOK_K={top_k}" \
              f" dereverb_prediction_delay={dereverb_prediction_delay} dereverb_num_iterations={dereverb_num_iterations} mc_filter_type={mc_filter_type} mc_filter_postfilter={mc_filter_postfilter} " \
              f" {NEMO_CHIME7_ROOT}/process/run_processing.sh '{scenarios}' " \
              f" {gpu_id} {diar_config} {diar_param} {diar_base_dir} {output_dir} " \
              f" {ESPNET_ROOT} {CHIME7_ROOT} {NEMO_CHIME7_ROOT} '{subsets}'"
              
    return command


def get_asr_eval_command(gpu_id, diar_config, diar_param, normalize_db, output_dir, scenarios=SCENARIOS, subsets=SUBSETS):
    command = f"EVAL_CHIME=True {NEMO_CHIME7_ROOT}/evaluation/run_asr.sh '{scenarios}' '{subsets}' " \
              f"{diar_config}-{diar_param} {output_dir}/processed {output_dir} {normalize_db} {ASR_MODEL_PATH} 1 4 {CHIME7_ROOT} {NEMO_CHIME7_ROOT} {gpu_id}"

    return command


def objective_gss_asr(
        trial: optuna.Trial,
        gpu_id: int,
        temp_dir: str,
        diar_config: str = DIAR_CONFIG,
        diar_param: str = DIAR_PARAM,
        diar_base_dir: str = DIAR_BASE_DIR,
        scenarios: str = SCENARIOS,
        subsets: str = SUBSETS,
):
    # Existing parameters
    mc_mask_min_db = trial.suggest_categorical("mc_mask_min_db", choices=[-200, -60])
    mc_postmask_min_db = trial.suggest_int("mc_postmask_min_db", -18, 0, 3)
    bss_iterations = trial.suggest_int("bss_iterations", 5, 20, 5)
    dereverb_filter_length = trial.suggest_int("dereverb_filter_length", 5, 15, 5)
    normalize_db = trial.suggest_int("normalize_db", -35, -20, 5)
    top_k = trial.suggest_int("top_k", 50, 100, 10)

    # New parameters
    dereverb_prediction_delay = trial.suggest_categorical("dereverb_prediction_delay", choices=[2, 3])
    dereverb_num_iterations = trial.suggest_categorical("dereverb_num_iterations", choices=[3, 5, 10])
    mc_filter_type = trial.suggest_categorical("mc_filter_type", choices=['pmwf', 'wmpdr'])
    mc_filter_postfilter = trial.suggest_categorical("mc_filter_postfilter", choices=['ban', 'None'])
    start_time = time.time()

    with tempfile.TemporaryDirectory(dir=temp_dir, prefix=str(trial.number)) as output_dir:
        command_gss = get_gss_command(
            gpu_id, 
            diar_config, 
            diar_param, 
            diar_base_dir, 
            output_dir, 
            mc_mask_min_db, 
            mc_postmask_min_db,  
            bss_iterations, 
            dereverb_filter_length, 
            top_k, 
            scenarios, 
            subsets, 
            dereverb_prediction_delay, 
            dereverb_num_iterations, 
            mc_filter_type, 
            mc_filter_postfilter
        )
        code = os.system(command_gss)
        if code != 0:
            raise RuntimeError(f"command failed: {command_gss}")
        command_asr = get_asr_eval_command(gpu_id, diar_config, diar_param, normalize_db, output_dir, scenarios, subsets)
        code = os.system(command_asr)
        if code != 0:
            raise RuntimeError(f"command failed: {command_asr}")
        eval_results = os.path.join(output_dir, f"eval_results_{diar_config}-{diar_param}_chime6_ft_rnnt_ln{normalize_db}/macro_wer.txt")
        with open(eval_results, "r") as f:
            wer = float(f.read().strip())
        print(f"WER={wer}. Time taken for trial {trial.number}: {time.time() - start_time:.2f}s")
        print(f"mc_mask_min_db={mc_mask_min_db}, mc_postmask_min_db={mc_postmask_min_db}, bss_iterations={bss_iterations}, dereverb_filter_length={dereverb_filter_length}, normalize_db={normalize_db}, top_k={top_k}")
    return wer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_name", help="Name of study.", type=str, default="optuna_chime7")
    parser.add_argument("--storage", help="Shared storage (i.e sqlite:///optuna.db).", type=str, default="sqlite:///optuna-gss-dev.db")
    parser.add_argument("--temp_dir", help="path to store temporary files", type=str, default="/raid/temp")
    parser.add_argument("--output_log", help="Where to store optuna output log", type=str, default="output.log")
    parser.add_argument(
        "--collar",
        help="collar used to be more forgiving at boundaries (usually set to 0 or 0.25)",
        type=float,
        default=0.0,
    )
    parser.add_argument("--n_trials", help="Number of trials to run optuna", type=int, default=100)
    parser.add_argument("--n_jobs", help="Number of parallel jobs to run, set -1 to use all GPUs", type=int, default=-1)
    parser.add_argument("--scenarios", help="Scenarios to run on", type=str, default=SCENARIOS)
    parser.add_argument("--subsets", help="Subsets to run on", type=str, default=SUBSETS)

    args = parser.parse_args()

    os.makedirs(args.temp_dir, exist_ok=True)

    nemo_logger.setLevel(logging.ERROR)

    func = lambda trial, gpu_id: objective_gss_asr(
        trial,
        gpu_id,
        temp_dir=args.temp_dir,
        scenarios=args.scenarios,
        subsets=args.subsets,
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
    print(f"Running {args.n_trials} trials on {n_jobs} GPUs")

    for i in range(0, n_jobs):
        p = Process(target=optimize, args=(i,))
        processes.append(p)
        p.start()
        time.sleep(5)

    for t in processes:
        t.join()

    study = optuna.load_study(
        study_name=args.study_name,
        storage=args.storage,
    )
    print(f"Best SA-WER {study.best_value}")
    print(f"Best Parameter Set: {study.best_params}")
