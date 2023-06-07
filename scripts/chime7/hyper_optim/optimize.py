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

import numpy as np
import optuna
import wget
from omegaconf import OmegaConf
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from nemo.utils import logging as nemo_logger

ESPNET_ROOT="/home/heh/github/espnet/egs2/chime7_task1/asr1"
CHIME7_ROOT="/media/data2/chime7-challenge/datasets/chime7_official_cleaned_v2"
NEMO_CHIME7_ROOT="/media/data2/chime7-challenge/nemo-gitlab-chime7/scripts/chime7"
ASR_MODEL_PATH="/media/data2/chime7-challenge/nemo_asr_chime6_finetuned_rnnt/checkpoints/rno_chime7_chime6_ft_ptDataSetasrset3_frontend_nemoGSSv1_prec32_layers24_heads8_conv5_d1024_dlayers2_dsize640_bs128_adamw_CosineAnnealing_lr0.0001_wd1e-2_spunigram1024.nemo"

DIAR_CONFIG="system_B_V05_D03"
DIAR_PARAM="T0.5"
DIAR_BASE_DIR="/media/data2/chime7-challenge/chime7_diar_results"

SCENARIOS = "chime6 dipco mixer6"

def scale_weights(r, K):
    return [r - kvar * (r - 1) / (K - 1) for kvar in range(K)]

def get_gss_command(gpu_id, diar_config, diar_param, diar_base_dir, output_dir, mc_mask_min_db, mc_postmask_min_db,
                    bss_iterations, dereverb_filter_length):
    command = f"BSS_ITERATION={bss_iterations} MC_MASK_MIN_DB={mc_mask_min_db} MC_POSTMASK_MIN_DB={mc_postmask_min_db} DEREVERB_FILTER_LENGTH={dereverb_filter_length} " \
              f" {NEMO_CHIME7_ROOT}/process/run_processing.sh {SCENARIOS} " \
              f" {gpu_id} {diar_config} {diar_param} {diar_base_dir} {output_dir} " \
              f" {ESPNET_ROOT} {CHIME7_ROOT} {NEMO_CHIME7_ROOT}"
              
    return command


def get_asr_eval_command(gpu_id, diar_config, diar_param, normalize_db, output_dir):
    command = f"EVAL_CHIME=True {NEMO_CHIME7_ROOT}/evaluation/run_asr.sh {SCENARIOS} dev " \
              f"{diar_config}-{diar_param} {output_dir}/processed {output_dir} {normalize_db} {ASR_MODEL_PATH} 1 4 {CHIME7_ROOT} {NEMO_CHIME7_ROOT} {gpu_id}"

    return command


def objective_gss_asr(
        trial: optuna.Trial,
        gpu_id: int,
        temp_dir: str,
        diar_config: str = DIAR_CONFIG,
        diar_param: str = DIAR_PARAM,
        diar_base_dir: str = DIAR_BASE_DIR,
):
    start_time = time.time()
    mc_mask_min_db = trial.suggest_int("mc_mask_min_db", -60, -5, 10)
    mc_postmask_min_db = trial.suggest_int("mc_postmask_min_db", -9, 0, 3)
    bss_iterations = trial.suggest_int("bss_iterations", 5, 30, 5)
    dereverb_filter_length = trial.suggest_int("dereverb_filter_length", 5, 20, 5)
    normalize_db = trial.suggest_int("normalize_db", -35, -20, 5)
    with tempfile.TemporaryDirectory(dir=temp_dir, prefix=str(trial.number)) as output_dir:
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
        print(f"Time taken for trial: {time.time() - start_time:.2f}s")
    return wer

def objective_diar(
    trial: optuna.Trial,
    gpu_id: int,
    manifest_path: str,
    config,
    pretrained_msdd_model: str,
    pretrained_speaker_model: str,
    pretrained_vad_model: str,
    temp_dir: str,
    collar: float,
    ignore_overlap: bool,
    oracle_vad: bool,
    optimize_segment_length: bool,
    optimize_clustering: bool,
):
    start_time = time.time()
    device = torch.device(f"cuda:{gpu_id}")
    with tempfile.TemporaryDirectory(dir=temp_dir, prefix=str(trial.number)) as output_dir:
        config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
        if not oracle_vad:
            config.diarizer.oracle_vad = False
            config.diarizer.clustering.parameters.oracle_num_speakers = False
            # Here, we use our in-house pretrained NeMo VAD model
            config.diarizer.vad.model_path = pretrained_vad_model
            config.diarizer.vad.parameters.onset = 0.8
            config.diarizer.vad.parameters.offset = 0.6
            config.diarizer.vad.parameters.pad_offset = -0.05

        config.diarizer.msdd_model.model_path = pretrained_msdd_model
        config.diarizer.msdd_model.parameters.sigmoid_threshold = [trial.suggest_float("sigmoid_threshold", 0.7, 1.0)]
        config.diarizer.msdd_model.parameters.diar_eval_settings = [
            (collar, ignore_overlap),
        ]
        config.diarizer.manifest_filepath = manifest_path
        config.diarizer.out_dir = output_dir  # Directory to store intermediate files and prediction outputs
        # todo: without setting this to 1, process hangs
        config.num_workers = 10  # Workaround for multiprocessing hanging
        config.prepared_manifest_vad_input = os.path.join(output_dir, 'manifest_vad.json')

        if optimize_segment_length:
            # Segmentation Optimization
            stt = trial.suggest_float("window_stt", 0.9, 2.5, step=0.05)
            end = trial.suggest_float("window_end", 0.25, 0.75, step=0.05)
            scale_n = trial.suggest_int("scale_n", 2, 11)
            _step = -1 * (stt - end) / (scale_n - 1)
            shift_ratio = 0.5
            window_mat = np.arange(stt, end - 0.01, _step).round(5)
            config.diarizer.speaker_embeddings.parameters.window_length_in_sec = window_mat.tolist()
            config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = (shift_ratio * window_mat / 2).tolist()
            config.diarizer.speaker_embeddings.parameters.multiscale_weights = [1] * scale_n
        if optimize_clustering:
            # Clustering Optimization
            r_value = round(trial.suggest_float("r_value", 0.0, 2.25, step=0.01), 4)
            max_rp_threshold = round(trial.suggest_float("max_rp_threshold", 0.05, 0.35, step=0.01), 2)
            sparse_search_volume = trial.suggest_int("sparse_search_volume", 2, 100, log=True)
            max_num_speakers = trial.suggest_int("max_num_speakers", 8, 27, step=1)
            config.diarizer.clustering.parameters.max_num_speakers = max_num_speakers
            config.diarizer.clustering.parameters.sparse_search_volume = sparse_search_volume
            config.diarizer.clustering.parameters.max_rp_threshold = max_rp_threshold
            scale_n = len(config.diarizer.speaker_embeddings.parameters.multiscale_weights)
            config.diarizer.speaker_embeddings.parameters.multiscale_weights = scale_weights(r_value, scale_n)

        system_vad_msdd_model = NeuralDiarizer(cfg=config).to(device)
        outputs = system_vad_msdd_model.diarize()
        # For each threshold, for each collar setting
        metric = outputs[0][0][0]
        DER = abs(metric)
        print(f"Time taken for trial: {time.time() - start_time:.2f}s")
        return DER


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_name", help="Name of study.", type=str, default="optuna_chime7")
    parser.add_argument("--storage", help="Shared storage (i.e sqlite:///testDB.db).", type=str, default="sqlite:///optuna.db")
    parser.add_argument("--manifest_path", help="path to the manifest file", type=str)
    parser.add_argument(
        "--config_url",
        help="path to the config yaml file to use",
        type=str,
        default="https://raw.githubusercontent.com/NVIDIA/NeMo/msdd_scripts_docs/"
        "examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml",
    )
    parser.add_argument(
        "--pretrained_speaker_model",
        help="path to the speaker embedding model",
        type=str,
        default="titanet_large",
    )
    parser.add_argument(
        "--pretrained_vad_model",
        help="path to the VAD model",
        type=str,
        default="vad_multilingual_marblenet",
    )
    parser.add_argument(
        "--pretrained_msdd_model",
        help="path to the Neural Diarizer model",
        type=str,
        default="diar_msdd_telephonic",
    )
    parser.add_argument("--temp_dir", help="path to store temporary files", type=str, default="optuna_output/")
    parser.add_argument("--output_log", help="Where to store optuna output log", type=str, default="output.log")
    parser.add_argument(
        "--collar",
        help="collar used to be more forgiving at boundaries (usually set to 0 or 0.25)",
        type=float,
        default=0.0,
    )
    parser.add_argument("--ignore_overlap", help="ignore overlap when evaluating", action="store_true")
    parser.add_argument("--n_trials", help="Number of trials to run optuna", type=int, default=100)
    parser.add_argument("--n_jobs", help="Number of parallel jobs to run, set -1 to use all GPUs", type=int, default=-1)
    parser.add_argument(
        "--oracle_vad",
        help="Enable oracle VAD (no need for VAD when enabled)",
        action="store_true",
    )
    parser.add_argument(
        "--optimize_clustering",
        help="Optimize clustering parameters",
        action="store_true",
    )
    parser.add_argument(
        "--optimize_segment_length",
        help="Optimize segment length parameters",
        action="store_true",
    )

    args = parser.parse_args()

    # if not (args.optimize_clustering or args.optimize_segment_length):
    #     raise MisconfigurationException(
    #         "When using the script you must pass one or both flags: --optimize_clustering, --optimize_segment_length"
    #     )

    os.makedirs(args.temp_dir, exist_ok=True)

    nemo_logger.setLevel(logging.ERROR)

    # model_config = os.path.basename(args.config_url)
    # if not os.path.exists(model_config):
    #     model_config = wget.download(args.config_url)
    # config = OmegaConf.load(model_config)

    func = lambda trial, gpu_id: objective_gss_asr(
        trial,
        gpu_id,
        temp_dir=args.temp_dir,
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
    print(f"Best SA-WER {study.best_value}")
    print(f"Best Parameter Set: {study.best_params}")
