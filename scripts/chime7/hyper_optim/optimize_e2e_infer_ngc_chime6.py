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
from pathlib import Path
from multiprocessing import Process
from typing import Optional

import torch
import optuna
from omegaconf import OmegaConf

from nemo.collections.asr.models.msdd_v2_models import NeuralDiarizer
from nemo.utils import logging as nemo_logger

# NGC workspace: nemo_asr_eval
NGC_WS_MOUNT="/ws"
ESPNET_ROOT="/workspace/espnet/egs2/chime7_task1/asr1"
NEMO_CHIME7_ROOT=f"{NGC_WS_MOUNT}/nemo-gitlab-chime7/scripts/chime7"
CHIME7_ROOT=f"{NGC_WS_MOUNT}/chime7_official_cleaned_v2"
# ASR_MODEL_PATH=f"{NGC_WS_MOUNT}/model_checkpoints/rno_chime7_chime6_ft_ptDataSetasrset3_frontend_nemoGSSv1_prec32_layers24_heads8_conv5_d1024_dlayers2_dsize640_bs128_adamw_CosineAnnealing_lr0.0001_wd1e-2_spunigram1024.nemo"
ASR_MODEL_PATH=f"{NGC_WS_MOUNT}/chime7/checkpoints/rnnt_ft_chime6ANDmixer6_26jun_avged.nemo"
LM_PATH=f"{NGC_WS_MOUNT}/chime7/checkpoints/rnnt_chime6_mixer6_dipco_train_dev.kenlm"

SCENARIOS = "chime6"
SUBSETS = "eval"

def scale_weights(r, K):
    return [r - kvar * (r - 1) / (K - 1) for kvar in range(K)]

def diar_config_setup(
    trial,
    config,
    diarizer_manifest_path: str,
    msdd_model_path: str,
    vad_model_path: str,
    output_dir: str,
    speaker_output_dir: Optional[str] = None,
    tune_vad: bool=True,
    ):

    config.diarizer.vad.model_path = vad_model_path
    config.diarizer.msdd_model.model_path = msdd_model_path
    config.diarizer.oracle_vad = False
    config.diarizer.msdd_model.parameters.diar_eval_settings = [
        (config.diarizer.collar, config.diarizer.ignore_overlap),
    ]
    config.diarizer.manifest_filepath = diarizer_manifest_path
    config.diarizer.out_dir = output_dir  # Directory to store intermediate files and prediction outputs
    config.diarizer.speaker_out_dir = speaker_output_dir if speaker_output_dir else output_dir # Directory to store speaker embeddings
    config.prepared_manifest_vad_input = os.path.join(output_dir, 'manifest_vad.json')

    config.diarizer.clustering.parameters.oracle_num_speakers = False

    # VAD Optimization
    config.diarizer.vad.model_path = vad_model_path
    
    config.diarizer.vad.parameters.frame_vad_threshold = 0.37 #trial.suggest_float("frame_vad_threshold", 0.15, 0.7, step=0.02)
    config.diarizer.vad.parameters.pad_onset = 0.01 #round(trial.suggest_float("pad_onset", 0.0, 0.2, step=0.01), 2)
    config.diarizer.vad.parameters.pad_offset = 0.0 #round(trial.suggest_float("pad_offset", 0.0, 0.2, step=0.01), 2)
    config.diarizer.vad.parameters.min_duration_on = 0.3 #round(trial.suggest_float("min_duration_on", 0.2, 0.4, step=0.05), 2)
    config.diarizer.vad.parameters.min_duration_off = 0.55 #round(trial.suggest_float("min_duration_off", 0.5, 0.95, step=0.05), 2)

    # MSDD Optimization
    config.diarizer.msdd_model.parameters.sigmoid_threshold = [0.55]
    config.diarizer.msdd_model.parameters.global_average_mix_ratio = 0.15 #trial.suggest_float("global_average_mix_ratio", low=0.1, high=0.95, step=0.05)
    config.diarizer.msdd_model.parameters.global_average_window_count = 190 #trial.suggest_int("global_average_window_count", low=10, high=500, step=20)

    # Clustering Optimization
    config.diarizer.clustering.parameters.max_rp_threshold = 0.07 #round(trial.suggest_float("max_rp_threshold", low=0.03, high=0.1, step=0.01), 2)
    config.diarizer.clustering.parameters.sparse_search_volume = 25 #trial.suggest_int("sparse_search_volume", low=25, high=25, step=1)
    r_value = 2.35  #round(trial.suggest_float("r_value", 0.5, 2.5, step=0.05), 4)
    scale_n = len(config.diarizer.speaker_embeddings.parameters.multiscale_weights)
    config.diarizer.speaker_embeddings.parameters.multiscale_weights = scale_weights(r_value, scale_n)
    return config


def get_gss_command(gpu_id, diar_config, diar_param, diar_base_dir, output_dir, mc_mask_min_db, mc_postmask_min_db,
                    bss_iterations, dereverb_filter_length, top_k, scenarios=SCENARIOS, subsets=SUBSETS, 
                    dereverb_prediction_delay=2, dereverb_num_iterations=3, mc_filter_type="pmwf", mc_filter_postfilter="ban"):
    command = f"MAX_SEGMENT_LENGTH=1000 MAX_BATCH_DURATION=20 BSS_ITERATION={bss_iterations} MC_MASK_MIN_DB={mc_mask_min_db} MC_POSTMASK_MIN_DB={mc_postmask_min_db} DEREVERB_FILTER_LENGTH={dereverb_filter_length} TOK_K={top_k}" \
              f" dereverb_prediction_delay={dereverb_prediction_delay} dereverb_num_iterations={dereverb_num_iterations} mc_filter_type={mc_filter_type} mc_filter_postfilter={mc_filter_postfilter} " \
              f" {NEMO_CHIME7_ROOT}/process/run_processing.sh '{scenarios}' " \
              f" {gpu_id} {diar_config} {diar_param} {diar_base_dir} {output_dir} " \
              f" {ESPNET_ROOT} {CHIME7_ROOT} {NEMO_CHIME7_ROOT} '{subsets}'"
              
    return command


def get_asr_eval_command(gpu_id, diar_config, diar_param, normalize_db, output_dir, lm_alpha, lm_beam_size, maes_num_steps, maes_alpha, maes_gamma, maes_beta, asr_output_dir, eval_results_dir, scenarios=SCENARIOS, subsets=SUBSETS):
    command = f"EVAL_CHIME=True {NEMO_CHIME7_ROOT}/evaluation/run_asr_lm.sh '{scenarios}' '{subsets}' " \
              f"{diar_config}-{diar_param} {output_dir}/processed {output_dir} {normalize_db} {ASR_MODEL_PATH} 1 4 {CHIME7_ROOT} {NEMO_CHIME7_ROOT} {gpu_id} " \
              f"{LM_PATH} {lm_alpha} {lm_beam_size} {maes_num_steps} {maes_alpha} {maes_gamma} {maes_beta} {asr_output_dir} {eval_results_dir}"

    return command


def objective_gss_asr(
        trial: optuna.Trial,
        gpu_id: int,
        output_dir: str,
        diar_config: str,
        diar_base_dir: str,
        diar_param: str = "T",
        scenarios: str = SCENARIOS,
        subsets: str = SUBSETS,
):
    mc_mask_min_db = -160 # trial.suggest_int("mc_mask_min_db", -160, -160, 20)
    mc_postmask_min_db = -16 #trial.suggest_int("mc_postmask_min_db", -25, -5, 3)
    bss_iterations = 5 #trial.suggest_int("bss_iterations", 5, 5, 5)
    dereverb_filter_length = 5 #trial.suggest_int("dereverb_filter_length", 5, 5, 5)
    normalize_db = -20 #trial.suggest_int("normalize_db", -20, -20, 5)
    top_k = 60 #trial.suggest_int("top_k", 60, 100, 20)

    # New parameters
    dereverb_prediction_delay = 3 #trial.suggest_categorical("dereverb_prediction_delay", choices=[3])
    dereverb_num_iterations = 5 #trial.suggest_categorical("dereverb_num_iterations", choices=[5])
    mc_filter_type = "pmwf" #trial.suggest_categorical("mc_filter_type", choices=['pmwf'])
    mc_filter_postfilter = trial.suggest_categorical("mc_filter_postfilter", choices=['ban'])

    lm_alpha = 0.0 #trial.suggest_float(name='lm_alpha', low=0, high=0.4, step=0.1)
    lm_beam_size = 8 # trial.suggest_int(name='lm_beam_size', low=6, high=8, step=1)
    maes_num_steps = 5
    maes_alpha = 3
    maes_gamma = 3.8 #trial.suggest_float(name="maes_expansion_gamma", low=0.3, high=5.3, step=0.5)
    maes_beta = 5

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
    asr_output_dir = os.path.join(output_dir, "nemo_asr_output")
    eval_results_dir = os.path.join(output_dir, "eval_results_output")
    logging.info(f"Start Trial {trial.number} with asr_output_dir: {asr_output_dir} and eval_results_dir: {eval_results_dir}")

    command_asr = get_asr_eval_command(gpu_id, diar_config, diar_param, normalize_db, output_dir, lm_alpha, lm_beam_size, maes_num_steps, maes_alpha, maes_gamma, maes_beta, asr_output_dir, eval_results_dir, scenarios, subsets)
    code = os.system(command_asr)
    if code != 0:
        raise RuntimeError(f"command failed: {command_asr}")

    eval_results = os.path.join(eval_results_dir, "macro_wer.txt")
    with open(eval_results, "r") as f:
        wer = float(f.read().strip())

        logging.info(f"-------------WER={wer:.4f}--------------------")
        logging.info(f"Trial: {trial.number}")
        logging.info(f"mc_mask_min_db: {mc_mask_min_db}, mc_postmask_min_db: {mc_postmask_min_db}, bss_iterations: {bss_iterations}, dereverb_filter_length: {dereverb_filter_length}, normalize_db: {normalize_db}")
        logging.info(f"lm_alpha: {lm_alpha}, lm_beam_size: {lm_beam_size}, maes_num_steps: {maes_num_steps}, maes_alpha: {maes_alpha}, maes_gamma: {maes_gamma}, maes_beta: {maes_beta}")
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
    tune_vad: bool = True,
    scenarios: str = SCENARIOS,
    subsets: str = SUBSETS,
    pattern: str = "*-dev.json",
):
    """
    [Note] Diarizaiton out `outputs`

    outputs = (metric, mapping_dict, itemized_erros)
    itemized_errors = (DER, CER, FA, MISS)
    """
    output_dir = temp_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    if True:
        logging.info(f"Start Trial {trial.number} with output_dir: {output_dir}")
        config.device = f"cuda:{gpu_id}"
        # Step:1-1 Configure Diarization
        config = diar_config_setup(
            trial,
            config,
            "manifest.json",
            msdd_model_path,
            vad_model_path,
            output_dir=output_dir,
            speaker_output_dir=speaker_output_dir,
            tune_vad=tune_vad,
        )

        with tempfile.TemporaryDirectory(dir=temp_dir, prefix=f"trial-{trial.number}-diar") as local_output_dir:
            start_time2 = time.time()
            for manifest_json in Path(diarizer_manifest_path).glob(pattern):
                logging.info(f"Start Diarization on {manifest_json}")
                scenario = manifest_json.stem.split("-")[0]
                curr_output_dir = os.path.join(output_dir, scenario)
                Path(curr_output_dir).mkdir(parents=True, exist_ok=True)

                # if "mixer6" in scenario and not keep_mixer6:  # don't save speaker outputs for mixer6
                #     curr_speaker_output_dir = os.path.join(local_output_dir, "speaker_outputs")
                # else:
                #     curr_speaker_output_dir = speaker_output_dir

                config.diarizer.out_dir = curr_output_dir  # Directory to store intermediate files and prediction outputs
                # config.diarizer.speaker_out_dir = curr_speaker_output_dir
                config.prepared_manifest_vad_input = os.path.join(curr_output_dir, 'manifest_vad.json')
                config.diarizer.manifest_filepath = str(manifest_json)

                # Step:1-2 Run Diarization
                diarizer_model = NeuralDiarizer(cfg=config).to(f"cuda:{gpu_id}")
                diarizer_model.diarize(verbose=False)
                move_diar_results(output_dir, config.diarizer.msdd_model.parameters.system_name, scenario=scenario)

                del diarizer_model
            logging.info(f"Diarization time taken for trial {trial.number}: {(time.time() - start_time2)/60:.2f} mins")


        WER = objective_gss_asr(
            trial,
            gpu_id,
            output_dir,
            diar_base_dir=output_dir,
            diar_config=config.diarizer.msdd_model.parameters.system_name,
            scenarios=scenarios,
            subsets=subsets,
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
    parser.add_argument(
        "--asr_model_path",
        help="path to the ASR model",
        type=str,
        default="diar_msdd_telephonic",
    )
    parser.add_argument(
        "--lm_path",
        help="path to the ngram language model",
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
    parser.add_argument("--tune_vad", help="whether to tune VAD", type=bool, default=True)
    parser.add_argument("--scenarios", help="Scenarios to run on", type=str, default=SCENARIOS)
    parser.add_argument("--subsets", help="Subsets to run on", type=str, default=SUBSETS)
    parser.add_argument("--pattern", help="Pattern to match manifest files", type=str, default="*-dev.json")

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
        tune_vad=args.tune_vad,
        scenarios=args.scenarios,
        subsets=args.subsets,
        pattern=args.pattern,
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
    os.system(f"rm -rf {args.storage}")
    os.system(f"rm -rf {args.output_log}")
    logging.info(f"Best SA-WER {study.best_value}")
    logging.info(f"Best Parameter Set: {study.best_params}")
