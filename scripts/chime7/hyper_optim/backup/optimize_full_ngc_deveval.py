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
ASR_MODEL_PATH=f"{NGC_WS_MOUNT}/model_checkpoints/rno_chime7_chime6_ft_ptDataSetasrset3_frontend_nemoGSSv1_prec32_layers24_heads8_conv5_d1024_dlayers2_dsize640_bs128_adamw_CosineAnnealing_lr0.0001_wd1e-2_spunigram1024.nemo"

SCENARIOS = "chime6 dipco mixer6"
SUBSETS = "dev eval"

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
    config.diarizer.msdd_model.parameters.diar_eval_settings = []
    config.diarizer.manifest_filepath = diarizer_manifest_path
    config.diarizer.out_dir = output_dir  # Directory to store intermediate files and prediction outputs
    config.diarizer.speaker_out_dir = speaker_output_dir if speaker_output_dir else output_dir # Directory to store speaker embeddings
    config.prepared_manifest_vad_input = os.path.join(output_dir, 'manifest_vad.json')
    
    config.diarizer.clustering.parameters.oracle_num_speakers = False
    
    # VAD Optimization
    config.diarizer.vad.model_path = vad_model_path
    if tune_vad: 
        config.diarizer.vad.parameters.frame_vad_threshold = trial.suggest_float("frame_vad_threshold", 0.03, 0.25, step=0.01)
        config.diarizer.vad.parameters.pad_onset = round(trial.suggest_float("pad_onset", 0.0, 0.2, step=0.05), 3)
        config.diarizer.vad.parameters.pad_offset = round(trial.suggest_float("pad_offset", 0.0, 0.5, step=0.05), 3)
        config.diarizer.vad.parameters.min_duration_on = round(trial.suggest_float("min_duration_on", 0.0, 0.5, step=0.05), 2)
        config.diarizer.vad.parameters.min_duration_off = round(trial.suggest_float("min_duration_off", 0.0, 1.2, step=0.05), 2)
    else: 
        config.diarizer.vad.parameters.frame_vad_threshold = trial.suggest_float("frame_vad_threshold", 0.04, 0.04, step=0.005)
        config.diarizer.vad.parameters.pad_onset = round(trial.suggest_float("pad_onset", 0.1, 0.1, step=0.01), 2)
        config.diarizer.vad.parameters.pad_offset = round(trial.suggest_float("pad_offset", 0.1, 0.1, step=0.01), 2)
        config.diarizer.vad.parameters.min_duration_on = round(trial.suggest_float("min_duration_on", 0.2, 0.2, step=0.05), 2)
        config.diarizer.vad.parameters.min_duration_off = round(trial.suggest_float("min_duration_off", 0.25, 0.25, step=0.05), 2)

    # MSDD Optimization
    config.diarizer.msdd_model.parameters.sigmoid_threshold = [trial.suggest_float("sigmoid_threshold", low=0.2, high=0.9, step=0.05)]
    config.diarizer.msdd_model.parameters.global_average_mix_ratio = trial.suggest_float("global_average_mix_ratio", low=0.6, high=1.0, step=0.05)

    # Clustering Optimization
    config.diarizer.clustering.parameters.max_rp_threshold = round(trial.suggest_float("max_rp_threshold", low=0.05, high=0.25, step=0.01), 2)
    config.diarizer.clustering.parameters.sparse_search_volume = trial.suggest_int("sparse_search_volume", low=25, high=25, step=1)
    r_value = round(trial.suggest_float("r_value", 0.05, 2.25, step=0.05), 4)
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


def get_asr_eval_command(gpu_id, diar_config, diar_param, normalize_db, output_dir, scenarios=SCENARIOS, subsets=SUBSETS):
    command = f"EVAL_CHIME=True {NEMO_CHIME7_ROOT}/evaluation/run_asr.sh '{scenarios}' '{subsets}' " \
              f"{diar_config}-{diar_param} {output_dir}/processed {output_dir} {normalize_db} {ASR_MODEL_PATH} 1 4 {CHIME7_ROOT} {NEMO_CHIME7_ROOT} {gpu_id}"

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
        mc_mask_min_db: int = -200,
        mc_postmask_min_db: int = -18,
        bss_iterations: int = 5,
        dereverb_filter_length: int = 5,
        top_k: int = 50,
        normalize_db: int = -35,
        dereverb_prediction_delay: int = 2,
        dereverb_num_iterations: int = 3,
        mc_filter_type: str = "pmwf",
        mc_filter_postfilter: str = "ban",
):
    
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
    
    logging.info(f"----------------------------------------------")
    logging.info(f"Trial: {trial.number}, WER={wer:.4f}")
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
    msdd_model_path: str,
    vad_model_path: str,
    speaker_output_dir: str,
    gpu_id: int,
    temp_dir: str,
    keep_mixer6: bool = False,
    tune_vad: bool = True,
    scenarios: str = SCENARIOS,
    pattern: str = "*-dev.json",
):
    """
    [Note] Diarizaiton out `outputs`
    
    outputs = (metric, mapping_dict, itemized_erros)
    itemized_errors = (DER, CER, FA, MISS)
    """
    wer_list = []
    start_time = time.time()

    # GSS params
    mc_mask_min_db = trial.suggest_int("mc_mask_min_db", -200, -60, 20)
    mc_postmask_min_db = trial.suggest_int("mc_postmask_min_db", -18, 0, 3)
    bss_iterations = trial.suggest_int("bss_iterations", 5, 20, 5)
    dereverb_filter_length = trial.suggest_int("dereverb_filter_length", 5, 15, 5)
    normalize_db = trial.suggest_int("normalize_db", -35, -20, 5)
    top_k = trial.suggest_int("top_k", 50, 100, 10)
    dereverb_prediction_delay = trial.suggest_categorical("dereverb_prediction_delay", choices=[2, 3])
    dereverb_num_iterations = trial.suggest_categorical("dereverb_num_iterations", choices=[3, 5, 10])
    mc_filter_type = trial.suggest_categorical("mc_filter_type", choices=['pmwf', 'wmpdr'])
    mc_filter_postfilter = trial.suggest_categorical("mc_filter_postfilter", choices=['ban', 'None'])

    with tempfile.TemporaryDirectory(dir=temp_dir, prefix=f"trial-{trial.number}") as output_dir:
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
            speaker_output_dir=output_dir,
            tune_vad=tune_vad,
        )
        origin_outputz_dir = str(output_dir)
        with tempfile.TemporaryDirectory(dir=temp_dir, prefix=f"trial-{trial.number}-diar") as local_output_dir:
            start_time2 = time.time()
            for subset in ["dev", "eval"]:
                output_dir = os.path.join(origin_outputz_dir, subset)
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                if subset == "dev":
                    pattern = "*-dev.json"
                    diarizer_manifest_path = "/ws/manifests_dev_ngc"
                else:
                    diarizer_manifest_path = "/ws/manifests_eval_ngc"
                for manifest_json in Path(diarizer_manifest_path).glob(pattern):
                    logging.info(f"Start Diarization on {manifest_json}")
                    scenario = manifest_json.stem.split("-")[0]
                    curr_output_dir = os.path.join(output_dir, scenario)
                    Path(curr_output_dir).mkdir(parents=True, exist_ok=True)

                    if "mixer6" in scenario and not keep_mixer6:  # don't save speaker outputs for mixer6
                        curr_speaker_output_dir = os.path.join(local_output_dir, "speaker_outputs")
                    else:
                        curr_speaker_output_dir = speaker_output_dir

                    config.diarizer.out_dir = curr_output_dir  # Directory to store intermediate files and prediction outputs
                    config.diarizer.speaker_out_dir = curr_speaker_output_dir
                    config.prepared_manifest_vad_input = os.path.join(curr_output_dir, 'manifest_vad.json')
                    config.diarizer.manifest_filepath = str(manifest_json)

                    # Step:1-2 Run Diarization 
                    diarizer_model = NeuralDiarizer(cfg=config).to(f"cuda:{gpu_id}")
                    diarizer_model.diarize(verbose=False)
                    move_diar_results(output_dir, config.diarizer.msdd_model.parameters.system_name, scenario=scenario)

                del diarizer_model
                logging.info(f"Diarization time taken for trial {trial.number}: {(time.time() - start_time2)/60:.2f} mins")
                start_time2 = time.time()
                WER = objective_gss_asr(
                    trial,
                    gpu_id,
                    output_dir,
                    diar_base_dir=output_dir,
                    diar_config=config.diarizer.msdd_model.parameters.system_name,
                    scenarios=scenarios,
                    subsets=subset,
                    mc_mask_min_db=mc_mask_min_db,
                    mc_postmask_min_db=mc_postmask_min_db,
                    bss_iterations=bss_iterations,
                    dereverb_filter_length=dereverb_filter_length,
                    top_k=top_k,
                    normalize_db=normalize_db,
                    dereverb_prediction_delay=dereverb_prediction_delay,
                    dereverb_num_iterations=dereverb_num_iterations,
                    mc_filter_type=mc_filter_type,
                    mc_filter_postfilter=mc_filter_postfilter,
                )
                wer_list.append(WER)
                logging.info(f"GSS+ASR time taken for trial {trial.number}: {(time.time() - start_time2)/60:.2f} mins")

    logging.info(f"Time taken for trial {trial.number}: {(time.time() - start_time)/60:.2f} mins")    
    WER_hmean = wer_list[0] * wer_list[1] / (wer_list[0] + wer_list[1])
    return WER_hmean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_name", help="Name of study.", type=str, default="optuna_chime7")
    parser.add_argument("--storage", help="Shared storage (i.e sqlite:///testDB.db).", type=str, default="sqlite:///optuna-msdd-gss-asr.db")
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
    parser.add_argument("--tune_vad", help="whether to tune VAD", type=bool, default=True)
    parser.add_argument("--scenarios", help="Scenarios to run on", type=str, default=SCENARIOS)
    parser.add_argument("--pattern", help="Pattern to match manifest files", type=str, default="*-eval-d03.json")

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
        msdd_model_path=args.msdd_model_path,
        vad_model_path=args.vad_model_path,
        speaker_output_dir=args.output_dir,
        keep_mixer6=args.keep_mixer6,
        tune_vad=args.tune_vad,
        scenarios=args.scenarios,
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
        time.sleep(5)

    for t in processes:
        t.join()

    study = optuna.load_study(
        study_name=args.study_name,
        storage=args.storage,
    )
    logging.info(f"Best SA-WER {study.best_value}")
    logging.info(f"Best Parameter Set: {study.best_params}")