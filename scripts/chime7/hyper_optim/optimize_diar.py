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
import glob
import time
from multiprocessing import Process

import numpy as np
import optuna
from omegaconf import OmegaConf
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from typing import Optional
from nemo.collections.asr.models.msdd_v2_models import NeuralDiarizer
from nemo.utils import logging as nemo_logger

import argparse
import logging
import os
import tempfile
import time
from multiprocessing import Process

import numpy as np
import optuna
import wget
from omegaconf import OmegaConf
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from nemo.collections.asr.models.clustering_diarizer import ClusteringDiarizer
from nemo.utils import logging as nemo_logger

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
    if tune_vad: 
        config.diarizer.vad.parameters.frame_vad_threshold = trial.suggest_float("frame_vad_threshold", 0.2, 0.6, step=0.05)
        config.diarizer.vad.parameters.pad_onset = round(trial.suggest_float("pad_onset", 0.0, 0.2, step=0.05), 3)
        config.diarizer.vad.parameters.pad_offset = round(trial.suggest_float("pad_offset", 0.0, 0.2, step=0.05), 3)
        config.diarizer.vad.parameters.min_duration_on = round(trial.suggest_float("min_duration_on", 0.0, 0.5, step=0.05), 2)
        config.diarizer.vad.parameters.min_duration_off = round(trial.suggest_float("min_duration_off", 0.0, 0.5, step=0.05), 2)
    else: 
        config.diarizer.vad.parameters.frame_vad_threshold = trial.suggest_float("frame_vad_threshold", 0.04, 0.04, step=0.005)
        config.diarizer.vad.parameters.pad_onset = round(trial.suggest_float("pad_onset", 0.1, 0.1, step=0.01), 2)
        config.diarizer.vad.parameters.pad_offset = round(trial.suggest_float("pad_offset", 0.1, 0.1, step=0.01), 2)
        config.diarizer.vad.parameters.min_duration_on = round(trial.suggest_float("min_duration_on", 0.2, 0.2, step=0.05), 2)
        config.diarizer.vad.parameters.min_duration_off = round(trial.suggest_float("min_duration_off", 0.25, 0.25, step=0.05), 2)

    # MSDD Optimization
    config.diarizer.msdd_model.parameters.sigmoid_threshold = [trial.suggest_float("sigmoid_threshold", low=0.4, high=0.6, step=0.01)]
    
    # Clustering Optimization
    config.diarizer.clustering.parameters.max_rp_threshold = round(trial.suggest_float("max_rp_threshold", low=0.05, high=0.25, step=0.01), 2)
    config.diarizer.clustering.parameters.sparse_search_volume = trial.suggest_int("sparse_search_volume", low=25, high=25, step=1)
    r_value = round(trial.suggest_float("r_value", 0.05, 2.25, step=0.05), 4)
    scale_n = len(config.diarizer.speaker_embeddings.parameters.multiscale_weights)
    config.diarizer.speaker_embeddings.parameters.multiscale_weights = scale_weights(r_value, scale_n)
    return config
    
def objective_diar(
    trial: optuna.Trial,
    config,
    diarizer_manifest_path: str,
    msdd_model_path: str,
    vad_model_path: str,
    speaker_output_dir: str,
):
    """
    [Note] Diarizaiton out `outputs`
    
    outputs = (metric, mapping_dict, itemized_erros)
    itemized_errors = (DER, CER, FA, MISS)
    """
    tmp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(tmp_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=tmp_dir, prefix=str(trial.number)) as output_dir:
        start_time = time.time()
        # Step:1-1 Configure Diarization
        config = diar_config_setup(
            trial, 
            config,
            diarizer_manifest_path,
            msdd_model_path,
            vad_model_path,
            output_dir,
            speaker_output_dir=speaker_output_dir,
        )
        # Step:1-2 Run Diarization 
        diarizer_model = NeuralDiarizer(cfg=config).to("cuda:0")
        outputs = diarizer_model.diarize(verbose=False)
        json_output_folder = os.path.join(output_dir, config.diarizer.msdd_model.parameters.system_name, "pred_jsons_T")
        # Glob all json files in json_output_folder:
        diar_segments_filelist = glob.glob(os.path.join(json_output_folder, "*.json"))
        print(f"[optuna] Diarization Segment Json saved in : {json_output_folder}")
        print(f"Time taken for trial: {time.time() - start_time:.2f}s")
        metric = outputs[0][0]
        DER = abs(metric)
    import ipdb; ipdb.set_trace()
    return DER
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_name", help="Name of study.", type=str, required=True)
    parser.add_argument("--storage", help="Shared storage (i.e sqlite:///testDB.db).", type=str, required=True)
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
    parser.add_argument("--output_dir", help="path to store temporary files", type=str, default="output/")
    parser.add_argument("--output_log", help="Where to store optuna output log", type=str, default="output.log")
    parser.add_argument("--n_trials", help="Number of trials to run optuna", type=int, default=5)
    parser.add_argument("--batch_size", help="Batch size for mc-embedings and MSDD", required=True, type=int, default=11)
    parser.add_argument("--n_jobs", help="Number of parallel jobs to run", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    nemo_logger.setLevel(logging.ERROR)
    model_config = args.config_url
    config = OmegaConf.load(model_config)
    config.batch_size = args.batch_size

    func = lambda trial: objective_diar(
        trial,
        config=config,
        diarizer_manifest_path=args.manifest_path,
        msdd_model_path=args.msdd_model_path,
        vad_model_path=args.vad_model_path,
        speaker_output_dir=args.output_dir,
    )

    def optimize(study):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # Setup the root logger.
        logger.addHandler(logging.FileHandler(args.output_log, mode="a"))
        logger.addHandler(logging.StreamHandler())
        optuna.logging.enable_propagation()  # Propagate logs to the root logger.
        study.optimize(func, n_trials=args.n_trials) 
        return study

    try:
        study = optuna.load_study(
            study_name=args.study_name,
            storage=args.storage,
        )
        print(f"[Optuna] Succesfully loaded study {args.study_name} from {args.storage}.")
    except:
        print(f"[Optuna] Failed to load study {args.study_name} from {args.storage}, creating a new study.")
        study = optuna.create_study(
            direction="minimize", study_name=args.study_name, storage=args.storage, load_if_exists=True
        )
    
    study_opt = optimize(study=study)
    print(f"[Optuna] Best DER {study_opt.best_value}")
    print(f"[Optuna] Best Parameter Set: {study_opt.best_params}")
