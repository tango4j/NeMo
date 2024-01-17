from asr.run_asr import run_asr
from eval.run_chime_eval import run_chime_evaluation
from gss_process.run_gss_process import run_gss_process
from omegaconf import DictConfig, OmegaConf

from nemo.core.config import hydra_runner
from nemo.utils import logging

from typing import Optional
import os

import torch
import optuna
from omegaconf import OmegaConf

from nemo.collections.asr.models.msdd_v2_models import NeuralDiarizer
from nemo.utils import logging as nemo_logger


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


def move_diar_results(diar_out_dir, system_name, scenario="chime6"):
    curr_diar_out_dir = Path(diar_out_dir, scenario, system_name, "pred_jsons_T")
    new_diar_out_dir = Path(diar_out_dir, system_name, scenario, "pred_jsons_T")
    new_diar_out_dir.mkdir(parents=True, exist_ok=True)
    for json_file in curr_diar_out_dir.glob("*.json"):
        os.rename(json_file, Path(new_diar_out_dir, json_file.name))
    return diar_out_dir

def scale_weights(r, K):
    return [r - kvar * (r - 1) / (K - 1) for kvar in range(K)]


def diar_config_setup(cfg):
    cfg.diarizer.oracle_vad = False
    cfg.diarizer.msdd_model.parameters.diar_eval_settings = [
        (cfg.diarizer.collar, cfg.diarizer.ignore_overlap),
    ]
    output_dir = cfg.diarizer.out_dir
    curr_output_dir = output_dir
    cfg.diarizer.speaker_out_dir = cfg.diarizer.out_dir 
    cfg.prepared_manifest_vad_input = os.path.join(output_dir, 'manifest_vad.json')

    cfg.diarizer.clustering.parameters.oracle_num_speakers = False
    cfg.diarizer.vad.parameters.frame_vad_threshold = 0.37 #trial.suggest_float("frame_vad_threshold", 0.15, 0.7, step=0.02)
    cfg.diarizer.vad.parameters.pad_onset = 0.14 #round(trial.suggest_float("pad_onset", 0.0, 0.2, step=0.01), 2)
    cfg.diarizer.vad.parameters.pad_offset = 0.17 #round(trial.suggest_float("pad_offset", 0.0, 0.2, step=0.01), 2)
    cfg.diarizer.vad.parameters.min_duration_on = 0.4 #round(trial.suggest_float("min_duration_on", 0.2, 0.4, step=0.05), 2)
    cfg.diarizer.vad.parameters.min_duration_off = 0.55 #round(trial.suggest_float("min_duration_off", 0.5, 0.95, step=0.05), 2)

    # MSDD Optimization
    cfg.diarizer.msdd_model.parameters.sigmoid_threshold = [0.55]
    cfg.diarizer.msdd_model.parameters.global_average_mix_ratio = 0.3 #trial.suggest_float("global_average_mix_ratio", low=0.1, high=0.95, step=0.05)
    cfg.diarizer.msdd_model.parameters.global_average_window_count = 190 #trial.suggest_int("global_average_window_count", low=10, high=500, step=20)

    # Clustering Optimization
    cfg.diarizer.clustering.parameters.max_rp_threshold = 0.06 #round(trial.suggest_float("max_rp_threshold", low=0.03, high=0.1, step=0.01), 2)
    cfg.diarizer.clustering.parameters.sparse_search_volume = 25 #trial.suggest_int("sparse_search_volume", low=25, high=25, step=1)
    r_value = 2.15  #round(trial.suggest_float("r_value", 0.5, 2.5, step=0.05), 4)
    scale_n = len(cfg.diarizer.speaker_embeddings.parameters.multiscale_weights)
    cfg.diarizer.speaker_embeddings.parameters.multiscale_weights = scale_weights(r_value, scale_n)
    
    cfg.prepared_manifest_vad_input = os.path.join(curr_output_dir, 'manifest_vad.json')
    return cfg

@hydra_runner(config_path="../", config_name="chime_config")
def main(cfg):
    cfg = DictConfig(OmegaConf.to_container(cfg, resolve=True))
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    
    # cfg.device = f"cuda:{gpu_id}"
    # Step:1-1 Configure Diarization
    cfg = diar_config_setup(cfg)
    
    # Step:1-2 Run Diarization
    gpu_id=0
    
    diarizer_model = NeuralDiarizer(cfg=cfg).to(f"cuda:{gpu_id}")
    diarizer_model.diarize(verbose=False)
    output_dir = cfg.diarizer.out_dir
    
    for scenario in cfg.scenarios: 
        move_diar_results(output_dir, cfg.diarizer.msdd_model.parameters.system_name, scenario=scenario)

    del diarizer_model
    
    # Run GSS
    logging.info("Running GSS")
    run_gss_process(cfg)

    # Run ASR
    logging.info("Running ASR")
    run_asr(cfg)

    # Run evaluation
    logging.info("Running evaluation")
    run_chime_evaluation(cfg)

if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
