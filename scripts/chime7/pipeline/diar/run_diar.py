# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import os
from omegaconf import DictConfig, OmegaConf
from nemo.core.config import hydra_runner
from nemo.collections.asr.models.msdd_v2_models import NeuralDiarizer
from nemo.utils import logging
from pathlib import Path
from scripts.chime7.manifests.prepare_nemo_diar_manifest import generate_annotations

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

def run_diarization(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    for subset in cfg.subsets: 
        generate_annotations(data_dir=cfg.chime_data_root,
                             subset=subset,
                             output_dir=os.path.join(cfg.diar_base_dir, "diar_manifests"),
                             scenarios_list=cfg.scenarios,
                         )
    # cfg.device = f"cuda:{gpu_id}"
    # Step:1-1 Configure Diarization
    cfg = diar_config_setup(cfg)
    
    # Step:1-2 Run Diarization
    gpu_id=0
    
    for scenario in cfg.scenarios:
        for subset in cfg.subsets: 
            cfg.diarizer.manifest_filepath = os.path.join(cfg.diar_base_dir, "diar_manifests", scenario, 'mulspk_asr_manifest', f"{scenario}-{subset}.json")
            diarizer_model = NeuralDiarizer(cfg=cfg).to(f"cuda:{gpu_id}")
            diarizer_model.diarize(verbose=False)
            output_dir = cfg.diarizer.out_dir
    
    for scenario in cfg.scenarios: 
        move_diar_results(output_dir, cfg.diarizer.msdd_model.parameters.system_name, scenario=scenario)

    del diarizer_model