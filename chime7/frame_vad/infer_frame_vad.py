# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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


import json
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
import torch
from pyannote.core import Annotation, Segment
from pyannote.metrics import detection
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from nemo.collections.asr.models.multi_classification_models import EncDecMultiClassificationModel
from nemo.collections.asr.parts.utils.vad_utils import (
    align_labels_to_frames,
    generate_vad_frame_pred,
    generate_vad_segment_table,
    prepare_manifest,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@hydra_runner(config_path="./configs", config_name="vad_inference_postprocessing.yaml")
def main(cfg):
    if not cfg.dataset:
        raise ValueError("You must input the path of json file of evaluation data")

    if not os.path.exists(cfg.frame_out_dir):
        os.mkdir(cfg.frame_out_dir)
    else:
        logging.info(f"Found existing dir: {cfg.frame_out_dir}, remove and create new one...")
        os.system(f"rm -rf {cfg.frame_out_dir}")
        os.mkdir(cfg.frame_out_dir)

    # init and load model
    torch.set_grad_enabled(False)
    vad_model = EncDecMultiClassificationModel.restore_from(restore_path=cfg.vad.model_path)

    manifest_list = cfg.dataset
    if isinstance(manifest_list, str):
        manifest_list = manifest_list.split(',')

    for manifest_file in manifest_list:
        filename = Path(manifest_file).stem
        out_dir = str(Path(cfg.frame_out_dir) / Path(f"vad_output_{filename}"))
        logging.info("====================================================")
        logging.info(f"Start inference on manifest: {manifest_file}")
        pred_dir = evaluate_single_manifest(manifest_file, cfg, vad_model, out_dir)
        logging.info(f"Output saved at: {pred_dir}")

    logging.info(cfg.vad.parameters.postprocessing)
    logging.info(f"Model path: {Path(cfg.vad.model_path).absolute()}")
    logging.info("Done.")


def evaluate_single_manifest(manifest_filepath, cfg, vad_model, out_dir):

    Path(out_dir).mkdir(exist_ok=True)

    # each line of dataset should be have different audio_filepath and unique name to simplify edge cases or conditions
    key_meta_map = {}
    with open(manifest_filepath, 'r') as manifest:
        for line in manifest.readlines():
            data = json.loads(line.strip())
            audio_filepath = data['audio_filepath']
            uniq_audio_name = audio_filepath.split('/')[-1].rsplit('.', 1)[0]
            if uniq_audio_name in key_meta_map:
                raise ValueError("Please make sure each line is with different audio name! ")
            key_meta_map[uniq_audio_name] = {'audio_filepath': audio_filepath}

    # Prepare manifest for streaming VAD
    manifest_vad_input = manifest_filepath
    if cfg.prepare_manifest.auto_split:
        logging.info("Split long audio file to avoid CUDA memory issue")
        logging.debug("Try smaller split_duration if you still have CUDA memory issue")
        if not cfg.prepared_manifest_vad_input:
            prepared_manifest_vad_input = os.path.join(out_dir, "manifest_vad_input.json")
        else:
            prepared_manifest_vad_input = cfg.prepared_manifest_vad_input
        config = {
            'input': manifest_vad_input,
            'window_length_in_sec': cfg.vad.parameters.window_length_in_sec,
            'split_duration': cfg.prepare_manifest.split_duration,
            'num_workers': cfg.num_workers,
            'prepared_manifest_vad_input': prepared_manifest_vad_input,
        }
        manifest_vad_input = prepare_manifest(config)
    else:
        logging.warning(
            "If you encounter CUDA memory issue, try splitting manifest entry by split_duration to avoid it."
        )

    # setup test data
    vad_model.setup_test_data(
        test_data_config={
            'batch_size': 1,
            'sample_rate': 16000,
            'manifest_filepath': manifest_vad_input,
            'labels': ['infer'],
            'num_workers': cfg.num_workers,
            'shuffle': False,
            'normalize_audio': cfg.vad.parameters.normalize_audio,
            'normalize_audio_target': cfg.vad.parameters.normalize_audio_target,
        }
    )

    vad_model = vad_model.to(device)
    vad_model.eval()

    logging.info("Generating frame-level prediction ")
    pred_dir, all_probs_map = generate_vad_frame_pred(
        vad_model=vad_model,
        window_length_in_sec=cfg.vad.parameters.window_length_in_sec,
        shift_length_in_sec=cfg.vad.parameters.shift_length_in_sec,
        manifest_vad_input=manifest_vad_input,
        out_dir=os.path.join(out_dir, "frames_predictions"),
    )
    logging.info(f"Finish generating VAD frame level prediction, output dir: {pred_dir}")

    logging.info("Generating segment tables for predictions")
    pred_segment_dir = str(Path(pred_dir) / Path("pred_segments"))
    pred_segment_dir = generate_vad_segment_table(
        pred_dir,
        cfg.vad.parameters.postprocessing,
        frame_length_in_sec=cfg.vad.parameters.shift_length_in_sec,
        num_workers=cfg.num_workers,
        out_dir=pred_segment_dir,
    )

    return pred_segment_dir


if __name__ == '__main__':
    main()
