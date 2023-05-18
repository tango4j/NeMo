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
from sklearn.metrics import classification_report, roc_auc_score

from nemo.collections.asr.models.multi_classification_models import EncDecMultiClassificationModel
from nemo.collections.asr.parts.utils.vad_utils import (
    align_labels_to_frames,
    generate_vad_frame_pred,
    generate_vad_segment_table,
    get_frame_labels,
    load_speech_segments_from_rttm,
    prepare_manifest,
    vad_frame_construct_pyannote_object_per_file,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@hydra_runner(config_path="./configs", config_name="vad_inference_postprocessing.yaml")
def main(cfg):
    if not cfg.dataset:
        raise ValueError("You must input the path of json file of evaluation data")

    if os.path.exists(cfg.frame_out_dir):
        logging.info(f"Found existing dir: {cfg.frame_out_dir}, remove and create new one...")
        os.system(f"rm -rf {cfg.frame_out_dir}")
    Path(cfg.frame_out_dir).mkdir(parents=True, exist_ok=True)

    # init and load model
    torch.set_grad_enabled(False)
    vad_model = EncDecMultiClassificationModel.restore_from(restore_path=cfg.vad.model_path)

    manifest_list = cfg.dataset
    if isinstance(manifest_list, str):
        manifest_list = manifest_list.split(',')

    probs_dict = {}
    labels_dict = {}
    reports_dict = {}
    pred_seg_dir_dict = {}
    gt_seg_dir_dict = {}
    for manifest_file in manifest_list:
        filename = Path(manifest_file).stem
        out_dir = str(Path(cfg.frame_out_dir) / Path(f"vad_output_{filename}"))
        logging.info("====================================================")
        logging.info(f"Start evaluating manifest: {manifest_file}")
        probs, labels, report, pred_segment_dir, gt_segment_dir = evaluate_single_manifest(
            manifest_file, cfg, vad_model, out_dir
        )
        probs_dict[filename] = probs
        labels_dict[filename] = labels
        reports_dict[filename] = report
        pred_seg_dir_dict[filename] = pred_segment_dir
        gt_seg_dir_dict[filename] = gt_segment_dir

    if cfg.get("infer_only", False):
        logging.info("=========================================================")
        for k, v in pred_seg_dir_dict.items():
            logging.info(f"VAD predictions for {k} are saved in {v}")
        logging.info("=========================================================")
        logging.info(cfg.vad.parameters.postprocessing)
        logging.info(Path(cfg.vad.model_path).absolute())
        logging.info("Done.")
        exit(0)

    logging.info("=========================================================")
    logging.info("Calculating aggregated Detection Error...")
    all_der_report = calculate_multi_detection_error(pred_seg_dir_dict, gt_seg_dir_dict)

    logging.info("====================================================")
    logging.info("Finalizing individual results...")
    threshold = cfg.vad.parameters.get("threshold", 0.5)

    all_probs = []
    all_labels = []
    for key in probs_dict:
        probs = probs_dict[key]
        labels = labels_dict[key]

        all_probs += probs
        all_labels += labels

        auroc = roc_auc_score(y_true=labels, y_score=probs)
        pred_labels = [int(x > threshold) for x in probs]
        clf_report = classification_report(y_true=labels, y_pred=pred_labels)
        logging.info(f"================= {key} =================")
        logging.info(f"AUROC: {auroc:0.4f}")
        logging.info(f"Classification report with threshold={threshold:.2f}")
        logging.info(clf_report)

        der_report = reports_dict[key]
        DetER = der_report.iloc[[-1]][('detection error rate', '%')].item()
        FA = der_report.iloc[[-1]][('false alarm', '%')].item()
        MISS = der_report.iloc[[-1]][('miss', '%')].item()
        logging.info(f"Detection Error Rate: DetER={DetER:0.4f}, False Alarm={FA:0.4f}, Miss={MISS:0.4f}")
        logging.info("==========================================\n\n")

    logging.info("================== Aggregrated Results ===================")
    DetER = all_der_report.iloc[[-1]][('detection error rate', '%')].item()
    FA = all_der_report.iloc[[-1]][('false alarm', '%')].item()
    MISS = all_der_report.iloc[[-1]][('miss', '%')].item()
    logging.info(f"============================================================")
    logging.info(f" DetER={DetER:0.4f}, False Alarm={FA:0.4f}, Miss={MISS:0.4f}")
    logging.info(f"============================================================")

    auroc = roc_auc_score(y_true=all_labels, y_score=all_probs)
    pred_labels = [int(x > threshold) for x in all_probs]
    clf_report = classification_report(y_true=all_labels, y_pred=pred_labels)
    logging.info(f"AUROC: {auroc:0.4f}")
    logging.info(f"Classification report with threshold={threshold:.2f}")
    logging.info(f"\n{clf_report}")

    logging.info(cfg.vad.parameters.postprocessing)
    logging.info("Done.")
    print(Path(cfg.vad.model_path).absolute())

def get_common_prefix(filelist: List[str]):
    filelist = [x.split('/')[-1] for x in filelist]
    common_prefix = os.path.commonprefix(filelist)
    return common_prefix

def load_pred(filepath):
    with open(filepath, 'r') as fin:
        probs = [float(x.strip()) for x in fin.readlines()]
    return probs

def merge_list(list1, list2):
    length = max(len(list1), len(list2))
    res = []
    for i in range(length):
        if i < len(list1) and i < len(list2):
            res.append(max(list1[i], list2[i]))
        elif i < len(list1):
            res.append(list1[i])
        else:
            res.append(list2[i])
    return res

def merge_multi_channel_pred(pred_dir, all_labels_map, sess_map):
    all_files = Path(pred_dir).glob("*.frame")
    all_preds_map = {}
    all_labels_map_new = {}
    for file in all_files:
        key = file.stem
        sess = sess_map[key]
        if sess not in all_preds_map:
            all_preds_map[sess] = load_pred(file)
        else:
            all_preds_map[sess] = merge_list(all_preds_map[sess], load_pred(file))
        all_labels_map_new[sess] = all_labels_map[key]
    
    out_dir = Path(pred_dir, "merged")
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in all_preds_map:
        with open(Path(out_dir, f"{key}.frame"), 'w') as fout:
            fout.write("\n".join([str(x) for x in all_preds_map[key]]))
    return str(out_dir), all_preds_map, all_labels_map_new


def evaluate_single_manifest(manifest_filepath, cfg, vad_model, out_dir):

    Path(out_dir).mkdir(exist_ok=True)

    # each line of dataset should be have different audio_filepath and unique name to simplify edge cases or conditions
    key_meta_map = {}
    all_labels_map = {}
    sess_map = {}
    with open(manifest_filepath, 'r') as manifest:
        for line in manifest.readlines():
            data = json.loads(line.strip())
            audio_filepath_list = data['audio_filepath']
            common_prefix = get_common_prefix(audio_filepath_list)
            for audio_filepath in audio_filepath_list:
                uniq_audio_name = audio_filepath.split('/')[-1].rsplit('.', 1)[0]
                sess_map[uniq_audio_name] = common_prefix
                if uniq_audio_name in key_meta_map:
                    raise ValueError("Please make sure each line is with different audio name! ")
                key_meta_map[uniq_audio_name] = {'audio_filepath': audio_filepath}
                if cfg.get("infer_only", False):
                    all_labels_map[uniq_audio_name] = [0]
                elif "rttm_filepath" in data or "rttm_file" in data or "label" not in data:
                    rttm_key = "rttm_filepath" if "rttm_filepath" in data else "rttm_file"
                    segments = load_speech_segments_from_rttm(data[rttm_key])
                    label_str = get_frame_labels(
                        segments=segments,
                        frame_length=cfg.vad.parameters.shift_length_in_sec,
                        duration=data['duration'],
                        offset=data['offset'],
                    )
                    all_labels_map[uniq_audio_name] = [int(x) for x in label_str.split()]
                else:
                    all_labels_map[uniq_audio_name] = [int(x) for x in data["label"].split()]

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

    # setup_test_data
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

    pred_dir, all_probs_map, all_labels_map = merge_multi_channel_pred(pred_dir, all_labels_map, sess_map)

    logging.info(
        f"Finish generating VAD frame level prediction with window_length_in_sec={cfg.vad.parameters.window_length_in_sec} and shift_length_in_sec={cfg.vad.parameters.shift_length_in_sec}"
    )

    # calculate AUROC
    predictions = []
    groundtruth = []
    for key in all_labels_map:
        probs = all_probs_map[key]
        labels = all_labels_map[key]
        labels_aligned = align_labels_to_frames(probs, labels)
        all_labels_map[key] = labels_aligned
        groundtruth += labels_aligned
        predictions += probs

    frame_length_in_sec = cfg.vad.parameters.shift_length_in_sec

    gt_frames_dir = dump_groundtruth_frames(out_dir, all_labels_map)

    report, pred_segment_dir, gt_segment_dir = calculate_detection_error(
        pred_dir,
        gt_frames_dir,
        post_params=cfg.vad.parameters.postprocessing,
        frame_length_in_sec=frame_length_in_sec,
        num_workers=cfg.num_workers,
        infer_only=cfg.get("infer_only", False),
    )

    return predictions, groundtruth, report, pred_segment_dir, gt_segment_dir


def calculate_multi_detection_error(pred_seg_dir_dict: dict, gt_seg_dir_dict: dict):
    all_paired_files = []
    for key in gt_seg_dir_dict:
        if key not in pred_seg_dir_dict:
            continue
        gt_seg_dir = gt_seg_dir_dict[key]
        pred_seg_dir = pred_seg_dir_dict[key]
        paired_files = find_paired_files(pred_dir=pred_seg_dir, gt_dir=gt_seg_dir)
        all_paired_files += paired_files

    metric = detection.DetectionErrorRate()
    for key, gt_file, pred_file in paired_files:
        reference, hypothesis = vad_frame_construct_pyannote_object_per_file(pred_file, gt_file)
        metric(reference, hypothesis)  # accumulation

    report = metric.report(display=False)
    return report


def calculate_detection_error(
    vad_pred_frame_dir: str,
    vad_gt_frame_dir: str,
    post_params: dict,
    frame_length_in_sec: float = 0.01,
    num_workers: int = 20,
    infer_only: bool = False,
):

    logging.info("Generating segment tables for predictions")
    pred_segment_dir = str(Path(vad_pred_frame_dir) / Path("pred_segments"))
    pred_segment_dir = generate_vad_segment_table(
        vad_pred_frame_dir,
        post_params,
        frame_length_in_sec=frame_length_in_sec,
        num_workers=num_workers,
        out_dir=pred_segment_dir,
    )

    if infer_only:
        logging.info("Infer only, skip calculating detection error metrics")
        return None, pred_segment_dir, None

    logging.info("Generating segment tables for groundtruths")
    gt_segment_dir = Path(vad_gt_frame_dir) / Path("gt_segments")
    gt_segment_dir = generate_gt_segment_table(
        vad_gt_frame_dir, frame_length_in_sec=frame_length_in_sec, num_workers=num_workers, out_dir=gt_segment_dir
    )

    paired_files = find_paired_files(pred_dir=pred_segment_dir, gt_dir=gt_segment_dir)
    metric = detection.DetectionErrorRate()

    logging.info("Calculating detection error metrics...")
    logging.info(f"Found {len(paired_files)} paired files")
    
    # add reference and hypothesis to metrics
    for key, gt_file, pred_file in paired_files:
        reference, hypothesis = vad_frame_construct_pyannote_object_per_file(pred_file, gt_file)
        metric(reference, hypothesis)  # accumulation

    # delete tmp table files
    # shutil.rmtree(pred_segment_dir, ignore_errors=True)
    # shutil.rmtree(gt_segment_dir, ignore_errors=True)

    report = metric.report(display=False)
    DetER = report.iloc[[-1]][('detection error rate', '%')].item()
    FA = report.iloc[[-1]][('false alarm', '%')].item()
    MISS = report.iloc[[-1]][('miss', '%')].item()
    total = report.iloc[[-1]]['total'].item()

    logging.info(f"parameter {post_params}, DetER={DetER:0.4f}, False Alarm={FA:0.4f}, Miss={MISS:0.4f}")
    metric.reset()  # reset internal accumulator
    return report, pred_segment_dir, gt_segment_dir


def dump_groundtruth_frames(out_dir, labels_map):
    out_dir = Path(out_dir) / Path("frames_groundtruth")
    out_dir.mkdir(exist_ok=True)
    for k, v in labels_map.items():
        out_file = out_dir / Path(f"{k}.frame")
        with out_file.open("a") as fout:
            for x in v:
                fout.write(f"{x}\n")
    return str(out_dir)


def generate_gt_segment_table(
    vad_pred_dir: str, frame_length_in_sec: float, num_workers: int, out_dir: str = None, save_rttm: bool = True
):
    params = {
        "onset": 0.5,  # onset threshold for detecting the beginning and end of a speech
        "offset": 0.5,  # offset threshold for detecting the end of a speech.
        "pad_onset": 0.0,  # adding durations before each speech segment
        "pad_offset": 0.0,  # adding durations after each speech segment
        "min_duration_on": 0.0,  # threshold for small non_speech deletion
        "min_duration_off": 0.0,  # threshold for short speech segment deletion
        "filter_speech_first": True,
        "save_rttm": save_rttm
    }
    vad_table_dir = generate_vad_segment_table(
        vad_pred_dir, params, frame_length_in_sec=frame_length_in_sec, num_workers=num_workers, out_dir=out_dir
    )
    return vad_table_dir


def find_paired_files(pred_dir, gt_dir, ext="rttm"):
    pred_files = list(Path(pred_dir).glob(f"*.{ext}"))
    gt_files = list(Path(gt_dir).glob(f"*.{ext}"))

    gt_file_map = {}
    for filepath in gt_files:
        fname = Path(filepath).stem
        gt_file_map[fname] = str(filepath)

    pred_file_map = {}
    for filepath in pred_files:
        fname = Path(filepath).stem
        pred_file_map[fname] = str(filepath)

    results = []
    for key in gt_file_map:
        if key in pred_file_map:
            results.append((key, gt_file_map[key], pred_file_map[key]))
    return results


if __name__ == '__main__':
    main()
