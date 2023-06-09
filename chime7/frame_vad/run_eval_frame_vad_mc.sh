#!/bin/bash
NEMO_BASEPATH="/media/data2/chime7-challenge/nemo-gitlab-chime7/"
echo $NEMO_BASEPATH
export PYTHONPATH=$NEMO_BASEPATH:$PYTHONPATH

model_path="./checkpoints/marblenet_3x2x64_Multilang_SynthEn1k_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_ep50_20ms_wce_featnorm-averaged.nemo"
data_dir=/media/data/projects/NeMo-fvad/vad_code/manifests_sd_eval_40ms
### Must have groundtruth in manifest
# dataset="[${data_dir}/dh3_dev_audiobooks_manifest.json,${data_dir}/dh3_dev_broadcast_interview_manifest.json]"
dataset="/media/data2/chime7-challenge/datasets/manifests_task1/dipco/mulspk_asr_manifest/dipco-dev.json"


CUDA_VISIBLE_DEVICES=0 python eval_frame_vad_mc.py \
    --config-path="./configs" --config-name="frame_vad_inference" \
    vad.model_path=$model_path \
    vad.parameters.normalize_audio=False \
    vad.parameters.normalize_audio_target=-25 \
    frame_out_dir="./nemo_experiments/frame_vad_output_nonorm" \
    dataset=${dataset}
