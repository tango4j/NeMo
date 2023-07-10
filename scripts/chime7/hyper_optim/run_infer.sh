#!/bin/bash

NEMO_ROOT="/media/data2/chime7-challenge/nemo-gitlab-chime7"
# NEMO_ROOT="/home/heh/nemo_asr_eval/nemo-gitlab-chime7"

export PYTHONPATH=${NEMO_ROOT}:${PYTHONPATH}

python run_full_inference.py --output_dir /media/data3/chime7_outputs/trial315_mixer6-debug1 --gpu 0 \
    --manifest_path "manifests_dev" \
    --msdd_model_path  "/media/data2/chime7-challenge/checkpoints/msdd_v2_PALO_bs6_a003_version6_e53.ckpt"
