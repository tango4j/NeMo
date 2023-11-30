#!/bin/bash

# NEMO_ROOT="/media/data2/chime7-challenge/nemo-gitlab-chime7"
NEMO_ROOT="/home/heh/nemo_asr_eval/nemo-gitlab-chime7"

export PYTHONPATH=${NEMO_ROOT}:${PYTHONPATH}

python run_full_inference.py --output_dir /media/data3/chime7_outputs2/trial315_eval --gpu 0 \
    --pattern "*-eval-d03.json" \
    --manifest_path "manifests_eval" \
    --msdd_model_path  "/media/data2/chime7-challenge/checkpoints/msdd_v2_PALO_bs6_a003_version6_e53.ckpt"
