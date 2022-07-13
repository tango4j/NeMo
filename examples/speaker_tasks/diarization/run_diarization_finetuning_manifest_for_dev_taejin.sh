#!/bin/bash

#generate dev set diarization manifest
branch_name="chooper_dl"
export PYTHONPATH=/home/taejin/projects/$branch_name/NeMo:$PYTHONPATH
python /home/taejin/projects/$branch_name/NeMo/scripts/speaker_tasks/diarization_finetuning_manifest.py \
    --step_count 50\
    --window 0.5 \
    --shift 0.25 \
    --input_manifest ami_mixheadset_dev_input_manifest.json \
    --output_manifest_path <path to output segment manifest file> \
