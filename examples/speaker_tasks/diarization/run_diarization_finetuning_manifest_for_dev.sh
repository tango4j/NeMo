#!/bin/bash

#generate dev set diarization manifest
branch_name="chooper_dl_msdiar"
export PYTHONPATH=/home/chooper/projects/$branch_name/NeMo:$PYTHONPATH
python /home/chooper/projects/$branch_name/NeMo/scripts/speaker_tasks/diarization_finetuning_manifest.py \
    --step_count 50\
    --window 0.5 \
    --shift 0.25 \
    --input_manifest /home/chooper/projects/$branch_name/NeMo/scripts/speaker_tasks/ami_mixheadset_dev_input_manifest.json \
    --output_manifest_path ./ami_mixheadset_dev_input_manifest.json \
