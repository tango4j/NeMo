#!/bin/bash

export PYTHONPATH=/home/chooper/projects/chooper_dl/NeMo:$PYTHONPATH

# Create manifest file with alignments
python create_alignment_manifest.py \
  --input_manifest_filepath /home/chooper/projects/datasets/LibriSpeech/dev_clean.json \
  --base_alignment_path /home/chooper/projects/branch_nemo/NeMo/scripts/speaker_tasks/LibriSpeech_Alignments/ \
  --dataset train-clean-100 \
  --output_path train-clean-100-align.json

# Create diarization session
python create_diarization_dataset_librispeech_multimic.py \
  data_simulator.random_seed=42 \
  data_simulator.manifest_path=./train-clean-100-align.json \
  data_simulator.output_dir=./test
