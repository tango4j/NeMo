#!/bin/bash

# Create manifest file with alignments
python create_alignment_manifest.py \
  --input_manifest_filepath <path to LibriSpeech train-clean-100 json manifest> \
  --base_alignment_path <path to LibriSpeech_Alignments folder from https://drive.google.com/file/d/1WYfgr31T-PPwMcxuAq09XZfHQO5Mw8fE> \
  --dataset train-clean-100 \
  --output_path train-clean-100-align.json

# Create diarization session
python create_diarization_dataset_librispeech_multimic.py \
  data_simulator.random_seed=42 \
  data_simulator.manifest_path=./train-clean-100-align.json \
  data_simulator.output_dir=./test
