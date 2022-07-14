#!/bin/bash
export PYTHONPATH=/home/chooper/projects/chooper_dl/NeMo:$PYTHONPATH

# Create initial diarization manifest file (used as input for run_diarization_finetuning_manifest_for_synthetic.sh)
python pathsfiles_to_manifest.py \
  --paths2audio_files <path to ami_mixheadset_dev_wav.list> \
  --paths2txt_files <path to ami_mixheadset_dev_transcript.list> \
  --paths2rttm_files <path to ami_mixheadset_dev_rttm.list> \
  --paths2ctm_files <path to ami_mixheadset_dev_ctm.list>
  --manifest_filepath ami_mixheadset_dev_input_manifest.json

# Create manifest file with alignments (used as data simulator input in run_msdd_train_synthetic_taejin.sh)
python create_alignment_manifest.py \
  --input_manifest_filepath <path to LibriSpeech train-clean-100 json manifest> \
  --base_alignment_path <path to LibriSpeech_Alignments folder from https://drive.google.com/file/d/1WYfgr31T-PPwMcxuAq09XZfHQO5Mw8fE> \
  --dataset train-clean-100 \
  --output_path train-clean-100-align.json
