#!/bin/bash

pred_dir="nemo_experiments/nemo_ctc/"
chime7_dir="/media/data2/chime7-challenge/datasets/chime7_official_cleaned"  
python evaluate_nemo_asr.py \
    --hyp_folder $pred_dir \
    --dasr_root $chime7_dir \
    --partition dev \
    --output_folder "nemo_experiments/eval_results"
