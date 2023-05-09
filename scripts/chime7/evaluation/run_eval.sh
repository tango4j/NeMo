#!/bin/bash

# pred_dir="nemo_experiments/nemo_chime6_ft_ctc/"
pred_dir="nemo_experiments/nemo_chime6_ft_rnnt/"

chime7_dir="/media/data2/chime7-challenge/datasets/chime7_official_cleaned"  
python evaluate_nemo_asr.py \
    --hyp_folder $pred_dir \
    --dasr_root $chime7_dir \
    --partition dev \
    --output_folder "nemo_experiments/eval_results"
