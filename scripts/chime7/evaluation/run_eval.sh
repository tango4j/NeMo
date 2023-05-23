#!/bin/bash

# pred_dir="nemo_experiments/nemo_chime6_ft_ctc/"
pred_dir="nemo_experiments/nemo_system_vA04D_chime6_ft_rnnt_v2/"

chime7_dir="/media/data2/chime7-challenge/datasets/chime7_official_cleaned_v2"  
python evaluate_nemo_asr.py \
    --hyp_folder $pred_dir \
    --dasr_root $chime7_dir \
    --partition dev \
    --output_folder "nemo_experiments/eval_results_nemo_system_vA04D_chime6_ft_rnnt_v2"
