#!/bin/bash

# pred_dir="nemo_experiments/nemo_chime6_ft_ctc/"
# system="baseline_diar-dev"  # system_vA04D system_vA01 system_vA04D-T0.5 system_B_V05_D03-T0.5 
# pred_dir="nemo_experiments/nemo_${system}_chime6_ft_rnnt_v2_ln25/"
# pred_dir="/home/heh/nemo_asr_eval/chime7_outputs/optuna-infer-e2e-t307-dev/nemo_asr_output"
# pred_dir="/home/heh/nemo_asr_eval/chime7_outputs/optuna-infer-e2e-t307-eval/nemo_asr_output"
# pred_dir="/home/heh/nemo_asr_eval/chime7_outputs/optuna-infer-e2e-t294-eval/nemo_asr_output"
pred_dir="/home/heh/nemo_asr_eval/chime7_outputs/optuna-infer-e2e-t385-eval/nemo_asr_output"

chime7_dir="/media/data2/chime7-challenge/datasets/chime7_official_cleaned_v2"  
python evaluate_nemo_asr.py \
    --hyp_folder $pred_dir \
    --dasr_root $chime7_dir \
    --partition eval \
    --output_folder "nemo_experiments/eval_results_e2e-t385-eval"
