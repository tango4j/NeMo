#!/bin/bash

chime7_dir="/data/chime7/chime7_official_cleaned"  

for tuning in nemo_v1
do
    transcribed_dir=nemo_experiments/nemo_chime6_ft_rnnt/oracle_diarization/${tuning}
    output_dir=nemo_experiments/eval_results/oracle_diarization/${tuning}

    mkdir -p $output_dir

    python evaluate_nemo_asr.py \
        --hyp_folder $transcribed_dir \
        --dasr_root $chime7_dir \
        --partition dev \
        --output_folder $output_dir
done