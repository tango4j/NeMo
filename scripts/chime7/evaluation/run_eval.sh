#!/bin/bash

chime7_dir="/data/chime7/chime7_official_cleaned"  
diarization_config=system_vA01

for tuning in nemo_v1 nemo_v1p1 nemo_v1p3 nemo_v1p4 nemo_v1_tune_block_one
do
    transcribed_dir=nemo_experiments/nemo_chime6_ft_rnnt/${diarization_config}/${tuning}
    output_dir=nemo_experiments/eval_results/${diarization_config}/${tuning}

    mkdir -p $output_dir

    python evaluate_nemo_asr.py \
        --hyp_folder $transcribed_dir \
        --dasr_root $chime7_dir \
        --partition dev \
        --output_folder $output_dir
done