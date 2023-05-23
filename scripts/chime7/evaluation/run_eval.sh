#!/bin/bash

chime7_dir="/data/chime7/chime7_official_cleaned"  
diarization_config=system_vA01

for tuning in nemo_v1
do
    for tag in processed_without_garbage_class processed_use_grabage_class
    do
        transcribed_dir=nemo_experiments/nemo_chime6_ft_rnnt/${tag}/${diarization_config}/${tuning}
        output_dir=nemo_experiments/eval_results/${tag}/${diarization_config}/${tuning}

        mkdir -p $output_dir

        python evaluate_nemo_asr.py \
            --hyp_folder $transcribed_dir \
            --dasr_root $chime7_dir \
            --partition dev \
            --output_folder $output_dir
    done
done