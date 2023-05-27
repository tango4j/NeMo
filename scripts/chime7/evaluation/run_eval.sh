#!/bin/bash
set -eou pipefail

chime7_dir="/data/chime7/chime7_official_cleaned"  

diarization_config=${1:-"oracle"}
tunings=${2:-"nemo_v1"}

for tuning in $tunings
do
    transcribed_dir=nemo_experiments/nemo_chime6_ft_rnnt/${diarization_config}/${tuning}
    output_dir=nemo_experiments/eval_results/${diarization_config}/${tuning}

    echo "Setup"
    echo "transcribed_dir: ${transcribed_dir}"
    echo "output_dir:      ${output_dir}"

    mkdir -p $output_dir

    python evaluate_nemo_asr.py \
        --hyp_folder $transcribed_dir \
        --dasr_root $chime7_dir \
        --partition dev \
        --output_folder $output_dir
done
