#!/bin/bash

set -eou pipefail

model_path="${HOME}/work/models/stt_chime7/ft_chime6_v2/rno_chime7_chime6_ft_ptDataSetasrset3_frontend_nemoGSSv1_prec32_layers24_heads8_conv5_d1024_dlayers2_dsize640_bs128_adamw_CosineAnnealing_lr0.0001_wd1e-2_spunigram1024.nemo"
# model_path="/media/data2/chime7-challenge/nemo_asr_chime6_finetuned_rnnt/checkpoints/rno_chime7_chime6_ft_ptDataSetasrset3_frontend_nemoGSSv1_prec32_layers24_heads8_conv5_d1024_dlayers2_dsize640_bs128_adamw_CosineAnnealing_lr0.0001_wd1e-2_spunigram1024.nemo"
# model_path="./rno_chime7_chime6_ft_ptDataSetasrset3_nr_prec32_layers24_heads8_conv5_d1024_bs128_adamw_CosineAnnealing_lr0.00007_wd1e-2_spunigram128.nemo"

scenario=${1} # mixer6 dipco chime6
diarization_config=${2:-"oracle"}
tunings=${3:-"nemo_v1"}
subset=dev

for tuning in $tunings
do
    data_dir=$(dirname ${PWD})/process/processed_0626/${diarization_config}/${scenario}/${subset}/${tuning}

    # Prepare manifests
    python ../process/prepare_nemo_manifests_for_processed.py --data-dir $data_dir

    echo "--"
    echo "Transcribe"
    echo "data_dir: ${data_dir}"
    echo "--"

    python transcribe_speech.py \
        batch_size=32 \
        num_workers=8 \
        model_path=$model_path \
        dataset_manifest=$data_dir \
        channel_selector="average" \
        output_filename="nemo_experiments/nemo_chime6_ft_rnnt/${diarization_config}/${tuning}/${scenario}/"
done