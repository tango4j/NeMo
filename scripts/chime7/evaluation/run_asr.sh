#!/bin/bash
model_path="/media/data2/chime7-challenge/nemo_asr_chime6_finetuned_rnnt/checkpoints/rno_chime7_chime6_ft_ptDataSetasrset3_frontend_nemoGSSv1_prec32_layers24_heads8_conv5_d1024_dlayers2_dsize640_bs128_adamw_CosineAnnealing_lr0.0001_wd1e-2_spunigram1024.nemo"
# model_path="./rno_chime7_chime6_ft_ptDataSetasrset3_nr_prec32_layers24_heads8_conv5_d1024_bs128_adamw_CosineAnnealing_lr0.00007_wd1e-2_spunigram128.nemo"

SCENARIOS=${1:-"chime6 dipco mixer6"} # select scenarios to run
SUBSETS=${2:-"dev"} # select subsets to run
SYSTEM=${3:-"baseline_diar-dev"} # select system to run, e.g., system_vA04D, system_vA01, system_B_V05_D03-T0.5
MANIFEST_DIR="/media/data2/chime7-challenge/nemo-gitlab-chime7/scripts/chime7/process/processed"

normalize_db=-25

# Process all scenarios
# =====================
for scenario in $SCENARIOS
do
for subset in $SUBSETS
do
    echo "--"
    echo $scenario/$subset
    echo "--"
    manifest="${MANIFEST_DIR}/${SYSTEM}/${scenario}/${subset}/nemo_v1"
    python transcribe_speech.py \
        batch_size=1 \
        num_workers=8 \
        model_path=$model_path \
        dataset_manifest=$manifest \
        channel_selector="average" \
        output_filename="nemo_experiments/nemo_${SYSTEM}_chime6_ft_rnnt_v2_ln25/${scenario}/" \
        normalize_db=$normalize_db
done
done