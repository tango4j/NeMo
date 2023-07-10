#!/bin/bash
# model_path="./rno_chime7_chime6_ft_ptDataSetasrset3_nr_prec32_layers24_heads8_conv5_d1024_bs128_adamw_CosineAnnealing_lr0.00007_wd1e-2_spunigram128.nemo"

SCENARIOS=${1:-"mixer6"} # select scenarios to run, chime6 dipco mixer6
SUBSETS=${2:-"eval"} # select subsets to run, dev eval
SYSTEM=${3:-"chime7_gt_diar_segs-eval"} # select system to run, e.g., system_vA04D, system_vA01, system_B_V05_D03-T0.5
MANIFEST_DIR_ROOT=${4:-"/media/data2/chime7-challenge/nemo-gitlab-chime7/scripts/chime7/process/processed"}
OUTPUT_DIR=${5:-"nemo_experiments"}
NORMALIZE_DB=${6:-"-30"}
MODEL_PATH=${7:-"/media/data2/chime7-challenge/nemo_asr_chime6_finetuned_rnnt/checkpoints/rno_chime7_chime6_ft_ptDataSetasrset3_frontend_nemoGSSv1_prec32_layers24_heads8_conv5_d1024_dlayers2_dsize640_bs128_adamw_CosineAnnealing_lr0.0001_wd1e-2_spunigram1024.nemo"}
BATCH_SIZE=${8:-"1"}
NUM_WORKERS=${9:-"8"}
CHIME7_ROOT=${10:-"/media/data2/chime7-challenge/datasets/chime7_official_cleaned_v2"} # For example, /data/chime7/chime7_official_cleaned
NEMO_CHIME7_ROOT=${11:-"/media/data2/chime7-challenge/nemo-gitlab-chime7/scripts/chime7"} # For example, /media/data2/chime7-challenge/nemo-gitlab-chime7/scripts/chime7
GPU_ID=${12:-"1"}
ASR_MODEL_TAG=${13:-"chime6_ft_rnnt"}

asr_output_dir="${OUTPUT_DIR}/nemo_${SYSTEM}_${ASR_MODEL_TAG}_ln${NORMALIZE_DB}"

echo "************************************************************"
echo "SCENARIOS:            $SCENARIOS"
echo "SYSTEM:               $SYSTEM"
echo "NORMALIZE_DB:         $NORMALIZE_DB"
echo "OUTPUT_DIR            $OUTPUT_DIR"
echo $asr_output_dir
echo "************************************************************"

if [ -z "$EVAL_CHIME" ]; then
    EVAL_CHIME=True
fi


# Process all scenarios
# =====================
for scenario in $SCENARIOS
do
for subset in $SUBSETS
do
    echo "--"
    echo $scenario/$subset
    echo "--"
    manifest="${MANIFEST_DIR_ROOT}/${SYSTEM}/${scenario}/${subset}/nemo_v1"
    output_filename="${asr_output_dir}/${scenario}/"
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ${NEMO_CHIME7_ROOT}/evaluation/transcribe_speech.py \
        batch_size=$BATCH_SIZE \
        num_workers=$NUM_WORKERS \
        model_path=$MODEL_PATH \
        dataset_manifest=$manifest \
        channel_selector="average" \
        output_filename=$output_filename \
        normalize_db=$NORMALIZE_DB
done
done

for subset in $SUBSETS
do
if [ $EVAL_CHIME == "True" ]; then
    eval_results_dir="${OUTPUT_DIR}/eval_results_${SYSTEM}_${ASR_MODEL_TAG}_ln${NORMALIZE_DB}"
    python ${NEMO_CHIME7_ROOT}/evaluation/evaluate_nemo_asr.py \
        --hyp_folder $asr_output_dir \
        --dasr_root $CHIME7_ROOT \
        --partition $subset \
        --output_folder $eval_results_dir
fi
done