#!/bin/bash
# model_path="./rno_chime7_chime6_ft_ptDataSetasrset3_nr_prec32_layers24_heads8_conv5_d1024_bs128_adamw_CosineAnnealing_lr0.00007_wd1e-2_spunigram128.nemo"

# SCENARIOS=${1:-"dipco"} # select scenarios to run
SCENARIOS=${1:-"chime6 dipco mixer6"} # select scenarios to run
SUBSETS=${2:-"dev"} # select subsets to run
SYSTEM=${3:-"sys-B-V07-T"} # select system to run, e.g., system_vA04D, system_vA01, system_B_V05_D03-T0.5, sys-B-V07-T
MANIFEST_DIR_ROOT=${4:-"/home/heh/nemo_asr_eval/chime7_outputs/optuna-infer-e2e-t294-dev/processed"}
OUTPUT_DIR=${5:-"./optuna_asr_lm"}
NORMALIZE_DB=${6:-"-20"}
MODEL_PATH=${7:-"/media/data2/chime7-challenge/checkpoints/rnnt_ft_chime6ANDmixer6_26jun_avged.nemo"}
BATCH_SIZE=${8:-"1"}
NUM_WORKERS=${9:-"8"}
CHIME7_ROOT=${10:-"/media/data2/chime7-challenge/datasets/chime7_official_cleaned_v2"} # For example, /data/chime7/chime7_official_cleaned
NEMO_CHIME7_ROOT=${11:-"/media/data2/chime7-challenge/nemo-gitlab-chime7/scripts/chime7"} # For example, /media/data2/chime7-challenge/nemo-gitlab-chime7/scripts/chime7
GPU_ID=${12:-"0"}
LM_PATH=${13:-"/media/data2/chime7-challenge/checkpoints/7gram_0001.kenlm"}
LM_ALPHA=${14:-"0.0"}
LM_BEAM_SIZE=${15:-"7"}
MAES_NUM_STEPS=${16:-"5"}
MAES_ALPHA=${17:-"3"}
MAES_GAMMA=${18:-"4.3"}
MAES_BETA=${19:-"5"}
ASR_OUTPUT_DIR=${20:-"./optuna_asr_lm/asr_optuna-infer-e2e-t294-dev"}
EVAL_RESULTS_DIR=${21:-"./optuna_asr_lm/eval_results_optuna-infer-e2e-t294-dev"}

# asr_output_dir="${OUTPUT_DIR}/nemo_${SYSTEM}_rnnt_ft_chime6ANDmixer6_LM_ln${NORMALIZE_DB}"

echo "************************************************************"
echo "SCENARIOS:            $SCENARIOS"
echo "SYSTEM:               $SYSTEM"
echo "NORMALIZE_DB:         $NORMALIZE_DB"
echo "OUTPUT_DIR            $OUTPUT_DIR"
echo $ASR_OUTPUT_DIR
echo "************************************************************"

if [ -z "$EVAL_CHIME" ]; then
    EVAL_CHIME=True
fi

# DECODER_PATH="/media/data3/decoders"
# export PYTHONPATH=$DECODER_PATH:$PYTHONPATH
# export PYTHONPATH=/workspace/nemo/decoders:$PYTHONPATH
# export PYTHONPATH=/usr/lib/python3.8/site-packages:$PYTHONPATH
# export PYTHONPATH=/usr/lib/python3.8/site-packages/kenlm-0.0.0-py3.8-linux-x86_64.egg:$PYTHONPATH

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
    output_filename="${ASR_OUTPUT_DIR}/${scenario}/"
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ${NEMO_CHIME7_ROOT}/evaluation/transcribe_speech.py \
        batch_size=$BATCH_SIZE \
        num_workers=$NUM_WORKERS \
        model_path=$MODEL_PATH \
        dataset_manifest=$manifest \
        channel_selector="average" \
        output_filename=$output_filename \
        normalize_db=$NORMALIZE_DB \
        rnnt_decoding.strategy="maes" \
        rnnt_decoding.beam.ngram_lm_model=$LM_PATH \
        rnnt_decoding.beam.ngram_lm_alpha=$LM_ALPHA \
        rnnt_decoding.beam.beam_size=$LM_BEAM_SIZE \
        rnnt_decoding.beam.return_best_hypothesis=true \
        rnnt_decoding.beam.maes_num_steps=$MAES_NUM_STEPS \
        rnnt_decoding.beam.maes_prefix_alpha=$MAES_ALPHA \
        rnnt_decoding.beam.maes_expansion_gamma=$MAES_GAMMA \
        rnnt_decoding.beam.maes_expansion_beta=$MAES_BETA
done
done



# eval_results_dir="${OUTPUT_DIR}/eval_results_${SYSTEM}_rnnt_ft_chime6ANDmixer6_LM_ln${NORMALIZE_DB}"
if [ $EVAL_CHIME == "True" ]; then
    python ${NEMO_CHIME7_ROOT}/evaluation/evaluate_nemo_asr.py \
        --hyp_folder $ASR_OUTPUT_DIR \
        --dasr_root $CHIME7_ROOT \
        --partition $SUBSETS \
        --output_folder $EVAL_RESULTS_DIR
fi
