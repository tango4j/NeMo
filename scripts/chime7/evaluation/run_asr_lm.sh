#!/bin/bash
# model_path="./rno_chime7_chime6_ft_ptDataSetasrset3_nr_prec32_layers24_heads8_conv5_d1024_bs128_adamw_CosineAnnealing_lr0.00007_wd1e-2_spunigram128.nemo"

# SCENARIOS=${1:-"dipco"} # select scenarios to run
SCENARIOS=${1:-"chime6 dipco mixer6"} # select scenarios to run
SUBSETS=${2:-"dev"} # select subsets to run
SYSTEM=${3:-"sys-B-V07-T-trial315"} # select system to run, e.g., system_vA04D, system_vA01, system_B_V05_D03-T0.5, sys-B-V07-T
MANIFEST_DIR_ROOT=${4:-"/home/kdhawan/work/diarization/chime7/data/after_frontend"}
OUTPUT_DIR=${5:-"/home/kdhawan/work/diarization/chime7/output/optuna_asr_lm"}
NORMALIZE_DB=${6:-"-25"}
MODEL_PATH=${7:-"/home/kdhawan/work/diarization/chime7/models/rnnt_ft_chime6ANDmixer6_26jun_avged.nemo"}
# MODEL_PATH=${7:-"/home/kdhawan/work/diarization/chime7/models/checkpoints_ctc/rno_chime7_chime6_ft_ptDataSetasrset3_nr_prec32_layers24_heads8_conv5_d1024_bs128_adamw_CosineAnnealing_lr0.0001_wd1e-2_spunigram128-averaged.nemo"}
BATCH_SIZE=${8:-"1"}
NUM_WORKERS=${9:-"8"}
CHIME7_ROOT=${10:-"/data/chime7_official_cleaned_v2"} # For example, /data/chime7/chime7_official_cleaned
NEMO_CHIME7_ROOT=${11:-"/home/kdhawan/work/diarization/chime7/NeMo/scripts/chime7"} # For example, /media/data2/chime7-challenge/nemo-gitlab-chime7/scripts/chime7
GPU_ID=${12:-"0"}
LM_PATH=${13:-"/home/kdhawan/work/diarization/chime7/models/7gram_0001.kenlm"}
LM_ALPHA=${14:-"0.1"}
LM_BEAM_SIZE=${15:-"8"}
MAES_NUM_STEPS=${16:-"2"}
MAES_ALPHA=${17:-"2"}
MAES_GAMMA=${18:-"4.3"}
MAES_BETA=${19:-"2"}
ASR_OUTPUT_DIR=${20:-"./optuna_asr_lm/nemo_sys-B-V07-T-trial315__rnnt_ft_chime6ANDmixer6_LM"}
EVAL_RESULTS_DIR=${21:-"./optuna_asr_lm/eval_results_nemo_sys-B-V07-T-trial315__rnnt_ft_chime6ANDmixer6_LM"}

# asr_output_dir="${OUTPUT_DIR}/nemo_${SYSTEM}_rnnt_ft_chime6ANDmixer6_LM_ln${NORMALIZE_DB}"

echo "************************************************************"
echo "SCENARIOS:            $SCENARIOS"
echo "SYSTEM:               $SYSTEM"
echo "NORMALIZE_DB:         $NORMALIZE_DB"
echo "OUTPUT_DIR            $OUTPUT_DIR"
echo "ASR_OUTPUT_DIR        $ASR_OUTPUT_DIR"
echo "ASR_OUTPUT_DIR        $EVAL_RESULTS_DIR"
echo "************************************************************"

if [ -z "$EVAL_CHIME" ]; then
    EVAL_CHIME=True
fi

export PYTHONPATH=/workspace/nemo/decoders:$PYTHONPATH
export PYTHONPATH=/usr/lib/python3.8/site-packages:$PYTHONPATH
export PYTHONPATH=/usr/lib/python3.8/site-packages/kenlm-0.0.0-py3.8-linux-x86_64.egg:$PYTHONPATH

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
        --partition ${SUBSETS} \
        --output_folder $EVAL_RESULTS_DIR
fi
