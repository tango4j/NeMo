#!/bin/bash

set -x

DEREBERB="d03"
SUBSETS="eval"
PATTERN="*-eval-${DEREBERB}.json"

NUM_TRIALS=1

NEMO_ROOT=/home/heh/nemo_asr_eval/nemo-gitlab-chime7

OPTUNA_JOB_NAME=optuna-debug
SCRIPT_NAME=optimize_full_local_debug.py

OPTUNA_LOG=${OPTUNA_JOB_NAME}.log
STORAGE=sqlite:///${OPTUNA_JOB_NAME}.db
DIAR_BATCH_SIZE=11


export PYTHONPATH=${NEMO_ROOT}:${PYTHONPATH}

python ${SCRIPT_NAME} --n_trials ${NUM_TRIALS} --n_jobs 1 --output_log ${OPTUNA_LOG} --storage ${STORAGE} --subsets ${SUBSETS} --pattern ${PATTERN} \
--manifest_path ./manifests_${SUBSETS} \
--config_url ${NEMO_ROOT}/examples/speaker_tasks/diarization/conf/inference/diar_infer_msdd_v2.yaml \
--vad_model_path /home/heh/nemo_asr_eval/chime7/checkpoints/frame_vad_chime7_acrobat.nemo \
--msdd_model_path /home/heh/nemo_asr_eval/chime7/checkpoints/msdd_v2_PALO_bs6_a003_version6_e53.ckpt \
--batch_size ${DIAR_BATCH_SIZE} \
--temp_dir /media/data3/chime7_outputs2

set +x
