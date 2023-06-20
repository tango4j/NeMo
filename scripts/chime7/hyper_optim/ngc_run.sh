#!/bin/bash

set -x

CONTAINER=nvcr.io/nv-maglev/nemo:chime7-gss
NGC_WORKSPACE=nemo_asr_eval
NGC_JOB_NAME=optuna-msdd-gss-asr
NGC_JOB_LABEL="ml___conformer"
NGC_NODE_TYPE="dgx1v.32g.8.norm"
NUM_TRIALS=100000

NEMO_ROOT=/ws/nemo-gitlab-chime7
OPTUNA_JOB_NAME=optuna-msdd-gss-asr
OPTUNA_LOG=${OPTUNA_JOB_NAME}.log
STORAGE=sqlite:///${OPTUNA_JOB_NAME}.db
DIAR_BATCH_SIZE=11

read -r -d '' cmd <<EOF
cd /ws/chime7_optuna \
&& export PYTHONPATH=${NEMO_ROOT}:${PYTHONPATH} \
&& echo "PYTHONPATH: ${PYTHONPATH}" \
&& python optimize_full_ngc.py --n_trials ${NUM_TRIALS} --n_jobs 6 --output_log ${OPTUNA_LOG} \
--manifest_path /ws/manifests_dev_ngc \
--config_url ${NEMO_ROOT}/examples/speaker_tasks/diarization/conf/inference/diar_infer_msdd_v2.yaml \
--vad_model_path /ws/chime7/checkpoints/frame_vad_chime7_acrobat.nemo \
--msdd_model_path /ws/chime7/checkpoints/msdd_v2_PALO_bs6_a003_version6_e53.ckpt \
--batch_size ${DIAR_BATCH_SIZE}          
EOF

echo "Launching evaluation job on NGC..."
ngc batch run \
  --instance ${NGC_NODE_TYPE} \
  --name ${NGC_JOB_NAME} \
  --image ${CONTAINER} \
  --result /result \
  --workspace ${NGC_WORKSPACE}:/ws \
  --commandline "$cmd" \
  --label "_wl___asr" \
  --label ${NGC_JOB_LABEL}

set + 
