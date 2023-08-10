#!/bin/bash

set -x

CONTAINER=nvcr.io/nvidia/nemo:22.12
NGC_WORKSPACE=nemo_asr_eval
NGC_JOB_NAME=optuna-infer-lmv2-eval-t385-s19
NGC_JOB_LABEL="ml___conformer"
NGC_NODE_TYPE="dgx1v.32g.8.norm"
NUM_TRIALS=1

NEMO_ROOT=/ws/nemo-gitlab-chime7-v2

TRIAL_ID=385
OPTUNA_JOB_NAME=optuna-infer-e2e-t${TRIAL_ID}-eval-s19p2
SPLIT=eval
PATTERN="*-evaladded-d03.json"
SCENARIOS="chime6"

SCRIPT_NAME=optimize_e2e_infer_ngc_t${TRIAL_ID}.py

OPTUNA_LOG=${OPTUNA_JOB_NAME}.log
STORAGE=sqlite:///${OPTUNA_JOB_NAME}.db
DIAR_BATCH_SIZE=11

LM_INSTALL_SCRIPT=${NEMO_ROOT}/stable/scripts/asr_language_modeling/ngram_lm/install_beamsearch_decoders.sh

read -r -d '' cmd <<EOF
cd /ws/chime7_optuna \
&& df -h \
&& pip install espnet \
&& git clone https://github.com/espnet/espnet.git /workspace/espnet \
&& pip uninstall -y 'cupy-cuda118' \
&& pip install --no-cache-dir -f https://pip.cupy.dev/pre/ "cupy-cuda11x[all]==12.1.0" \
&& pip install git+http://github.com/desh2608/gss \
&& pip install optuna \
&& pip install lhotse==1.14.0 \
&& pip install --upgrade jiwer \
&& ./ngc_install_lm.sh \
&& export PYTHONPATH=${NEMO_ROOT}:${PYTHONPATH} \
&& python ${SCRIPT_NAME} --n_trials ${NUM_TRIALS} --n_jobs 1 --output_log ${OPTUNA_LOG} --storage ${STORAGE} --output_dir ./speaker_outputs_v3 \
--subsets ${SPLIT} --pattern ${PATTERN} --scenarios $SCENARIOS \
--manifest_path /ws/manifests_${SPLIT}_ngc/chime6-s19 \
--config_url ${NEMO_ROOT}/examples/speaker_tasks/diarization/conf/inference/diar_infer_msdd_v2.yaml \
--vad_model_path /ws/chime7/checkpoints/frame_vad_chime7_acrobat.nemo \
--msdd_model_path /ws/chime7/checkpoints/msdd_v2_PALO_bs6_a003_version6_e53.ckpt \
--batch_size ${DIAR_BATCH_SIZE} \
--temp_dir /ws/chime7_outputs/${OPTUNA_JOB_NAME}   
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
