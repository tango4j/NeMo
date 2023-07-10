#!/bin/bash

set -x

CONTAINER=nvcr.io/nvidia/nemo:22.12
NGC_WORKSPACE=nemo_asr_eval
NGC_JOB_LABEL="ml___conformer"
NGC_NODE_TYPE="dgx1v.32g.8.norm"
NUM_TRIALS=1000000

NEMO_ROOT=/ws/nemo-gitlab-chime7


dereberb_version="d03"
NGC_JOB_NAME=optuna-full-eval2-${dereberb_version}
OPTUNA_JOB_NAME="optuna-msdd-gss-asr-eval2-${dereberb_version}"


SCRIPT_NAME=optimize_full_ngc_eval.py

PATTERN="*-eval-${dereberb_version}.json"
SUBSETS="eval"
OPTUNA_LOG=${OPTUNA_JOB_NAME}.log
STORAGE=sqlite:///${OPTUNA_JOB_NAME}.db
DIAR_BATCH_SIZE=11


read -r -d '' cmd <<EOF
cd /ws/chime7_optuna \
&& df -h \
&& export PYTHONPATH=${NEMO_ROOT}:${PYTHONPATH} \
&& pip install espnet \
&& git clone https://github.com/espnet/espnet.git /workspace/espnet \
&& pip uninstall -y 'cupy-cuda118' \
&& pip install --no-cache-dir -f https://pip.cupy.dev/pre/ "cupy-cuda11x[all]==12.1.0" \
&& pip install git+http://github.com/desh2608/gss \
&& pip install optuna \
&& pip install lhotse==1.14.0 \
&& pip install --upgrade jiwer \
&& python ${SCRIPT_NAME} --n_trials ${NUM_TRIALS} --n_jobs 5 --output_log ${OPTUNA_LOG} --storage ${STORAGE} --pattern ${PATTERN} --subsets ${SUBSETS} \
--manifest_path /ws/manifests_eval_ngc \
--config_url ${NEMO_ROOT}/examples/speaker_tasks/diarization/conf/inference/diar_infer_msdd_v2.yaml \
--vad_model_path /ws/chime7/checkpoints/frame_vad_chime7_acrobat.nemo \
--msdd_model_path /ws/chime7/checkpoints/msdd_v2_PALO_bs6_a003_version6_e53.ckpt \
--batch_size ${DIAR_BATCH_SIZE} \
--temp_dir /raid/temp \
--keep_mixer6         
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
