#!/bin/bash

set -x

CONTAINER=nvcr.io/nvidia/nemo:22.12
NGC_WORKSPACE=nemo_asr_eval
NGC_JOB_NAME=optuna-gss-dev
NGC_JOB_LABEL="ml___conformer"
NGC_NODE_TYPE="dgx1v.32g.8.norm"
NUM_TRIALS=1000000

NEMO_ROOT=/ws/nemo-gitlab-chime7


###############################
SUBSETS="dev"
OPTUNA_JOB_NAME=optuna-gss-dev
SCENARIOS="chime6 dipco mixer6"
STUDY_NAME=chime7_dev_trial315_all
###############################


SCRIPT_NAME=optimize_gss_ngc.py
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
&& python ${SCRIPT_NAME} --n_trials ${NUM_TRIALS} --n_jobs 8 --output_log ${OPTUNA_LOG} --storage ${STORAGE} --temp_dir /raid/temp --study_name ${STUDY_NAME} --scenarios $SCENARIOS --subsets $SUBSETS      
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
