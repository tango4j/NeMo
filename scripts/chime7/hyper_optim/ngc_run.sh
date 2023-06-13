#!/bin/bash

set -x

CONTAINER=nvcr.io/nv-maglev/nemo:chime7-gss
NGC_WORKSPACE=nemo_asr_eval
NGC_JOB_NAME=optuna-gss-asr
NGC_JOB_LABEL="ml___conformer"
NGC_NODE_TYPE="dgx1v.32g.8.norm"
NUM_TRIALS=1000

read -r -d '' cmd <<EOF
cd /ws/chime7 \
&& python optimize_ngc.py --n_trials ${NUM_TRIALS}                                             
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
