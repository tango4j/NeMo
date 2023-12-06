#!/bin/bash

set -x


### GSS files are saved here:
# /disk_b/chime7_outputs/infer-e2e-t385-chime8-baseline/processed/sys-B-V07-T/mixer6/dev/nemo_v1/20090714_134807_LDC_120290-dev-mdm

NUM_TRIALS=1
WORK_SPACE=/disk_b
CHECKPOINTS=/disk_b/models/checkpoints

# NEMO_ROOT=${WORK_SPACE}/nemo-gitlab-chime7-v2
NEMO_ROOT="/home/taejinp/projects/infer_gitlab_dev_chime7/NeMo"
NEMO_PATH=${NEMO_ROOT}/nemo

# OPTUNA_JOB_NAME=optuna-infer-e2e-t307-eval-part2
OPTUNA_JOB_NAME=infer-e2e-t385-chime8-baseline
SCENARIOS="chime6"
# SPLIT=eval
SPLIT=dev
# PATTERN="*-evaladded-d03.json"
# PATTERN="*-dev.json"
# PATTERN="*-dev.short1.json"
# PATTERN="*-dev-d03-short1.json"
PATTERN="*-dev-d03.first1.json"

# SCRIPT_NAME=optimize_e2e_infer_ngc.py
SCRIPT_NAME=infer_e2e_t385_chime7.py

OPTUNA_LOG=${OPTUNA_JOB_NAME}.log
STORAGE=sqlite:///${OPTUNA_JOB_NAME}.db
DIAR_BATCH_SIZE=11


export KENLM_LIB=$NEMO_PATH/decoders/kenlm/build/bin
export KENLM_ROOT=$NEMO_PATH/decoders/kenlm
export PYTHONPATH=$NEMO_PATH/decoders:$PYTHONPATH
export PYTHONPATH=/usr/lib/python3.8/site-packages:$PYTHONPATH
export PYTHONPATH=/usr/lib/python3.8/site-packages/kenlm-0.0.0-py3.8-linux-x86_64.egg:$PYTHONPATH

MANIFEST_BASE_PATH="/disk_d/datasets/nemo_chime7_diar_manifests/mixer6/mulspk_asr_manifest"

python -c "import kenlm; print('kenlm imported successfully')"
python -c "import kenlm; print(kenlm.__file__)"


python ${SCRIPT_NAME} --n_trials ${NUM_TRIALS} --n_jobs 1 --output_log ${OPTUNA_LOG} --storage ${STORAGE} --output_dir ./speaker_outputs_v3 \
--subsets ${SPLIT} --pattern ${PATTERN} --scenarios $SCENARIOS \
--gpu_id 0 \
--manifest_path ${MANIFEST_BASE_PATH} \
--config_url ${NEMO_ROOT}/examples/speaker_tasks/diarization/conf/inference/diar_infer_msdd_v2.yaml \
--vad_model_path ${CHECKPOINTS}/frame_vad_chime7_acrobat.nemo \
--msdd_model_path ${CHECKPOINTS}/msdd_v2_PALO_bs6_a003_version6_e53.ckpt \
--batch_size ${DIAR_BATCH_SIZE} \
--scenarios "mixer6" \
--subsets "dev" \
--temp_dir ${WORK_SPACE}/chime7_outputs/${OPTUNA_JOB_NAME}   

