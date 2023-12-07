#!/bin/bash

set -x

NUM_TRIALS=1
WORK_SPACE=/disk_b
CHECKPOINTS=/disk_b/models/checkpoints

NEMO_ROOT="/home/taejinp/projects/challenge_nemo/NeMo"
NEMO_PATH=${NEMO_ROOT}/nemo

OPTUNA_JOB_NAME=infer-e2e-t385-tango4j-NOd-dev_chime7
PATTERN="*-dev.first1.json"
# OPTUNA_JOB_NAME=infer-e2e-t385-tango4j-d03-dev_chime7
# PATTERN="*-dev-d03.first1.json"
SCENARIOS=mixer6
SPLIT=dev


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
--subsets ${SPLIT} --pattern ${PATTERN} --scenarios ${SCENARIOS} \
--gpu_id 0 \
--manifest_path ${MANIFEST_BASE_PATH} \
--config_url ${NEMO_ROOT}/examples/speaker_tasks/diarization/conf/inference/diar_infer_msdd_v2.yaml \
--vad_model_path ${CHECKPOINTS}/frame_vad_chime7_acrobat.nemo \
--msdd_model_path ${CHECKPOINTS}/msdd_v2_PALO_bs6_a003_version6_e53.ckpt \
--asr_model_path ${CHECKPOINTS}/rnnt_ft_chime6ANDmixer6_26jun_avged.nemo \
--lm_path ${CHECKPOINTS}/rnnt_chime6_mixer6_dipco_train_dev.kenlm \
--batch_size ${DIAR_BATCH_SIZE} \
--scenarios ${SCENARIOS} \
--subsets "dev" \
--temp_dir ${WORK_SPACE}/chime7_outputs/${OPTUNA_JOB_NAME}   

### GSS files are saved here:
# /disk_b/chime7_outputs/infer-e2e-t385-chime8-baseline/processed/sys-B-V07-T/mixer6/dev/nemo_v1/20090714_134807_LDC_120290-dev-mdm

