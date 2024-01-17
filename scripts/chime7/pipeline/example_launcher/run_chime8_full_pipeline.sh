
CHECKPOINTS=/home/taejinp/projects/challenge_nemo/checkpoints
branch_name="challenge_nemo"
cd /home/taejinp/projects/$branch_name/NeMo
NEMO_ROOT=/home/taejinp/projects/$branch_name/NeMo
NEMO_MSASR_MANIFEST="/disk_d_nvd/datasets/nemo_chime7_diar_manifests"
NEMO_PATH=${NEMO_ROOT}/nemo

OPTUNA_JOB_NAME=infer-e2e-t385-tango4j-NOd-dev_chime7
DATA_SPLIT=dev
PATTERN="*-"${DATA_SPLIT}".json"

TEMP_DIR=/disk_c/temporary_data/chime7_outputs/${OPTUNA_JOB_NAME}
SCRIPT_NAME=${NEMO_ROOT}/scripts/chime7/pipeline/run_full_pipeline.py

OPTUNA_LOG=${OPTUNA_JOB_NAME}.log
STORAGE=sqlite:///${OPTUNA_JOB_NAME}.db
DIAR_BATCH_SIZE=11

MANIFEST_BASE_PATH="${NEMO_MSASR_MANIFEST}/mixer6/mulspk_asr_manifest"

# CHIME_DATA_ROOT=/disk_d/datasets/chime7_official_cleaned_v2 
CHIME_DATA_ROOT=/disk_d/datasets/chime7_official_cleaned_v2_short1mixer6
python -c "import kenlm; print('kenlm imported successfully')" || exit 1

CONFIG_PATH=${NEMO_ROOT}/scripts/chime7/pipeline
YAML_NAME="chime_config_t385.yaml"
ASR_MODEL_PATH=${CHECKPOINTS}/rnnt_ft_chime6ANDmixer6_26jun_avged.nemo
LM_MODEL_PATH=${CHECKPOINTS}/7gram_0001.kenlm
VAD_MODEL_PATH=${CHECKPOINTS}/frame_vad_chime7_acrobat.nemo
MSDD_MODEL_PATH=${CHECKPOINTS}/msdd_v2_PALO_bs6_a003_version6_e53.ckpt 
DIAR_MANIFEST_FILEPATH=/disk_d_nvd/datasets/nemo_chime7_diar_manifests/mixer6/mulspk_asr_manifest/mixer6-dev-d03-short1.json

diar_config="chime8-baseline"

DIAR_BASE_DIR="/disk_c/temporary_data/chime7_outputs/diar_results/"
DIAR_OUT_DIR=/disk_c/temporary_data/chime7_outputs/diar_results/${diar_config}

SITE_PACKAGES=`$(which python) -c 'import site; print(site.getsitepackages()[0])'`

export KENLM_ROOT=$NEMO_ROOT/decoders/kenlm
export KENLM_LIB=$NEMO_ROOT/decoders/kenlm/build/bin
export PYTHONPATH=$NEMO_ROOT/decoders:$PYTHONPATH
export PYTHONPATH=$SITE_PACKAGES/kenlm-0.2.0-py3.10-linux-x86_64.egg:$PYTHONPATH

python ${SCRIPT_NAME} --config-path="${CONFIG_PATH}" --config-name="$YAML_NAME" \
chime_data_root=${CHIME_DATA_ROOT} \
output_root=${TEMP_DIR} \
scenarios="[mixer6]" \
subsets="[dev]" \
diar_base_dir=${DIAR_BASE_DIR} \
diar_config="$diar_config" \
diar_param="T0.5" \
asr_model_path=${ASR_MODEL_PATH} \
lm_model_path=${LM_MODEL_PATH} \
diarizer.vad.model_path=${VAD_MODEL_PATH} \
diarizer.use_saved_embeddings=true \
diarizer.msdd_model.model_path=${MSDD_MODEL_PATH} \
diarizer.manifest_filepath=${DIAR_MANIFEST_FILEPATH} \
diarizer.out_dir=${DIAR_BASE_DIR} \