
# TODO: Administrative Section

[] Convert two bash scripts (GSS, RNNT-ASR-BSD) to python based script.
[] Add Whisper text-normalization (as in the original challenge implementation)
[] Add Ante's pre-dereverbration to NeMo Multichannel diarization.
[] Make inference script in (1) Class based structure (2) And make separate yaml file or dataConfig-class
[] Clean and organize environment setting again 
[] Setup and check the training script of NeMo multichannel diarization
[] Setup Optuna script and re-optimize including the new dataset
[] Plug-in 3rd party ASR (Whisper-v3, wavLM) and report the dev/eval-set performance.

# CHiME-8 DASR Baseline Environment Setup

## SubTracks

### SubTrack-1
### SubTrack-2


## Environment Setup

This README outlines the steps to set up your environment for the required operations. Please follow these steps in the order presented to ensure a proper setup.

### Prerequisites

- Ensure that you have `git`, `pip`, and `bash` installed on your system.
- It's assumed that you have CUDA 11.x compatible hardware and drivers installed for `cupy-cuda11x` to work properly.

## TLDR; 

This environment is based on the assumption that you installed the latest `NeMo` on your conda environment named `chime8_baseline`

```bash
conda activate chime8_baseline
pip install espnet
git clone https://github.com/espnet/espnet.git /workspace/espnet
pip uninstall -y 'cupy-cuda118'
pip install --no-cache-dir -f https://pip.cupy.dev/pre/ "cupy-cuda11x[all]==12.1.0"
pip install git+http://github.com/desh2608/gss
pip install optuna
pip install lhotse==1.14.0
pip install --upgrade jiwer
./ngc_install_lm.sh "/your/path/to/NeMo"
```

### Detailed Installation Steps for ESPnet and Related Tools

Use pip to install ESPnet, a toolkit for end-to-end speech processing.

```bash
pip install espnet
```

Clone the ESPnet repository into the `/workspace/espnet` directory.

```bash
git clone https://github.com/espnet/espnet.git /workspace/espnet
```

If you have `cupy-cuda118` installed, uninstall it.

```bash
pip uninstall -y 'cupy-cuda118'
```

Install version `12.1.0` of `cupy-cuda11x`. This is done from a pre-release channel.

```bash
pip install --no-cache-dir -f https://pip.cupy.dev/pre/ "cupy-cuda11x[all]==12.1.0"
```

Install GSS from the provided GitHub repository.

```bash
pip install git+http://github.com/desh2608/gss
```

Optuna, a hyperparameter optimization framework, is installed via pip.

```bash
pip install optuna
```

Install a specific version of Lhotse (`1.14.0`).
Lhotse is mainly needed for Guided Source Separation (GSS) part.

```bash
pip install lhotse==1.14.0
```

Upgrade Jiwer, a package for evaluating automatic speech recognition.
Jiwer is needed for evaluating and normalizing the text.

```bash
pip install --upgrade jiwer
```

Run the script to install the language model.

```bash
./ngc_install_lm.sh
```

# How to launch NeMo CHiME-8 Baseline

## 1. Create NeMo style manifest files

This script will generate NeMo style 

```bash
NEMO_ROOT="/your/path/to/NeMo" # cloned NeMo branch: dev/chime7
NEMO_MSASR_MANIFEST="/your/path/to/nemo_msasr_manifest" # Folder that contains nemo manifest .json files
CHIME7_CLEANED_DATA_FOLDER="/your/path/to/chime7_official_cleaned" # Folder that contains sub-folders named: chime6, dipco, mixer6
python ${NEMO_ROOT}/scripts/chime7/manifests/prepare_nemo_manifest_rttm_ctm_for_infer.py --data-dir ${CHIME7_CLEANED_DATA_FOLDER} --subset ${DATA_SPLIT} --output-dir ${NEMO_MSASR_MANIFEST} \
```

## 2. Setup global varialbes (This is temporary, will be removed in the final format)

Modify the path in the inference script: `<NeMo Root>/scripts/chime7/hyper_optim/infer_e2e_t385_chime7.py`

```python
NEMO_ROOT="/your/path/to/NeMo" 
ESPNET_ROOT="/your/path/to/espnet/egs2/chime7_task1/asr1"
CHIME7_ROOT="/your/path/to/chime7_official_cleaned" 
```


## 3. Launch CHiME-8 Baseline 

```bash
CHECKPOINTS=/your/path/to/checkpoints

NEMO_ROOT="/your/path/to/NeMo"
NEMO_PATH=${NEMO_ROOT}/nemo

OPTUNA_JOB_NAME=infer-e2e-t385-tango4j-NOd-dev_chime7
DATA_SPLIT=dev
PATTERN="*-"${DATA_SPLIT}".json"
SCENARIOS=mixer6


TEMP_DIR=/path/to/temp_dir/chime7_outputs/${OPTUNA_JOB_NAME}  
SCRIPT_NAME=infer_e2e_t385_chime7.py

OPTUNA_LOG=${OPTUNA_JOB_NAME}.log
STORAGE=sqlite:///${OPTUNA_JOB_NAME}.db
DIAR_BATCH_SIZE=11

export KENLM_LIB=$NEMO_PATH/decoders/kenlm/build/bin
export KENLM_ROOT=$NEMO_PATH/decoders/kenlm
export PYTHONPATH=$NEMO_PATH/decoders:$PYTHONPATH
export PYTHONPATH=/usr/lib/python3.8/site-packages:$PYTHONPATH
export PYTHONPATH=/usr/lib/python3.8/site-packages/kenlm-0.0.0-py3.8-linux-x86_64.egg:$PYTHONPATH

NEMO_MSASR_MANIFEST="/your/path/to/nemo_msasr_manifest"
MANIFEST_BASE_PATH="${NEMO_MSASR_MANIFEST}/mixer6/mulspk_asr_manifest"

python -c "import kenlm; print('kenlm imported successfully')" || exit 1

python ${SCRIPT_NAME} --n_trials 1 --n_jobs 1 --output_log ${OPTUNA_LOG} --storage ${STORAGE} --output_dir ./speaker_outputs_v3 \
--subsets ${DATA_SPLIT} --pattern ${PATTERN} --scenarios ${SCENARIOS} \
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
--temp_dir $TEMP_DIR
```