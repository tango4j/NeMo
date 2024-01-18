
### TODO: Administrative Section

- [x] Clone internal NVIDIA NeMo Gitlab repo branch to `github.com/tango4j/NeMo/tree/dev/chime7`.  

- [x] Push NeMo manifest creation script to `https://github.com/tango4j/NeMo/tree/dev/chime7/scripts/chime7`.  

- [x] Convert two bash scripts (GSS, RNNT-ASR-BSD) to python based script.  

- [ ] Add Whisper text-normalization (as in the original challenge implementation)

- [ ] Add Ante's pre-dereverbration to NeMo Multichannel diarization.

- [ ] Make inference script in (1) Class based structure (2) And make separate yaml file or dataConfig-class

- [x] Clean and organize environment setting again 

- [ ] Setup and check the training script of NeMo multichannel diarization

- [ ] Setup Optuna script and re-optimize including the new dataset

- [ ] Plug-in 3rd party ASR (Whisper-v3, wavLM) and report the dev/eval-set performance.

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
./hyper_optim/ngc_install_lm.sh "/your/path/to/NeMo"
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
./hyper_optim/ngc_install_lm.sh
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

Before launch the following script, make sure to activate your Conda environment.
```bash
conda activate chime8_baseline
```

Make sure to setup your CHIME8 Data path, temporary directory with write permissions and NeMo root path where NeMo toolkit is cloned.

```bash
###########################################################################
### YOUR CUSTOMIZED CONFIGURATIONS HERE ###################################
NEMO_ROOT=/path/to/NeMo
CHECKPOINTS=/path/to/checkpoints
TEMP_DIR=/temp/path/to/chime8_baseline_each1sess
CHIME_DATA_ROOT=/path/to/chime7_official_cleaned
SCENARIOS="[mixer6,chime6,dipco]"
DIAR_CONFIG="chime8-baseline-mixer6-short1"
###########################################################################
cd $NEMO_ROOT
export CUDA_VISIBLE_DEVICES="0"

SCRIPT_NAME=${NEMO_ROOT}/scripts/chime7/pipeline/run_full_pipeline.py
python -c "import kenlm; print('kenlm imported successfully')" || exit 1

CONFIG_PATH=${NEMO_ROOT}/scripts/chime7/pipeline
YAML_NAME="chime_config_t385.yaml"

ASR_MODEL_PATH=${CHECKPOINTS}/rnnt_ft_chime6ANDmixer6_26jun_avged.nemo
LM_MODEL_PATH=${CHECKPOINTS}/7gram_0001.kenlm
VAD_MODEL_PATH=${CHECKPOINTS}/frame_vad_chime7_acrobat.nemo
MSDD_MODEL_PATH=${CHECKPOINTS}/msdd_v2_PALO_bs6_a003_version6_e53.ckpt 

SITE_PACKAGES=`$(which python) -c 'import site; print(site.getsitepackages()[0])'`
export KENLM_ROOT=$NEMO_ROOT/decoders/kenlm
export KENLM_LIB=$NEMO_ROOT/decoders/kenlm/build/bin
export PYTHONPATH=$NEMO_ROOT/decoders:$PYTHONPATH
export PYTHONPATH=$SITE_PACKAGES/kenlm-0.2.0-py3.10-linux-x86_64.egg:$PYTHONPATH
export PYTHONPATH=$NEMO_ROOT:$PYTHONPATH

python ${SCRIPT_NAME} --config-path="${CONFIG_PATH}" --config-name="$YAML_NAME" \
diar_config=${DIAR_CONFIG} \
chime_data_root=${CHIME_DATA_ROOT} \
output_root=${TEMP_DIR} \
scenarios=${SCENARIOS} \
subsets="[dev]" \
asr_model_path=${ASR_MODEL_PATH} \
lm_model_path=${LM_MODEL_PATH} \
diarizer.vad.model_path=${VAD_MODEL_PATH} \
diarizer.msdd_model.model_path=${MSDD_MODEL_PATH} \
```