

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

```bash
pip install lhotse==1.14.0
```

Upgrade Jiwer, a package for evaluating automatic speech recognition.

```bash
pip install --upgrade jiwer
```

Run the script to install the language model.

```bash
./ngc_install_lm.sh
```