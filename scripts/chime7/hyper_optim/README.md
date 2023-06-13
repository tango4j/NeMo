# Hyper-parameter Optimization Using Optuna

## Usage

First modify the paths in `optimize.py` to point to the correct data and model directories, for example:
```python
ESPNET_ROOT="/home/heh/github/espnet/egs2/chime7_task1/asr1"
CHIME7_ROOT="/media/data2/chime7-challenge/datasets/chime7_official_cleaned_v2"
NEMO_CHIME7_ROOT="/media/data2/chime7-challenge/nemo-gitlab-chime7/scripts/chime7"
ASR_MODEL_PATH="/media/data2/chime7-challenge/nemo_asr_chime6_finetuned_rnnt/checkpoints/rno_chime7_chime6_ft_ptDataSetasrset3_frontend_nemoGSSv1_prec32_layers24_heads8_conv5_d1024_dlayers2_dsize640_bs128_adamw_CosineAnnealing_lr0.0001_wd1e-2_spunigram1024.nemo"
DIAR_CONFIG="system_B_V05_D03"
DIAR_PARAM="T0.5"
DIAR_BASE_DIR="/media/data2/chime7-challenge/chime7_diar_results"
```

Diarization Model (MSDD v2) 
Access these files at:   
lab@10.110.43.14:/disk_c  
[PassWord](https://docs.google.com/document/d/1IT07_3YkgshtMGrBLW6vrUjRBl_LwaFlseQBjELhZAY/edit?usp=sharing)
```
CONFIG_URL=<dev/chime7 branch NeMo Path>/examples/speaker_tasks/diarization/conf/inference/diar_infer_msdd_v2.yaml
MSDD_MODEL_PATH=/disk_c/models/msdd_v2_models/msdd_v2_PALO_bs6_a003_version6_e53.ckpt
VAD_MODEL_PATH=/disk_c/taejinp/gdrive/model/VAD_models/frame_vad_chime7_acrobat.nemo
test_manifest=/disk_d/datasets/nemo_chime7_diar_manifests/chdipmixALL-dev-d03.json
```

In a terminal, start the optimization:
```bash
python optimize.py --n_jobs -1 --n_trials 100 --study_name chime7
```
Here `n_jobs=-1` will automatically set the number of parallel jobs to the number of GPUs.
