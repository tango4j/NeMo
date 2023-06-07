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

In one terminal, start the sqlite3 server:

```bash
sqlite3 optuna.db

# write .quit to close CLI
```

In another terminal, start the optimization:
```bash
python optimize.py --n_jobs -1 --n_trials 100 --study_name chime7
```
Here `n_jobs=-1` will automatically set the number of parallel jobs to the number of GPUs.
