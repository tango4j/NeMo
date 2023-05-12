#!/bin/bash

set -eou pipefail

model_path="${HOME}/work/models/stt_chime7/ft_chime6_v2/rno_chime7_chime6_ft_ptDataSetasrset3_frontend_nemoGSSv1_prec32_layers24_heads8_conv5_d1024_dlayers2_dsize640_bs128_adamw_CosineAnnealing_lr0.0001_wd1e-2_spunigram1024.nemo"
# model_path="/media/data2/chime7-challenge/nemo_asr_chime6_finetuned_rnnt/checkpoints/rno_chime7_chime6_ft_ptDataSetasrset3_frontend_nemoGSSv1_prec32_layers24_heads8_conv5_d1024_dlayers2_dsize640_bs128_adamw_CosineAnnealing_lr0.0001_wd1e-2_spunigram1024.nemo"
# model_path="./rno_chime7_chime6_ft_ptDataSetasrset3_nr_prec32_layers24_heads8_conv5_d1024_bs128_adamw_CosineAnnealing_lr0.00007_wd1e-2_spunigram128.nemo"

subset=dev

for scenario in mixer6 dipco chime6
do
for tuning in nemo_v1
do
    data_dir=/home/ajukic/scratch/chime7/processed/nemo_dev/complete_dev/gss_rerun_0329/${scenario}/${tuning}/

    # Prepare manifests
    python ../process/prepare_nemo_manifests_for_processed.py --data-dir ${data_dir}

    #    pretrained_name=stt_en_conformer_ctc_large \
    #     model_path=$model_path \
    CUDA_VISIBLE_DEVICES=0 python transcribe_speech.py \
        batch_size=32 \
        num_workers=8 \
        model_path=$model_path \
        dataset_manifest=$data_dir \
        channel_selector="average" \
        output_filename="nemo_experiments/nemo_chime6_ft_rnnt/oracle_diarization/${tuning}/${scenario}/"
done
done