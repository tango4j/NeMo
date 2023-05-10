#!/bin/bash
model_path="/media/data2/chime7-challenge/nemo_asr_chime6_finetuned_rnnt/checkpoints/rno_chime7_chime6_ft_ptDataSetasrset3_frontend_nemoGSSv1_prec32_layers24_heads8_conv5_d1024_dlayers2_dsize640_bs128_adamw_CosineAnnealing_lr0.0001_wd1e-2_spunigram1024.nemo"
# model_path="./rno_chime7_chime6_ft_ptDataSetasrset3_nr_prec32_layers24_heads8_conv5_d1024_bs128_adamw_CosineAnnealing_lr0.00007_wd1e-2_spunigram128.nemo"

scenario="dipco"  # mixer6 dipco chime6
diarout_dir="diarout_dipco-dev.short4_cuda1"  # diarout_mixer6-dev.json_cuda1 diarout_dipco-dev.short4_cuda1 diarout_chime6-dev.json_cuda0
manifest="/media/data2/chime7-challenge/chime7_diar_results/system_vA01/${diarout_dir}/pred_jsons_with_overlap/"

#    pretrained_name=stt_en_conformer_ctc_large \
#     model_path=$model_path \
python transcribe_speech.py \
    batch_size=32 \
    num_workers=8 \
    model_path=$model_path \
    dataset_manifest=$manifest \
    channel_selector="average" \
    output_filename="nemo_experiments/nemo_chime6_ft_rnnt/${scenario}/"
