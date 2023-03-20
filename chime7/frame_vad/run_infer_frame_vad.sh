
#!/bin/bash
NEMO_BASEPATH="/media/data2/chime7-challenge/nemo-gitlab-chime7/"
echo $NEMO_BASEPATH
export PYTHONPATH=$NEMO_BASEPATH:$PYTHONPATH


model_path="./checkpoints/marblenet_3x2x64_Multilang_SynthEn1k_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_ep50_20ms_wce_featnorm-averaged.nemo"
data_dir=/media/data/projects/NeMo-fvad/vad_code/manifests_sd_eval_40ms
dataset="[${data_dir}/dh3_dev_audiobooks_manifest.json,${data_dir}/dh3_dev_broadcast_interview_manifest.json]"

CUDA_VISIBLE_DEVICES=0 python infer_frame_vad.py \
    --config-path="./configs" --config-name="frame_vad_inference" \
    vad.model_path=$model_path \
    frame_out_dir="./frame_vad_output" \
    dataset=${dataset}
