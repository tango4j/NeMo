


### Must have groundtruth in manifest

model_path="xxxxxxx.nemo"
data_dir=/media/data/projects/NeMo-fvad/vad_code/manifests_sd_eval_40ms
dataset="[${data_dir}/dh3_dev_audiobooks_manifest.json,${data_dir}/dh3_dev_broadcast_interview_manifest.json,${data_dir}/dh3_dev_clinical_manifest.json,${data_dir}/dh3_dev_court_manifest.json,${data_dir}/dh3_dev_cts_manifest.json,${data_dir}/dh3_dev_maptask_manifest.json,${data_dir}/dh3_dev_meeting_manifest.json,${data_dir}/dh3_dev_restaurant_manifest.json,${data_dir}/dh3_dev_socio_field_manifest.json,${data_dir}/dh3_dev_socio_lab_manifest.json,${data_dir}/dh3_dev_webvideo_manifest.json]"

CUDA_VISIBLE_DEVICES=0 python eval_frame_vad.py \
    --config-path="./configs" --config-name="frame_vad_inference" \
    vad.model_path=$model_path \
    frame_out_dir="${ckpt_dir}/frame_vad_output" \
    dataset=${dataset}
