
branch_name="gitlab_chime7_NeMo"

cd /home/taejinp/projects/$branch_name/NeMo
BASEPATH=/home/taejinp/projects/$branch_name/NeMo
echo "Running MSDD v2 in the path $PWD"

export PYTHONPATH=/home/taejinp/projects/$branch_name/NeMo:$PYTHONPATH
MEMO='MS_'$SPLIT'_'$branch_name_"$CUDA_VISIBLE_DEVICES"

################################################################################################
### < Need Your Change 1 >
####### Diarization model path
# lab@10.110.43.14:/disk_c_nvd/models/msdd_v2_models/msdd_v2_PALO_bs6_a003_version6_e53.ckpt

####### VAD model path (This is needed to replace the fine-tuned VAD model in the diarization model)
# lab@10.110.43.14:/disk_c_nvd/models/frame_vad_models/wnc_frame_vad.nemo

# PASSWORD for lab@10.110.43.14 is in the Google Docs : https://docs.google.com/document/d/1IT07_3YkgshtMGrBLW6vrUjRBl_LwaFlseQBjELhZAY/edit?usp=sharing

# MSDD_MODEL_PATH='/disk_c_nvd/taejinp_backup/msdd_model_train/MSDD_v2_PALO_BS6_a0.003/version_6/checkpoints/epoch53.ckpt' # 3.0 scale libVox, TnFrz
MSDD_MODEL_PATH="/disk_c/models/msdd_v2_models/msdd_v2_PALO_bs6_a003_version6_e53.ckpt"
VAD_MODEL_PATH="/disk_c/models/frame_vad_models/wnc_frame_vad.nemo"
################################################################################################

DEREV="-d03"
#DEREV=""
DISK_D="disk_d"
MY_DISK_NAME="disk_d_nvd"


################################################################################################
### < Need Your Change 2 >
### Go download the dev/eval set data from the following server
### lab@10.110.43.14:/disk_d/datasets/nemo_chime7_diar_manifests/
# PASSWORD for lab@10.110.43.14 is in the Google Docs : https://docs.google.com/document/d/1IT07_3YkgshtMGrBLW6vrUjRBl_LwaFlseQBjELhZAY/edit?usp=sharing

#### Mixer6 
# test_manifest="/"$DISK_D"/datasets/nemo_chime7_diar_manifests/mixer6/mulspk_asr_manifest/mixer6-dev"$DEREV".json" # Full set
# test_manifest="/"$DISK_D"/datasets/nemo_chime7_diar_manifests/mixer6/mulspk_asr_manifest/mixer6-dev.short1.json"
# test_manifest="/"$DISK_D"/datasets/nemo_chime7_diar_manifests/mixer6/mulspk_asr_manifest/mixer6-dev.short1.dur120.json"
# test_manifest="/"$DISK_D"/datasets/nemo_chime7_diar_manifests/mixer6/mulspk_asr_manifest/mixer6-dev.short2.dur240.json"
# test_manifest="/"$DISK_D"/datasets/nemo_chime7_diar_manifests/mixer6/mulspk_asr_manifest/mixer6-dev.short2.json"
# test_manifest="/"$DISK_D"/datasets/nemo_chime7_diar_manifests/mixer6/mulspk_asr_manifest/mixer6-dev"$DEREV".short6.json"

#### Dipco 
# test_manifest="/"$DISK_D"/datasets/nemo_chime7_diar_manifests/dipco/mulspk_asr_manifest/dipco-dev"$DEREV".json" # Full set
#test_manifest="/"$DISK_D"/datasets/nemo_chime7_diar_manifests/dipco/mulspk_asr_manifest/dipco-dev"$DEREV".short1.json"

#### Chime6
test_manifest="/"$DISK_D"/datasets/nemo_chime7_diar_manifests/chime6/mulspk_asr_manifest/chime6-dev"$DEREV".json" # Full set
#test_manifest="/"$DISK_D"/datasets/nemo_chime7_diar_manifests/chime6/mulspk_asr_manifest/chime6-dev"$DEREV".dur600.json"

### If you need to change the disk name, please change the following line
sed -i 's/disk_d\/data/'$MY_DISK_NAME'\/data/g' $test_manifest
################################################################################################


################################################################################################
### < Need Your Change 3 >
### Use batch size 11 if GRAM is 32GB, 14 if GRAM is 48GB
BATCH_SIZE=11
################################################################################################

echo "----BASEPATH" $BASEPATH
export CUDA_VISIBLE_DEVICES="0"


################################################################################################
### < Need Your Change 4 >
# Get the base name of the test_manifest and remove extension
UNIQ_MEMO=$(basename $test_manifest | cut -d'.' -f1-2)
norm_mc_audio="true"

MY_TEMP_PATH="/disk_a/temporary_data"
DIAR_OUT_DOWNLOAD="$MY_TEMP_PATH"/diar_results_v2_norm-"$norm_mc_audio"/diarout_"$UNIQ_MEMO""$DEREV"
echo "DIAR_OUT_DOWNLOAD: " $DIAR_OUT_DOWNLOAD 
mkdir -p $DIAR_OUT_DOWNLOAD
################################################################################################


DIAR_OUT_PATH=$DIAR_OUT_DOWNLOAD
mkdir -p $DIAR_OUT_PATH
SIGMOID_THRESHOLD='[0.5,0.9,1.0]'

NUM_WORKERS=20
python $BASEPATH/examples/speaker_tasks/diarization/neural_diarizer/multiscale_diar_decoder_v2_infer.py --config-path='../conf/inference' --config-name='diar_infer_msdd_v2.yaml' \
    device="cuda" \
    num_workers=$NUM_WORKERS \
    diarizer.oracle_vad=false \
    batch_size=$BATCH_SIZE \
    diarizer.multichannel.parameters.mc_audio_normalize=$norm_mc_audio \
    diarizer.vad.parameters.shift_length_in_sec=0.02 \
    diarizer.msdd_model.model_path=$MSDD_MODEL_PATH \
    diarizer.vad.model_path="$VAD_MODEL_PATH" \
    diarizer.msdd_model.parameters.infer_batch_size=$BATCH_SIZE \
    diarizer.manifest_filepath=$test_manifest \
    diarizer.clustering.parameters.max_num_speakers=4 \
    +diarizer.clustering.parameters.min_num_speakers=2 \
    diarizer.clustering.parameters.oracle_num_speakers=false \
    diarizer.msdd_model.parameters.sigmoid_threshold=$SIGMOID_THRESHOLD \
    diarizer.out_dir="$DIAR_OUT_PATH" || exit 1
