
################################################################################################
### < Need Your Change 1>
branch_name="dev_chime7_gitlab"
BASEPATH=/home/taejinp/projects/$branch_name/NeMo/scripts/chime7/hyper_optim
cd $BASEPATH

DEREV="-d03"
DISK_D="disk_d"
MY_DISK_NAME="disk_d_nvd"
DISK_D=$MY_DISK_NAME

################################################################################################
### < Need Your Change 2 >
### Go download the dev/eval set data from the following server
### lab@10.110.43.14:/disk_d/datasets/nemo_chime7_diar_manifests/
# PASSWORD for lab@10.110.43.14 is in the Google Docs : https://docs.google.com/document/d/1IT07_3YkgshtMGrBLW6vrUjRBl_LwaFlseQBjELhZAY/edit?usp=sharing

#### Mixer6 
# test_manifest="/"$DISK_D"/datasets/nemo_chime7_diar_manifests/mixer6/mulspk_asr_manifest/mixer6-dev"$DEREV".json" # Full set

#### Dipco 
# test_manifest="/"$DISK_D"/datasets/nemo_chime7_diar_manifests/dipco/mulspk_asr_manifest/dipco-dev"$DEREV".json" # Full set

#### Chime6
# test_manifest="/"$DISK_D"/datasets/nemo_chime7_diar_manifests/chime6/mulspk_asr_manifest/chime6-dev"$DEREV".json" # Full set

#### Chime6 + Dipco + Mixer6
test_manifest="/"$DISK_D"/datasets/nemo_chime7_diar_manifests/chdipmixShort180s-dev-d03.json"
# test_manifest="/"$DISK_D"/datasets/nemo_chime7_diar_manifests/chdipmixShort1-dev-d03.json"
# test_manifest="/"$DISK_D"/datasets/nemo_chime7_diar_manifests/chdipmixALL-dev-d03.json"


emb_tag='ch109_full_main'
pwd
titanet_path="/home/taejinp/Downloads/titanet-l.nemo"
export PYTHONPATH=/home/taejinp/projects/$branch_name/NeMo:$PYTHONPATH

BATCH_SIZE=11
MEMO="$emb_tag"
echo "----BASEPATH" $BASEPATH

# STUDY_NAME="chime7_pilot_v01"
STORAGE="sqlite:///"$STUDY_NAME".db"
MANIFEST_PATH="$test_manifest"
CONFIG_URL=/home/taejinp/projects/$branch_name/NeMo/examples/speaker_tasks/diarization/conf/inference/diar_infer_msdd_v2.yaml
MSDD_MODEL_PATH="/disk_c/models/msdd_v2_models/msdd_v2_PALO_bs6_a003_version6_e53.ckpt"
VAD_MODEL_PATH="/disk_c/taejinp/gdrive/model/VAD_models/frame_vad_chime7_acrobat.nemo"
TEMP_DIR="output/"
OUTPUT_LOG="$STUDY_NAME".log
COLLAR=0.25
N_TRIALS=1000000
N_JOBS=1

################################################################################################
### < Need Your Change 3 >
### Create a directory for the output of the hyper-parameter optimization
UNIQ_MEMO=$(basename $test_manifest | cut -d'.' -f1)
MODEL_VER="sys-B-optV07D03"
norm_mc_audio="true"
MY_TEMP_PATH="/disk_c/temporary_data"
DIAR_OUT_PATH="$MY_TEMP_PATH"/diar_results_v2_norm-"$norm_mc_audio"/"$MODEL_VER"_"$UNIQ_MEMO"
mkdir -p $DIAR_OUT_PATH || exit 1

echo ">>>> DIAR_OUT_PATH: " $DIAR_OUT_PATH

for cuda_device in {0..$LAST_CUDA_DEVICE}
do
# cuda_device=2
export CUDA_VISIBLE_DEVICES=$cuda_device

python $BASEPATH/optimize_diar.py \
            --study_name $STUDY_NAME \
            --storage $STORAGE \
            --manifest_path $MANIFEST_PATH \
            --config_url $CONFIG_URL \
            --msdd_model_path $MSDD_MODEL_PATH \
            --vad_model_path $VAD_MODEL_PATH \
            --batch_size $BATCH_SIZE \
            --output_dir $DIAR_OUT_PATH \
            --output_log $OUTPUT_LOG \
            --n_trials $N_TRIALS \
            --n_jobs $N_JOBS 
done
