
# branch_name="challenge_nemo"
branch_name="diar_sortformer"
cd /home/taejinp/projects/$branch_name/NeMo
#bash ../copy_from_mc_audio_gitlab.sh 
pwd
BASEPATH=/home/taejinp/projects/$branch_name/NeMo/examples/speaker_tasks/diarization
noise_manifest_paths='/disk_d/temporary_data/musan/manifest_files/only1ch/musan_noise_and_music_manifest.only1ch.json'


# train_manifest='/disk_b/datasets/simulated_data/diar_sim_libri_beta_v100/05Movl/data_splits/diar_sim_libri_beta_v101_05Movl_train.v2seg.json'
# train_manifest='/disk_a/datasets/eend_train_data_2xCtSSpks_2xMovl_1-4spks/train_beta_v201_libri.shuf.json' # 56112 files x 60sec

train_manifest='/disk_a/datasets/eend_train_data_2xCtSSpks_2xMovl_1-4spks/train_beta_v201_libri.v2seg.json' # Benchmark test 1x data
#train_manifest='/disk_d_nvd/datasets/nemo_chime7_manifests_train/chime6/mulspk_asr_manifest/chime6-train.v2seg.json'


# train_manifest='/disk_a/datasets/eend_train_data_2xCtSSpks_2xMovl_1-4spks/libriTrVox12_train_beta_v201_171919.no_bal.v2seg.json'


# cat $train_manifest | awk 'NR % 200 == 0' | shuf  > "/home/taejinp/test_manifest/train_temp.json"  # Benchmark test 1x data 
# cat $train_manifest | awk 'NR % 100 == 0' | shuf  > "/home/taejinp/test_manifest/train_temp.json"  # Benchmark test 2x data 
# cat $train_manifest | awk 'NR % 50 == 0' | shuf  > "/home/taejinp/test_manifest/train_temp.json"  # Benchmark test 10x data 
cat $train_manifest | awk 'NR % 20 == 0' | shuf  > "/home/taejinp/test_manifest/train_temp.json"  # Benchmark test 10x data 
# cat $train_manifest | awk 'NR % 10 == 0' | shuf  > "/home/taejinp/test_manifest/train_temp.json"  # Benchmark test 10x data 
# cat $train_manifest | awk 'NR % 5 == 0' | shuf  > "/home/taejinp/test_manifest/train_temp.json"  # Benchmark test 10x data ###TESTED-longest
# cat $train_manifest | awk 'NR % 2 == 0' | shuf  > "/home/taejinp/test_manifest/train_temp.json" # Benchmark 100x data BS 180

train_manifest="/home/taejinp/test_manifest/train_temp.json"
TRAIN_LINES=`cat $train_manifest | wc -l`
SAVE_TOP_K=3

#cat $train_manifest | shuf | head -5000 > "/home/taejinp/test_manifest/train_temp.json"
#train_manifest="/home/taejinp/test_manifest/train_temp.json"


# dev_manifest='/disk_a/datasets/eend_train_data_2xCtSSpks_2xMovl_1-4spks/dev_beta_v201.json' 
# dev_manifest="/disk_b/datasets/simulated_data/diar_sim_libri_beta_v100/05Movl/data_splits/diar_sim_libri_beta_v101_05Movl_dev.v2seg.json"
dev_manifest='/disk_a/datasets/eend_train_data_2xCtSSpks_2xMovl_1-4spks/dev_beta_v201.v2seg.json' # Benchmark test

# dev_manifest='/disk_a/datasets/eend_train_data_2xCtSSpks_2xMovl_1-4spks/libriTrVox12_dev_beta_v201_4045.json'
#dev_manifest='/disk_d_nvd/datasets/nemo_chime7_manifests/chime6/mulspk_asr_manifest/chime6-dev.no_bal.v2seg.json'

# cat $dev_manifest | awk 'NR % 2 == 0' | shuf > "/home/taejinp/test_manifest/dev_temp.json"
# cat $dev_manifest | awk 'NR % 5 == 0' | shuf > "/home/taejinp/test_manifest/dev_temp.json"
# cat $dev_manifest | awk 'NR % 10 == 0' | shuf > "/home/taejinp/test_manifest/dev_temp.json"
cat $dev_manifest | awk 'NR % 20 == 0' | shuf > "/home/taejinp/test_manifest/dev_temp.json"
# cat $dev_manifest | awk 'NR % 50 == 0' | shuf > "/home/taejinp/test_manifest/dev_temp.json" # Benchmark test 1x data
# cat $dev_manifest | awk 'NR % 100 == 0' | shuf > "/home/taejinp/test_manifest/dev_temp.json" # Benchmark test 0.5x data
# cat $dev_manifest | awk 'NR % 200 == 0' | shuf > "/home/taejinp/test_manifest/dev_temp.json" # Benchmark test 0.5x data
dev_manifest="/home/taejinp/test_manifest/dev_temp.json"
DEV_LINES=`cat $dev_manifest | wc -l`

cat $dev_manifest | head -1 > "/home/taejinp/test_manifest/test_temp.json"

test_manifest="/home/taejinp/test_manifest/test_temp.json"

PATHADD=''
train_emb_dir='/disk_c/taejinp_backup/data/diar_manifest_input'$PATHADD'/msdd_meta_data/'$emb_tag'/diar_train'
dev_emb_dir='/disk_c/taejinp_backup/data/diar_manifest_input'$PATHADD'/msdd_meta_data/'$emb_tag'/diar_dev'
test_emb_dir='/disk_c/taejinp_backup/data/diar_manifest_input'$PATHADD'/msdd_meta_data/'$emb_tag'/diar_test'
mkdir -p $dev_emb_dir
# VAL_BATCHSIZE=8
# VAL_BATCHSIZE=90
VAL_BATCHSIZE=180

# TRAIN_BATCHSIZE=6
TRAIN_BATCHSIZE=$VAL_BATCHSIZE
#TRAIN_BATCHSIZE=300
EXP_NAME=SFmr_MixMockEmbsTest
# EXP_NAME=SortformerDiar_PermResolver
# EXP_NAME=SortformerDiar_PermResolver

YAML_NAME=sortformer_train_develop.yaml
yaml_path=$BASEPATH/conf/neural_diarizer/$YAML_NAME
ls $yaml_path || exit 1
yaml_hash=$(md5sum $yaml_path | cut -d" " -f1 | cut -c 1-2)
exp_hash=$(openssl rand -hex 16 | md5sum | cut -d" " -f1 | cut -c 1-2)
#TRIAL_MEMO="$yaml_hash":"$exp_hash"
# UNIT_DIM=3
# UNIT_DIM=192
# UNIT_DIM=96
# UNIT_DIM=5
UNIT_DIM=192
# UNIT_DIM=7
# UNIT_DIM=9

VARIANCE_IN_MOCK_EMBS=0.02
MOCK_EMB_NOISE_STD=0.03
PERM_MOCK_EMBS=false
TRAIN_DOF=4
VALID_DOF=4

LAYER_ARRIVAL_TIME_SORT=false
# LAYER_ARRIVAL_TIME_SORT=true

ALPHA=0.5

# 
# SORT_LAYER_ON=true
# SORTFORMER_NUM_LAYER=6
# TRANSFORMER_NUM_LAYER=6

# SORT_LAYER_ON=false
# SORT_LAYER_ON=true

### King Sortformer=True
# SORT_LAYER_ON="pre" # pre, post, false
# SORT_BIN_ORDER=true
# SORTFORMER_NUM_LAYER=4
# TRANSFORMER_NUM_LAYER=2

### King Non-Sortformer
SORT_LAYER_ON=false
SORT_BIN_ORDER=false
SORTFORMER_NUM_LAYER=0
TRANSFORMER_NUM_LAYER=6

ATS_SORT=true
# ATS_SORT=false

# SORTED_PREDS=true
SORTED_PREDS=false

CLASS_NORMALIZATION=false
# CLASS_NORMALIZATION='binary'
# CLASS_NORMALIZATION='class'
# CLASS_NORMALIZATION='class_binary'

# SORT_FINAL_LAYER_ON=true
SORT_FINAL_LAYER_ON=false # This should be false 

SEQ_VAR_SORT=true
# SEQ_VAR_SORT=false

# DETACH_PREDS=true
DETACH_PREDS=false

# CLASS_NORMALIZATION=true
LR=0.0003
# LR=0.00025
# LR=0.001
HEADS=16
INNER=`echo "192*2" | bc`
HASH_MEMO="$yaml_hash":"$exp_hash":Tr"$TRAIN_LINES"dev"$DEV_LINES"-ATSsort:"$ATS_SORT"-PredSort:"$SORTED_PREDS"-SortLayer:"$SORT_LAYER_ON"
TRANSFORMER_MEMO=-var"$VARIANCE_IN_MOCK_EMBS"-TrDOF:"$TRAIN_DOF"-ValDOF:"$VALID_DOF"-noiseVar:"$MOCK_EMB_NOISE_STD"-Perm:"$PERM_MOCK_EMBS"-s"$SORTFORMER_NUM_LAYER"t"$TRANSFORMER_NUM_LAYER"-head"$HEADS"-inner"$INNER"-unD"$UNIT_DIM"
PROCESSING_MEMO=-ClsNorm:"$CLASS_NORMALIZATION"-lr"$LR"-bs"$TRAIN_BATCHSIZE"-SeqVarSort:"$SEQ_VAR_SORT"-dtchPrd:"$DETACH_PREDS"-SortBinOrdr:"$SORT_BIN_ORDER"
TRIAL_MEMO="$HASH_MEMO""$TRANSFORMER_MEMO""$PROCESSING_MEMO"
echo "============= Experiment Name: " $TRIAL_MEMO
EXP_DIR="/disk_c/taejinp_backup/msdd_model_train/"
export PYTHONPATH=/home/taejinp/projects/$branch_name/NeMo:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export CUDA_VISIBLE_DEVICES="1,2,3"
# export CUDA_VISIBLE_DEVICES="2,3"
# export CUDA_VISIBLE_DEVICES="3"
MEMO=$emb_tag
NUM_WORKERS=18
# NUM_WORKERS=0
VAD_MODEL_PATH="/home/taejinp/Downloads/wnc_frame_vad.nemo"
MAX_NUM_OF_SPKS=4
EMB_MODEL_PATH="titanet_large"
MAX_STEPS=500000
MAX_EPOCHS=5000
python $BASEPATH/neural_diarizer/sortformer_diar_encoder_train.py --config-path='../conf/neural_diarizer' --config-name="$YAML_NAME" \
    trainer.devices="[0,1]" \
    model.session_len_sec=15 \
    model.lr=$LR \
    model.alpha=$ALPHA \
    model.optim.sched.warmup_steps=500 \
    model.heads=$HEADS \
    trainer.max_epochs=$MAX_EPOCHS \
    trainer.max_steps=$MAX_STEPS \
    model.sort_layer_on=$SORT_LAYER_ON \
    model.sort_bin_order=$SORT_BIN_ORDER \
    model.layer_arrival_time_sort=$LAYER_ARRIVAL_TIME_SORT \
    model.perm_mock_embs=$PERM_MOCK_EMBS \
    model.variance_in_mock_embs=$VARIANCE_IN_MOCK_EMBS \
    model.mock_emb_noise_std=$MOCK_EMB_NOISE_STD \
    model.sortformer_encoder.unit_dim=$UNIT_DIM \
    model.sortformer_encoder.num_layers=${SORTFORMER_NUM_LAYER} \
    model.sortformer_encoder.seq_var_sort=${SEQ_VAR_SORT} \
    model.sortformer_encoder.detach_preds=${DETACH_PREDS} \
    model.transformer_encoder.num_layers=${TRANSFORMER_NUM_LAYER} \
    trainer.strategy="ddp_find_unused_parameters_true" \
    num_workers=$NUM_WORKERS \
    model.diarizer.speaker_embeddings.model_path=$EMB_MODEL_PATH \
    model.diarizer.speaker_embeddings.parameters.save_embeddings=True \
    model.diarizer.oracle_vad=True \
    batch_size=$TRAIN_BATCHSIZE\
    model.augmentor.noise.manifest_path=$noise_manifest_paths \
    trainer.max_epochs=$MAX_EPOCHS\
    model.interpolated_scale=0.1 \
    model.max_num_of_spks=$MAX_NUM_OF_SPKS \
    model.train_ds.batch_size=$TRAIN_BATCHSIZE \
    model.validation_ds.batch_size=$VAL_BATCHSIZE \
    model.test_ds.batch_size=$VAL_BATCHSIZE \
    model.test_ds.shuffle=false \
    model.train_ds.manifest_filepath="$train_manifest" \
    model.validation_ds.manifest_filepath="$dev_manifest" \
    model.test_ds.manifest_filepath="$test_manifest" \
    model.train_ds.mock_emb_degree_of_freedom=$TRAIN_DOF \
    model.validation_ds.mock_emb_degree_of_freedom=$VALID_DOF \
    model.test_ds.mock_emb_degree_of_freedom=$TRAIN_DOF \
    model.train_ds.emb_dir="$train_emb_dir" \
    model.test_ds.emb_dir="$test_emb_dir" \
    model.validation_ds.emb_dir="$dev_emb_dir" \
    trainer.max_steps=-1 \
    model.loss.sorted_loss=$ATS_SORT \
    model.loss.sorted_preds=$SORTED_PREDS \
    model.loss.class_normalization=$CLASS_NORMALIZATION \
    model.diarizer_module.num_spks=$MAX_NUM_OF_SPKS \
    model.diarizer_module.sort_final_layer_on=$SORT_FINAL_LAYER_ON \
    model.soft_label_thres=0.5 \
    exp_manager.exp_dir=$EXP_DIR \
    exp_manager.checkpoint_callback_params.save_top_k=$SAVE_TOP_K  \
    +exp_manager.use_datetime_version=False \
    exp_manager.name=$EXP_NAME \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="$TRIAL_MEMO" \
    exp_manager.wandb_logger_kwargs.project=$EXP_NAME \