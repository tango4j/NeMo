
branch_name="chooper_dl_msdiar"
cd /home/chooper/projects/$branch_name/NeMo
pwd
BASEPATH=/home/chooper/projects/$branch_name/NeMo/examples/speaker_tasks/diarization

# emb_tag='ami_4_50segf_cos'
# train_manifest='/home/chooper/projects/branch_nemo/NeMo/scripts/speaker_tasks/ami_mixheadset_test_input_manifest.json'
# dev_manifest='/home/chooper/projects/branch_nemo/NeMo/scripts/speaker_tasks/ami_mixheadset_dev_input_manifest.json'

train_manifest='/home/chooper/projects/chooper_dl_msdiar/NeMo/examples/speaker_tasks/diarization/ami_mixheadset_dev_input_manifest_short.json'
dev_manifest='/home/chooper/projects/chooper_dl_msdiar/NeMo/examples/speaker_tasks/diarization/ami_mixheadset_dev_input_manifest_short.json'

# PATHADD=''
# train_emb_dir='/home/chooper/projects/data/diar_manifest_input'$PATHADD'/ts_vad_embs/'$emb_tag'/diar_train'
# dev_emb_dir='/home/chooper/projects/data/diar_manifest_input'$PATHADD'/ts_vad_embs/'$emb_tag'/diar_dev'
# test_emb_dir='/home/chooper/projects/data/diar_manifest_input'$PATHADD'/ts_vad_embs/'$emb_tag'/diar_test'

train_emb_dir='./speaker_outputs_train'
dev_emb_dir='./speaker_outputs_dev'
test_emb_dir='./speaker_outputs_test'

emb_tag='sd'
EXP_NAME="$emb_tag"
EMB_BATCH_SIZE=0
TRAIN_BATCHSIZE=2
BATCHSIZE=$TRAIN_BATCHSIZE
MAX_NUM_OF_SPKS=2
export CUDA_VISIBLE_DEVICES="0,1"
export PYTHONPATH=/home/chooper/projects/$branch_name/NeMo:$PYTHONPATH
MEMO=$emb_tag

wandb login 7aff50100c43f8efc06542e5a9b0c5aaff35b4c1 \
&& python $BASEPATH/diarizer_end2end_finetune.py --config-path='conf' --config-name='msdd_training.synthetic.yaml' \
    diarizer.speaker_embeddings.model_path="/home/chooper/Downloads/titanet-m.nemo" \
    diarizer.speaker_embeddings.parameters.save_embeddings=True \
    diarizer.clustering.parameters.oracle_num_speakers=True \
    diarizer.clustering.parameters.max_num_speakers=$MAX_NUM_OF_SPKS\
    diarizer.oracle_vad=True \
    batch_size=$EMB_BATCH_SIZE\
    trainer.max_epochs=200\
    trainer.max_steps=-1\
    trainer.reload_dataloaders_every_n_epochs=1\
    msdd_model.use_longest_scale_clus_avg_emb=False \
    msdd_model.train_ds.soft_label_thres=0.5 \
    msdd_model.validation_ds.soft_label_thres=0.5 \
    msdd_model.max_num_of_spks=$MAX_NUM_OF_SPKS \
    msdd_model.train_ds.batch_size=$TRAIN_BATCHSIZE \
    msdd_model.validation_ds.batch_size=$TRAIN_BATCHSIZE \
    msdd_model.train_ds.manifest_filepath="$train_manifest" \
    msdd_model.validation_ds.manifest_filepath="$dev_manifest" \
    msdd_model.train_ds.emb_dir="$train_emb_dir" \
    msdd_model.validation_ds.emb_dir="$dev_emb_dir" \
    msdd_model.emb_batch_size=0\
    msdd_model.end_to_end_train=True\
    msdd_model.data_simulator.manifest_path="/home/chooper/projects/chooper_dl_msdiar/NeMo/scripts/speaker_tasks/train-clean-100-align.json"\
    msdd_model.data_simulator.outputs.output_dir="/home/chooper/projects/chooper_dl_msdiar/NeMo/examples/speaker_tasks/diarization/test"\
    msdd_model.data_simulator.session_config.num_sessions=1\
    msdd_model.data_simulator.session_config.session_length=600\
    msdd_model.data_simulator.background_noise.background_dir='/home/chooper/projects/rir_isotropic_noises/train/'\
    num_workers=24\
    trainer.gpus=2\
    exp_manager.name=$EXP_NAME \
    +exp_manager.use_datetime_version=False \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="$EXP_NAME" \
    exp_manager.wandb_logger_kwargs.project="synthetic-diarization" \
    exp_manager.exp_dir="expdir" \
    msdd_model.base.diarizer.speaker_embeddings.model_path="/home/chooper/Downloads/titanet-m.nemo" \
