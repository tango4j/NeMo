#!/bin/bash

# runs training on synthetic data

branch_name="<branch name>"
cd /home/taejin/projects/$branch_name/NeMo
pwd
BASEPATH=/home/taejin/projects/$branch_name/NeMo/examples/speaker_tasks/diarization

# emb_tag='ami_4_50segf_cos'
# train_manifest='/home/chooper/projects/branch_nemo/NeMo/scripts/speaker_tasks/ami_mixheadset_test_input_manifest.json'
# dev_manifest='/home/chooper/projects/branch_nemo/NeMo/scripts/speaker_tasks/ami_mixheadset_dev_input_manifest.json'

train_manifest='' #not needed here
dev_manifest='<path to ami_mixheadset_dev_input_manifest.json from run_diarization_finetuning_manifest_for_synthetic_taejin.sh>'

# PATHADD=''
# train_emb_dir='/home/chooper/projects/data/diar_manifest_input'$PATHADD'/ts_vad_embs/'$emb_tag'/diar_train'
# dev_emb_dir='/home/chooper/projects/data/diar_manifest_input'$PATHADD'/ts_vad_embs/'$emb_tag'/diar_dev'
# test_emb_dir='/home/chooper/projects/data/diar_manifest_input'$PATHADD'/ts_vad_embs/'$emb_tag'/diar_test'

train_emb_dir='<path to tmp directory for train embeddings>'
dev_emb_dir='<path to tmp directory for dev embeddings>'
test_emb_dir='<path to tmp directory for test embeddings>'

EXP_NAME=$emb_tag
EMB_BATCH_SIZE=0
TRAIN_BATCHSIZE=4
BATCHSIZE=$TRAIN_BATCHSIZE
MAX_NUM_OF_SPKS=2
export CUDA_VISIBLE_DEVICES="0,1"
export PYTHONPATH=/home/taejin/projects/$branch_name/NeMo:$PYTHONPATH
MEMO=$emb_tag
python $BASEPATH/diarizer_end2end_finetune.py --config-path='conf' --config-name='msdd_training.synthetic.yaml' \
    diarizer.speaker_embeddings.model_path="<path to titanet-m.nemo>" \
    diarizer.speaker_embeddings.parameters.save_embeddings=True \
    diarizer.clustering.parameters.oracle_num_speakers=True \
    diarizer.clustering.parameters.max_num_speakers=$MAX_NUM_OF_SPKS\
    diarizer.oracle_vad=True \
    diarizer.collar=0.0\
    diarizer.ignore_overlap=False\
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
    msdd_model.data_simulator.manifest_path="<path to train-clean-100-align.json from create_diarization_manifests_taejin.sh>"\
    msdd_model.data_simulator.outputs.output_dir="<temporary directory>"\
    num_workers=24\
    trainer.gpus=2\
    exp_manager.name=$EXP_NAME \
    +exp_manager.use_datetime_version=False \
    exp_manager.create_wandb_logger=False \
    exp_manager.wandb_logger_kwargs.name="taejinp" \
    exp_manager.wandb_logger_kwargs.project=$EXP_NAME \
    exp_manager.exp_dir=$emb_tag \
    msdd_model.base.diarizer.speaker_embeddings.model_path="<path to titanet-m.nemo>" \
