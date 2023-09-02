branch_name="diar_chime7"

BASEPATH=/home/taejinp/projects/$branch_name/NeMo

# pwd
export PYTHONPATH=/home/taejinp/projects/$branch_name/NeMo:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="2"

# ASR_LM_PATH="/disk_c/models/lm/lowercase_3-gram.pruned.1e-7.arpa"
ASR_LM_PATH=null

MSDD_MODEL_PATH='/disk_c/taejinp_backup/msdd_model_train/MSDD_v2_PALO_BS6_a0.003/version_6/checkpoints/epoch53.ckpt' # 3.0 scale libVox, TnFrz
VAD_MODEL_PATH="/disk_c/taejinp/gdrive/model/VAD_models/vad_multilingual_frame_marblenet.nemo"

# DIAR_LM_PATH="/disk_c/models/lm/lowercase_3-gram.pruned.1e-7.arpa" # 3.8M entries
# DIAR_LM_PATH="/disk_c/models/lm/4-gram.arpa" 145.3M entries
DIAR_LM_PATH="/disk_c/models/lm/4gram_small.arpa" # 2M entries
# DIAR_LM_PATH="/disk_c/models/lm/4gram_big.arpa" # 10.5M entries
# DIAR_LM_PATH=null
USE_NGRAM=True
# USE_NGRAM=False

ASR_MODEL_PATH=/disk_c/taejinp/gdrive/model/ASR_models/Conformer-CTC-BPE_large_Riva_ASR_set_3.0_ep60.nemo
# ASR_MODEL_PATH=stt_en_conformer_ctc_xlarge
# stt_en_conformer_ctc_large

# test_manifest="/home/taejinp/projects/data/diar_manifest_input/ch109.en_0638.json"
# test_manifest="/home/taejinp/projects/data/diar_manifest_input/ch109.en_4065.json"
# test_manifest="/home/taejinp/projects/data/diar_manifest_input/ch109.en_4074.json"
# test_manifest="/home/taejinp/projects/data/diar_manifest_input/ch109.en4077.json"
# test_manifest="/home/taejinp/projects/data/diar_manifest_input/ch109.en_4093.json"
# test_manifest="/home/taejinp/projects/data/diar_manifest_input/ch109.short13.json"
# test_manifest="/home/taejinp/projects/data/diar_manifest_input/ch109.short14.json"
# test_manifest="/home/taejinp/projects/data/diar_manifest_input/ch109.short3.json"
# test_manifest="/home/taejinp/projects/data/diar_manifest_input/ch109.short2.json"
# test_manifest="/home/taejinp/projects/data/diar_manifest_input/ch109.json"

# test_manifest="/disk_c/taejinp/gdrive/datasets/riva_nemo_sd_eval_data/CH109_CHAES/CHAES_splits/list_files/chaes_ow_dev.json"
# test_manifest="/disk_c/taejinp/gdrive/datasets/riva_nemo_sd_eval_data/CH109_CHAES/CHAES_splits/list_files/chaes_ow_eval.json"
# FRAME_VAD_THRESHOLD=0.92 # Best for CHAES dev DER, 0.93 for lowest cpWER


# test_manifest="/disk_c/taejinp/gdrive/audio_data/AMI_landini/list_files/ami_mh_silintro_dev.json"
# test_manifest="/disk_c/taejinp/gdrive/audio_data/AMI_landini/list_files/ami_mh_silintro_test.json"
# test_manifest="/disk_c/taejinp/gdrive/audio_data/AMI_landini/list_files/ami_mh_silintro_en2002c.json"
# FRAME_VAD_THRESHOLD=0.02

# test_manifest="/disk_c/taejinp/gdrive/datasets/riva_nemo_sd_eval_data/CH109_CHAES/CHAES_splits/list_files/chaes_ow_ch11.json"
test_manifest="/disk_c/taejinp/gdrive/datasets/riva_nemo_sd_eval_data/CH109_CHAES/CHAES_splits/list_files/chaes_ow_ch109.json"
# alpha=0.05263436896340909, beta= 0.062, parallel_chunk_world_len=250, beam_width=9, word_window=28 = ch11 dev best params
FRAME_VAD_THRESHOLD=0.92 # Best for CHAES dev DER, 0.93 for lowest cpWER
### CH11 best params [ngram]
# ALPHA=0.05263437
# BETA=0.062
# PARALLEL_CHUNK_WORD_LEN=250
# BEAM_WIDTH=9
# WORD_WINDOW=28
# USE_NGRAM=True
# LM_METHOD=ngram

### CH11 best params [LLM]
ALPHA=0.003412946476816792
BETA=0.156
PARALLEL_CHUNK_WORD_LEN=100
BEAM_WIDTH=5
WORD_WINDOW=17
USE_NGRAM=False
LM_METHOD=llm

### AMI dev best params
# ALPHA=0.09011967946761944
# BETA=1.654085658772723e-06
# PARALLEL_CHUNK_WORD_LEN=450
# BEAM_WIDTH=6
# WORD_WINDOW=20
# USE_NGRAM=True
# LM_METHOD=ngram


# export CUDA_VISIBLE_DEVICES="0"; FRAME_VAD_THRESHOLD=0.01
# export CUDA_VISIBLE_DEVICES="1"; FRAME_VAD_THRESHOLD=0.02
# export CUDA_VISIBLE_DEVICES="2"; FRAME_VAD_THRESHOLD=0.03
# export CUDA_VISIBLE_DEVICES="3"; FRAME_VAD_THRESHOLD=0.04

# org_test_manifest="/home/taejinp/projects/data/diar_manifest_input/ch109.json"
# base_name=$(basename "$org_test_manifest" .json)
# dir_name=$(dirname "$org_test_manifest")
# new_name="$dir_name/$base_name.target.json"

# sed -n '3p' $org_test_manifest > $new_name
# test_manifest=$new_name
echo "----BASEPATH" $BASEPATH

# Get the base name of the test_manifest and remove extension
UNIQ_MEMO=$(basename "${test_manifest}" .json | sed 's/\./_/g') 

echo "UNIQ MEMO:" $UNIQ_MEMO


# TRIAL=trial221
# YAML_FILE="diar_infer_msdd_v2_db5_"$TRIAL"_mod.yaml"
TRIAL=telephonic
YAML_FILE="diar_infer_msdd_v2.yaml"

DIAR_OUT_DOWNLOAD=/disk_b/chime7_diar_mono_ch/"$TRIAL"-V100/diarout_"$UNIQ_MEMO"


echo "DIAR_OUT_DOWNLOAD: " $DIAR_OUT_DOWNLOAD 
# echo "Do you really want to delete [  $DIAR_OUT_DOWNLOAD  ]? (y/n)"
# read answer
# if [[ $answer == 'y' || $answer == 'Y' ]]; then
#     rm -rf $DIAR_OUT_DOWNLOAD
#     echo "Deleted - $DIAR_OUT_DOWNLOAD"
# else 
#     echo "Not deleted, keeping - $DIAR_OUT_DOWNLOAD"
# fi


### AMI setting
# ALPHA=0.2
# BETA=0.002


mkdir -p $DIAR_OUT_DOWNLOAD
BATCH_SIZE=11
python $BASEPATH/examples/speaker_tasks/diarization/neural_diarizer/msdd_diar_with_asr_infer.py --config-path='../conf/inference' --config-name=$YAML_FILE \
    device=cuda \
    batch_size=$BATCH_SIZE \
    diarizer.vad.model_path="$VAD_MODEL_PATH" \
    diarizer.vad.parameters.frame_vad_threshold=$FRAME_VAD_THRESHOLD \
    diarizer.msdd_model.model_path="$MSDD_MODEL_PATH" \
    diarizer.msdd_model.parameters.system_name="$TRIAL"_mono \
    diarizer.manifest_filepath=$test_manifest \
    diarizer.asr.realigning_lm_parameters.arpa_language_model=$DIAR_LM_PATH \
    diarizer.asr.model_path=$ASR_MODEL_PATH \
    diarizer.asr.parameters.fix_word_ts_with_VAD=True \
    diarizer.asr.ctc_decoder_parameters.pretrained_language_model=$ASR_LM_PATH \
    diarizer.asr.ctc_decoder_parameters.beam_width=50 \
    diarizer.asr.ctc_decoder_parameters.alpha=0.4 \
    diarizer.asr.ctc_decoder_parameters.beta=1.5 \
    diarizer.asr.realigning_lm_parameters.use_ngram=$USE_NGRAM \
    diarizer.asr.realigning_lm_parameters.alpha=$ALPHA \
    diarizer.asr.realigning_lm_parameters.beta=$BETA \
    diarizer.asr.realigning_lm_parameters.beam_width=$BEAM_WIDTH \
    diarizer.asr.realigning_lm_parameters.word_window=$WORD_WINDOW \
    diarizer.asr.realigning_lm_parameters.port=[5501,5502,5503,5511,5512,5513,5521,5522,5523,5531,5532,5533] \
    diarizer.asr.realigning_lm_parameters.parallel_chunk_word_len=$PARALLEL_CHUNK_WORD_LEN\
    diarizer.asr.realigning_lm_parameters.use_mp=True \
    diarizer.asr.realigning_lm_parameters.use_chunk_mp=True \
    diarizer.asr.realigning_lm_parameters.lm_method=$LM_METHOD \
    diarizer.speaker_out_dir=$DIAR_OUT_DOWNLOAD \
    diarizer.out_dir="$DIAR_OUT_DOWNLOAD" || exit 1
    
    # diarizer.asr.realigning_lm_parameters.beam_width=9 \
    # diarizer.asr.realigning_lm_parameters.word_window=36 \

    # diarizer.asr.realigning_lm_parameters.port=[5501,5502,5503,5511,5512,5513,5521,5522,5523,5531,5532,5533] \
    # diarizer.asr.realigning_lm_parameters.port=[5501,5502,5511,5512,5521,5522,5531,5532] \
    # diarizer.asr.realigning_lm_parameters.port=[5501,5502,5511,5512,5521,5522] \
    # diarizer.asr.realigning_lm_parameters.port=[5501,5500,5502,5503] \

    ### CHAES    
    # diarizer.asr.realigning_lm_parameters.alpha=0.01 \
    # diarizer.asr.realigning_lm_parameters.beta=0.07 \
    # diarizer.asr.realigning_lm_parameters.beam_width=9 \
    # diarizer.asr.realigning_lm_parameters.word_window=36 \

    ### AMI 
    # diarizer.asr.realigning_lm_parameters.alpha=0.2 \
    # diarizer.asr.realigning_lm_parameters.beta=0.002 \

echo FRAME_VAD_THRESHOLD: $FRAME_VAD_THRESHOLD