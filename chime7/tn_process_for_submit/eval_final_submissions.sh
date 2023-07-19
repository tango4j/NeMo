#!/bin/bash


# pred_dir="nemo_experiments/nemo_chime6_ft_ctc/"
system="baseline_diar-dev"  # system_vA04D system_vA01 system_vA04D-T0.5 system_B_V05_D03-T0.5 
pred_dir="nemo_experiments/nemo_${system}_chime6_ft_rnnt_v2_ln25/"
track="main" 
# track="subtrack" 
# system_num="1"
system_num="2"
split="dev"
# split="eval"

track=$1
split=$2
system_num=$3

# sub_json_foldername=final_submission_nemo_json
sub_json_foldername="final_submission_nemo_json_added_s19s20"
# pred_dir="/disk_b/datasets/chime7_final_submission/final_submission_nemo_json/${track}/system${system_num}/${split}"
# pred_dir="/disk_b/datasets/chime7_final_submission/final_submission_nemo_json_added_s19s20
pred_dir="/disk_b/datasets/chime7_final_submission/${sub_json_foldername}/${track}/system${system_num}/${split}"
eval_log_path="/disk_b/datasets/chime7_final_submission/sa_wer_output_dir"
pred_dir_normalized="/disk_b/datasets/chime7_final_submission/${sub_json_foldername}_normalized/${track}/system${system_num}/${split}"

mkdir -p $pred_dir_normalized || exit 1
ls $pred_dir 
ls $pred_dir || exit 1

# python /home/taejinp/projects/fp32_dev_chime7/
python ./text_normalize_krishna.py --input_dir $pred_dir  --sub_json_foldername $sub_json_foldername || exit 1

# eval_log_path="/disk_b/datasets/chime7_final_submission/sa_wer_no_text_norm"
# pred_dir_normalized=$pred_dir

ls $pred_dir_normalized 
ls $pred_dir_normalized || exit 1


CHIME7_ROOT="/disk_d/datasets/chime7_official_cleaned_v2"  
eval_results_dir="${eval_log_path}/${track}_${split}_system${system_num}"
mkdir -p $eval_results_dir

CHIME7_SUBMISSION_ROOT="/disk_b/datasets/chime7_final_submission"

echo "Evaluation of ${track}_${split}_system${system_num}"
echo "Evaluating jsons in $pred_dir_normalized ...."
echo ""
echo hyp_folder $pred_dir_normalized
echo dasr_root $CHIME7_ROOT
echo partition $SPLIT
echo output_folder $eval_results_dir
echo ""
python ${CHIME7_SUBMISSION_ROOT}/evaluate_nemo_submissions.py \
                    --hyp_folder $pred_dir_normalized \
                    --dasr_root $CHIME7_ROOT \
                    --partition $split \
                    --output_folder $eval_results_dir 2>&1 | tee $eval_results_dir/${track}_${split}_system${system_num}_pyannote_eval.log || exit 1