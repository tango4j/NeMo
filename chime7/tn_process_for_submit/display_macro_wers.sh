#!/bin/bash

eval_log_path="/disk_b/datasets/chime7_final_submission/sa_wer_output_dir"

track="subtrack"
# track="main"
split=("dev" "eval")
# split=("dev")
system_num=("1" "2" "3")

# Loop over the array variables
for s in "${split[@]}"; do
    for sn in "${system_num[@]}"; do
        # Construct the directory path
        eval_results_dir="${eval_log_path}/${track}_${s}_system${sn}"
        
        # # Check if the file exists before trying to print its contents
        # if [[ -f "${eval_results_dir}/macro_wer.txt" ]]; then
        #     # "macro-avg SA-WER: "
        #     echo -n ${s}","system-$sn","
        #     cat "${eval_results_dir}/macro_wer.txt"
        #     echo ""
        # else
        #     echo "File not found: ${eval_results_dir}/macro_wer.txt"
        # fi
        echo ${track}_${s}_system${sn}
        grep -A 7 "### Metrics for all Scenarios ###" ${eval_log_path}/${track}_${s}_system${sn}/${track}_${s}_system${sn}_pyannote_eval.log | tail -5 | awk -F '|' '{print $3 $13 $16 $21}'
        echo ""
    done
done
