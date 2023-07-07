# Current best: Trial 315
# {
#     'bss_iterations': 5,
#     'dereverb_filter_length': 5,
#     'frame_vad_threshold': 0.14,
#     'max_rp_threshold': 0.17,
#     'mc_mask_min_db': -60,
#     'mc_postmask_min_db': -9,
#     'min_duration_off': 0.4,
#     'min_duration_on': 0.35,
#     'normalize_db': -30,
#     'pad_offset': 0.2,
#     'pad_onset': 0.1,
#     'r_value': 0.9,
#     'sigmoid_threshold': 0.6,
#     'sparse_search_volume': 25,
#     'top_k': 80
# }

# Use as:
# ./process_multistream.sh "chime6 mixer6" 0
# ./process_multistream.sh "dipco" 1

set -eou pipefail

SCENARIOS="chime6 dipco mixer6"
SUBSETS="dev"
SYSTEM="sys-B-V07_trial_315-pred_jsons_T"

for top_k in 80 100
do
    for mc_postmask_min_db in -9 0
    do
        MANIFEST_DIR_ROOT=${HOME}/scratch/chime7/multistream_v1/top_${top_k}_postmask_${mc_postmask_min_db}/processed
        OUTPUT_DIR=${HOME}/scratch/chime7/multistream_v1_eval/top_${top_k}_postmask_${mc_postmask_min_db}

        ./run_asr.sh \
            "$SCENARIOS" \
            "$SUBSETS" \
            "$SYSTEM" \
            "$MANIFEST_DIR_ROOT" \
            "$OUTPUT_DIR"
    done
done