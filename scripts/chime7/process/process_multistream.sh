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

SCENARIOS="$1"
GPU_ID="$2"
DIARIZATION_CONFIG=sys-B-V07_trial_315
DIARIZATION_PARAMS="pred_jsons_T"
DIARIZATION_BASE_DIR="${HOME}/scratch/chime7/chime7_diar_results"

for top_k in 80 100
do
    for mc_postmask_min_db in -9 0
    do
        OUTPUT_ROOT=${HOME}/scratch/chime7/multistream_v1/top_${top_k}_postmask_${mc_postmask_min_db}

        BSS_ITERATION=5 DEREVERB_FILTER_LENGTH=5 MC_MASK_MIN_DB=-60 MC_POSTMASK_MIN_DB=${mc_postmask_min_db} TOP_K=${top_k} ./run_processing.sh \
            "$SCENARIOS" \
            "$GPU_ID" \
            "$DIARIZATION_CONFIG" \
            "$DIARIZATION_PARAMS" \
            "$DIARIZATION_BASE_DIR" \
            "$OUTPUT_ROOT"
    done
done