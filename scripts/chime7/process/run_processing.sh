#!/usr/bin/env bash
#
# This script aims to use diarization output, convert it to appropriate format,
# run multichannel processing and prepare NeMo manifests for the outputs.
#
set -eou pipefail

# Arguments
# =========
SCENARIOS=${1:-"chime6 dipco mixer6"} # select scenarios to run
GPU_ID=${2:-0} # for example, 0 or 1
DIARIZATION_CONFIG=${3:-system_vA01} # for example, system_vA01
DIARIZATION_BASE_DIR=${4:-"./chime7_diar_results"} # for example, ${HOME}/scratch/chime7/chime7_diar_results

echo "************************************************************"
echo "SCENARIOS:            $SCENARIOS"
echo "GPU_ID:               $GPU_ID"
echo "DIARIZATION_CONFIG:   $DIARIZATION_CONFIG"
echo "DIARIZATION_BASE_DIR: $DIARIZATION_BASE_DIR"
echo "************************************************************"

# Manual path setup
# =================
espnet_root= # For example, ${HOME}/work/repos/espnet-mirror/egs2/chime7_task1/asr1
chime7_root= # For example, /data/chime7/chime7_official_cleaned

if [ -z "$espnet_root" ]
then
    echo "ERROR: espnet_root variable is not set"
    echo "==> Set espnet_root variable to point to chime7_task1/asr1"
    exit -1
fi

if [ -z "$chime7_root" ]
then
    echo "ERROR: chime7_root variable is not set"
    echo "==> Set chime7_root variable to point to chime7_official_cleaned"
    exit -1
fi

# Diarization output
# ==================
diarization_base_dir=${DIARIZATION_BASE_DIR}
diarization_config=${DIARIZATION_CONFIG}
diarization_output_dir=${diarization_base_dir}/${diarization_config}

# Output
# ======
base_output_dir=./processed/${diarization_config}
num_workers=4

# Processing setup
# ================
# parameters for channel selection
sel_nj=16
# select top 80% channels
top_k=80
# tunings to evaluate
tunings=nemo_v1

# Alignment
# =========
# Convert diarization output to falign format
python convert_diarization_result_to_falign.py --diarization-dir $diarization_output_dir

# Process all scenarios
# =====================
for scenario in $SCENARIOS
do
for subset in dev
do
    echo "--"
    echo $scenario/$subset
    echo "--"

    # Alignments generated from diarization
    alignments_dir=./alignments/${diarization_config}/${scenario} # do not include subset in this path

    # Prepare lhotse manifests from diarization output
    manifests_root=./manifests/lhotse/${diarization_config}

    # NOTE:
    # Unfortunately, this scripts runs for "mdm" and "ihm" for the dev set.
    # This means it will always fail for "ihm", since we only have diarization output for "mdm".
    # It's safe to ignore the error, since we only need the "mdm" manifests.
    # The error in question: "AssertionError: No recordings left after fixing the manifests."
    # If the error appears twice for the same scenario, then something is wrong.
    #
    # NOTE 2:
    # The script will ignore segments shorter than 0.2 seconds.
    # This is the default behavior in Espnet.
    python ${espnet_root}/local/get_lhotse_manifests.py -c $chime7_root \
        -d $scenario \
        -p $subset \
        -o $manifests_root --diar_jsons_root "$alignments_dir" \
        --ignore_shorter 0.2 || true

    # These manifests need to be created from diarization output
    manifests_dir=${manifests_root}/${scenario}/${subset}

    exp_dir=${base_output_dir}/${scenario}/${subset}
    mkdir -p ${exp_dir}

    # mic selection
    echo "Stage 0: Selecting a subset of channels"
    python ${espnet_root}/local/gss_micrank.py -r ${manifests_dir}/${scenario}-mdm_recordings_${subset}.jsonl.gz \
        -s ${manifests_dir}/${scenario}-mdm_supervisions_${subset}.jsonl.gz \
        -o  ${exp_dir}/${scenario}_${subset}_selected \
        -k $top_k --nj $sel_nj

    recordings=${exp_dir}/${scenario}_${subset}_selected_recordings.jsonl.gz
    supervisions=${exp_dir}/${scenario}_${subset}_selected_supervisions.jsonl.gz

    echo "Stage 1: Prepare cut set"
    lhotse cut simple --force-eager \
        -r $recordings \
        -s $supervisions \
        ${exp_dir}/cuts.jsonl.gz

    echo "Stage 2: Trim cuts to supervisions (1 cut per supervision segment)"
    lhotse cut trim-to-supervisions --discard-overlapping \
        ${exp_dir}/cuts.jsonl.gz  \
        ${exp_dir}/cuts_per_segment.jsonl.gz

    cuts_per_recording=${exp_dir}/cuts.jsonl.gz
    cuts_per_segment=${exp_dir}/cuts_per_segment.jsonl.gz

    echo "--"
    ls -l $cuts_per_recording
    ls -l $cuts_per_segment
    echo "--"

    for enhancer_impl in $tunings
    do
        # Processed output directory
        enhanced_dir=${exp_dir}/${enhancer_impl}

        # Run processing
        CUDA_VISIBLE_DEVICES=${GPU_ID} python ../enhance_cuts/enhance_cuts.py \
            --enhancer-impl ${enhancer_impl} \
            --cuts-per-recording ${cuts_per_recording} \
            --cuts-per-segment ${cuts_per_segment} \
            --enhanced-dir ${enhanced_dir} \
            --num-workers ${num_workers}

        # Prepare manifests
        python prepare_nemo_manifests_for_processed.py --data-dir $enhanced_dir
    done
done
done