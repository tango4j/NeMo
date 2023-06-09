#!/usr/bin/env bash
set -eou pipefail

# To prepare short manifests for test
# 1) gzip -dk cuts_per_segment.jsonl.gz
# 2) head -n 10 cuts_per_segment.jsonl > cuts_per_segment_short.jsonl
# 3) gzip -k cuts_per_segment_short.jsonl


cuts_per_recording=$PWD/manifests/chime6/dev/cuts.jsonl.gz
cuts_per_segment=$PWD/manifests/chime6/dev/cuts_per_segment_short.jsonl.gz

# enhancer_impl=gss # baseline
enhancer_impl=nemo_v1 # NeMo baseline

enhanced_dir=${PWD}/output/${enhancer_impl}
python enhance_cuts.py --enhancer-impl ${enhancer_impl} --cuts-per-recording ${cuts_per_recording} --cuts-per-segment ${cuts_per_segment} --enhanced-dir ${enhanced_dir} --use-garbage-class