#!/bin/bash
# Set HF_TOKEN in your shell before running this script; do not commit real tokens.
# Hugging Face cache on the HOST (must be an absolute path). Mount it into the
# container so weights survive `docker run --rm`. Without this, every new
# container downloads the full model again.
# Override: HOST_HF_CACHE=/your/path ./run_vllm_server_alt2.sh
HOST_HF_CACHE="${HOST_HF_CACHE:-${HOME}/.cache/huggingface}"
mkdir -p "${HOST_HF_CACHE}"
export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm
export VLLM_FLASHINFER_MOE_BACKEND=throughput
export VLLM_USE_FLASHINFER_MOE_FP8=1

# DOCKER_IMAGE=vllm/vllm-openai:cu130-nightly
# DOCKER_IMAGE=vllm/vllm-openai:v0.17.1-cu130
DOCKER_IMAGE=vllm/vllm-openai:v0.19.0-cu130
docker run \
  -e HF_TOKEN=${HF_TOKEN} \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_CACHE=/root/.cache/huggingface/hub \
  -v "${HOST_HF_CACHE}:/root/.cache/huggingface" \
  -e VLLM_FLASHINFER_MOE_BACKEND=${VLLM_FLASHINFER_MOE_BACKEND} \
  -e VLLM_USE_FLASHINFER_MOE_FP8=${VLLM_USE_FLASHINFER_MOE_FP8} \
  --gpus all \
  -it \
  --rm \
  --ipc=host \
  -p 8000:8000 \
  -v /home/lab:/home/lab \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  ${DOCKER_IMAGE} \
  --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
  --async-scheduling \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 1 \
  --trust-remote-code \
  --attention-backend TRITON_ATTN \
  --gpu-memory-utilization 0.9 \
  --enable-chunked-prefill \
  --max-num-seqs 1 \
  --host 0.0.0.0 \
  --port 8000 \
  --mamba_ssm_cache_dtype float32 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3
