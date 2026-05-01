#!/bin/bash

# Hugging Face cache on the HOST (must be an absolute path). Mount it into the
# container so weights survive `docker run --rm`.
# Override: HOST_HF_CACHE=/your/path ./run_vllm_server_heh1.sh
HOST_HF_CACHE="${HOST_HF_CACHE:-${HOME}/.cache/huggingface}"
mkdir -p "${HOST_HF_CACHE}"

# Configuration for vLLM
export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm
export VLLM_FLASHINFER_MOE_BACKEND=throughput
export VLLM_USE_FLASHINFER_MOE_FP4=1

# Define the Docker image
# DOCKER_IMAGE=vllm/vllm-openai:cu130-nightly
# DOCKER_IMAGE=vllm/vllm-openai:v0.17.1-cu130
DOCKER_IMAGE=vllm/vllm-openai:v0.19.0-cu130

# Define the model to be served
# MODEL=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
# MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4

# Execute the vLLM Docker container with specified parameters
docker run \
  -e HF_TOKEN=${HF_TOKEN} \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_CACHE=/root/.cache/huggingface/hub \
  -v "${HOST_HF_CACHE}:/root/.cache/huggingface" \
  -e VLLM_FLASHINFER_ALLREDUCE_BACKEND=${VLLM_FLASHINFER_ALLREDUCE_BACKEND} \
  -e VLLM_FLASHINFER_MOE_BACKEND=${VLLM_FLASHINFER_MOE_BACKEND} \
  -e VLLM_USE_FLASHINFER_MOE_FP4=${VLLM_USE_FLASHINFER_MOE_FP4} \
  --gpus all \
  -it \
  --rm \
  --ipc=host \
  -p 8000:8000 \
  -v /home/lab:/home/lab \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  ${DOCKER_IMAGE} \
  --model ${MODEL} \
  --async-scheduling \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 1 \
  --trust-remote-code \
  --attention-backend TRITON_ATTN \
  --gpu-memory-utilization 0.7 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --max-num-seqs 4 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-cudagraph-capture-size 128 \
  --mamba-ssm-cache-dtype float16 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 \
  --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "I have to finalize the answer now.</think>"}'

