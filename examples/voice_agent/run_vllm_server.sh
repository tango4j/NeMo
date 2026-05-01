# Run this under <NeMo Repo>/examples/voice_agent/ 
VLLM_USE_FLASHINFER_MOE_FP4=1 \
VLLM_FLASHINFER_MOE_BACKEND=throughput \
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
  --enable-log-requests \
  --enable-log-outputs \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.7 \
  --tensor-parallel-size 1 \
  --max-model-len 20000 \
  --port 8000 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser-plugin server/parsers/nano_v3_reasoning_parser.py \
  --reasoning-parser nano_v3 \
  --kv-cache-dtype fp8