MODEL_NAME="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"

curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'$MODEL_NAME'",
        "messages":[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Tell me a joke about new york"}],
        "max_tokens": 256,
        "temperature": 0.6,
        "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": false}
    }'
