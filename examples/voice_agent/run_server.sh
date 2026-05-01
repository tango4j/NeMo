NEMO_PATH=/home/lab/projects/nemo_voice_agent/NeMo  # Use your local NeMo path with the latest main branch from: https://github.com/NVIDIA-NeMo/NeMo
export PYTHONPATH=$NEMO_PATH:$PYTHONPATH
# export HF_TOKEN="hf_..."  # Set this in your shell if needed; do not commit real tokens.
# # export HF_HUB_CACHE="/path/to/your/huggingface/cache"  # change where HF cache is stored if you don't want to use the default cache
# # export SERVER_CONFIG_PATH="/path/to/your/server/config.yaml"  # change to the server config you want to use, otherwise it will use the default config in `server/server_configs/default.yaml`
SERVER_CONFIG_PATH=/home/lab/projects/nemo_voice_agent/NeMo/examples/voice_agent/server/server_configs/default.yaml
python ./server/server.py

