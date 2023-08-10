docker run --gpus all -it -v /media:/media -v /home/heh:/home/heh \
--shm-size=64g --ulimit memlock=-1 --ulimit \
stack=67108864 --device=/dev/snd nvcr.io/nv-maglev/nemo:chime7-gss

# -v ~/nemo_asr_eval:/ws