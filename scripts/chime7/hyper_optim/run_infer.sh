NEMO_ROOT="/media/data2/chime7-challenge/nemo-gitlab-chime7"
export PYTHONPATH=${NEMO_ROOT}:${PYTHONPATH}

python run_full_inference.py --output_dir /media/data3/chime7_outputs_3809
