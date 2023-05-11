
This is a script to run processing from input.

### Setup

To setup

1) Install NeMo as usual

2) Setup GSS -- this is used for the baseline implementation and dataloaders

```
pip install 'cupy-cuda11x<12' # 12 removed cp.bool and had other changes breaking gss

pip install git+http://github.com/desh2608/gss

# this may be necessary if you get numpy errors
pip install numpy==1.21
```

3) Set paths in `run_processing.sh`

```
espnet_root= path to espnet/egs2/chime7_task1/asr1
chime7_root= path to chime7_official_cleaned
```

### Example

Assume diarization output is in
```
/path/to/diarization_outputs/system_vA01
```

Run the following

```
SCENARIOS="chime6" # run only on chime6
GPU_ID=1 # use only GPU 1
DIARIZATION_CONFIG=system_vA01
DIARIZATION_BASE_DIR=/path/to/diarization_outputs

bash run_processing.sh $SCENARIOS $GPU_ID $DIARIZATION_CONFIG $DIARIZATION_BASE_DIR
```

After this, you can use `../evaluation/run_asr.sh` on the manifests in
```
./processed/${DIARIZATION_CONFIG}/chime6/dev/nemo_v1
```

