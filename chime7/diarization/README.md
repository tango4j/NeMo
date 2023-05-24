# Manual for running Chime7 Multichannel Diarization


Please use the following launcher script.
`chime7/diarization/run_msdd_v2_infer_sysVB01.sh`


### Step 1 - Models
- Download models from the path in the following workstation 
- Diarization model path
`lab@10.110.43.14:/disk_c_nvd/models/msdd_v2_models/msdd_v2_PALO_bs6_a003_version6_e53.ckpt`
- VAD model path (This is needed to replace the fine-tuned VAD model in the diarization model)
`lab@10.110.43.14:/disk_c_nvd/models/frame_vad_models/wnc_frame_vad.nemo`
- password for lab@10.110.43.14 is in this [Google Docs page](https://docs.google.com/document/d/1IT07_3YkgshtMGrBLW6vrUjRBl_LwaFlseQBjELhZAY/edit?usp=sharing) :

- Plug these into `MSDD_MODEL_PATH` and `VAD_MODEL_PATH`.

### Step 2 - Dev/Eval set manifest

- Go download the dev/eval set data from the following server
- `lab@10.110.43.14:/disk_d/datasets/nemo_chime7_diar_manifests`
- password for lab@10.110.43.14 is in this [Google Docs page](https://docs.google.com/document/d/1IT07_3YkgshtMGrBLW6vrUjRBl_LwaFlseQBjELhZAY/edit?usp=sharing)
- Plug these manifest path files to `test_manifest`.
- If you want to do a quick sanity check, use the files with `short1` postfix. 

### Step 3 - Batch size
- You need to setup a right batch size for your machine.
- Use batch size 11 if GRAM is 32GB, 14 if GRAM is 48GB
- For example, `BATCH_SIZE=11`

### Step 4 - Setup a temporary path

- Multi-channel diarization takes up a massive memory size since it extracts embedding and VAD logits for every 0.05 second for every channel.
- Please provide a path that can store more than 300~400GB memory to `$DIAR_OUT_DOWNLOAD`.



