
#!/bin/bash


diarout_dir="diarout_chime6-dev.json_cuda0"  # diarout_mixer6-dev.json_cuda1 diarout_dipco-dev.short4_cuda1 diarout_chime6-dev.json_cuda0
manifest="/media/data2/chime7-challenge/chime7_diar_results/system_vA01/${diarout_dir}/pred_jsons_with_overlap/"

python clean_diar_results.py --input_manifest $manifest --dataset chime6 --output_dir ./nemo_experiments/system_vA01/chime6/dev


diarout_dir="diarout_dipco-dev.short4_cuda1"  # diarout_mixer6-dev.json_cuda1 diarout_dipco-dev.short4_cuda1 diarout_chime6-dev.json_cuda0
manifest="/media/data2/chime7-challenge/chime7_diar_results/system_vA01/${diarout_dir}/pred_jsons_with_overlap/"

python clean_diar_results.py --input_manifest $manifest --dataset dipco --output_dir ./nemo_experiments/system_vA01/dipco/dev


diarout_dir="diarout_mixer6-dev.json_cuda1"  # diarout_mixer6-dev.json_cuda1 diarout_dipco-dev.short4_cuda1 diarout_chime6-dev.json_cuda0
manifest="/media/data2/chime7-challenge/chime7_diar_results/system_vA01/${diarout_dir}/pred_jsons_with_overlap/"

python clean_diar_results.py --input_manifest $manifest --dataset mixer6 --output_dir ./nemo_experiments/system_vA01/mixer6/dev

