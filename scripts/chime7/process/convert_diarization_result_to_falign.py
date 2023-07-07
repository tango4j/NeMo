import argparse
import os
import glob
import tqdm
import json
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest

# diarization_system = 'system_vA04D'
# diarization_dir = os.path.expanduser(f'~/scratch/chime7/chime7_diar_results/{diarization_system}')

# output_dir = f'./alignments/{diarization_system}'

def main(diarization_dir: str, diarization_params: str, output_dir: str = "", subsets: list = ['dev']):
    # Assumption:
    # Output of diarization is organized in 3 subdirectories, with each subdirectory corresponding to one scenario (chime6, dipco, mixer6)
    diarization_system = diarization_dir.split('/')[-1]
    print('diarization_dir:', diarization_dir)

    if output_dir == "":
        output_dir = f'./alignments/{diarization_system}-{diarization_params}'
    print('output_dir:', output_dir)

    scenario_dirs = glob.glob(diarization_dir + '/*')
    # assert len(scenario_dirs) == 3, f'Expected 3 subdirectories, found {len(scenario_dirs)}'
    none_useful_fields = ['audio_filepath', 'words', 'text', 'duration', 'offset']
    for scenario in ['chime6', 'dipco', 'mixer6']:
        for subset in subsets:
            # Currently, subdirectories don't have a uniform naming scheme
            # Therefore, we pick the subdirectory that has both scenario and subset in its name
            scenario_subset_dir = [sd for sd in scenario_dirs if scenario in sd]
            if len(scenario_subset_dir) == 0:
                print(f'No subdirectory found for {scenario} and {subset}')
                break
            else:
                scenario_subset_dir = scenario_subset_dir[0]

            # Grab manifests from the results of diarization
            manifests_dir =  os.path.join(scenario_subset_dir, diarization_params)
            if not os.path.isdir(manifests_dir):
                manifests_dir = os.path.join(scenario_subset_dir, f"pred_jsons_{diarization_params}")

            manifests = list(glob.glob(manifests_dir + '/*.json'))
            if len(manifests) == 0:
                print(f'No manifests found in {manifests_dir}')
                continue
            
            # Process each manifest
            for manifest in manifests:
                manifest_name = os.path.basename(manifest)
                session_name = manifest_name.replace(scenario, '').replace('dev', '').replace('.json', '').strip('-')
                new_manifest = os.path.join(output_dir, scenario, subset, session_name + '.json')
                
                if not os.path.isdir(os.path.dirname(new_manifest)):
                    os.makedirs(os.path.dirname(new_manifest))

                # read manifest
                try:
                    data = read_manifest(manifest)
                except json.decoder.JSONDecodeError:
                    data = json.load(open(manifest, 'r'))

                for item in data:
                    # not required
                    for k in none_useful_fields:
                        if k in item:
                            item.pop(k)
                    # set these to be consistent with the baseline falign manifests
                    item['session_id'] = session_name
                    item['words'] = 'placeholder'

                # dump the list in a json file (not JSONL as our manifests)
                with open(new_manifest, 'w') as f:
                    json.dump(data, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--diarization-dir',
        type=str,
        required=True,
        help='Directory with output of diarization',
    )
    parser.add_argument(
        '--diarization-params',
        type=str,
        default='pred_jsons_with_overlap',
        help='Name of the subdirectory with diarization results',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='',
        help='Directory to store the output',
    )
    args = parser.parse_args()

    main(diarization_dir=args.diarization_dir, diarization_params=args.diarization_params, output_dir=args.output_dir)
