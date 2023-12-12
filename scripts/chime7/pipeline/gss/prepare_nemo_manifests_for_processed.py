import glob
import os
import argparse
from collections import defaultdict
from nemo.collections.asr.parts.utils.manifest_utils import write_manifest

def main(data_dir: str, audio_type: str = 'flac'):
    # Make sure we know scenario
    if 'chime6' in data_dir:
        scenario = 'chime6'
    elif 'dipco' in data_dir:
        scenario = 'dipco'
    elif 'mixer6' in data_dir:
        scenario = 'mixer6'
    else:
        raise ValueError(f'Unknown setup: {data_dir}')

    # Make sure we know subset
    if 'dev' in data_dir and 'eval' in data_dir:
        raise ValueError(f'Unknown subset: both dev and eval are in {data_dir}')
    elif 'dev' in data_dir:
        subset = 'dev'
    elif 'eval' in data_dir:
        subset = 'eval'
    else:
        raise ValueError(f'Unknown subset: {data_dir}')
    
    # Find all audio files
    audio_files = glob.glob(data_dir + f'/**/*.{audio_type}', recursive=True)
    print('Found', len(audio_files), f'*.{audio_type} files in', data_dir)

    session_to_data = defaultdict(list)
    for audio_file in audio_files:
        # Each audio files is named session_id-speaker_id-start_time-end_time.{audio_type}
        # with start and end times in 1/100 seconds
        filename = os.path.basename(audio_file)
        if scenario == 'mixer6':
            # session_id has '-' in it
            parts = filename.replace(f'.{audio_type}', '').split('-')
            session_id = parts[0] # keep only session, drop dev and mdm
            speaker_id = parts[-2]
            start_end_time = parts[-1]
        else:
            session_id, speaker_id, start_end_time = filename.replace(f'.{audio_type}', '').split('-')
        start_time, end_time = start_end_time.split('_')
        start_time = int(start_time) / 100
        end_time = int(end_time) / 100
        session_to_data[session_id].append({
            'speaker': speaker_id,
            'session_id': session_id,
            'start_time': str(start_time),
            'end_time': str(end_time),
            'audio_filepath': os.path.relpath(path=audio_file, start=data_dir),
        })

    for session_id, data in session_to_data.items():
        manifest_file = os.path.join(data_dir, f'{scenario}-{subset}-{session_id}.json')
        write_manifest(manifest_file, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory with processed data',
    )
    args = parser.parse_args()

    main(data_dir=args.data_dir)