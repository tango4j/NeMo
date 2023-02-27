import subprocess
import argparse
import os
import json
import glob
import tqdm
import numpy as np
import soundfile as sf
from nemo.utils import logging
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment

this_dir = os.path.dirname(os.path.abspath(__file__))

datasets = ['chime6', 'dipco', 'mixer6']


def to_seconds(time: str):

    time_sec = 0
    for tt in time.split(':'):
        time_sec = time_sec * 60 + float(tt)

    return time_sec


def get_duration_num_channels(path: str):

    duration = subprocess.check_output(['soxi', '-d', path]).decode("utf-8").strip()
    num_channels = subprocess.check_output(['soxi', '-c', path])
    return to_seconds(duration), int(num_channels)


def main(data_dir: str, subset: str, output_dir: str, overwrite: bool):
    """Take original CHiME-7 data and prepare
    multichannel audio files and the corresponding manifests.
    """
    for dataset in datasets:
        dataset_dir = os.path.join(data_dir, dataset)
        dataset_output_dir = os.path.join(output_dir, dataset)

        if not os.path.isdir(dataset_output_dir):
            logging.info('Creating dir: %s', dataset_output_dir)
            os.makedirs(dataset_output_dir)
        elif not overwrite:
            raise RuntimeError(f'Directory {dataset_output_dir} already exists. Consider using --overwrite.')

        transcriptions_scoring_dir = os.path.join(dataset_dir, 'transcriptions_scoring', subset)
        transcriptions_scoring_files = glob.glob(transcriptions_scoring_dir + '/*.json')

        logging.info('transcriptions_scoring')
        logging.info('\tdir:       %s', transcriptions_scoring_dir)
        logging.info('\tnum files: %s', len(transcriptions_scoring_files))

        # Prepare manifest data
        manifest_data = []

        for file_path in transcriptions_scoring_files:
            logging.info('Process: %s', file_path)

            with open(file_path, 'r') as f:
                file_data = json.load(f)

                for data in file_data:
                    if dataset in ['chime6', 'dipco']:
                        audio_filename = data['session_id']
                    elif dataset in ['mixer6']:
                        # drop json extension
                        audio_filename, _ = os.path.splitext(os.path.basename(file_path))

                    # path is relative to dataset_output_dir
                    audio_filepath = os.path.join('audio', subset, audio_filename)
                    offset = float(data['start_time'])
                    duration = float(data['end_time']) - offset
                    text = data['words']

                    for key in ['audio_filepath', 'offset', 'duration', 'text']:
                        assert key not in data

                    data.update(
                        {'audio_filepath': audio_filepath, 'offset': offset, 'duration': duration, 'text': text,}
                    )

                    manifest_data.append(data)

        # Prepare audio files for each example
        multichannel_audio_files = [data['audio_filepath'] for data in manifest_data]
        multichannel_audio_files = set(multichannel_audio_files)
        mc_audio_to_list_of_sc_files = dict()

        for mc_audio_file in multichannel_audio_files:
            logging.info('Preparing list of single-channel audio files: %s', mc_audio_file)

            # drop the extension
            filepath_base, _ = os.path.splitext(mc_audio_file)

            if dataset in ['chime6', 'dipco']:
                file_ext = '_U??.CH?.wav'
            elif dataset in ['mixer6']:
                file_ext = '_CH??.wav'

            # Get single-channel files
            sc_audio_files = glob.glob(os.path.join(dataset_dir, filepath_base + file_ext))
            sc_audio_files.sort()

            # Double-check that the channel count is correct
            if dataset == 'chime6':
                assert len(sc_audio_files) in [20, 24], f'Expected 20 or 24 files, found {len(sc_audio_files)}'
            elif dataset == 'dipco':
                assert len(sc_audio_files) == 35, f'Expected 35 files, found {len(sc_audio_files)}'
            elif dataset == 'mixer6':
                assert len(sc_audio_files) == 13, f'Expected 13 files, found {len(sc_audio_files)}'
            else:
                raise ValueError(f'Unknown dataset: {dataset}')

            # Make filepaths relative
            sc_audio_files = [os.path.relpath(path, start=dataset_dir) for path in sc_audio_files]

            mc_audio_to_list_of_sc_files[mc_audio_file] = sc_audio_files

        # Replace each audio_filepath with a list of files
        # Instead of creating a large multichannel file, we provide a list with one file per channel
        for data in manifest_data:
            data['audio_filepath'] = mc_audio_to_list_of_sc_files[data['audio_filepath']]

        # Save manifest
        manifest_filepath = os.path.join(dataset_output_dir, f'{dataset}-{subset}_manifest.json')
        if os.path.isfile(manifest_filepath) and not overwrite:
            raise RuntimeError(f'File {manifest_filepath} already exists. Consider using --overwrite.')
        write_manifest(manifest_filepath, manifest_data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir', type=str, required=True, help='Directory with CHiME-7 data',
    )
    parser.add_argument(
        '--subset', choices=['dev'], default='dev', help='Data subset',
    )
    parser.add_argument(
        '--output-dir', type=str, required=True, help='Output dir',
    )
    parser.add_argument(
        '--overwrite', action='store_true', help='Overwrite existing files',
    )
    args = parser.parse_args()

    main(data_dir=args.data_dir, subset=args.subset, output_dir=args.output_dir, overwrite=args.overwrite)
