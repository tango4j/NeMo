import subprocess
import argparse
import os
import json
import glob
import tqdm
import numpy as np
import soundfile as sf
import librosa
from nemo.utils import logging
from collections import Counter
from nemo.collections.asr.parts.utils.manifest_utils import (
    get_path_dict,
    read_file,
    write_file,
)
from nemo.collections.asr.parts.utils.diarization_utils import (
    convert_word_dict_seq_to_ctm,
    write_txt
)
from nemo.collections.asr.parts.utils.speaker_utils import (
    rttm_to_labels,
    labels_to_rttmfile,
)
from typing import Dict, List, Tuple

this_dir = os.path.dirname(os.path.abspath(__file__))

datasets = ['chime6', 'dipco', 'mixer6']

def get_rttm_info(rttm_filepath: str) -> Tuple[int, float, float]:
    """
    Get the number of speakers, the minimum and maximum time from an RTTM file
    """
    labels = rttm_to_labels(rttm_filepath)
    label_list, spk_list = [], []
    for l in labels:
        label_list.append((float(l.split()[0]), float(l.split()[1])))
        spk_list.append(l.split()[-1])
    label_array = np.array(label_list)
    num_speakers = Counter(spk_list).keys().__len__()
    rttm_min = np.min(label_array[:, 0])
    rttm_max = np.max(label_array[:, 1])
    return num_speakers, rttm_min, rttm_max

def create_multichannel_manifest(
    uniq_id_path: str,
    wav_path: str,
    manifest_filepath: str,
    text_path: str = None,
    rttm_path: str = None,
    uem_path: str = None,
    ctm_path: str = None,
    add_duration: bool = False,
):
    """
    Create base manifest file

    Args:
        wav_path (str): Path to list of wav files
        manifest_filepath (str): Path to output manifest file
        text_path (str): Path to list of text files
        rttm_path (str): Path to list of rttm files
        uem_path (str): Path to list of uem files
        ctm_path (str): Path to list of ctm files
        add_duration (bool): Whether to add durations to the manifest file
    """
    uniq_id_list = [ x.strip() for x in read_file(uniq_id_path) ]
    uniqids = sorted(uniq_id_list)
    len_wavs = len(uniq_id_list)

    text_pathdict = get_path_dict(text_path, uniqids, len_wavs)
    rttm_pathdict = get_path_dict(rttm_path, uniqids, len_wavs)
    uem_pathdict = get_path_dict(uem_path, uniqids, len_wavs)
    ctm_pathdict = get_path_dict(ctm_path, uniqids, len_wavs)

    if os.path.exists(manifest_filepath):
        os.remove(manifest_filepath)
    wav_pathlist = [ x.strip() for x in read_file(wav_path) ]
    wav_pathdict = {}
    for uniq_id, cs_wav_paths in zip(uniqids,wav_pathlist):
        wav_pathdict[uniq_id] = cs_wav_paths

    lines = []
    total_file_count = len(uniqids)
    for idx, uid in enumerate(uniqids):
        count = idx + 1

        wav, text, rttm, uem, ctm = (
            wav_pathdict[uid],
            text_pathdict[uid],
            rttm_pathdict[uid],
            uem_pathdict[uid],
            ctm_pathdict[uid],
        )

        audio_line_list = wav.split(',')

        if rttm is not None:
            rttm = rttm.strip()
            num_speakers, rttm_min, rttm_max = get_rttm_info(rttm_filepath=rttm)
            rttm_dur = rttm_max - rttm_min
        else:
            num_speakers = None

        if uem is not None:
            uem = uem.strip()

        if text is not None:
            with open(text.strip()) as f:
                text = f.readlines()[0].strip()
        else:
            text = "-"

        if ctm is not None:
            ctm = ctm.strip()

        duration = None
        audio_duration_list = []
        for audio_line in tqdm.tqdm(audio_line_list, desc=f"Measuring multichannel audio duration for {uid} {count}/{total_file_count}", unit=" files"):
            audio_line = audio_line.strip()
            data, samplerate = sf.read(audio_line)
            duration = float(data.shape[0] / samplerate)
            audio_duration_list.append(duration)
        min_duration, max_duration = min(audio_duration_list), max(audio_duration_list)
        meta = [
            {
                "audio_filepath": audio_line_list,
                "offset": rttm_min,
                "duration": rttm_dur,
                "min_duration": min_duration,
                "max_duration": max_duration,
                "uniq_id": uid,
                "label": "infer",
                "text": text,
                "num_speakers": num_speakers,
                "rttm_filepath": rttm,
                "uem_filepath": uem,
                "ctm_filepath": ctm,
            }
        ]
        lines.extend(meta)

    write_file(manifest_filepath, lines, range(len(lines)))

def create_speaker_line(start: int, end: int, speaker_id: int, output_precision: int = 3) -> List[str]:
    """
    Create new RTTM entries from the segments input

    Args:
        start (int): Current start of the audio file being inserted.
        end (int): End of the audio file being inserted.
        speaker_id (int): LibriSpeech speaker ID for the current entry.
    
    Returns:
        rttm_list (list): List of rttm entries
    """
    new_start = start
    t_stt = float(round(new_start, output_precision))
    t_end = float(round(end, output_precision))
    speaker_segment_str = f"{t_stt} {t_end} {speaker_id}"
    return speaker_segment_str

def create_word_dict_lines(data):
    """
    Create new word dict lines from the segments input.
    NOTE: This function assumes a segment contains multiple words and 
    the start and end time of the segment is the same as the start and end time of the words.

    Args:
        data (dict): Dictionary containing the start_time, end_time, speaker, and words of the segment.

    Returns:
        new_word_dict_lines (list): List of word dict lines
    """
    new_word_dict_lines = []
    words_list = data['words'].split(' ')
    start_time = data['start_time']
    end_time = data['end_time']
    for word in words_list:
        word_dict_line = {}
        word_dict_line['start_time'] = float(start_time)
        word_dict_line['end_time'] = float(end_time)
        word_dict_line['word'] = word
        word_dict_line['speaker'] = data['speaker']
        new_word_dict_lines.append(word_dict_line)
    return new_word_dict_lines

def parse_chime7_json_file(dataset: str, data: Dict, file_path: str, subset: str):
    """
    This function was originated from prepare_manifest_dev_eval.py.
    
    Args:
        dataset (str): Name of the dataset.
        data (dict): Dictionary containing the start_time, end_time, speaker, and words of the segment.
        file_path (str): Path to the json file.
        subset (str): Name of the subset.

    Returns:
        offset (float): Start time of the segment.
        duration (float): Duration of the segment.
        end_time (float): End time of the segment.
        audio_filename (str): Name of the audio file.
    """
    if dataset in ['chime6', 'dipco']:
        audio_filename = data['session_id']
    elif dataset in ['mixer6']:
        # drop json extension
        audio_filename, _ = os.path.splitext(os.path.basename(file_path))
        # Mixer6 has a different naming convention for "'".
        data['words'] = data['words'].replace("\u2019", "'")

    # path is relative to dataset_output_dir
    offset = float(data['start_time'])
    duration = float(data['end_time']) - offset
    end_time = float(offset + duration)
    return offset, duration, end_time, audio_filename

def get_mc_audio_filepaths(multichannel_audio_files: str, dataset: str, dataset_dir: str):
    """
    This function was originated from prepare_manifest_dev_eval.py.
    In this function, we use absolute path to get the single-channel audio files.

    Args:
        multichannel_audio_files (str): Path to multichannel audio files.
        dataset (str): Dataset name. One of ['chime6', 'dipco', 'mixer6']
        dataset_dir (str): Absolute path to the dataset directory.

    Returns:
         (dict): Dictionary mapping multichannel audio file to list of single-channel audio files.
    """
    mc_audio_to_list_of_sc_files = dict()
    for mc_audio_file in multichannel_audio_files:
        logging.info('Preparing list of single-channel audio files: %s', mc_audio_file)

        # drop the absolute
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

        # Make filepaths absolute
        mc_audio_to_list_of_sc_files[mc_audio_file] = sc_audio_files
    
    mc_audio_file_list = []
    for base_name, sc_audio_files in mc_audio_to_list_of_sc_files.items(): 
        mc_audio_file_list.append(",".join(sc_audio_files))

    return mc_audio_file_list


def main(data_dir: str, subset: str, output_dir: str, output_precision: int=2):
    """
    Take original CHiME-7 data and prepare
    multichannel audio files and the corresponding manifests for inference and evaluation.
    """
    total_data_stats = {}
    for dataset in datasets:
        dataset_dir = os.path.join(data_dir, dataset)
        dataset_output_dir = os.path.join(output_dir, dataset)

        if not os.path.isdir(dataset_output_dir):
            logging.info('Creating dir: %s', dataset_output_dir)
            os.makedirs(dataset_output_dir)

        transcriptions_scoring_dir = os.path.join(dataset_dir, 'transcriptions_scoring', subset)
        transcriptions_scoring_files = glob.glob(transcriptions_scoring_dir + '/*.json')

        logging.info('transcriptions_scoring')
        logging.info('\tnum files: %s', len(transcriptions_scoring_files))

        # Prepare manifest data
        manifest_data, session_duration_list = [], []
        rttm_path_list, ctm_path_list, text_path_list, mc_audio_path_list, uniq_id_list = [], [], [], [], []
        transcriptions_scoring_files = sorted(transcriptions_scoring_files)
        for file_path in transcriptions_scoring_files:
            dataset_dir = os.path.join(data_dir, dataset)
            logging.info('Process: %s', file_path)

            with open(file_path, 'r') as f:
                file_data = json.load(f)
            speaker_line_list, word_dict_list = [], []
            session_end_time = float(file_data[-1]['end_time'])
            session_duration_list.append(session_end_time)
            for data in file_data:
                offset, duration, end_time, audio_filename = parse_chime7_json_file(dataset, data, file_path, subset)
                speaker_line = create_speaker_line(start=offset, end=end_time, speaker_id=data['speaker'], output_precision=output_precision)
                speaker_line_list.append(speaker_line)
                new_word_dict_lines = create_word_dict_lines(data)
                word_dict_list.extend(new_word_dict_lines)
                
                # path is relative to dataset_output_dir
                for key in ['audio_filepath', 'offset', 'duration', 'text']:
                    assert key not in data
                data.update(
                    {'audio_filepath': os.path.join('audio', subset, audio_filename),
                     'offset': float(data['start_time']),
                     'duration': float(data['end_time']) - offset,
                     'text': data['words'],}
                )
                manifest_data.append(data)

            if 'session_id' in data:
                session_id = data['session_id']
            else:
                # mixer6 does not have session_id in the json file.
                # Get session_id from the basename of the file path
                session_id = os.path.basename(file_path).split('.')[0]

            # If dataset_output_dir does not exist, create it
            os.makedirs(os.path.join(dataset_output_dir, 'rttm'), exist_ok=True)
            os.makedirs(os.path.join(dataset_output_dir, 'ctm'), exist_ok=True)

            # Write RTTM file with the name convention: <dataset>-<subset>-<session_id>.rttm
            uniq_id = f"{dataset}-{subset}-{session_id}" 
            out_rttm_path = os.path.join(dataset_output_dir, 'rttm', f'{uniq_id}.rttm')
            labels_to_rttmfile(labels=speaker_line_list, uniq_id=uniq_id, out_rttm_dir=os.path.join(dataset_output_dir, 'rttm'))

            # Write CTM file with the name convention: <dataset>-<subset>-<session_id>.ctm
            ctm_lines_list = convert_word_dict_seq_to_ctm(word_dict_seq_list=word_dict_list, uniq_id=uniq_id)
            out_ctm_path=os.path.join(os.path.join(dataset_output_dir, 'ctm'), f'{uniq_id}.ctm')
            write_txt(w_path=out_ctm_path, val='\n'.join(ctm_lines_list))
            
            rttm_path_list.append(out_rttm_path)
            ctm_path_list.append(out_ctm_path)
            uniq_id_list.append(uniq_id)
        
        # End of dataset loop

        # Get duration stats
        data_stats = get_data_stats(session_duration_list, output_precision=output_precision)
        total_data_stats[dataset] = data_stats

        # Get multichannel audio file paths
        multichannel_audio_files = [data['audio_filepath'] for data in manifest_data]
        multichannel_audio_files = sorted(set(multichannel_audio_files))
        mc_audio_path_list = get_mc_audio_filepaths(multichannel_audio_files, dataset, dataset_dir)
        
        # If dataset_output_dir does not exist, create it
        os.makedirs(os.path.join(dataset_output_dir, 'filelist'), exist_ok=True)
        os.makedirs(os.path.join(dataset_output_dir, 'mulspk_asr_manifest'), exist_ok=True)

        split_id = f"{dataset}-{subset}" 
        out_mc_audio_file_list_path = os.path.join(dataset_output_dir, 'filelist', f'{split_id}.mc_audio.list')
        out_rttm_file_list_path = os.path.join(dataset_output_dir, 'filelist', f'{split_id}.rttm.list')
        out_ctm_file_list_path = os.path.join(dataset_output_dir, 'filelist', f'{split_id}.ctm.list')
        out_uniqid_list_path = os.path.join(dataset_output_dir, 'filelist', f'{split_id}.uniq_id.list')
        
        out_manifest_path = os.path.join(dataset_output_dir, 'mulspk_asr_manifest', f'{split_id}.json')
        
        write_txt(w_path=out_mc_audio_file_list_path, val='\n'.join(mc_audio_path_list))
        write_txt(w_path=out_rttm_file_list_path, val='\n'.join(rttm_path_list))
        write_txt(w_path=out_ctm_file_list_path, val='\n'.join(ctm_path_list))
        write_txt(w_path=out_uniqid_list_path, val='\n'.join(uniq_id_list))

        create_multichannel_manifest(uniq_id_path=out_uniqid_list_path,
                                     wav_path=out_mc_audio_file_list_path,
                                     manifest_filepath=out_manifest_path,
                                     rttm_path=out_rttm_file_list_path,
                                     ctm_path=out_ctm_file_list_path,
                                     add_duration = True
        )

    # Display data statistics 
    print(json.dumps(total_data_stats, indent=4, default=str))

def get_data_stats(session_duration_list: list, output_precision: int) -> Dict[str, float]:
    """
    Get data statistics from the list of session durations.

    Args:
        session_duration_list (list): List of session durations.

    Returns:
        data_stats (dict): Dictionary of data statistics.
        keys:
            max_session_duration (float): Maximum session duration.
            min_session_duration (float): Minimum session duration.
            avg_session_duration (float): Average session duration.
            total_duration (float): Total duration of all sessions.
            num_files (int): Number of sessions.
    """
    data_stats = {}
    if len(session_duration_list) == 0:
        raise ValueError('session_duration_list is empty.')
    data_stats['max_session_duration'] = round(max(session_duration_list), output_precision)
    data_stats['min_session_duration'] = round(min(session_duration_list), output_precision)
    data_stats['avg_session_duration'] = round(np.mean(np.array(session_duration_list)), output_precision)
    data_stats['total_duration'] = round(sum(session_duration_list), output_precision)
    data_stats['num_files'] = len(session_duration_list)
    return data_stats

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
    args = parser.parse_args()

    main(data_dir=args.data_dir, subset=args.subset, output_dir=args.output_dir)
