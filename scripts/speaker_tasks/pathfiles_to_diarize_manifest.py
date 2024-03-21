# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import random
import os 
from tqdm import tqdm
import concurrent

from nemo.collections.asr.parts.utils.manifest_utils import create_subsegment_manifest, get_path_dict, read_file, write_file, get_dict_from_wavlist

random.seed(42)

"""
This script creates manifest file for speaker diarization inference purposes.
Useful to get manifest when you have list of audio files and optionally rttm and uem files for evaluation

Note: make sure basename for each file is unique and rttm files also has the corresponding base name for mapping
"""

def create_manifest_mp(
    wav_path: str,
    manifest_filepath: str,
    text_path: str = None,
    rttm_path: str = None,
    uem_path: str = None,
    ctm_path: str = None,
    use_enclosing_folder_as_label: bool = True,
    add_duration: bool = False,
    num_workers: int = 20,
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
    if os.path.exists(manifest_filepath):
        os.remove(manifest_filepath)
    wav_pathlist = read_file(wav_path)
    wav_pathdict = get_dict_from_wavlist(wav_pathlist, use_enclosing_folder_as_label=use_enclosing_folder_as_label)
    len_wavs = len(wav_pathlist)
    uniqids = sorted(wav_pathdict.keys())

    text_pathdict = get_path_dict(text_path, uniqids, len_wavs, use_enclosing_folder_as_label, EXT='txt')
    rttm_pathdict = get_path_dict(rttm_path, uniqids, len_wavs, use_enclosing_folder_as_label, EXT='rttm')
    uem_pathdict = get_path_dict(uem_path, uniqids, len_wavs, use_enclosing_folder_as_label, EXT='uem')
    ctm_pathdict = get_path_dict(ctm_path, uniqids, len_wavs, use_enclosing_folder_as_label, EXT='ctm')

    multiprocessing = True
    if multiprocessing:
        chunk_size = len(uniqids) // num_workers + (len(uniqids) % num_workers > 0)
        uniqids_chunks = [uniqids[i:i + chunk_size] for i in range(0, len(uniqids), chunk_size)]
    
        lines = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submitting tasks
            futures = [executor.submit(create_subsegment_manifest, chunk, wav_pathdict, text_pathdict, rttm_pathdict, uem_pathdict, ctm_pathdict, add_duration) for chunk in uniqids_chunks]
            
            # Progress bar for completed futures
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing chunks", unit="chunk"):
                lines.extend(future.result()) 
    else:
        lines = create_subsegment_manifest(uniqids, wav_pathdict, text_pathdict, rttm_pathdict, uem_pathdict, ctm_pathdict, add_duration)
        write_file(manifest_filepath, lines, range(len(lines)))

    write_file(manifest_filepath, lines, range(len(lines)))

def main(
    wav_path, text_path=None, rttm_path=None, uem_path=None, ctm_path=None, manifest_filepath=None, use_enclosing_folder_as_label=False, add_duration=False, num_workers=20
):
    create_manifest_mp(
        wav_path,
        manifest_filepath,
        text_path=text_path,
        rttm_path=rttm_path,
        uem_path=uem_path,
        ctm_path=ctm_path,
        use_enclosing_folder_as_label=use_enclosing_folder_as_label,
        add_duration=add_duration,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths2audio_files", help="path to text file containing list of audio files", type=str, required=True
    )
    parser.add_argument("--paths2txt_files", help="path to text file containing list of transcription files", type=str)
    parser.add_argument("--paths2rttm_files", help="path to text file containing list of rttm files", type=str)
    parser.add_argument("--paths2uem_files", help="path to uem files", type=str)
    parser.add_argument("--paths2ctm_files", help="path to ctm files", type=str)
    parser.add_argument("--manifest_filepath", help="path to output manifest file", type=str, required=True)
    parser.add_argument("--num_workers", help="number of workers for multiprocessing", type=int, required=True)
    parser.add_argument("--use_enclosing_folder_as_label", help="path to output manifest file", type=bool, required=False)
    parser.add_argument(
        "--add_duration", help="add duration of audio files to output manifest files.", action='store_true',
    )
    args = parser.parse_args()
    main(
        args.paths2audio_files,
        args.paths2txt_files,
        args.paths2rttm_files,
        args.paths2uem_files,
        args.paths2ctm_files,
        args.manifest_filepath,
        args.use_enclosing_folder_as_label,
        args.add_duration,
        args.num_workers,
    )
