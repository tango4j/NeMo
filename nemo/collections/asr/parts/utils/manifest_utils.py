# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os
import numpy as np

from collections import Counter
from collections import OrderedDict as od
from nemo.collections.asr.parts.utils.speaker_utils import (
    rttm_to_labels,
    get_subsegments,
    get_uniqname_from_filepath,
    audio_rttm_map,
    segments_manifest_to_subsegments_manifest,
    write_rttm2manifest
)

def rreplace(s, old, new):
    li = s.rsplit(old, 1)
    return new.join(li)

def get_uniq_id_with_period(path):
    split_path = os.path.basename(path).split('.')[:-1]
    uniq_id = '.'.join(split_path) if len(split_path) > 1 else split_path[0]
    return uniq_id

def get_subsegment_dict(subsegments_manifest_file, window, shift, deci):
    _subsegment_dict = {}
    with open(subsegments_manifest_file, 'r') as subsegments_manifest:
        # print(f"Reading subsegments_manifest_file: {subsegments_manifest_file}")
        segments = subsegments_manifest.readlines()
        for segment in segments:
            segment = segment.strip()
            dic = json.loads(segment)
            audio, offset, duration, label = dic['audio_filepath'], dic['offset'], dic['duration'], dic['label']
            subsegments = get_subsegments(offset=offset, window=window, shift=shift, duration=duration)
            uniq_id = get_uniq_id_with_period(audio)
            if uniq_id not in _subsegment_dict:
                _subsegment_dict[uniq_id] = {'ts' : [], 'json_dic': []}
            for subsegment in subsegments:
                start, dur = subsegment
            _subsegment_dict[uniq_id]['ts'].append([round(start, deci), round(start+dur, deci)])
            _subsegment_dict[uniq_id]['json_dic'].append(dic)
    return _subsegment_dict

def get_input_manifest_dict(input_manifest_path):
    input_manifest_dict = {}
    with open(input_manifest_path, 'r') as input_manifest_fp:
        json_lines = input_manifest_fp.readlines()
        for json_line in json_lines:
            dic = json.loads(json_line)
            dic["text"] = "-"
            uniq_id = get_uniqname_from_filepath(dic["audio_filepath"])
            input_manifest_dict[uniq_id] = dic
    return input_manifest_dict

def write_truncated_subsegments(input_manifest_dict, _subsegment_dict, output_manifest_path, step_count, deci):
    with open(output_manifest_path, 'w') as output_manifest_fp:
        for uniq_id, subseg_dict in _subsegment_dict.items():
            # print(f"Writing {uniq_id}")
            subseg_array = np.array(subseg_dict['ts'])
            subseg_array_idx = np.argsort(subseg_array, axis=0)
            chunked_set_count = subseg_array_idx.shape[0] // step_count

            for idx in range(chunked_set_count-1):
                chunk_index_stt = subseg_array_idx[:, 0][idx * step_count]
                chunk_index_end = subseg_array_idx[:, 1][(idx+1)* step_count]
                offset_sec = subseg_array[chunk_index_stt, 0]
                end_sec = subseg_array[chunk_index_end, 1]
                dur = round(end_sec - offset_sec, deci)
                meta = input_manifest_dict[uniq_id]
                meta['offset'] = offset_sec
                meta['duration'] = dur
                json.dump(meta, output_manifest_fp)
                output_manifest_fp.write("\n")

def write_file(name, lines, idx):
    with open(name, 'w') as fout:
        for i in idx:
            dic = lines[i]
            json.dump(dic, fout)
            fout.write('\n')

def read_file(pathlist):
    pathlist = open(pathlist, 'r').readlines()
    return sorted(pathlist)

def get_dict_from_wavlist(pathlist):
    path_dict = od()
    pathlist = sorted(pathlist)
    for line_path in pathlist:
        uniq_id = os.path.basename(line_path).split('.')[0]
        path_dict[uniq_id] = line_path
    return path_dict

def get_dict_from_list(data_pathlist, uniqids):
    path_dict = {}
    for line_path in data_pathlist:
        uniq_id = os.path.basename(line_path).split('.')[0]
        if uniq_id in uniqids:
            path_dict[uniq_id] = line_path
        else:
            raise ValueError(f'uniq id {uniq_id} is not in wav filelist')
    return path_dict

def get_path_dict(data_path, uniqids, len_wavs=None):
    if data_path is not None:
        data_pathlist = read_file(data_path)
        if len_wavs is not None:
            assert len(data_pathlist) == len_wavs
            data_pathdict = get_dict_from_list(data_pathlist, uniqids)
    elif len_wavs is not None:
        data_pathdict = {uniq_id: None for uniq_id in uniqids}
    return data_pathdict

def create_segment_manifest(input_manifest_path, output_manifest_path, window, shift, step_count, deci):
    if '.json' not in input_manifest_path:
        raise ValueError("input_manifest_path file should be .json file format")
    if output_manifest_path and '.json' not in output_manifest_path:
        raise ValueError("output_manifest_path file should be .json file format")
    elif not output_manifest_path:
        output_manifest_path = rreplace(input_manifest_path, '.json', f'_{step_count}seg.json')

    input_manifest_dict = get_input_manifest_dict(input_manifest_path)
    segment_manifest_path = rreplace(input_manifest_path, '.json', '_seg.json')
    subsegment_manifest_path = rreplace(input_manifest_path, '.json', '_subseg.json')
    min_subsegment_duration=0.05
    step_count = int(step_count)

    input_manifest_file = open(input_manifest_path, 'r').readlines()
    input_manifest_file = sorted(input_manifest_file)
    AUDIO_RTTM_MAP = audio_rttm_map(input_manifest_path)
    segments_manifest_file = write_rttm2manifest(AUDIO_RTTM_MAP, segment_manifest_path, deci)
    # print(segments_manifest_file)
    subsegments_manifest_file = subsegment_manifest_path
    segments_manifest_to_subsegments_manifest(
        segments_manifest_file,
        subsegments_manifest_file,
        window,
        shift,
        min_subsegment_duration,
    )
    subsegments_dict = get_subsegment_dict(subsegments_manifest_file, window, shift, deci)
    write_truncated_subsegments(input_manifest_dict, subsegments_dict, output_manifest_path, step_count, deci)
    os.remove(segment_manifest_path)
    os.remove(subsegment_manifest_path)

def create_manifest(wav_path, manifest_filepath, text_path=None, rttm_path=None, uem_path=None, ctm_path=None):
    if os.path.exists(manifest_filepath):
        os.remove(manifest_filepath)
    wav_pathlist = read_file(wav_path)
    wav_pathdict = get_dict_from_wavlist(wav_pathlist)
    len_wavs = len(wav_pathlist)
    uniqids = sorted(wav_pathdict.keys())

    text_pathdict = get_path_dict(text_path, uniqids, len_wavs)
    rttm_pathdict = get_path_dict(rttm_path, uniqids, len_wavs)
    uem_pathdict = get_path_dict(uem_path, uniqids, len_wavs)
    ctm_pathdict = get_path_dict(ctm_path, uniqids, len_wavs)

    lines = []
    for uid in uniqids:
        wav, text, rttm, uem, ctm = (
            wav_pathdict[uid],
            text_pathdict[uid],
            rttm_pathdict[uid],
            uem_pathdict[uid],
            ctm_pathdict[uid],
        )

        audio_line = wav.strip()
        if rttm is not None:
            rttm = rttm.strip()
            labels = rttm_to_labels(rttm)
            num_speakers = Counter([l.split()[-1] for l in labels]).keys().__len__()
        else:
            num_speakers = None

        if uem is not None:
            uem = uem.strip()

        if text is not None:
            text = open(text.strip()).readlines()[0].strip()
        else:
            text = "-"

        if ctm is not None:
            ctm = ctm.strip()

        meta = [
            {
                "audio_filepath": audio_line,
                "offset": 0,
                "duration": None,
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

def read_manifest(manifest):
    data = []
    with open(manifest, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data

def write_manifest(output_path, target_manifest):
    with open(output_path, "w") as outfile:
        for tgt in target_manifest:
            json.dump(tgt, outfile)
            outfile.write('\n')
