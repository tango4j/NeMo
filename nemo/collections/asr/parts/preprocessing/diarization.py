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

import json
import os
import random
import warnings
import shutil

import librosa
import numpy as np
from scipy.stats import halfnorm
from scipy.signal.windows import hamming, hann, cosine
from scipy.signal import convolve
import soundfile as sf
from omegaconf import OmegaConf
from gpuRIR import att2t_SabineEstimator, beta_SabineEstimation, simulateRIR, t2n

from nemo.collections.asr.parts.utils.speaker_utils import labels_to_rttmfile

#######################################################################
from collections import Counter
from collections import OrderedDict as od
from nemo.collections.asr.parts.utils.speaker_utils import (
    get_rttm_speaker_index,
    audio_rttm_map,
    get_subsegments,
    get_embs_and_timestamps,
    get_uniqname_from_filepath,
    parse_scale_configs,
    perform_clustering,
    score_labels,
    segments_manifest_to_subsegments_manifest,
    write_rttm2manifest,
    rttm_to_labels,
    labels_to_pyannote_object
)

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
#######################################################################


# from scripts/speaker_tasks/filelist_to_manifest.py - move function?
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

def write_ctm(output_path, target_ctm):
    target_ctm.sort(key=lambda y: y[0])
    with open(output_path, "w") as outfile:
        for pair in target_ctm:
            tgt = pair[1]
            outfile.write(tgt)

def write_text(output_path, target_ctm):
    target_ctm.sort(key=lambda y: y[0])
    with open(output_path, "w") as outfile:
        for pair in target_ctm:
            tgt = pair[1]
            word = tgt.split(' ')[4]
            outfile.write(word + ' ')
        outfile.write('\n')


class LibriSpeechGenerator(object):
    """
    Librispeech Diarization Session Generator.

    Args:
        manifest_path (str): Manifest file with paths to librispeech audio files
        sr (int): sampling rate of the audio files
        num_speakers (int): number of unique speakers per diarization session
        session_length (int): length of each diarization session (seconds)
        output_dir (str): output directory
        output_filename (str): output filename for the wav and rttm files
        sentence_length_params (list): k,p values for negative_binomial distribution
                              initial values are from page 209 of
                              https://www.researchgate.net/publication/318396023_How_will_text_size_influence_the_length_of_its_linguistic_constituents
        alignment_type (str): input alignment format
                              end - end alignments passed
                              start - start alignments passed
                              tuple - alignments expected in (start,end) pairs
        dominance_var (float): variance in speaker dominance
        min_dominance (float): minimum percentage of speaking time per speaker
        turn_prob (float): probability of switching speakers
        mean_overlap (float): mean proportion of overlap to speaking time
        mean_silence (float): mean proportion of silence to speaking time
        outputs (str): which files to output (r - rttm, j - json, c - ctm)
        enforce_num_speakers (bool): enforce that all requested speakers are present in the output wav file
        num_sessions (int): number of sessions
        random_seed (int): random seed
    """

    def __init__(self, cfg):
        self._params = cfg

        # internal params
        self._manifest = read_manifest(self._params.data_simulator.manifest_path)
        self._sentence = None
        self._text = ""
        self._words = []
        self._alignments = []
        #keep track of furthest sample per speaker to avoid overlapping same speaker
        self._furthest_sample = [0 for n in range(0,self._params.data_simulator.num_speakers)]
        #use to ensure overlap percentage is correct
        self._missing_overlap = 0

        #debugging stats
        self._speaking_time = 0
        self._overlap_amount = 0
        self._desired_overlap_amount = 0
        self._total_missing_overlap = 0

        #creating manifests
        self.base_manifest_filepath = None
        self.segment_manifest_filepath = None

        #for multidevice training
        self.device = None

    # randomly select speaker ids from loaded dict
    def _get_speaker_ids(self):
        speaker_ids = []
        s = 0
        while s < self._params.data_simulator.num_speakers:
            file = self._manifest[np.random.randint(0, len(self._manifest) - 1)]
            fn = file['audio_filepath'].split('/')[-1]
            speaker_id = fn.split('-')[0]
            # ensure speaker ids are not duplicated
            if speaker_id not in speaker_ids:
                speaker_ids.append(speaker_id)
                s += 1
        return speaker_ids

    # get a list of the samples for the specified speakers
    def _get_speaker_samples(self, speaker_ids):
        speaker_lists = {}
        for i in range(0, self._params.data_simulator.num_speakers):
            spid = speaker_ids[i]
            speaker_lists[str(spid)] = []

        for file in self._manifest:
            fn = file['audio_filepath'].split('/')[-1]
            new_speaker_id = fn.split('-')[0]
            for spid in speaker_ids:
                if spid == new_speaker_id:
                    speaker_lists[str(spid)].append(file)

        return speaker_lists

    # load a sample for the selected speaker id
    def _load_speaker_sample(self, speaker_lists, speaker_ids, speaker_turn):
        speaker_id = speaker_ids[speaker_turn]
        file_id = np.random.randint(0, len(speaker_lists[str(speaker_id)]) - 1)
        file = speaker_lists[str(speaker_id)][file_id]
        return file

    # get dominance for each speaker
    def _get_speaker_dominance(self):
        dominance_mean = 1.0/self._params.data_simulator.num_speakers
        dominance = np.random.normal(loc=dominance_mean, scale=self._params.data_simulator.dominance_var, size=self._params.data_simulator.num_speakers)
        for i in range(0,len(dominance)):
          if dominance[i] < 0:
            dominance[i] = 0
        #normalize while maintaining minimum dominance
        total = np.sum(dominance)
        if total == 0:
          for i in range(0,len(dominance)):
            dominance[i]+=min_dominance
        #scale accounting for min_dominance which has to be added after
        dominance = (dominance / total)*(1-self._params.data_simulator.min_dominance*self._params.data_simulator.num_speakers)
        for i in range(0,len(dominance)):
          dominance[i]+=self._params.data_simulator.min_dominance
          if i > 0:
            dominance[i] = dominance[i] + dominance[i-1]
        return dominance

    def _increase_speaker_dominance(self, increase_percent, base_speaker_dominance, factor):
        #extract original per-speaker probabilities
        dominance = np.copy(base_speaker_dominance)
        for i in range(len(dominance)-1,0,-1):
            dominance[i] = dominance[i] - dominance[i-1]
        #increase specified speakers by the desired factor and renormalize
        for i in increase_percent:
            dominance[i] = dominance[i] * factor
        dominance = dominance / np.sum(dominance)
        for i in range(1,len(dominance)):
            dominance[i] = dominance[i] + dominance[i-1]
        return dominance

    # get next speaker (accounting for turn probability, dominance distribution)
    def _get_next_speaker(self, prev_speaker, dominance):
        if np.random.uniform(0, 1) > self._params.data_simulator.turn_prob and prev_speaker != None:
            return prev_speaker
        else:
            speaker_turn = prev_speaker
            while speaker_turn == prev_speaker:
                rand = np.random.uniform(0, 1)
                speaker_turn = 0
                while rand > dominance[speaker_turn]:
                    speaker_turn += 1
            return speaker_turn

    # add audio file to current sentence
    def _add_file(self, file, audio_file, sentence_duration, max_sentence_duration, max_sentence_duration_sr):
        sentence_duration_sr = len(self._sentence)
        remaining_duration_sr = max_sentence_duration_sr - sentence_duration_sr
        remaining_duration = max_sentence_duration - sentence_duration
        prev_dur_sr = dur_sr = 0
        nw = i = 0

        #ensure the desired number of words are added and the length of the output session isn't exceeded
        while (nw < remaining_duration and dur_sr < remaining_duration_sr and i < len(file['words'])):
            dur_sr = int(file['alignments'][i] * self._params.data_simulator.sr)
            if dur_sr > remaining_duration_sr:
                break

            word = file['words'][i]
            self._words.append(word)

            if self._params.data_simulator.alignment_type == 'start':
                self._alignments.append(int(sentence_duration_sr / self._params.data_simulator.sr) + file['alignments'][i])
            elif self._params.data_simulator.alignment_type == 'end':
                self._alignments.append(int(sentence_duration_sr / self._params.data_simulator.sr) + file['alignments'][i])
            elif self._params.data_simulator.alignment_type == 'tuple':
                start = int(sentence_duration_sr / self._params.data_simulator.sr) + file['alignments'][i][0]
                end = int(sentence_duration_sr / self._params.data_simulator.sr) + file['alignments'][i][1]
                self._alignments.append((start,end))

            if word == "":
                i+=1
                continue
            elif self._text == "":
                self._text += word
            else:
                self._text += " " + word
            i+=1
            nw+=1
            prev_dur_sr = dur_sr

        # add audio clip up to the final alignment
        self._sentence = np.append(self._sentence, audio_file[:prev_dur_sr])

        #windowing
        if i < len(file['words']) and self._params.data_simulator.window_type != None:
            window_amount = int(self._params.data_simulator.window_size*self._params.data_simulator.sr)
            if prev_dur_sr+window_amount > remaining_duration_sr:
                window_amount = remaining_duration_sr - prev_dur_sr
            if self._params.data_simulator.window_type == 'hamming':
                window = hamming(window_amount*2)[window_amount:]
            elif self._params.data_simulator.window_type == 'hann':
                window = hann(window_amount*2)[window_amount:]
            elif self._params.data_simulator.window_type == 'cosine':
                window = cosine(window_amount*2)[window_amount:]
            if len(audio_file[prev_dur_sr:]) < window_amount:
                audio_file = np.pad(audio_file, (0, window_amount - len(audio_file[prev_dur_sr:])))
            self._sentence = np.append(self._sentence, np.multiply(audio_file[prev_dur_sr:prev_dur_sr+window_amount], window))

        #zero pad if close to end of the clip
        if dur_sr > remaining_duration_sr:
            self._sentence = np.pad(self._sentence, (0, max_sentence_duration_sr - len(self._sentence)))
        return sentence_duration+nw, len(self._sentence)

    # returns new overlapped (or shifted) start position
    def _add_silence_or_overlap(self, speaker_turn, prev_speaker, start, length, session_length_sr, prev_length_sr, enforce):
        overlap_prob = self._params.data_simulator.overlap_prob / (self._params.data_simulator.turn_prob)  # accounting for not overlapping the same speaker
        mean_overlap_percent = (self._params.data_simulator.mean_overlap / (1+self._params.data_simulator.mean_overlap)) /  self._params.data_simulator.overlap_prob #overlap_prob
        mean_silence_percent = self._params.data_simulator.mean_silence / (1-self._params.data_simulator.overlap_prob) #(1-overlap_prob)

        self._speaking_time += length
        orig_end = start + length

        # overlap
        if prev_speaker != speaker_turn and prev_speaker != None and np.random.uniform(0, 1) < overlap_prob:
            overlap_percent = halfnorm(loc=0, scale=mean_overlap_percent*np.sqrt(np.pi)/np.sqrt(2)).rvs()
            desired_overlap_amount = int(prev_length_sr * overlap_percent) #/ (1+self._params.data_simulator.mean_overlap)
            self._desired_overlap_amount += desired_overlap_amount
            new_start = start - desired_overlap_amount

            if self._missing_overlap > 0 and overlap_percent < 1:
                rand = int(prev_length_sr * np.random.uniform(0, 1 - overlap_percent / (1+self._params.data_simulator.mean_overlap)))
                if rand > self._missing_overlap:
                    new_start -= self._missing_overlap
                    desired_overlap_amount += self._missing_overlap
                    self._missing_overlap = 0
                else:
                    new_start -= rand
                    desired_overlap_amount += rand
                    self._missing_overlap -= rand

            #avoid overlap at start of clip
            if new_start < 0:
                desired_overlap_amount -= 0 - new_start
                self._missing_overlap += 0 - new_start
                new_start = 0

            #if same speaker ends up overlapping from any previous clip, pad with silence instead
            if (new_start < self._furthest_sample[speaker_turn]):
                desired_overlap_amount -= self._furthest_sample[speaker_turn] - new_start
                self._missing_overlap += self._furthest_sample[speaker_turn] - new_start
                new_start = self._furthest_sample[speaker_turn]

            if desired_overlap_amount < 0:
                desired_overlap_amount = 0

            prev_start = start - prev_length_sr
            prev_end = start
            new_end = new_start + length
            overlap_amount = 0
            if prev_start < new_start and new_end > prev_end: # 111111 2121 222222
                overlap_amount = prev_end - new_start
            elif prev_start < new_start and new_end < prev_end: # 1111 21212 11111
                overlap_amount = new_end - new_start
            elif prev_start > new_start and new_end < prev_end: # 2222 1212 1111
                overlap_amount = new_end - prev_start
            elif prev_start > new_start and new_end > prev_end: # 2222 12121 2222
                overlap_amount = prev_end - prev_start

            if overlap_amount < 0:
                overlap_amount = 0

            if overlap_amount < desired_overlap_amount:
                self._missing_overlap += desired_overlap_amount - overlap_amount

            self._speaking_time -= overlap_amount
            self._overlap_amount += overlap_amount

            return new_start
        else:
            # add silence
            silence_percent = halfnorm(loc=0, scale=mean_silence_percent*np.sqrt(np.pi)/np.sqrt(2)).rvs()
            silence_amount = int(length * silence_percent)

            if start + length + silence_amount > session_length_sr and not enforce:
                return session_length_sr - length
            else:
                return start + silence_amount

    # add new entry to dict (to write to output rttm file)
    def _create_new_rttm_entry(self, start, dur, speaker_id):
        start = float(round(start,3))
        dur = float(round(dur,3))
        return str(start) + ' ' + str(dur) + ' ' + str(speaker_id)

    # add new entry to dict (to write to output json file)
    def _create_new_json_entry(self, wav_filename, start, dur, speaker_id, text, rttm_filepath, ctm_filepath):
        start = float(round(start,3))
        dur = float(round(dur,3))
        dict = {"audio_filepath": wav_filename,
                "offset": start,
                "duration": dur,
                "label": speaker_id,
                "text": text,
                "num_speakers": self._params.data_simulator.num_speakers,
                "rttm_filepath": rttm_filepath,
                "ctm_filepath": ctm_filepath,
                "uem_filepath": None}
        return dict

    # add new entry to dict (to write to output ctm file)
    def _create_new_ctm_entry(self, session_name, speaker_id, start):
        arr = []
        start = float(round(start,3))
        for i in range(0, len(self._words)):
            word = self._words[i]
            if self._params.data_simulator.alignment_type == 'start':
                align1 = float(round(self._alignments[i] + start, 3))
                align2 = float(round(self._alignments[i+1] - self._alignments[i], 3))
            elif self._params.data_simulator.alignment_type == 'end':
                align1 = float(round(self._alignments[i-1] + start, 3))
                align2 = float(round(self._alignments[i] - self._alignments[i-1], 3))
            elif self._params.data_simulator.alignment_type == 'tuple':
                align1 = float(round(self._alignments[i][0] + start, 3))
                align2 = float(round(self._alignments[i][1] - self._alignments[i][0], 3))
            if word != "": #note that using the current alignments the first word is always empty, so there is no error from indexing the array with i-1
                text = str(session_name) + ' ' + str(speaker_id) + ' ' + str(align1) + ' ' + str(align2) + ' ' + str(word) + ' ' + '0' + '\n'
                arr.append((align1, text))
        return arr

    """
    Generate diarization session
    """
    def generate_session(self):
        print(f"Generating Diarization Sessions")
        np.random.seed(self._params.data_simulator.random_seed)

        output_dir = self._params.data_simulator.output_dir
        if self.device != None:
            output_dir += f"_{self.device}"

        #delete output directory if it exists or throw warning
        if os.path.isdir(output_dir) and os.listdir(output_dir):
            if self._params.data_simulator.overwrite_output:
                shutil.rmtree(output_dir)
                os.mkdir(output_dir)
            else:
                raise Exception("Output directory is nonempty and overwrite_output = false")
        elif not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        self._speaking_time = 0
        self._overlap_amount = 0
        self._desired_overlap_amount = 0
        self._total_missing_overlap = 0

        # only add root if paths are relative?
        if not os.path.isabs(output_dir):
            ROOT = os.getcwd()
            basepath = os.path.join(ROOT, output_dir)
        else:
            basepath = output_dir

        if 'l' in self._params.data_simulator.outputs:
            wavlist = open(os.path.join(basepath, "synthetic_wav.list"), "w")
            if 'r' in self._params.data_simulator.outputs:
                rttmlist = open(os.path.join(basepath, "synthetic_rttm.list"), "w")
            if 'j' in self._params.data_simulator.outputs:
                jsonlist = open(os.path.join(basepath, "synthetic_json.list"), "w")
            if 'c' in self._params.data_simulator.outputs:
                ctmlist = open(os.path.join(basepath, "synthetic_ctm.list"), "w")
            if 't' in self._params.data_simulator.outputs:
                textlist = open(os.path.join(basepath,"synthetic_txt.list"), "w")

        for i in range(0, self._params.data_simulator.num_sessions):
            print(f"Generating Session Number {i}")
            speaker_ids = self._get_speaker_ids()  # randomly select speaker ids
            speaker_dominance = self._get_speaker_dominance()  # randomly determine speaker dominance
            base_speaker_dominance = np.copy(speaker_dominance)
            speaker_lists = self._get_speaker_samples(speaker_ids)  # get list of samples per speaker

            filename = self._params.data_simulator.output_filename + f"_{i}"
            speaker_turn = 0  # assume alternating between speakers 1 & 2
            running_length_sr = 0  # starting point for each sentence
            prev_length_sr = 0  # for overlap
            start = end = 0
            prev_speaker = None
            rttm_list = []
            json_list = []
            ctm_list = []
            self._furthest_sample = [0 for n in range(0,self._params.data_simulator.num_speakers)]
            self._missing_overlap = 0

            #hold enforce until all speakers have spoken
            enforce_counter = 2
            enforce_time = np.random.uniform(0.25, 0.75)
            if self._params.data_simulator.enforce_num_speakers:
                enforce = True
            else:
                enforce = False

            wavpath = os.path.join(basepath, filename + '.wav')
            rttm_filepath = os.path.join(basepath, filename + '.rttm')
            json_filepath = os.path.join(basepath, filename + '.json')
            ctm_filepath = os.path.join(basepath, filename + '.ctm')
            text_filepath = os.path.join(basepath, filename + '.txt')

            if 'l' in self._params.data_simulator.outputs:
                wavlist.write(wavpath + '\n')
                if 'r' in self._params.data_simulator.outputs:
                    rttmlist.write(rttm_filepath + '\n')
                if 'j' in self._params.data_simulator.outputs:
                    jsonlist.write(json_filepath + '\n')
                if 'c' in self._params.data_simulator.outputs:
                    ctmlist.write(ctm_filepath + '\n')
                if 't' in self._params.data_simulator.outputs:
                    textlist.write(text_filepath + '\n')

            session_length_sr = int((self._params.data_simulator.session_length * self._params.data_simulator.sr))
            array = np.zeros(session_length_sr)

            while running_length_sr < session_length_sr or enforce:
                #enforce num_speakers
                if running_length_sr > enforce_time*session_length_sr and enforce:
                    increase_percent = []
                    for i in range(0,self._params.data_simulator.num_speakers):
                        if self._furthest_sample[i] == 0:
                            increase_percent.append(i)
                    #ramp up enforce counter until speaker is sampled, then reset once all speakers have spoken
                    if len(increase_percent) > 0:
                        speaker_dominance = self._increase_speaker_dominance(increase_percent, base_speaker_dominance, enforce_counter)
                        enforce_counter += 1
                    else:
                        enforce = False
                        speaker_dominance = base_speaker_dominance

                # select speaker
                speaker_turn = self._get_next_speaker(prev_speaker, speaker_dominance)

                # select speaker length
                sl = np.random.negative_binomial(
                    self._params.data_simulator.sentence_length_params[0], self._params.data_simulator.sentence_length_params[1]
                ) + 1
                max_sentence_duration_sr = session_length_sr - running_length_sr

                # only add if remaining length > 0.5 second
                if max_sentence_duration_sr < 0.5 * self._params.data_simulator.sr and not enforce:
                    break
                if enforce:
                    max_sentence_duration_sr = float('inf')

                # initialize sentence, text, words, alignments
                self._sentence = np.zeros(0)
                self._text = ""
                self._words = []
                self._alignments = []
                sentence_duration = sentence_duration_sr = 0

                # build sentence
                while sentence_duration < sl and sentence_duration_sr < max_sentence_duration_sr:
                    file = self._load_speaker_sample(speaker_lists, speaker_ids, speaker_turn)
                    audio_file, sr = librosa.load(file['audio_filepath'], sr=self._params.data_simulator.sr)
                    sentence_duration,sentence_duration_sr = self._add_file(file, audio_file, sentence_duration, sl, max_sentence_duration_sr)

                #per-speaker normalization
                if self._params.data_simulator.normalization == 'equal':
                    if  np.max(np.abs(self._sentence)) > 0:
                        self._sentence = self._sentence / (1.0 * np.max(np.abs(self._sentence)))
                #TODO fix randomized speaker variance (per-speaker volume selected at start of sentence)

                length = len(self._sentence)
                start = self._add_silence_or_overlap(
                    speaker_turn, prev_speaker, running_length_sr, length, session_length_sr, prev_length_sr, enforce
                )
                end = start + length
                if end > len(array):
                    array = np.pad(array, (0, end - len(array)))
                array[start:end] += self._sentence

                #build entries for output files
                if 'r' in self._params.data_simulator.outputs:
                    new_rttm_entry = self._create_new_rttm_entry(start / self._params.data_simulator.sr, end / self._params.data_simulator.sr, speaker_ids[speaker_turn])
                    rttm_list.append(new_rttm_entry)
                if 'j' in self._params.data_simulator.outputs:
                    new_json_entry = self._create_new_json_entry(wavpath, start / self._params.data_simulator.sr, length / self._params.data_simulator.sr, speaker_ids[speaker_turn], self._text, rttm_filepath, ctm_filepath)
                    json_list.append(new_json_entry)
                if 'c' in self._params.data_simulator.outputs:
                    new_ctm_entries = self._create_new_ctm_entry(filename, speaker_ids[speaker_turn], start / self._params.data_simulator.sr)
                    for entry in new_ctm_entries:
                        ctm_list.append(entry)

                running_length_sr = np.maximum(running_length_sr, end)
                self._furthest_sample[speaker_turn] = running_length_sr
                prev_speaker = speaker_turn
                prev_length_sr = length

            #throw error if number of speakers is less than requested
            num_missing = 0
            for k in range(0,len(self._furthest_sample)):
                if self._furthest_sample[k] == 0:
                    num_missing += 1
            if num_missing != 0:
                warnings.warn(f"{self._params.data_simulator.num_speakers-num_missing} speakers were included in the clip instead of the requested amount of {self._params.data_simulator.num_speakers}")

            array = array / (1.0 * np.max(np.abs(array)))  # normalize wav file
            sf.write(wavpath, array, self._params.data_simulator.sr)
            if 'r' in self._params.data_simulator.outputs:
                labels_to_rttmfile(rttm_list, filename, output_dir)
            if 'j' in self._params.data_simulator.outputs:
                write_manifest(json_filepath, json_list)
            if 'c' in self._params.data_simulator.outputs:
                write_ctm(ctm_filepath, ctm_list)
            if 't' in self._params.data_simulator.outputs:
                write_text(text_filepath, ctm_list)

            #CHECK OVERLAP
            timeline = np.zeros(len(array))
            for line in rttm_list:
                l = line.split(' ')
                sp = l[2]
                start = float(l[0])
                start = int(start * self._params.data_simulator.sr)
                end = float(l[1])
                end = int(end * self._params.data_simulator.sr)

                # end = start+dur
                timeline[start:end] += 1

            self._total_missing_overlap += self._missing_overlap

            speaking_time = np.sum(timeline > 0)
            overlap_time = np.sum(timeline > 1)
            double_overlap = np.sum(timeline > 2)
            overlap_percent = overlap_time / speaking_time

            # print('self._overlap_percent: ', 1.0*(self._overlap_amount + self._total_missing_overlap) / self._speaking_time)
            # print('self._desired_overlap_amount: ', 1.0*self._desired_overlap_amount / self._speaking_time)
            #END CHECK OVERLAP

        if 'l' in self._params.data_simulator.outputs:
            wavlist.close()
            if 'r' in self._params.data_simulator.outputs:
                rttmlist.close()
            if 'j' in self._params.data_simulator.outputs:
                jsonlist.close()
            if 'c' in self._params.data_simulator.outputs:
                ctmlist.close()
            if 't' in self._params.data_simulator.outputs:
                textlist.close()

    def create_base_manifest(self):
        basepath = self._params.data_simulator.output_dir
        if self.device != None:
            basepath += f"_{self.device}"
        wav_path = os.path.join(basepath, 'synthetic_wav.list')
        text_path = os.path.join(basepath, 'synthetic_txt.list')
        rttm_path = os.path.join(basepath, 'synthetic_rttm.list')
        ctm_path = os.path.join(basepath, 'synthetic_ctm.list')
        uem_path = None
        manifest_filepath = os.path.join(basepath, 'base_manifest.json')

        ###################################### create_base_manifest(wav_path, text_path=text_path, rttm_path=rttm_path, ctm_path=ctm_path, manifest_filepath=manifest_filepath)

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
        #####################################3
        self.base_manifest_filepath = manifest_filepath
        return self.base_manifest_filepath

    def create_segment_manifest(self):
        basepath = self._params.data_simulator.output_dir
        if self.device != None:
            basepath += f"_{self.device}"
        output_manifest_path = os.path.join(basepath, 'segment_manifest.json')
        input_manifest_path = self.base_manifest_filepath
        window = self._params.data_simulator.segment_manifest_window
        shift = self._params.data_simulator.segment_manifest_shift
        step_count = self._params.data_simulator.segment_manifest_step_count
        deci = self._params.data_simulator.segment_manifest_deci

        ##############################create_segment_manifest(base_manifest, segment_manifest, window, shift, step_count, deci)
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
        ##############################

        print('len subsegments_dict: ', len(subsegments_dict))

        self.segment_manifest_filepath = output_manifest_path
        return self.segment_manifest_filepath


class MultiMicLibriSpeechGenerator(LibriSpeechGenerator):
    """
    Multi Microphone Librispeech Diarization Session Generator.
    """

    def __init__(self, cfg):
        self._params = cfg

        # internal params
        self._manifest = read_manifest(self._params.data_simulator.manifest_path)
        self._sentence = None
        self._text = ""
        self._words = []
        self._alignments = []
        #keep track of furthest sample per speaker to avoid overlapping same speaker
        self._furthest_sample = [0 for n in range(0,self._params.data_simulator.num_speakers)]
        #use to ensure overlap percentage is correct
        self._missing_overlap = 0

    def _generate_rir(self):
        room_sz = np.array(self._params.data_simulator.rir_generation.room_sz)
        pos_src = np.array(self._params.data_simulator.rir_generation.pos_src)
        pos_rcv = np.array(self._params.data_simulator.rir_generation.pos_rcv)
        orV_rcv = self._params.data_simulator.rir_generation.orV_rcv
        if orV_rcv: #not needed for omni mics
            orV_rcv = np.array(orV_rcv)
        mic_pattern = self._params.data_simulator.rir_generation.mic_pattern
        abs_weights = self._params.data_simulator.rir_generation.abs_weights
        T60 = self._params.data_simulator.rir_generation.T60
        att_diff = self._params.data_simulator.rir_generation.att_diff
        att_max = self._params.data_simulator.rir_generation.att_max
        fs = self._params.data_simulator.rir_generation.fs

        beta = beta_SabineEstimation(room_sz, T60, abs_weights=abs_weights)  # Reflection coefficients
        Tdiff = att2t_SabineEstimator(att_diff, T60)  # Time to start the diffuse reverberation model [s]
        Tmax = att2t_SabineEstimator(att_max, T60)  # Time to stop the simulation [s]
        nb_img = t2n(Tdiff, room_sz)  # Number of image sources in each dimension
        RIR = simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, fs, Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern)
        return RIR

    def _convolve_rir(self, speaker_turn, RIR):
        output_sound = []
        for channel in range(0,self._params.data_simulator.num_channels):
            out_channel = convolve(self._sentence, RIR[speaker_turn, channel, : len(self._sentence)]).tolist()
            output_sound.append(out_channel)
        output_sound = np.array(output_sound).T
        return output_sound

    """
    Generate diarization session
    """
    def generate_session(self):
        print("Generating Diarization Sessions")
        np.random.seed(self._params.data_simulator.random_seed)

        #delete output directory if it exists or throw warning
        if os.path.isdir(self._params.data_simulator.output_dir) and os.listdir(self._params.data_simulator.output_dir):
            if self._params.data_simulator.overwrite_output:
                shutil.rmtree(self._params.data_simulator.output_dir)
                os.mkdir(self._params.data_simulator.output_dir)
            else:
                raise Exception("Output directory is nonempty and overwrite_output = false")
        elif not os.path.isdir(self._params.data_simulator.output_dir):
            os.mkdir(self._params.data_simulator.output_dir)

        if 'l' in self._params.data_simulator.outputs:
            wavlist = open(os.path.join(basepath, "synthetic_wav.list"), "w")
            if 'r' in self._params.data_simulator.outputs:
                rttmlist = open(os.path.join(basepath, "synthetic_rttm.list"), "w")
            if 'j' in self._params.data_simulator.outputs:
                jsonlist = open(os.path.join(basepath, "synthetic_json.list"), "w")
            if 'c' in self._params.data_simulator.outputs:
                ctmlist = open(os.path.join(basepath, "synthetic_ctm.list"), "w")
            if 't' in self._params.data_simulator.outputs:
                textlist = open(os.path.join(basepath,"synthetic_txt.list"), "w")

        for i in range(0, self._params.data_simulator.num_sessions):
            # print(f"Generating Session Number {i}")
            speaker_ids = self._get_speaker_ids()  # randomly select speaker ids
            speaker_dominance = self._get_speaker_dominance()  # randomly determine speaker dominance
            base_speaker_dominance = np.copy(speaker_dominance)
            speaker_lists = self._get_speaker_samples(speaker_ids)  # get list of samples per speaker

            filename = self._params.data_simulator.output_filename + f"_{i}"
            speaker_turn = 0  # assume alternating between speakers 1 & 2
            running_length_sr = 0  # starting point for each sentence
            prev_length_sr = 0  # for overlap
            start = end = 0
            prev_speaker = None
            rttm_list = []
            json_list = []
            ctm_list = []
            self._furthest_sample = [0 for n in range(0,self._params.data_simulator.num_speakers)]
            self._missing_overlap = 0

            #Room Imulse Response
            # RIR = self._select_rir()
            RIR = self._generate_rir()

            #hold enforce until all speakers have spoken
            enforce_counter = 2
            enforce_time = np.random.uniform(0.25, 0.75)
            if self._params.data_simulator.enforce_num_speakers:
                enforce = True
            else:
                enforce = False

            # only add root if paths are relative?
            if not os.path.isabs(self._params.data_simulator.output_dir):
                ROOT = os.getcwd()
                basepath = os.path.join(ROOT, self._params.data_simulator.output_dir)
            else:
                basepath = self._params.data_simulator.output_dir
            wavpath = os.path.join(basepath, filename + '.wav')
            rttm_filepath = os.path.join(basepath, filename + '.rttm')
            json_filepath = os.path.join(basepath, filename + '.json')
            ctm_filepath = os.path.join(basepath, filename + '.ctm')
            text_filepath = os.path.join(basepath, filename + '.txt')

            if 'l' in self._params.data_simulator.outputs:
                wavlist.write(wavpath + '\n')
                if 'r' in self._params.data_simulator.outputs:
                    rttmlist.write(rttm_filepath + '\n')
                if 'j' in self._params.data_simulator.outputs:
                    jsonlist.write(json_filepath + '\n')
                if 'c' in self._params.data_simulator.outputs:
                    ctmlist.write(ctm_filepath + '\n')
                if 't' in self._params.data_simulator.outputs:
                    textlist.write(text_filepath + '\n')

            session_length_sr = int((self._params.data_simulator.session_length * self._params.data_simulator.sr))
            array = np.zeros((session_length_sr, self._params.data_simulator.num_channels))

            while running_length_sr < session_length_sr or enforce:
                #enforce num_speakers
                if running_length_sr > enforce_time*session_length_sr and enforce:
                    increase_percent = []
                    for i in range(0,self._params.data_simulator.num_speakers):
                        if self._furthest_sample[i] == 0:
                            increase_percent.append(i)
                    #ramp up enforce counter until speaker is sampled, then reset once all speakers have spoken
                    if len(increase_percent) > 0:
                        speaker_dominance = self._increase_speaker_dominance(increase_percent, base_speaker_dominance, enforce_counter)
                        enforce_counter += 1
                    else:
                        enforce = False
                        speaker_dominance = base_speaker_dominance

                # select speaker
                speaker_turn = self._get_next_speaker(prev_speaker, speaker_dominance)

                # select speaker length
                sl = np.random.negative_binomial(
                    self._params.data_simulator.sentence_length_params[0], self._params.data_simulator.sentence_length_params[1]
                ) + 1

                #sentence will be RIR_len-1 longer than selected
                RIR_pad = (RIR.shape[2] - 1)
                max_sentence_duration_sr = session_length_sr - running_length_sr - RIR_pad

                # only add if remaining length > 0.5 second
                if max_sentence_duration_sr < 0.5 * self._params.data_simulator.sr and not enforce:
                    break
                if enforce:
                    max_sentence_duration_sr = float('inf')

                # initialize sentence, text, words, alignments
                self._sentence = np.zeros(0)
                self._text = ""
                self._words = []
                self._alignments = []
                sentence_duration = sentence_duration_sr = 0

                # build sentence
                while sentence_duration < sl and sentence_duration_sr < max_sentence_duration_sr:
                    file = self._load_speaker_sample(speaker_lists, speaker_ids, speaker_turn)
                    audio_file, sr = librosa.load(file['audio_filepath'], sr=self._params.data_simulator.sr)
                    sentence_duration,sentence_duration_sr = self._add_file(file, audio_file, sentence_duration, sl, max_sentence_duration_sr)

                #augment sentence
                augmented_sentence = self._convolve_rir(speaker_turn, RIR)

                #per-speaker normalization
                if self._params.data_simulator.normalization == 'equal':
                    if  np.max(np.abs(augmented_sentence)) > 0:
                        augmented_sentence = augmented_sentence / (1.0 * np.max(np.abs(augmented_sentence)))

                length = augmented_sentence.shape[0]
                start = self._add_silence_or_overlap(
                    speaker_turn, prev_speaker, running_length_sr, length, session_length_sr, prev_length_sr, enforce
                )
                end = start + length

                if end > len(array):
                    array = np.pad(array, (0, end - len(array)))

                array[start:end, :] += augmented_sentence

                #build entries for output files
                if 'r' in self._params.data_simulator.outputs:
                    new_rttm_entry = self._create_new_rttm_entry(start / self._params.data_simulator.sr, end / self._params.data_simulator.sr, speaker_ids[speaker_turn])
                    rttm_list.append(new_rttm_entry)
                if 'j' in self._params.data_simulator.outputs:
                    new_json_entry = self._create_new_json_entry(wavpath, start / self._params.data_simulator.sr, length / self._params.data_simulator.sr, speaker_ids[speaker_turn], self._text, rttm_filepath, ctm_filepath)
                    json_list.append(new_json_entry)
                if 'c' in self._params.data_simulator.outputs:
                    new_ctm_entries = self._create_new_ctm_entry(filename, speaker_ids[speaker_turn], start / self._params.data_simulator.sr)
                    for entry in new_ctm_entries:
                        ctm_list.append(entry)

                running_length_sr = np.maximum(running_length_sr, end)
                self._furthest_sample[speaker_turn] = running_length_sr
                prev_speaker = speaker_turn
                prev_length_sr = length

            #throw error if number of speakers is less than requested
            num_missing = 0
            for k in range(0,len(self._furthest_sample)):
                if self._furthest_sample[k] == 0:
                    num_missing += 1
            if num_missing != 0:
                warnings.warn(f"{self._params.data_simulator.num_speakers-num_missing} speakers were included in the clip instead of the requested amount of {self._params.data_simulator.num_speakers}")

            array = array / (1.0 * np.max(np.abs(array)))  # normalize wav file
            sf.write(wavpath, array, self._params.data_simulator.sr)
            if 'r' in self._params.data_simulator.outputs:
                labels_to_rttmfile(rttm_list, filename, self._params.data_simulator.output_dir)
            if 'j' in self._params.data_simulator.outputs:
                write_manifest(json_filepath, json_list)
            if 'c' in self._params.data_simulator.outputs:
                write_ctm(ctm_filepath, ctm_list)
            if 't' in self._params.data_simulator.outputs:
                write_text(text_filepath, ctm_list)

        if 'l' in self._params.data_simulator.outputs:
            wavlist.close()
            if 'r' in self._params.data_simulator.outputs:
                rttmlist.close()
            if 'j' in self._params.data_simulator.outputs:
                jsonlist.close()
            if 'c' in self._params.data_simulator.outputs:
                ctmlist.close()
            if 't' in self._params.data_simulator.outputs:
                textlist.close()
