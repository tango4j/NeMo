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

from collections import Counter
from nemo.collections.asr.parts.utils.speaker_utils import labels_to_rttmfile
from nemo.collections.asr.parts.utils.manifest_utils import (
    create_manifest,
    create_segment_manifest,
    read_manifest,
    write_manifest
)

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
        cfg: OmegaConf configuration loaded from yaml file.
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
        self._furthest_sample = [0 for n in range(0,self._params.data_simulator.session_config.num_speakers)]
        #use to ensure overlap percentage is correct
        self._missing_overlap = 0
        #creating manifests
        self.base_manifest_filepath = None
        self.segment_manifest_filepath = None

    # randomly select speaker ids from loaded dict
    def _get_speaker_ids(self):
        speaker_ids = []
        s = 0
        while s < self._params.data_simulator.session_config.num_speakers:
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
        for i in range(0, self._params.data_simulator.session_config.num_speakers):
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
        dominance_mean = 1.0/self._params.data_simulator.session_config.num_speakers
        dominance = np.random.normal(loc=dominance_mean, scale=self._params.data_simulator.session_params.dominance_var, size=self._params.data_simulator.session_config.num_speakers)
        for i in range(0,len(dominance)):
          if dominance[i] < 0:
            dominance[i] = 0
        #normalize while maintaining minimum dominance
        total = np.sum(dominance)
        if total == 0:
          for i in range(0,len(dominance)):
            dominance[i]+=min_dominance
        #scale accounting for min_dominance which has to be added after
        dominance = (dominance / total)*(1-self._params.data_simulator.session_params.min_dominance*self._params.data_simulator.session_config.num_speakers)
        for i in range(0,len(dominance)):
          dominance[i]+=self._params.data_simulator.session_params.min_dominance
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
        if np.random.uniform(0, 1) > self._params.data_simulator.session_params.turn_prob and prev_speaker != None:
            return prev_speaker
        else:
            speaker_turn = prev_speaker
            #ensure another speaker goes next
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
            self._alignments.append(int(sentence_duration_sr / self._params.data_simulator.sr) + file['alignments'][i])

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
        if i < len(file['words']) and self._params.data_simulator.session_params.window_type != None:
            window_amount = int(self._params.data_simulator.session_params.window_size*self._params.data_simulator.sr)
            if prev_dur_sr+window_amount > remaining_duration_sr:
                window_amount = remaining_duration_sr - prev_dur_sr
            if self._params.data_simulator.session_params.window_type == 'hamming':
                window = hamming(window_amount*2)[window_amount:]
            elif self._params.data_simulator.session_params.window_type == 'hann':
                window = hann(window_amount*2)[window_amount:]
            elif self._params.data_simulator.session_params.window_type == 'cosine':
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
        #NOTE: turn_prob & overlap_prob should be restricted so that overlap_prob/turn_prob <= 1
        overlap_prob = self._params.data_simulator.session_params.overlap_prob / (self._params.data_simulator.session_params.turn_prob)  #accounting for not overlapping the same speaker
        mean_overlap_percent = (self._params.data_simulator.session_params.mean_overlap / (1+self._params.data_simulator.session_params.mean_overlap)) /  self._params.data_simulator.session_params.overlap_prob
        mean_silence_percent = self._params.data_simulator.session_params.mean_silence / (1-self._params.data_simulator.session_params.overlap_prob)
        orig_end = start + length

        # overlap
        if prev_speaker != speaker_turn and prev_speaker != None and np.random.uniform(0, 1) < overlap_prob:
            overlap_percent = halfnorm(loc=0, scale=mean_overlap_percent*np.sqrt(np.pi)/np.sqrt(2)).rvs()
            desired_overlap_amount = int(prev_length_sr * overlap_percent)
            new_start = start - desired_overlap_amount

            if self._missing_overlap > 0 and overlap_percent < 1:
                rand = int(prev_length_sr * np.random.uniform(0, 1 - overlap_percent / (1+self._params.data_simulator.session_params.mean_overlap)))
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

            prev_start = start - prev_length_sr
            prev_end = start
            new_end = new_start + length
            overlap_amount = 0
            if prev_start < new_start and new_end > prev_end:
                overlap_amount = prev_end - new_start
            elif prev_start < new_start and new_end < prev_end:
                overlap_amount = new_end - new_start
            elif prev_start > new_start and new_end < prev_end:
                overlap_amount = new_end - prev_start
            elif prev_start > new_start and new_end > prev_end:
                overlap_amount = prev_end - prev_start

            if overlap_amount < 0:
                overlap_amount = 0

            if overlap_amount < desired_overlap_amount:
                self._missing_overlap += desired_overlap_amount - overlap_amount

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
                "num_speakers": self._params.data_simulator.session_config.num_speakers,
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
            align1 = float(round(self._alignments[i-1] + start, 3))
            align2 = float(round(self._alignments[i] - self._alignments[i-1], 3))
            if word != "": #note that using the current alignments the first word is always empty, so there is no error from indexing the array with i-1
                text = str(session_name) + ' ' + str(speaker_id) + ' ' + str(align1) + ' ' + str(align2) + ' ' + str(word) + ' ' + '0' + '\n'
                arr.append((align1, text))
        return arr

    """
    Generate diarization session
    """
    def _generate_session(self, idx, filename):
        print(f"Generating Session Number {i}")
        wavpath = os.path.join(basepath, filename + '.wav')
        rttm_filepath = os.path.join(basepath, filename + '.rttm')
        json_filepath = os.path.join(basepath, filename + '.json')
        ctm_filepath = os.path.join(basepath, filename + '.ctm')
        text_filepath = os.path.join(basepath, filename + '.txt')

        speaker_ids = self._get_speaker_ids()  # randomly select speaker ids
        speaker_dominance = self._get_speaker_dominance()  # randomly determine speaker dominance
        base_speaker_dominance = np.copy(speaker_dominance)
        speaker_lists = self._get_speaker_samples(speaker_ids)  # get list of samples per speaker

        speaker_turn = 0  # assume alternating between speakers 1 & 2
        running_length_sr = 0  # starting point for each sentence
        prev_length_sr = 0  # for overlap
        start = end = 0
        prev_speaker = None
        rttm_list = []
        json_list = []
        ctm_list = []

        #hold enforce until all speakers have spoken
        enforce_counter = 2
        enforce_time = np.random.uniform(0.25, 0.75)
        if self._params.data_simulator.enforce_num_speakers:
            enforce = True
        else:
            enforce = False

        session_length_sr = int((self._params.data_simulator.session_config.session_length * self._params.data_simulator.sr))
        array = np.zeros(session_length_sr)

        while running_length_sr < session_length_sr or enforce:
            #enforce num_speakers
            if running_length_sr > enforce_time*session_length_sr and enforce:
                increase_percent = []
                for i in range(0,self._params.data_simulator.session_config.num_speakers):
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
                self._params.data_simulator.session_params.sentence_length_params[0], self._params.data_simulator.session_params.sentence_length_params[1]
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
            if self._params.data_simulator.session_params.normalization == 'equal':
                if  np.max(np.abs(self._sentence)) > 0:
                    self._sentence = self._sentence / (1.0 * np.max(np.abs(self._sentence)))
            #TODO add variable speaker volume (per-speaker volume selected at start of sentence)

            length = len(self._sentence)
            start = self._add_silence_or_overlap(
                speaker_turn, prev_speaker, running_length_sr, length, session_length_sr, prev_length_sr, enforce
            )
            end = start + length
            if end > len(array):
                array = np.pad(array, (0, end - len(array)))
            array[start:end] += self._sentence

            #build entries for output files
            if 'r' in self._params.data_simulator.outputs.output_files:
                new_rttm_entry = self._create_new_rttm_entry(start / self._params.data_simulator.sr, end / self._params.data_simulator.sr, speaker_ids[speaker_turn])
                rttm_list.append(new_rttm_entry)
            if 'j' in self._params.data_simulator.outputs.output_files:
                new_json_entry = self._create_new_json_entry(wavpath, start / self._params.data_simulator.sr, length / self._params.data_simulator.sr, speaker_ids[speaker_turn], self._text, rttm_filepath, ctm_filepath)
                json_list.append(new_json_entry)
            if 'c' in self._params.data_simulator.outputs.output_files:
                new_ctm_entries = self._create_new_ctm_entry(filename, speaker_ids[speaker_turn], start / self._params.data_simulator.sr)
                for entry in new_ctm_entries:
                    ctm_list.append(entry)

            running_length_sr = np.maximum(running_length_sr, end)
            self._furthest_sample[speaker_turn] = running_length_sr
            prev_speaker = speaker_turn
            prev_length_sr = length

        array = array / (1.0 * np.max(np.abs(array)))  # normalize wav file
        sf.write(wavpath, array, self._params.data_simulator.sr)
        if 'r' in self._params.data_simulator.outputs.output_files:
            labels_to_rttmfile(rttm_list, filename, output_dir)
        if 'j' in self._params.data_simulator.outputs.output_files:
            write_manifest(json_filepath, json_list)
        if 'c' in self._params.data_simulator.outputs.output_files:
            write_ctm(ctm_filepath, ctm_list)
        if 't' in self._params.data_simulator.outputs.output_files:
            write_text(text_filepath, ctm_list)

    """
    Generate diarization sessions
    """
    def generate_sessions(self):
        print(f"Generating Diarization Sessions")
        np.random.seed(self._params.data_simulator.random_seed)
        output_dir = self._params.data_simulator.outputs.output_dir

        #delete output directory if it exists or throw warning
        if os.path.isdir(output_dir) and os.listdir(output_dir):
            if self._params.data_simulator.outputs.overwrite_output:
                shutil.rmtree(output_dir)
                os.mkdir(output_dir)
            else:
                raise Exception("Output directory is nonempty and overwrite_output = false")
        elif not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        # only add root if paths are relative
        if not os.path.isabs(output_dir):
            ROOT = os.getcwd()
            basepath = os.path.join(ROOT, output_dir)
        else:
            basepath = output_dir

        #create output files
        if 'l' in self._params.data_simulator.outputs.output_files:
            wavlist = open(os.path.join(basepath, "synthetic_wav.list"), "w")
            if 'r' in self._params.data_simulator.outputs.output_files:
                rttmlist = open(os.path.join(basepath, "synthetic_rttm.list"), "w")
            if 'j' in self._params.data_simulator.outputs.output_files:
                jsonlist = open(os.path.join(basepath, "synthetic_json.list"), "w")
            if 'c' in self._params.data_simulator.outputs.output_files:
                ctmlist = open(os.path.join(basepath, "synthetic_ctm.list"), "w")
            if 't' in self._params.data_simulator.outputs.output_files:
                textlist = open(os.path.join(basepath,"synthetic_txt.list"), "w")

        for i in range(0, self._params.data_simulator.session_config.num_sessions):
            self._furthest_sample = [0 for n in range(0,self._params.data_simulator.session_config.num_speakers)]
            self._missing_overlap = 0

            filename = self._params.data_simulator.outputs.output_filename + f"_{i}"
            array = self.generate_session(i, filename)

            wavpath = os.path.join(basepath, filename + '.wav')
            rttm_filepath = os.path.join(basepath, filename + '.rttm')
            json_filepath = os.path.join(basepath, filename + '.json')
            ctm_filepath = os.path.join(basepath, filename + '.ctm')
            text_filepath = os.path.join(basepath, filename + '.txt')

            if 'l' in self._params.data_simulator.outputs.output_files:
                wavlist.write(wavpath + '\n')
                if 'r' in self._params.data_simulator.outputs.output_files:
                    rttmlist.write(rttm_filepath + '\n')
                if 'j' in self._params.data_simulator.outputs.output_files:
                    jsonlist.write(json_filepath + '\n')
                if 'c' in self._params.data_simulator.outputs.output_files:
                    ctmlist.write(ctm_filepath + '\n')
                if 't' in self._params.data_simulator.outputs.output_files:
                    textlist.write(text_filepath + '\n')

            #throw error if number of speakers is less than requested
            num_missing = 0
            for k in range(0,len(self._furthest_sample)):
                if self._furthest_sample[k] == 0:
                    num_missing += 1
            if num_missing != 0:
                warnings.warn(f"{self._params.data_simulator.session_config.num_speakers-num_missing} speakers were included in the clip instead of the requested amount of {self._params.data_simulator.session_config.num_speakers}")

        if 'l' in self._params.data_simulator.outputs.output_files:
            wavlist.close()
            if 'r' in self._params.data_simulator.outputs.output_files:
                rttmlist.close()
            if 'j' in self._params.data_simulator.outputs.output_files:
                jsonlist.close()
            if 'c' in self._params.data_simulator.outputs.output_files:
                ctmlist.close()
            if 't' in self._params.data_simulator.outputs.output_files:
                textlist.close()

    def create_base_manifest_ds(self):
        basepath = self._params.data_simulator.outputs.output_dir
        wav_path = os.path.join(basepath, 'synthetic_wav.list')
        text_path = os.path.join(basepath, 'synthetic_txt.list')
        rttm_path = os.path.join(basepath, 'synthetic_rttm.list')
        ctm_path = os.path.join(basepath, 'synthetic_ctm.list')
        manifest_filepath = os.path.join(basepath, 'base_manifest.json')

        create_manifest(wav_path, manifest_filepath, text_path=text_path, rttm_path=rttm_path, ctm_path=ctm_path)

        self.base_manifest_filepath = manifest_filepath
        return self.base_manifest_filepath

    def create_segment_manifest_ds(self):
        basepath = self._params.data_simulator.outputs.output_dir
        output_manifest_path = os.path.join(basepath, 'segment_manifest.json')
        input_manifest_path = self.base_manifest_filepath
        window = self._params.data_simulator.segment_manifest.window
        shift = self._params.data_simulator.segment_manifest.shift
        step_count = self._params.data_simulator.segment_manifest.step_count
        deci = self._params.data_simulator.segment_manifest.deci

        create_segment_manifest(input_manifest_path, output_manifest_path, window, shift, step_count, deci)

        self.segment_manifest_filepath = output_manifest_path
        return self.segment_manifest_filepath


class MultiMicLibriSpeechGenerator(LibriSpeechGenerator):
    """
    Multi Microphone Librispeech Diarization Session Generator.
    """

    """
    Create simulated RIR
    """
    def _generate_rir(self):
        room_sz = np.array(self._params.data_simulator.rir_generation.room_config.room_sz)
        pos_src = np.array(self._params.data_simulator.rir_generation.room_config.pos_src)
        pos_rcv = np.array(self._params.data_simulator.rir_generation.mic_config.pos_rcv)
        orV_rcv = self._params.data_simulator.rir_generation.mic_config.orV_rcv
        if orV_rcv: #not needed for omni mics
            orV_rcv = np.array(orV_rcv)
        mic_pattern = self._params.data_simulator.rir_generation.mic_config.mic_pattern
        abs_weights = self._params.data_simulator.rir_generation.absorbtion_params.abs_weights
        T60 = self._params.data_simulator.rir_generation.absorbtion_params.T60
        att_diff = self._params.data_simulator.rir_generation.absorbtion_params.att_diff
        att_max = self._params.data_simulator.rir_generation.absorbtion_params.att_max
        sr = self._params.data_simulator.sr

        beta = beta_SabineEstimation(room_sz, T60, abs_weights=abs_weights)  # Reflection coefficients
        Tdiff = att2t_SabineEstimator(att_diff, T60)  # Time to start the diffuse reverberation model [s]
        Tmax = att2t_SabineEstimator(att_max, T60)  # Time to stop the simulation [s]
        nb_img = t2n(Tdiff, room_sz)  # Number of image sources in each dimension
        RIR = simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, sr, Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern)
        return RIR

    """
    Augment sample using synthetic RIR
    """
    def _convolve_rir(self, speaker_turn, RIR):
        output_sound = []
        for channel in range(0,self._params.data_simulator.rir_generation.mic_config.num_channels):
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

        output_dir = self._params.data_simulator.outputs.output_dir

        #delete output directory if it exists or throw warning
        if os.path.isdir(output_dir) and os.listdir(output_dir):
            if self._params.data_simulator.outputs.overwrite_output:
                shutil.rmtree(output_dir)
                os.mkdir(output_dir)
            else:
                raise Exception("Output directory is nonempty and overwrite_output = false")
        elif not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        # only add root if paths are relative?
        if not os.path.isabs(output_dir):
            ROOT = os.getcwd()
            basepath = os.path.join(ROOT, output_dir)
        else:
            basepath = output_dir

        if 'l' in self._params.data_simulator.outputs.output_files:
            wavlist = open(os.path.join(basepath, "synthetic_wav.list"), "w")
            if 'r' in self._params.data_simulator.outputs.output_files:
                rttmlist = open(os.path.join(basepath, "synthetic_rttm.list"), "w")
            if 'j' in self._params.data_simulator.outputs.output_files:
                jsonlist = open(os.path.join(basepath, "synthetic_json.list"), "w")
            if 'c' in self._params.data_simulator.outputs.output_files:
                ctmlist = open(os.path.join(basepath, "synthetic_ctm.list"), "w")
            if 't' in self._params.data_simulator.outputs.output_files:
                textlist = open(os.path.join(basepath,"synthetic_txt.list"), "w")

        for i in range(0, self._params.data_simulator.session_config.num_sessions):
            print(f"Generating Session Number {i}")
            speaker_ids = self._get_speaker_ids()  # randomly select speaker ids
            speaker_dominance = self._get_speaker_dominance()  # randomly determine speaker dominance
            base_speaker_dominance = np.copy(speaker_dominance)
            speaker_lists = self._get_speaker_samples(speaker_ids)  # get list of samples per speaker

            filename = self._params.data_simulator.outputs.output_filename + f"_{i}"
            speaker_turn = 0  # assume alternating between speakers 1 & 2
            running_length_sr = 0  # starting point for each sentence
            prev_length_sr = 0  # for overlap
            start = end = 0
            prev_speaker = None
            rttm_list = []
            json_list = []
            ctm_list = []
            self._furthest_sample = [0 for n in range(0,self._params.data_simulator.session_config.num_speakers)]
            self._missing_overlap = 0

            #Room Impulse Response Generation (performed once per batch of sessions)
            RIR = self._generate_rir()

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

            if 'l' in self._params.data_simulator.outputs.output_files:
                wavlist.write(wavpath + '\n')
                if 'r' in self._params.data_simulator.outputs.output_files:
                    rttmlist.write(rttm_filepath + '\n')
                if 'j' in self._params.data_simulator.outputs.output_files:
                    jsonlist.write(json_filepath + '\n')
                if 'c' in self._params.data_simulator.outputs.output_files:
                    ctmlist.write(ctm_filepath + '\n')
                if 't' in self._params.data_simulator.outputs.output_files:
                    textlist.write(text_filepath + '\n')

            session_length_sr = int((self._params.data_simulator.session_config.session_length * self._params.data_simulator.sr))
            array = np.zeros((session_length_sr, self._params.data_simulator.rir_generation.mic_config.num_channels))

            while running_length_sr < session_length_sr or enforce:
                #enforce num_speakers
                if running_length_sr > enforce_time*session_length_sr and enforce:
                    increase_percent = []
                    for i in range(0,self._params.data_simulator.session_config.num_speakers):
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
                    self._params.data_simulator.session_params.sentence_length_params[0], self._params.data_simulator.session_params.sentence_length_params[1]
                ) + 1

                #sentence will be RIR_len - 1 longer than the audio was pre-augmentation
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
                if self._params.data_simulator.session_params.normalization == 'equal':
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
                if 'r' in self._params.data_simulator.outputs.output_files:
                    new_rttm_entry = self._create_new_rttm_entry(start / self._params.data_simulator.sr, end / self._params.data_simulator.sr, speaker_ids[speaker_turn])
                    rttm_list.append(new_rttm_entry)
                if 'j' in self._params.data_simulator.outputs.output_files:
                    new_json_entry = self._create_new_json_entry(wavpath, start / self._params.data_simulator.sr, length / self._params.data_simulator.sr, speaker_ids[speaker_turn], self._text, rttm_filepath, ctm_filepath)
                    json_list.append(new_json_entry)
                if 'c' in self._params.data_simulator.outputs.output_files:
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
                warnings.warn(f"{self._params.data_simulator.session_config.num_speakers-num_missing} speakers were included in the clip instead of the requested amount of {self._params.data_simulator.session_config.num_speakers}")

            array = array / (1.0 * np.max(np.abs(array)))  # normalize wav file
            sf.write(wavpath, array, self._params.data_simulator.sr)
            if 'r' in self._params.data_simulator.outputs.output_files:
                labels_to_rttmfile(rttm_list, filename, self._params.data_simulator.outputs.output_dir)
            if 'j' in self._params.data_simulator.outputs.output_files:
                write_manifest(json_filepath, json_list)
            if 'c' in self._params.data_simulator.outputs.output_files:
                write_ctm(ctm_filepath, ctm_list)
            if 't' in self._params.data_simulator.outputs.output_files:
                write_text(text_filepath, ctm_list)

        if 'l' in self._params.data_simulator.outputs.output_files:
            wavlist.close()
            if 'r' in self._params.data_simulator.outputs.output_files:
                rttmlist.close()
            if 'j' in self._params.data_simulator.outputs.output_files:
                jsonlist.close()
            if 'c' in self._params.data_simulator.outputs.output_files:
                ctmlist.close()
            if 't' in self._params.data_simulator.outputs.output_files:
                textlist.close()
