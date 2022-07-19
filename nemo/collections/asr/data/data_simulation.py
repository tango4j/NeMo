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
import torch
import time

import librosa
import numpy as np
from scipy.stats import halfnorm
from scipy.signal.windows import hamming, hann, cosine
from scipy.signal import convolve
import soundfile as sf
from omegaconf import OmegaConf
from gpuRIR import att2t_SabineEstimator, beta_SabineEstimation, simulateRIR, t2n
import pyroomacoustics as pra
from pyroomacoustics.directivities import (
    DirectivityPattern,
    DirectionVector,
    CardioidFamily,
)

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
        #error check arguments
        self._check_args()
        # internal params
        try:
            self._manifest = read_manifest(self._params.data_simulator.manifest_path)
        except:
            raise Exception("Manifest file could not be opened")
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

    def _check_args(self):
        """
        Checks arguments to ensure they are within valid ranges
        """
        if self._params.data_simulator.session_config.num_speakers < 2:
            raise Exception("Atleast two speakers are required for multispeaker audio sessions (num_speakers < 2)")
        if self._params.data_simulator.session_params.turn_prob < 0 or self._params.data_simulator.session_params.turn_prob > 1:
            raise Exception("Turn probability is outside of [0,1]")
        if self._params.data_simulator.session_params.overlap_prob < 0 or self._params.data_simulator.session_params.overlap_prob > 1:
            raise Exception("Overlap probability is outside of [0,1]")

        if self._params.data_simulator.session_params.mean_overlap < 0 or self._params.data_simulator.session_params.mean_overlap > 1:
            raise Exception("Mean overlap is outside of [0,1]")
        if self._params.data_simulator.session_params.mean_silence < 0 or self._params.data_simulator.session_params.mean_silence > 1:
            raise Exception("Mean silence is outside of [0,1]")
        if self._params.data_simulator.session_params.min_dominance < 0 or self._params.data_simulator.session_params.min_dominance > 1:
            raise Exception("Minimum dominance is outside of [0,1]")
        if self._params.data_simulator.speaker_enforcement.enforce_time[0] < 0 or self._params.data_simulator.speaker_enforcement.enforce_time[0] > 1:
            raise Exception("Speaker enforcement start is outside of [0,1]")
        if self._params.data_simulator.speaker_enforcement.enforce_time[1] < 0 or self._params.data_simulator.speaker_enforcement.enforce_time[1] > 1:
            raise Exception("Speaker enforcement end is outside of [0,1]")

        if self._params.data_simulator.session_params.min_dominance*self._params.data_simulator.session_config.num_speakers > 1:
            raise Exception("Number of speakers times minimum dominance is greater than 1")

        if self._params.data_simulator.session_params.overlap_prob / self._params.data_simulator.session_params.turn_prob > 1:
            raise Exception("Overlap probability / turn probability is greater than 1")
        if self._params.data_simulator.session_params.overlap_prob / self._params.data_simulator.session_params.turn_prob == 1 and self._params.data_simulator.session_params.mean_silence > 0:
            raise Exception("Overlap probability / turn probability is equal to 1 and mean silence is greater than 0")

        if self._params.data_simulator.session_params.window_type not in ['hamming', 'hann', 'cosine'] and self._params.data_simulator.session_params.window_type != None:
            raise Exception("Incorrect window type provided")

    def _get_speaker_ids(self):
        """
        Randomly select speaker IDs from loaded dict
        """
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

    def _get_speaker_samples(self, speaker_ids):
        """
        Get a list of the samples for the specified speakers

        Args:
            speaker_ids (list): LibriSpeech speaker IDs for each speaker in the current session.
        """
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

    def _load_speaker_sample(self, speaker_lists, speaker_ids, speaker_turn):
        """
        Load a sample for the selected speaker id

        Args:
            speaker_lists (list): List of samples for each speaker in the session.
            speaker_ids (list): LibriSpeech speaker IDs for each speaker in the current session.
            speaker_turn (int): Current speaker turn.
        """
        speaker_id = speaker_ids[speaker_turn]
        file_id = np.random.randint(0, len(speaker_lists[str(speaker_id)]) - 1)
        file = speaker_lists[str(speaker_id)][file_id]
        return file

    def _get_speaker_dominance(self):
        """
        Get the dominance value for each speaker
        """
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

    def _increase_speaker_dominance(self, base_speaker_dominance, factor):
        """
        Increase speaker dominance (used only in enforce mode)

        Args:
            base_speaker_dominance (list): Dominance values for each speaker.
            factor (int): Factor to increase dominance of unrepresented speakers by.
        """
        increase_percent = []
        for i in range(0,self._params.data_simulator.session_config.num_speakers):
            if self._furthest_sample[i] == 0:
                increase_percent.append(i)
        #ramp up enforce counter until speaker is sampled, then reset once all speakers have spoken
        if len(increase_percent) > 0:
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
            enforce = True
        else:
            dominance = base_speaker_dominance
            enforce = False
        return dominance, enforce

    def _get_next_speaker(self, prev_speaker, dominance):
        """
        Get next speaker (accounting for turn probability, dominance distribution)

        Args:
            prev_speaker (int): Previous speaker turn.
            dominance (list): Dominance values for each speaker.
        """
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

    def _get_window(self, window_amount, start=False):
        """
        Get window

        Args:
            window_amount (int): Window length (in terms of number of samples).
            start (bool): If true, return first half of the window.
        """
        if self._params.data_simulator.session_params.window_type == 'hamming':
            window = hamming(window_amount*2)
        elif self._params.data_simulator.session_params.window_type == 'hann':
            window = hann(window_amount*2)
        elif self._params.data_simulator.session_params.window_type == 'cosine':
            window = cosine(window_amount*2)
        else:
            raise Exception("Incorrect window type provided")

        if start:
            return window[:window_amount]
        else:
            return window[window_amount:]

    def _get_start_buffer_and_window(self, first_alignment):
        """
        Get the start cutoff and window length

        Args:
            first_alignment (int): Start of the first word (in terms of number of samples).
        """
        window_amount = int(self._params.data_simulator.session_params.window_size*self._params.data_simulator.sr)
        start_buffer = int(self._params.data_simulator.session_params.start_buffer*self._params.data_simulator.sr)

        if first_alignment < start_buffer:
            window_amount = 0
            start_cutoff = 0
        elif first_alignment < start_buffer + window_amount:
            window_amount = first_alignment - start_buffer
            start_cutoff = 0
        else:
            start_cutoff = first_alignment - start_buffer - window_amount

        return start_cutoff, window_amount

    def _get_end_buffer_and_window(self, current_sr, remaining_duration_sr, remaining_len_audio_file):
        """
        Get the end buffer and window length

        Args:
            current_sr (int): Current location in the target file (in terms of number of samples).
            remaining_duration_sr (int): Remaining duration in the target file (in terms of number of samples).
            remaining_len_audio_file (int): Length remaining in audio file (in terms of number of samples).
        """

        window_amount = int(self._params.data_simulator.session_params.window_size*self._params.data_simulator.sr)
        release_buffer = int(self._params.data_simulator.session_params.release_buffer*self._params.data_simulator.sr)

        if current_sr + release_buffer > remaining_duration_sr:
            release_buffer = remaining_duration_sr - current_sr
            window_amount = 0
        elif current_sr + window_amount + release_buffer > remaining_duration_sr:
            window_amount = remaining_duration_sr - current_sr - release_buffer

        if remaining_len_audio_file < release_buffer:
            release_buffer = remaining_len_audio_file
            window_amount = 0
        elif remaining_len_audio_file < release_buffer + window_amount:
            window_amount = remaining_len_audio_file - release_buffer

        return release_buffer, window_amount

    def _add_file(self, file, audio_file, sentence_duration, max_sentence_duration, max_sentence_duration_sr):
        """
        Add audio file to current sentence

        Args:
            file (dict): Line from manifest file for current audio file
            audio_file (tensor): Current loaded audio file
            sentence_duration (int): Running count for number of words in sentence
            max_sentence_duration (int): Maximum count for number of words in sentence
            max_sentence_duration_sr (int): Maximum length for sentence in terms of samples
        """
        sentence_duration_sr = len(self._sentence)
        remaining_duration_sr = max_sentence_duration_sr - sentence_duration_sr
        remaining_duration = max_sentence_duration - sentence_duration
        prev_dur_sr = dur_sr = 0
        nw = i = 0

        if (sentence_duration == 0) and self._params.data_simulator.session_params.start_window: #cut off the start of the sentence
            first_alignment = int(file['alignments'][0]*self._params.data_simulator.sr)
            start_cutoff, start_window_amount = self._get_start_buffer_and_window(first_alignment)
        else:
            start_cutoff = 0

        remaining_duration_sr = max_sentence_duration_sr - sentence_duration_sr

        #ensure the desired number of words are added and the length of the output session isn't exceeded
        while (nw < remaining_duration and dur_sr < remaining_duration_sr and i < len(file['words'])):
            dur_sr = int(file['alignments'][i] * self._params.data_simulator.sr) - start_cutoff
            if dur_sr > remaining_duration_sr:
                break

            word = file['words'][i]
            self._words.append(word)
            self._alignments.append(float(sentence_duration_sr * 1.0 / self._params.data_simulator.sr) - float(start_cutoff * 1.0 / self._params.data_simulator.sr) + file['alignments'][i])

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
        if (sentence_duration == 0) and self._params.data_simulator.session_params.window_type != None: #cut off the start of the sentence
            if (start_window_amount > 0): #include window
                window = self._get_window(start_window_amount, start=True)
                self._sentence = torch.cat((self._sentence, np.multiply(audio_file[start_cutoff:start_cutoff+start_window_amount], window)), 0)
            self._sentence = torch.cat((self._sentence, audio_file[start_cutoff+start_window_amount:start_cutoff+prev_dur_sr]), 0)
        else:
            self._sentence = torch.cat((self._sentence, audio_file[:prev_dur_sr]), 0)

        #windowing at the end of the sentence
        if (nw == remaining_duration or dur_sr >= remaining_duration_sr) and self._params.data_simulator.session_params.window_type != None:
            release_buffer, end_window_amount = self._get_end_buffer_and_window(prev_dur_sr, remaining_duration_sr, len(audio_file[start_cutoff+prev_dur_sr:]))
            self._sentence = torch.cat((self._sentence, audio_file[start_cutoff+prev_dur_sr:start_cutoff+prev_dur_sr+release_buffer]), 0)
            if (end_window_amount > 0): #include window
                window = self._get_window(end_window_amount, start=False)
                self._sentence = torch.cat((self._sentence, np.multiply(audio_file[start_cutoff+prev_dur_sr+release_buffer:start_cutoff+prev_dur_sr+release_buffer+end_window_amount], window)), 0)

        return sentence_duration+nw, len(self._sentence)

    # returns new overlapped (or shifted) start position
    def _add_silence_or_overlap(self, speaker_turn, prev_speaker, start, length, session_length_sr, prev_length_sr, enforce):
        """
        Returns new overlapped (or shifted) start position

        Args:
            speaker_turn (int): Current speaker turn.
            prev_speaker (int): Previous speaker turn.
            start (int): Current start of the audio file being inserted.
            length (int): Length of the audio file being inserted.
            session_length_sr (int): Running length of the session in terms of number of samples
            prev_length_sr (int): Length of previous sentence (in terms of number of samples)
            enforce (bool): Whether speaker enforcement mode is being used
        """
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

        else:
            # add silence
            silence_percent = halfnorm(loc=0, scale=mean_silence_percent*np.sqrt(np.pi)/np.sqrt(2)).rvs()
            silence_amount = int(length * silence_percent)

            if start + length + silence_amount > session_length_sr and not enforce:
                new_start = session_length_sr - length
            else:
                new_start = start + silence_amount

        return new_start

    def _create_new_rttm_entry(self, start, end, speaker_id):
        """
        Create new RTTM entries (to write to output rttm file)

        Args:
            start (int): Current start of the audio file being inserted.
            end (int): End of the audio file being inserted.
            speaker_id (int): LibriSpeech speaker ID for the current entry.
        """
        rttm_list = []
        new_start = start
        #look for split locations
        for i in range(0, len(self._words)):
            if self._words[i] == "" and i != 0 and i != len(self._words) - 1:
                silence_length = self._alignments[i] - self._alignments[i-1]
                if silence_length > 2 * self._params.data_simulator.session_params.split_buffer: #split utterance on silence
                    new_end = start + self._alignments[i-1] + self._params.data_simulator.session_params.split_buffer
                    s = float(round(new_start,self._params.data_simulator.outputs.output_precision))
                    e = float(round(new_end,self._params.data_simulator.outputs.output_precision))
                    utterance = f"{s} {e} {speaker_id}"
                    rttm_list.append(utterance)
                    new_start = start + self._alignments[i] - self._params.data_simulator.session_params.split_buffer

        s = float(round(new_start,self._params.data_simulator.outputs.output_precision))
        e = float(round(end,self._params.data_simulator.outputs.output_precision))
        utterance = f"{s} {e} {speaker_id}"
        rttm_list.append(utterance)
        return rttm_list

    def _create_new_json_entry(self, wav_filename, start, length, speaker_id, text, rttm_filepath, ctm_filepath):
        """
        Create new JSON entry (to write to output json file)

        Args:
            wav_filename (str): Output wav filepath.
            start (int): Current start of the audio file being inserted.
            length (int): Length of the audio file being inserted.
            speaker_id (int): LibriSpeech speaker ID for the current entry.
            text (str): Transcript for the current utterance.
            rttm_filepath (str): Output rttm filepath.
            ctm_filepath (str): Output ctm filepath.
        """
        start = float(round(start,self._params.data_simulator.outputs.output_precision))
        length = float(round(length,self._params.data_simulator.outputs.output_precision))
        dict = {"audio_filepath": wav_filename,
                "offset": start,
                "duration": length,
                "label": speaker_id,
                "text": text,
                "num_speakers": self._params.data_simulator.session_config.num_speakers,
                "rttm_filepath": rttm_filepath,
                "ctm_filepath": ctm_filepath,
                "uem_filepath": None}
        return dict

    def _create_new_ctm_entry(self, session_name, speaker_id, start):
        """
        Create new CTM entry (to write to output ctm file)

        Args:
            session_name (str): Current session name.
            start (int): Current start of the audio file being inserted.
            speaker_id (int): LibriSpeech speaker ID for the current entry.
        """
        arr = []
        start = float(round(start,self._params.data_simulator.outputs.output_precision))
        for i in range(0, len(self._words)):
            word = self._words[i]
            if word != "": #note that using the current alignments the first word is always empty, so there is no error from indexing the array with i-1
                align1 = float(round(self._alignments[i-1] + start, self._params.data_simulator.outputs.output_precision))
                align2 = float(round(self._alignments[i] - self._alignments[i-1], self._params.data_simulator.outputs.output_precision))
                text = f"{session_name} {speaker_id} {align1} {align2} {word} 0\n"
                arr.append((align1, text))
        return arr

    def _build_sentence(self, speaker_turn, speaker_ids, speaker_lists, max_sentence_duration_sr):
        """
        Build new sentence

        Args:
            speaker_turn (int): Current speaker turn.
            speaker_ids (list): LibriSpeech speaker IDs for each speaker in the current session.
            speaker_lists (list): List of samples for each speaker in the session.
            max_sentence_duration_sr (int): Maximum length for sentence in terms of samples
        """
        # select speaker length
        sl = np.random.negative_binomial(
            self._params.data_simulator.session_params.sentence_length_params[0], self._params.data_simulator.session_params.sentence_length_params[1]
        ) + 1

        # initialize sentence, text, words, alignments
        self._sentence = torch.zeros(0)
        self._text = ""
        self._words = []
        self._alignments = []
        sentence_duration = sentence_duration_sr = 0

        # build sentence
        while sentence_duration < sl and sentence_duration_sr < max_sentence_duration_sr:
            file = self._load_speaker_sample(speaker_lists, speaker_ids, speaker_turn)
            audio_file, sr = librosa.load(file['audio_filepath'], sr=self._params.data_simulator.sr)
            audio_file = torch.from_numpy(audio_file)
            sentence_duration,sentence_duration_sr = self._add_file(file, audio_file, sentence_duration, sl, max_sentence_duration_sr)

        #look for split locations
        splits = []
        new_start = 0
        for i in range(0, len(self._words)):
            if self._words[i] == "" and i != 0 and i != len(self._words) - 1:
                silence_length = self._alignments[i] - self._alignments[i-1]
                if silence_length > 2 * self._params.data_simulator.session_params.split_buffer: #split utterance on silence
                    new_end = self._alignments[i-1] + self._params.data_simulator.session_params.split_buffer
                    splits.append([int(new_start * self._params.data_simulator.sr), int(new_end * self._params.data_simulator.sr)])
                    new_start = self._alignments[i] - self._params.data_simulator.session_params.split_buffer
        splits.append([int(new_start * self._params.data_simulator.sr), len(self._sentence)])

        #per-speaker normalization
        if self._params.data_simulator.session_params.normalization == 'equal':
            if torch.max(torch.abs(self._sentence)) > 0:
                split_length = split_sum = 0
                for split in splits:
                    split_length += len(self._sentence[split[0]:split[1]])
                    split_sum += torch.sum(self._sentence[split[0]:split[1]]**2)
                average_rms = torch.sqrt(split_sum*1.0/split_length)
                self._sentence = self._sentence / (1.0 * average_rms)
        #TODO add variable speaker volume (per-speaker volume selected at start of sentence)

    def _get_background(self, len_array, power_array):
        """
        Augment with background noise

        Args:
            len_array (int): Length of background noise required.
            avg_power_array (float): Average power of the audio file.
        """

        bg_dir = self._params.data_simulator.background_noise.background_dir
        desired_snr = self._params.data_simulator.background_noise.snr
        ratio = 10 ** (desired_snr / 20)
        desired_avg_power_noise = power_array / ratio

        print('power_array: ', power_array)
        print('desired_avg_power_noise: ', desired_avg_power_noise)

        bg_files = os.listdir(bg_dir)
        bg_array = torch.zeros(len_array)
        running_len = 0
        while running_len < len_array:
            file_id = np.random.randint(0, len(bg_files) - 1)
            file = bg_files[file_id]
            audio_file, sr = librosa.load(os.path.join(bg_dir, file), sr=self._params.data_simulator.sr)
            audio_file = torch.from_numpy(audio_file)

            if running_len+len(audio_file) < len_array:
                end_audio_file = running_len+len(audio_file)
            else:
                end_audio_file = len_array

            pow_audio_file = torch.mean(audio_file[:end_audio_file-running_len]**2)
            print('pow_audio_file: ', pow_audio_file)

            scaled_audio_file = audio_file[:end_audio_file-running_len] * torch.sqrt(desired_avg_power_noise / pow_audio_file)
            bg_array[running_len:end_audio_file] = scaled_audio_file
            running_len = end_audio_file

        return bg_array

    def _generate_session(self, idx, basepath, filename):
        """
        Generate diarization session

        Args:
            idx (int): Index for current session (out of total number of sessions).
            basepath (str): Path to output directory.
            filename (str): Filename for output files.
        """
        speaker_ids = self._get_speaker_ids()  # randomly select speaker ids
        speaker_dominance = self._get_speaker_dominance()  # randomly determine speaker dominance
        base_speaker_dominance = np.copy(speaker_dominance)
        speaker_lists = self._get_speaker_samples(speaker_ids)  # get list of samples per speaker

        running_length_sr = prev_length_sr = 0  # starting point for each sentence
        start = end = 0
        prev_speaker = None
        rttm_list = []
        json_list = []
        ctm_list = []
        self._furthest_sample = [0 for n in range(0,self._params.data_simulator.session_config.num_speakers)]
        self._missing_overlap = 0

        #hold enforce until all speakers have spoken
        enforce_counter = 2 # dominance is increased by a factor of enforce_counter
        enforce_time = np.random.uniform(self._params.data_simulator.speaker_enforcement.enforce_time[0], self._params.data_simulator.speaker_enforcement.enforce_time[1])
        enforce = self._params.data_simulator.speaker_enforcement.enforce_num_speakers

        session_length_sr = int((self._params.data_simulator.session_config.session_length * self._params.data_simulator.sr))
        array = torch.zeros(session_length_sr)

        while running_length_sr < session_length_sr or enforce:
            #enforce num_speakers
            if running_length_sr > enforce_time*session_length_sr and enforce:
                speaker_dominance, enforce = self._increase_speaker_dominance(base_speaker_dominance, enforce_counter)
                if enforce:
                    enforce_counter += 1

            # select speaker
            speaker_turn = self._get_next_speaker(prev_speaker, speaker_dominance)

            # build sentence (only add if remaining length >  specific time)
            max_sentence_duration_sr = session_length_sr - running_length_sr
            if enforce:
                max_sentence_duration_sr = float('inf')
            elif max_sentence_duration_sr < self._params.data_simulator.session_params.end_buffer * self._params.data_simulator.sr:
                break
            self._build_sentence(speaker_turn, speaker_ids, speaker_lists, max_sentence_duration_sr)

            length = len(self._sentence)
            start = self._add_silence_or_overlap(speaker_turn, prev_speaker, running_length_sr, length, session_length_sr, prev_length_sr, enforce)
            end = start + length
            if end > len(array): #only occurs in enforce mode
                array = torch.nn.functional.pad(array, (0, end - len(array)))
            array[start:end] += self._sentence

            #build entries for output files
            new_rttm_entries = self._create_new_rttm_entry(start / self._params.data_simulator.sr, end / self._params.data_simulator.sr, speaker_ids[speaker_turn])
            for entry in new_rttm_entries:
                rttm_list.append(entry)
            new_json_entry = self._create_new_json_entry(os.path.join(basepath, filename + '.wav'), start / self._params.data_simulator.sr, length / self._params.data_simulator.sr, speaker_ids[speaker_turn], self._text, os.path.join(basepath, filename + '.rttm'), os.path.join(basepath, filename + '.ctm'))
            json_list.append(new_json_entry)
            new_ctm_entries = self._create_new_ctm_entry(filename, speaker_ids[speaker_turn], start / self._params.data_simulator.sr)
            for entry in new_ctm_entries:
                ctm_list.append(entry)

            running_length_sr = np.maximum(running_length_sr, end)
            self._furthest_sample[speaker_turn] = running_length_sr
            prev_speaker = speaker_turn
            prev_length_sr = length

        #background noise augmentation
        if self._params.data_simulator.background_noise.add_bg:
            avg_power_array = torch.mean(array**2)
            bg = self._get_background(len(array), avg_power_array)
            array += bg

        array = array / (1.0 * torch.max(torch.abs(array)))  # normalize wav file to avoid clipping
        sf.write(os.path.join(basepath, filename + '.wav'), array, self._params.data_simulator.sr)
        labels_to_rttmfile(rttm_list, filename, self._params.data_simulator.outputs.output_dir)
        write_manifest(os.path.join(basepath, filename + '.json'), json_list)
        write_ctm(os.path.join(basepath, filename + '.ctm'), ctm_list)
        write_text(os.path.join(basepath, filename + '.txt'), ctm_list)

    def generate_sessions(self):
        """
        Generate diarization sessions
        """
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

        wavlist = open(os.path.join(basepath, "synthetic_wav.list"), "w")
        rttmlist = open(os.path.join(basepath, "synthetic_rttm.list"), "w")
        jsonlist = open(os.path.join(basepath, "synthetic_json.list"), "w")
        ctmlist = open(os.path.join(basepath, "synthetic_ctm.list"), "w")
        textlist = open(os.path.join(basepath,"synthetic_txt.list"), "w")

        for i in range(0, self._params.data_simulator.session_config.num_sessions):
            self._furthest_sample = [0 for n in range(0,self._params.data_simulator.session_config.num_speakers)]
            self._missing_overlap = 0

            print(f"Generating Session Number {i}")
            filename = self._params.data_simulator.outputs.output_filename + f"_{i}"
            self._generate_session(i, basepath, filename)

            wavlist.write(os.path.join(basepath, filename + '.wav\n'))
            rttmlist.write(os.path.join(basepath, filename + '.rttm\n'))
            jsonlist.write(os.path.join(basepath, filename + '.json\n'))
            ctmlist.write(os.path.join(basepath, filename + '.ctm\n'))
            textlist.write(os.path.join(basepath, filename + '.txt\n'))

            #throw error if number of speakers is less than requested
            num_missing = 0
            for k in range(0,len(self._furthest_sample)):
                if self._furthest_sample[k] == 0:
                    num_missing += 1
            if num_missing != 0:
                warnings.warn(f"{self._params.data_simulator.session_config.num_speakers-num_missing} speakers were included in the clip instead of the requested amount of {self._params.data_simulator.session_config.num_speakers}")

        wavlist.close()
        rttmlist.close()
        jsonlist.close()
        ctmlist.close()
        textlist.close()

    def create_base_manifest_ds(self):
        """
        Create base diarization manifest file
        """
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
        """
        Create segmented diarization manifest file
        """
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

    def _check_args(self):
        """
        Checks arguments to ensure they are within valid ranges
        """
        #check base arguments
        super()._check_args()

        if self._params.data_simulator.rir_generation.toolkit != 'pyroomacoustics' and self._params.data_simulator.rir_generation.toolkit != 'gpuRIR':
            raise Exception("Toolkit must be pyroomacoustics or gpuRIR")
        if len(self._params.data_simulator.rir_generation.room_config.room_sz) != 3:
            raise Exception("Incorrect room dimensions provided")
        if self._params.data_simulator.rir_generation.mic_config.num_channels == 0:
            raise Exception("Number of channels should be greater or equal to 1")
        if len(self._params.data_simulator.rir_generation.room_config.pos_src) < 2:
            raise Exception("Less than 2 provided source positions")
        for sublist in self._params.data_simulator.rir_generation.room_config.pos_src:
            if len(sublist) != 3:
                raise Exception("Three coordinates must be provided for sources positions")
        if len(self._params.data_simulator.rir_generation.mic_config.pos_rcv) == 0:
            raise Exception("No provided mic positions")
        for sublist in self._params.data_simulator.rir_generation.room_config.pos_src:
            if len(sublist) != 3:
                raise Exception("Three coordinates must be provided for mic positions")

        if self._params.data_simulator.session_config.num_speakers != len(self._params.data_simulator.rir_generation.room_config.pos_src):
            raise Exception("Number of speakers is not equal to the number of provided source positions")
        if self._params.data_simulator.rir_generation.mic_config.num_channels != len(self._params.data_simulator.rir_generation.mic_config.pos_rcv):
            raise Exception("Number of channels is not equal to the number of provided microphone positions")

        if not self._params.data_simulator.rir_generation.mic_config.orV_rcv and self._params.data_simulator.rir_generation.mic_config.mic_pattern != 'omni':
            raise Exception("Microphone orientations must be provided if mic_pattern != omni")
        if self._params.data_simulator.rir_generation.mic_config.orV_rcv != None:
            if len(self._params.data_simulator.rir_generation.mic_config.orV_rcv) != len(self._params.data_simulator.rir_generation.mic_config.pos_rcv):
                raise Exception("A different number of microphone orientations and microphone positions were provided")
            for sublist in self._params.data_simulator.rir_generation.mic_config.orV_rcv:
                if len(sublist) != 3:
                    raise Exception("Three coordinates must be provided for orientations")

    def _generate_rir_gpuRIR(self):
        """
        Create simulated RIR
        """
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

    def _generate_rir_pyroomacoustics(self):
        """
        Create simulated RIR
        """

        rt60 = self._params.data_simulator.rir_generation.absorbtion_params.T60  # The desired reverberation time
        room_dim = np.array(self._params.data_simulator.rir_generation.room_config.room_sz)
        sr = self._params.data_simulator.sr

        # We invert Sabine's formula to obtain the parameters for the ISM simulator
        e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
        room = pra.ShoeBox(room_dim, fs=sr, materials=pra.Material(e_absorption), max_order=max_order)

        pos_src = np.array(self._params.data_simulator.rir_generation.room_config.pos_src)
        for pos in pos_src:
            room.add_source(pos)

        orV_rcv = self._params.data_simulator.rir_generation.mic_config.orV_rcv
        mic_pattern = self._params.data_simulator.rir_generation.mic_config.mic_pattern
        if self._params.data_simulator.rir_generation.mic_config.mic_pattern == 'omni':
            mic_pattern = DirectivityPattern.OMNI
            dir_vec = DirectionVector(azimuth=0, colatitude=90, degrees=True)
        dir_obj = CardioidFamily(
            orientation=dir_vec,
            pattern_enum=mic_pattern,
        )
        room.add_microphone_array(np.array(self._params.data_simulator.rir_generation.mic_config.pos_rcv).T, directivity=dir_obj)

        room.compute_rir()
        rir_pad = 0
        for channel in room.rir:
            for pos in channel:
                if pos.shape[0] > rir_pad:
                    rir_pad = pos.shape[0]
        return room.rir,rir_pad

    def _convolve_rir_gpuRIR(self, speaker_turn, RIR):
        """
        Augment sample using synthetic RIR

        Args:
            speaker_turn (int): Current speaker turn.
            RIR (torch.tensor): Room Impulse Response.
        """
        output_sound = []
        for channel in range(0,self._params.data_simulator.rir_generation.mic_config.num_channels):
            out_channel = convolve(self._sentence, RIR[speaker_turn, channel, : len(self._sentence)]).tolist()
            output_sound.append(out_channel)
        output_sound = torch.tensor(output_sound)
        output_sound = torch.transpose(output_sound, 0, 1)
        return output_sound

    def _convolve_rir_pyroomacoustics(self, speaker_turn, RIR):
        """
        Augment sample using synthetic RIR

        Args:
            speaker_turn (int): Current speaker turn.
            RIR (torch.tensor): Room Impulse Response.
        """
        output_sound = []
        length = 0
        for channel in range(0,self._params.data_simulator.rir_generation.mic_config.num_channels):
            out_channel = convolve(self._sentence, RIR[channel][speaker_turn][:len(self._sentence)]).tolist()
            if len(out_channel) > length:
                length = len(out_channel)
            output_sound.append(torch.tensor(out_channel))
        return output_sound,length

    def _generate_session(self, idx, basepath, filename):
        """
        Generate diarization session

        Args:
            idx (int): Index for current session (out of total number of sessions).
            basepath (str): Path to output directory.
            filename (str): Filename for output files.
        """
        speaker_ids = self._get_speaker_ids()  # randomly select speaker ids
        speaker_dominance = self._get_speaker_dominance()  # randomly determine speaker dominance
        base_speaker_dominance = np.copy(speaker_dominance)
        speaker_lists = self._get_speaker_samples(speaker_ids)  # get list of samples per speaker

        running_length_sr = prev_length_sr = 0  # starting point for each sentence
        start = end = 0
        prev_speaker = None
        rttm_list = []
        json_list = []
        ctm_list = []
        self._furthest_sample = [0 for n in range(0,self._params.data_simulator.session_config.num_speakers)]
        self._missing_overlap = 0

        #Room Impulse Response Generation (performed once per batch of sessions)
        if self._params.data_simulator.rir_generation.toolkit == 'gpuRIR':
            RIR = self._generate_rir_gpuRIR()
            RIR_pad = RIR.shape[2] - 1
        elif self._params.data_simulator.rir_generation.toolkit == 'pyroomacoustics':
            RIR,RIR_pad = self._generate_rir_pyroomacoustics()
        else:
            raise Exception("Toolkit must be pyroomacoustics or gpuRIR")

        #hold enforce until all speakers have spoken
        enforce_counter = 2
        enforce_time = np.random.uniform(self._params.data_simulator.speaker_enforcement.enforce_time[0], self._params.data_simulator.speaker_enforcement.enforce_time[1])
        enforce = self._params.data_simulator.speaker_enforcement.enforce_num_speakers

        session_length_sr = int((self._params.data_simulator.session_config.session_length * self._params.data_simulator.sr))
        array = torch.zeros((session_length_sr, self._params.data_simulator.rir_generation.mic_config.num_channels))

        while running_length_sr < session_length_sr or enforce:
            #enforce num_speakers
            if running_length_sr > enforce_time*session_length_sr and enforce:
                speaker_dominance, enforce = self._increase_speaker_dominance(base_speaker_dominance, enforce_counter)
                if enforce:
                    enforce_counter += 1

            # select speaker
            speaker_turn = self._get_next_speaker(prev_speaker, speaker_dominance)

            # build sentence (only add if remaining length >  specific time)
            max_sentence_duration_sr = session_length_sr - running_length_sr - RIR_pad #sentence will be RIR_len - 1 longer than the audio was pre-augmentation
            if enforce:
                max_sentence_duration_sr = float('inf')
            elif max_sentence_duration_sr < self._params.data_simulator.session_params.end_buffer * self._params.data_simulator.sr:
                break
            self._build_sentence(speaker_turn, speaker_ids, speaker_lists, max_sentence_duration_sr)

            #augment sentence
            if self._params.data_simulator.rir_generation.toolkit == 'gpuRIR':
                augmented_sentence = self._convolve_rir_gpuRIR(speaker_turn, RIR)
                length = augmented_sentence.shape[0]
            elif self._params.data_simulator.rir_generation.toolkit == 'pyroomacoustics':
                augmented_sentence, length = self._convolve_rir_pyroomacoustics(speaker_turn, RIR)

            start = self._add_silence_or_overlap(speaker_turn, prev_speaker, running_length_sr, length, session_length_sr, prev_length_sr, enforce)
            end = start + length
            if end > len(array):
                array = torch.nn.functional.pad(array, (0, end - len(array)))

            if self._params.data_simulator.rir_generation.toolkit == 'gpuRIR':
                array[start:end, :] += augmented_sentence
            elif self._params.data_simulator.rir_generation.toolkit == 'pyroomacoustics':
                for channel in range(0,self._params.data_simulator.rir_generation.mic_config.num_channels):
                    len_ch = len(augmented_sentence[channel]) #acounts for how channels are slightly different lengths
                    array[start:start+len_ch, channel] += augmented_sentence[channel]

            #build entries for output files
            new_rttm_entries = self._create_new_rttm_entry(start / self._params.data_simulator.sr, end / self._params.data_simulator.sr, speaker_ids[speaker_turn])
            for entry in new_rttm_entries:
                rttm_list.append(new_rttm_entries)
            new_json_entry = self._create_new_json_entry(os.path.join(basepath, filename + '.wav'), start / self._params.data_simulator.sr, length / self._params.data_simulator.sr, speaker_ids[speaker_turn], self._text, os.path.join(basepath, filename + '.rttm'), os.path.join(basepath, filename + '.ctm'))
            json_list.append(new_json_entry)
            new_ctm_entries = self._create_new_ctm_entry(filename, speaker_ids[speaker_turn], start / self._params.data_simulator.sr)
            for entry in new_ctm_entries:
                ctm_list.append(entry)

            running_length_sr = np.maximum(running_length_sr, end)
            self._furthest_sample[speaker_turn] = running_length_sr
            prev_speaker = speaker_turn
            prev_length_sr = length

        array = array / (1.0 * torch.max(torch.abs(array)))  # normalize wav file to avoid clipping
        sf.write(os.path.join(basepath, filename + '.wav'), array, self._params.data_simulator.sr)
        labels_to_rttmfile(rttm_list, filename, self._params.data_simulator.outputs.output_dir)
        write_manifest(os.path.join(basepath, filename + '.json'), json_list)
        write_ctm(os.path.join(basepath, filename + '.ctm'), ctm_list)
        write_text(os.path.join(basepath, filename + '.txt'), ctm_list)
