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

"""
This script demonstrates how to run offline speaker diarization with asr.
Usage:
python offline_diar_with_asr_infer.py \
    diarizer.manifest_filepath=<path to manifest file> \
    diarizer.out_dir='demo_asr_output' \
    diarizer.speaker_embeddings.model_path=<pretrained modelname or path to .nemo> \
    diarizer.asr.model_path=<pretrained modelname or path to .nemo> \
    diarizer.asr.parameters.asr_based_vad=True \
    diarizer.speaker_embeddings.parameters.save_embeddings=False

Check out whole parameters in ./conf/offline_diarization_with_asr.yaml and their meanings.
For details, have a look at <NeMo_git_root>/tutorials/speaker_tasks/Speaker_Diarization_Inference.ipynb
Currently, the following NGC models are supported:

    stt_en_quartznet15x5
    stt_en_citrinet*
    stt_en_conformer_ctc*

"""
from omegaconf import OmegaConf

from nemo.collections.asr.parts.utils.decoder_timestamps_utils import ASRDecoderTimeStamps
from nemo.collections.asr.parts.utils.diarization_utils import OfflineDiarWithASR
from nemo.core.config import hydra_runner
from nemo.utils import logging
from tqdm import tqdm

import copy 
import os
import json 
import pickle
import numpy as np
import concurrent.futures
from copy import deepcopy
from nemo.collections.asr.metrics.der import score_labels
from nemo.collections.asr.parts.utils.speaker_utils import (
    audio_rttm_map,
    rttm_to_labels,
    labels_to_pyannote_object,
)
SPLITSYM = "@"
import meeteval

import jiwer
from jiwer.transforms import RemoveKaldiNonWords


jiwer_chime6_scoring = jiwer.Compose(
    [
        RemoveKaldiNonWords(),
        jiwer.SubstituteRegexes({r"\"": " ", "^[ \t]+|[ \t]+$": "", r"\u2019": "'"}),
        jiwer.RemoveEmptyStrings(),
        jiwer.RemoveMultipleSpaces(),
    ]
)
jiwer_chime7_scoring = jiwer.Compose(
    [
        jiwer.SubstituteRegexes(
            {
                "(?:^|(?<= ))(hm|hmm|mhm|mmh|mmm)(?:(?= )|$)": "hmmm",
                "(?:^|(?<= ))(uhm|um|umm|umh|ummh)(?:(?= )|$)": "ummm",
                "(?:^|(?<= ))(uh|uhh)(?:(?= )|$)": "uhhh",
            }
        ),
        jiwer.RemoveEmptyStrings(),
        jiwer.RemoveMultipleSpaces(),
    ]
)
def chime7_norm_scoring(txt):
    return jiwer_chime7_scoring(
        jiwer_chime6_scoring(txt)  # noqa: E731
    )  # noqa: E731


def divide_chunks(trans_info_dict, win_len, word_window, port):
    """
    Divide word sequence into chunks of length `win_len` for parallel processing.    

    Args:
        trans_info_dict (_type_): _description_
        diar_logits (_type_): _description_
        win_len (int, optional): _description_. Defaults to 250.
    """
    if len(port) > 1:
        num_workers = len(port) 
    else:
        num_workers = 1
    div_trans_info_dict = {}
    # div_trans_info_dict, div_diar_logits = {}, {}
    for uniq_id in trans_info_dict.keys():
        uniq_trans = trans_info_dict[uniq_id]
        del uniq_trans['status']
        del uniq_trans['transcription']
        del uniq_trans['sentences']
        word_seq = uniq_trans['words']
        # logit_seq = diar_logits[uniq_id]['pred_mat']

        div_word_seq = [] 
        # div_word_seq, div_diar_logits_seq = [], []
        if win_len is None:
            win_len = int(np.ceil(len(word_seq)/num_workers))
        n_chunks = int(np.ceil(len(word_seq)/win_len))
        
        for k in range(n_chunks):
            div_word_seq.append(word_seq[max(k*win_len - word_window, 0):(k+1)*win_len])
            # div_diar_logits_seq.append(logit_seq[max(k*win_len - word_window, 0):(k+1)*win_len])
        
        total_count = len(div_word_seq)
        for k, w_seq in enumerate(div_word_seq):
            seq_id = uniq_id + f"{SPLITSYM}{k}{SPLITSYM}{total_count}"
            div_trans_info_dict[seq_id] = dict(uniq_trans)
            div_trans_info_dict[seq_id]['words'] = w_seq
            # div_diar_logits[seq_id] = dict(diar_logits[uniq_id])
            # div_diar_logits[seq_id]['pred_mat'] = div_diar_logits_seq[k]
    return div_trans_info_dict
    # , div_diar_logits

def merge_div_inputs(div_trans_info_dict, org_trans_info_dict, win_len=250, word_window=16):
    """
    Merge the outputs of parallel processing.
    """
    
    uniq_id_list = list(org_trans_info_dict.keys())
    sub_div_dict = {}
    for seq_id in div_trans_info_dict.keys():
        div_info = seq_id.split(SPLITSYM)
        uniq_id, sub_idx, total_count = div_info[0], int(div_info[1]), int(div_info[2])
        if uniq_id not in sub_div_dict:
            sub_div_dict[uniq_id] = [None] * total_count
        sub_div_dict[uniq_id][sub_idx] = div_trans_info_dict[seq_id]['words']
            
    for uniq_id in uniq_id_list:
        org_trans_info_dict[uniq_id]['words'] = []
        for k, div_words in enumerate(sub_div_dict[uniq_id]):
            if k == 0:
                div_words = div_words[:win_len]
            else:
                div_words = div_words[word_window:]
            org_trans_info_dict[uniq_id]['words'].extend(div_words)
    return org_trans_info_dict
            
# def run_mp_beam_search_decoding(asr_diar_offline, trans_info_dict, diar_logits, org_trans_info_dict, div_mp, win_len, word_window, port=None, use_ngram=False):
def run_mp_beam_search_decoding(asr_diar_offline, trans_info_dict, org_trans_info_dict, div_mp, win_len, word_window, port=None, use_ngram=False):
    # uniq_id_list = sorted(list(asr_diar_offline.AUDIO_RTTM_MAP.keys() ))
    if len(port) > 1:
        port = [int(p) for p in port]
        # raise ValueError(f"port must be a list of integers but got {port}")
    if use_ngram:
        port = [None]
        num_workers = 36
    else:
        num_workers = len(port)
    
    uniq_id_list = sorted(list(trans_info_dict.keys() ))
    tp = concurrent.futures.ProcessPoolExecutor(max_workers=num_workers)
    futures = []

    count = 0
    
    for uniq_id in uniq_id_list:
        print(f"running beam search decoding for {uniq_id}")
        if port is not None:
            port_num = port[count % len(port)]    
        else:
            port_num = None
        count += 1
        uniq_trans_info_dict = {uniq_id: trans_info_dict[uniq_id]}
        # uniq_diar_logits = {uniq_id: diar_logits[uniq_id]}
        futures.append(tp.submit(asr_diar_offline.beam_search_diarization, uniq_trans_info_dict, port_num=port_num))

    pbar = tqdm(total=len(uniq_id_list), desc="Running beam search decoding", unit="files")
    count = 0
    output_trans_info_dict = {}
    for done_future in concurrent.futures.as_completed(futures):
        count += 1
        pbar.update()
        output_trans_info_dict.update(done_future.result())
    pbar.close() 
    tp.shutdown()
    if div_mp:
        output_trans_info_dict = merge_div_inputs(div_trans_info_dict=output_trans_info_dict, 
                                                  org_trans_info_dict=org_trans_info_dict, 
                                                  win_len=win_len, word_window=word_window)
    return output_trans_info_dict

def count_num_of_spks(json_trans_list):
    spk_set = set()
    for sentence_dict in json_trans_list:
        try:
            spk_set.add(sentence_dict['speaker'])
        except:
            import ipdb; ipdb.set_trace()
    speaker_map = { spk_str: idx for idx, spk_str in enumerate(spk_set)}
    return speaker_map

def add_speaker_softmax(json_trans_list, peak_prob=0.94 ,max_spks=4): 
    nemo_json_dict = {}
    word_dict_seq_list = []
    if peak_prob > 1 or peak_prob < 0:
        raise ValueError(f"peak_prob must be between 0 and 1 but got {peak_prob}")
    speaker_map = count_num_of_spks(json_trans_list)
    base_array = np.ones(max_spks) * (1 - peak_prob)/(max_spks-1)
    stt_sec, end_sec = None, None
    for sentence_dict in json_trans_list:
        word_list = sentence_dict['words'].split()
        speaker = sentence_dict['speaker']
        for word in word_list:
            speaker_softmax = copy.deepcopy(base_array)
            speaker_softmax[speaker_map[speaker]] = peak_prob
            word_dict_seq_list.append({'word': word, 
                                    'start_time': stt_sec, 
                                    'end_time': end_sec, 
                                    'speaker': speaker_map[speaker], 
                                    'speaker_softmax': speaker_softmax}
                                    )
    nemo_json_dict.update({'words': word_dict_seq_list, 
                           'status': "success", 
                           'sentences': json_trans_list, 
                           'speaker_count': len(speaker_map), 
                           'transcription': None}
                        )
    return nemo_json_dict

def convert_nemo_json_to_seglst(trans_info_dict):
    seglst_seq_list = []
    seg_lst_dict, spk_wise_trans_sessions = {}, {}
    for uniq_id in trans_info_dict.keys():
        spk_wise_trans_sessions[uniq_id] = {}
        seglst_seq_list = []
        word_seq_list = trans_info_dict[uniq_id]['words']
        prev_speaker, sentence = None, ''
        for widx, word_dict in enumerate(word_seq_list):
            curr_speaker = word_dict['speaker']

            # For making speaker wise transcriptions
            word = word_dict['word']
            if curr_speaker not in spk_wise_trans_sessions[uniq_id]:
                spk_wise_trans_sessions[uniq_id][curr_speaker] = word
            elif curr_speaker in spk_wise_trans_sessions[uniq_id]:
                spk_wise_trans_sessions[uniq_id][curr_speaker] = f"{spk_wise_trans_sessions[uniq_id][curr_speaker]} {word_dict['word']}"

            # For making segment wise transcriptions
            if curr_speaker!= prev_speaker and prev_speaker is not None:
                sentence = chime7_norm_scoring(sentence)
                seglst_seq_list.append({'session_id': uniq_id, 
                                        'words': sentence, 
                                        'start_time': 0.0,
                                        'end_time': 0.0,
                                        'speaker': prev_speaker, 
                })
                sentence = word_dict['word']
            else:
                sentence = f"{sentence} {word_dict['word']}"
            prev_speaker = curr_speaker

        # For the last word:
        # (1) If there is no speaker change, add the existing sentence and exit the loop
        # (2) If there is a speaker change, add the last word and exit the loop
        if widx == len(word_seq_list) - 1:
            sentence = chime7_norm_scoring(sentence)
            seglst_seq_list.append({'session_id': uniq_id, 
                                    'words': sentence, 
                                    'start_time': 0.0,
                                    'end_time': 0.0,
                                    'speaker': curr_speaker, 
            })
        seg_lst_dict[uniq_id] = seglst_seq_list
    return seg_lst_dict, spk_wise_trans_sessions

def load_input_jsons(cfg):
    trans_info_dict = {}
    json_filepath_list = open(cfg.diarizer.manifest_filepath).readlines()
    for json_path in json_filepath_list:
        json_path = json_path.strip()
        uniq_id = os.path.split(json_path)[-1].split(".")[0]
        if os.path.exists(json_path):
            with open(json_path, "r") as file:
                json_trans = json.load(file)
        else:
            logging.info(f"JSON transcript not found for {uniq_id}")
            return None, None, None
        nemo_json_dict = add_speaker_softmax(json_trans)
        trans_info_dict[uniq_id] = nemo_json_dict
    json_trans[0]['words'].split()
    return trans_info_dict

def write_seglst_jsons(seg_lst_sessions_dict, diar_out_path):
    for session_id, seg_lst_list in seg_lst_sessions_dict.items():
        print(f"Writing {diar_out_path}/{session_id}.seglst.json")
        with open(f'{diar_out_path}/{session_id}.seglst.json', 'w') as file:
            json.dump(seg_lst_list, file, indent=4)  # indent=4 for pretty printing

@hydra_runner(config_path="../conf/inference", config_name="diar_infer_meeting.yaml")
def main(cfg):
    cfg_copy = copy.deepcopy(cfg)
    
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    # trans_info_dict, diar_scores, diar_logits = load_input_jsons(cfg)
    trans_info_dict = load_input_jsons(cfg)
    diar_scores = None
    if trans_info_dict is None:
        # ASR inference for words and word timestamps
        asr_decoder_ts = ASRDecoderTimeStamps(cfg.diarizer)
        asr_model = asr_decoder_ts.set_asr_model()
        word_hyp, word_ts_hyp = asr_decoder_ts.run_ASR(asr_model)

        # Create a class instance for matching ASR and diarization results
        asr_diar_offline = OfflineDiarWithASR(cfg.diarizer)
        asr_diar_offline.word_ts_anchor_offset = asr_decoder_ts.word_ts_anchor_offset
            
        # Diarization inference for speaker labels
        diar_hyp, diar_logits, diar_scores = asr_diar_offline.run_diarization(cfg_copy, word_ts_hyp)
        metric = diar_scores[0]
        
        trans_info_dict = asr_diar_offline.get_transcript_with_speaker_labels(diar_hyp, word_hyp, word_ts_hyp)
    else:
        asr_diar_offline = OfflineDiarWithASR(cfg.diarizer) 
        use_mp = True
        use_mp = False
        # div_mp = True
        # div_mp = False
        if cfg.diarizer.asr.realigning_lm_parameters.use_mp and use_mp:
            if cfg.diarizer.asr.realigning_lm_parameters.use_chunk_mp:
                div_trans_info_dict = divide_chunks(trans_info_dict=trans_info_dict, 
                                                    win_len=cfg.diarizer.asr.realigning_lm_parameters.parallel_chunk_word_len, 
                                                    word_window=cfg.diarizer.asr.realigning_lm_parameters.word_window,
                                                    port=cfg.diarizer.asr.realigning_lm_parameters.port,)
                
                trans_info_dict = run_mp_beam_search_decoding(asr_diar_offline, 
                                                              trans_info_dict=div_trans_info_dict, 
                                                              org_trans_info_dict=trans_info_dict, 
                                                              div_mp=True,
                                                              win_len=cfg.diarizer.asr.realigning_lm_parameters.parallel_chunk_word_len,
                                                              word_window=cfg.diarizer.asr.realigning_lm_parameters.word_window,
                                                              port=cfg.diarizer.asr.realigning_lm_parameters.port,
                                                              use_ngram=cfg.diarizer.asr.realigning_lm_parameters.use_ngram,
                                                              )

            else:
                div_trans_info_dict = trans_info_dict
                trans_info_dict = run_mp_beam_search_decoding(asr_diar_offline, 
                                                              trans_info_dict=div_trans_info_dict, 
                                                              diar_logits=diar_logits, 
                                                              org_trans_info_dict=trans_info_dict, 
                                                              div_mp=False,
                                                              win_len=cfg.diarizer.asr.realigning_lm_parameters.parallel_chunk_word_len,
                                                              use_ngram=cfg.diarizer.asr.realigning_lm_parameters.use_ngram,
                                                              )
        else:
            if len(cfg.diarizer.asr.realigning_lm_parameters.port) > 1:
                port_num = cfg.diarizer.asr.realigning_lm_parameters.port[0]
            else:
                port_num = int(cfg.diarizer.asr.realigning_lm_parameters.port[0])
            _trans_info_dict = deepcopy(trans_info_dict)
            trans_info_dict = asr_diar_offline.beam_search_diarization(trans_info_dict, port_num=port_num)

    seg_lst_sessions_dict, spk_wise_trans_sessions = convert_nemo_json_to_seglst(trans_info_dict) 
    # Write SegLST jsons to output folder
    write_seglst_jsons(seg_lst_sessions_dict, diar_out_path=cfg.diarizer.out_dir)
    print(f"ALPHA: {cfg.diarizer.asr.realigning_lm_parameters.alpha} BETA: {cfg.diarizer.asr.realigning_lm_parameters.beta} BEAM WIDTH: {cfg.diarizer.asr.realigning_lm_parameters.beam_width} Word Window: {cfg.diarizer.asr.realigning_lm_parameters.word_window} Use Ngram: {cfg.diarizer.asr.realigning_lm_parameters.use_ngram} Chunk Word Len: {cfg.diarizer.asr.realigning_lm_parameters.parallel_chunk_word_len} \
            \nSpeakerLM Model: {cfg.diarizer.asr.realigning_lm_parameters.arpa_language_model} \
            \nASR MODEL: {cfg.diarizer.asr.model_path}")

if __name__ == '__main__':
    main()