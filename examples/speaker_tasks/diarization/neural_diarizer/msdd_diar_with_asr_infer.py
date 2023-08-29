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


from omegaconf import OmegaConf

from nemo.collections.asr.parts.utils.decoder_timestamps_utils import ASRDecoderTimeStamps
from nemo.collections.asr.parts.utils.diarization_utils import OfflineDiarWithASR
from nemo.core.config import hydra_runner
from nemo.utils import logging
from tqdm import tqdm

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

def divide_chunks(trans_info_dict, diar_logits, win_len, word_window, port):
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
    div_trans_info_dict, div_diar_logits = {}, {}
    for uniq_id in trans_info_dict.keys():
        uniq_trans = trans_info_dict[uniq_id]
        del uniq_trans['status']
        del uniq_trans['transcription']
        del uniq_trans['sentences']
        word_seq = uniq_trans['words']
        logit_seq = diar_logits[uniq_id]['pred_mat']
        
        div_word_seq, div_diar_logits_seq = [], []
        if win_len is None:
            win_len = int(np.ceil(len(word_seq)/num_workers))
        n_chunks = int(np.ceil(len(word_seq)/win_len))
        
        for k in range(n_chunks):
            div_word_seq.append(word_seq[max(k*win_len - word_window, 0):(k+1)*win_len])
            div_diar_logits_seq.append(logit_seq[max(k*win_len - word_window, 0):(k+1)*win_len])
        
        total_count = len(div_word_seq)
        for k, w_seq in enumerate(div_word_seq):
            seq_id = uniq_id + f"{SPLITSYM}{k}{SPLITSYM}{total_count}"
            div_trans_info_dict[seq_id] = dict(uniq_trans)
            div_trans_info_dict[seq_id]['words'] = w_seq
            div_diar_logits[seq_id] = dict(diar_logits[uniq_id])
            div_diar_logits[seq_id]['pred_mat'] = div_diar_logits_seq[k]
    return div_trans_info_dict, div_diar_logits

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
            
def run_mp_beam_search_decoding(asr_diar_offline, trans_info_dict, diar_logits, org_trans_info_dict, div_mp, win_len, word_window, port=None, use_ngram=False):
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
    for uniq_id in tqdm(uniq_id_list, desc="Submitting uniq_id futures", unit="file"):
        if port is not None:
            port_num = port[count % len(port)]    
        else:
            port_num = None
        count += 1
        uniq_trans_info_dict = {uniq_id: trans_info_dict[uniq_id]}
        uniq_diar_logits = {uniq_id: diar_logits[uniq_id]}
        futures.append(tp.submit(asr_diar_offline.beam_search_diarization, uniq_trans_info_dict, uniq_diar_logits, port_num=port_num))

    pbar = tqdm(total=len(uniq_id_list), desc="Running beam search decoding", unit="files")
    count = 0
    output_trans_info_dict = {}
    for done_future in concurrent.futures.as_completed(futures):
        count += 1
        pbar.update()
        output_trans_info_dict.update(done_future.result())
    tp.shutdown()
    pbar.close() 
    if div_mp:
        output_trans_info_dict = merge_div_inputs(div_trans_info_dict=output_trans_info_dict, 
                                                  org_trans_info_dict=org_trans_info_dict, 
                                                  win_len=win_len, word_window=word_window)
    return output_trans_info_dict
     

def check_results_dir(cfg):
    trans_info_dict, diar_logits = {}, {}
    all_reference, all_hypothesis = [], []
    result_path = cfg.diarizer.out_dir
    manifest_json = audio_rttm_map(manifest=cfg.diarizer.manifest_filepath)
    
    for uniq_id in manifest_json:
        json_trans_path = os.path.join(result_path, "pred_rttms", uniq_id + ".json")
        # logits_path = os.path.join(result_path, cfg.diarizer.msdd_model.parameters.system_name, "pred_logits", uniq_id + ".pkl")
        logits_path = os.path.join(result_path, "pred_logits", uniq_id + ".pkl")
        hyp_rttm_path = os.path.join(result_path, "pred_rttms", uniq_id + ".rttm")
        ref_rttm_path = manifest_json[uniq_id]["rttm_filepath"]
        if os.path.exists(json_trans_path):
            with open(json_trans_path, "r") as file:
                # Load the JSON content into a Python dictionary
                json_trans = json.load(file)
        else:
            logging.info(f"JSON transcript not found for {uniq_id}")
            return None, None, None
        trans_info_dict[uniq_id] = json_trans
        
        ref_labels = rttm_to_labels(ref_rttm_path)
        reference = labels_to_pyannote_object(ref_labels, uniq_name=uniq_id)
        all_reference.append([uniq_id, reference])
        
        hyp_labels = rttm_to_labels(hyp_rttm_path)
        hypothesis = labels_to_pyannote_object(hyp_labels, uniq_name=uniq_id)
        all_hypothesis.append([uniq_id, hypothesis])    
        
        if os.path.exists(logits_path):
            with open(logits_path, 'rb') as f:
                logit_dict = pickle.load(f)
        else:
            logging.info(f"Diarizaiton Logits not found for {uniq_id}")
            return None, None, None
        logit_dict['diar_labels'] = hyp_labels
        diar_logits[uniq_id] = logit_dict
        
        
    diar_scores = score_labels(
            AUDIO_RTTM_MAP=manifest_json,
            all_reference=all_reference,
            all_hypothesis=all_hypothesis,
            collar=0.25,
            ignore_overlap=False,
            verbose=True,
        )
    return trans_info_dict, diar_scores, diar_logits

@hydra_runner(config_path="../conf/inference", config_name="diar_infer_meeting.yaml")
def main(cfg):
    cfg_copy = copy.deepcopy(cfg)
    
    # logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
       
    trans_info_dict, diar_scores, diar_logits = check_results_dir(cfg)
    # trans_info_dict, diar_scores, diar_logits = None, None, None
    org_trans_info_dict = deepcopy(trans_info_dict) 
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
        
        trans_info_dict = asr_diar_offline.get_transcript_with_speaker_labels(diar_hyp, word_hyp, word_ts_hyp, diar_logits=diar_logits)
    else:
        asr_diar_offline = OfflineDiarWithASR(cfg.diarizer) 
        use_mp = True
        # use_mp = False
        div_mp = True
        # div_mp = False
        if cfg.diarizer.asr.realigning_lm_parameters.use_mp:
            if cfg.diarizer.asr.realigning_lm_parameters.use_chunk_mp:
                div_trans_info_dict, div_diar_logits = divide_chunks(trans_info_dict, 
                                                                 diar_logits, 
                                                                 win_len=cfg.diarizer.asr.realigning_lm_parameters.parallel_chunk_word_len, 
                                                                 word_window=cfg.diarizer.asr.realigning_lm_parameters.word_window,
                                                                 port=cfg.diarizer.asr.realigning_lm_parameters.port,)
                
                trans_info_dict = run_mp_beam_search_decoding(asr_diar_offline, 
                                                              trans_info_dict=div_trans_info_dict, 
                                                              diar_logits=div_diar_logits, 
                                                              org_trans_info_dict=trans_info_dict, 
                                                              div_mp=True,
                                                              win_len=cfg.diarizer.asr.realigning_lm_parameters.parallel_chunk_word_len,
                                                              word_window=cfg.diarizer.asr.realigning_lm_parameters.word_window,
                                                              port=cfg.diarizer.asr.realigning_lm_parameters.port,
                                                              use_ngram=cfg.diarizer.asr.realigning_lm_parameters.use_ngram,
                                                              )
                for uniq_id, session_trans_dict in trans_info_dict.items():
                    asr_diar_offline._make_json_output(uniq_id=uniq_id, 
                                                       diar_labels=diar_logits[uniq_id]['diar_labels'], 
                                                       word_dict_seq_list=trans_info_dict[uniq_id]['words'], 
                                                       write_files=True)
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
            trans_info_dict = asr_diar_offline.beam_search_diarization(trans_info_dict, diar_logits, port_num=port_num)
                
    # If RTTM is provided and DER evaluation
    if diar_scores is not None:
        metric, mapping_dict, _ = diar_scores

        # Get session-level diarization error rate and speaker counting error
        der_results = OfflineDiarWithASR.gather_eval_results(
            diar_score=diar_scores,
            audio_rttm_map_dict=asr_diar_offline.AUDIO_RTTM_MAP,
            trans_info_dict=trans_info_dict,
            root_path=asr_diar_offline.root_path,
        )

        # Calculate WER and cpWER if reference CTM files exist
        wer_results = OfflineDiarWithASR.evaluate(
            hyp_trans_info_dict=trans_info_dict,
            audio_file_list=asr_diar_offline.audio_file_list,
            ref_ctm_file_list=asr_diar_offline.ctm_file_list,
            mapping_dict=mapping_dict,
        )

        # Print average DER, WER and cpWER
        OfflineDiarWithASR.print_errors(der_results=der_results, wer_results=wer_results)

        # Save detailed session-level evaluation results in `root_path`.
        OfflineDiarWithASR.write_session_level_result_in_csv(
            der_results=der_results,
            wer_results=wer_results,
            root_path=asr_diar_offline.root_path,
            csv_columns=asr_diar_offline.csv_columns,
        )
        
        print(f"ALPHA: {cfg.diarizer.asr.realigning_lm_parameters.alpha} BETA: {cfg.diarizer.asr.realigning_lm_parameters.beta} BEAM WIDTH: {cfg.diarizer.asr.realigning_lm_parameters.beam_width} Word Window: {cfg.diarizer.asr.realigning_lm_parameters.word_window} Use Ngram: {cfg.diarizer.asr.realigning_lm_parameters.use_ngram} Chunk Word Len: {cfg.diarizer.asr.realigning_lm_parameters.parallel_chunk_word_len} \
                \nSpeakerLM Model: {cfg.diarizer.asr.realigning_lm_parameters.arpa_language_model} \
                \nASR MODEL: {cfg.diarizer.asr.model_path}")

if __name__ == '__main__':
    main()
