# Copyright 2021-present Kensho Technologies, LLC.
from __future__ import annotations, division

import functools
import heapq
import hashlib
import logging
import math
import multiprocessing as mp
from multiprocessing.pool import Pool
import os
from pathlib import Path
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from copy import deepcopy
import numpy as np
from numpy.typing import NBitBase, NDArray
from scipy.special import softmax
import re
from tqdm import tqdm 
from functools import wraps
import time


from .alphabet import BPE_TOKEN, Alphabet, verify_alphabet_coverage
from .constants import (
    DEFAULT_ALPHA,
    DEFAULT_BEAM_WIDTH,
    DEFAULT_BETA,
    DEFAULT_HOTWORD_WEIGHT,
    DEFAULT_MIN_TOKEN_LOGP,
    DEFAULT_PRUNE_BEAMS,
    DEFAULT_PRUNE_LOGP,
    DEFAULT_SCORE_LM_BOUNDARY,
    DEFAULT_UNK_LOGP_OFFSET,
    MIN_TOKEN_CLIP_P,
)
DEFAULT_BEAM_WIDTH=10
LOG_MIN_VAL = 1e-12
from .language_model import (
    AbstractLanguageModel,
    HotwordScorer,
    LanguageModel,
    load_unigram_set_from_arpa,
)


logger = logging.getLogger(__name__)

try:
    import kenlm  # type: ignore
except ImportError:
    logger.warning(
        "kenlm python bindings are not installed. Most likely you want to install it using: "
        "pip install https://github.com/kpu/kenlm/archive/master.zip"
    )
try:
    import kenlm

    ARPA = True
    ARPA_STT = '<s>'
    ARPA_END = '</s>'
except ImportError:
    ARPA = False

# type hints
# store frame information for each word, where frame is the logit index of (start_frame, end_frame)
Frames = Tuple[int, int]
WordFrames = Tuple[str, Frames]
# all the beam information we need to keep track of during decoding
# text, next_word, partial_word, last_char, text_frames, part_frames, logit_score
Beam = Tuple[str, str, str, Optional[str], List[Frames], Frames, float]
# same as BEAMS but with current lm score that will be discarded again after sorting
LMBeam = Tuple[str, str, str, Optional[str], List[Frames], Frames, float, float]
# lm state supports single and multi language model
LMState = Optional[Union["kenlm.State", List["kenlm.State"]]]
# for output beams we return the text, the scores, the lm state and the word frame indices
# text, last_lm_state, text_frames, logit_score, lm_score
OutputBeam = Tuple[str, LMState, List[WordFrames], float, float]
# for multiprocessing we need to remove kenlm state since it can't be pickled
OutputBeamMPSafe = Tuple[str, List[WordFrames], float, float]
# Key for the language model score cache
# text, is_eos
LMScoreCacheKey = Tuple[str, bool]
# LM score with hotword score, raw LM score, LMState
LMScoreCacheValue = Tuple[float, float, LMState]

# constants
NULL_FRAMES: Frames = (-1, -1)  # placeholder that gets replaced with positive integer frame indices
EMPTY_START_BEAM: Beam = ("", "", "", None, [], NULL_FRAMES, 0.0)


# Generic float type
if sys.version_info < (3, 8):
    NpFloat = Any
else:
    if sys.version_info < (3, 9) and not TYPE_CHECKING:
        NpFloat = Any
    else:
        NpFloat = np.floating[NBitBase]

FloatVar = TypeVar("FloatVar", bound=NpFloat)
Shape = TypeVar("Shape")




import json
import requests
import time
from pprint import pprint 

def rindex(mylist, myvalue):
    return len(mylist) - mylist[::-1].index(myvalue) - 1

def rindex_incl(mylist, myvalue):
    for i, item in enumerate(reversed(mylist)):
        if myvalue in item:
            return len(mylist) - i - 1
    return 0

def request_data(data, port_num=5555):
    headers = {"Content-Type": "application/json"}
    resp = requests.put('http://localhost:{}/generate'.format(port_num),
			data=json.dumps(data),
			headers=headers)
    if type(resp.json()) != dict:
        raise ValueError("Error: {}".format(resp.json()))
    
    if 'word_probs' in resp.json():
        probs = resp.json()['word_probs']
    elif 'spk_probs' in resp.json():
        probs = resp.json()['spk_probs']
    else:
        raise ValueError("Error: No such keys as word_probs or spk_probs {}".format(resp.json()))

    resp_dict = resp.json()
    return resp_dict

def send_chat(prompt_var, tokens_to_generate, port):
    task_hash = [] 
    # for prompt in prompt_var:
    #     task_hash.append(hashlib.md5(prompt.encode()).hexdigest())
    if type(prompt_var) == list:
        prompt = prompt_var
    else:
        prompt = [prompt_var]
    data = {
        "sentences": prompt,
        # "task_ids": task_hash,
        "tokens_to_generate": tokens_to_generate,
        "temperature": 0.75,
        "add_BOS": True,
        "top_k": 0.99,
        "top_p": 0.0,
        "greedy": True,
        "all_probs": True,
        "repetition_penalty": 2.5,
        "compute_logprob": True,
        "min_tokens_to_generate": 2,
    }
    # sentences, resp_dict = request_data(data, port_num=port)
    # return sentences
    resp_dict = request_data(data, port_num=port)
    return resp_dict



def timeit(func):
    @wraps(func)
    def wrapper(*args):
        start_time = time.time();
        retval = func(*args)
        # print(f"the function ends in {time.time()-start_time:.6f}, secs")
        return retval
    return wrapper

class SpeakerToWordAlignerLM:
    def __init__(self, realigning_lm, beta: float=0.1) -> None:
        if type(realigning_lm) == str:
            self.realigning_lm = kenlm.LanguageModel(realigning_lm)
        else:
            self.realigning_lm = realigning_lm
        self.beta = beta
    
    def update_transcript_status(self, spk_trans_dict, spk_label, prev_spk, word):
        if spk_trans_dict[spk_label] == '': # If the sentence is empty
            spk_trans_dict[spk_label] = f" {ARPA_STT} {word}"
        elif spk_label == prev_spk:
            spk_trans_dict[spk_label] += f" {word}"
        elif spk_label != prev_spk and prev_spk is not None:
            if spk_trans_dict[prev_spk].split()[-1] != ARPA_END:
                spk_trans_dict[prev_spk] += f" {ARPA_END}"
            
            if spk_trans_dict[spk_label].split()[-1] == ARPA_END:
                spk_trans_dict[spk_label] += f" {ARPA_STT} {word}"
            else:
                spk_trans_dict[spk_label] += f" {word}"
        return spk_trans_dict
    
    def _get_windowed_context_ngram(self, all_spk_trans: dict , num_spks: int , max_word: int):
        # trunc_trans = all_spk_trans.split()[-(num_spks*max_word):]
        trunc_trans = all_spk_trans.split()[-(max_word):]
        if trunc_trans[0] not in [ARPA_STT, ARPA_END]:
            trunc_trans.insert(0, ARPA_STT)
        elif trunc_trans[0] == ARPA_END:
            trunc_trans = trunc_trans[1:]
        return trunc_trans
    
    def _get_windowed_context_llm(self, prompt_trans: dict , num_spks: int , max_word: int):
        split_trans = prompt_trans.split()
        # ridx = rindex_incl(split_trans[:-(num_spks*max_word)], '[speaker')
        ridx = rindex_incl(split_trans[:-(max_word)], '[speaker')
        speaker_symbol = split_trans[ridx]
        trunc_trans = split_trans[-(num_spks*max_word):]
        if '[speaker' not in trunc_trans[0]:
            trunc_trans.insert(0, speaker_symbol)
        return trunc_trans
    
    def get_spk_wise_probs(
        self, 
        next_word: str, 
        last_char: str,
        next_char: str,
        spk_trans_dict: Dict[str, str],
        max_word: int = 30,
        alpha: float = 0.2
        ):
        """
        Get speaker-wise probabilities for the given word.

        Args:
            word (str): The next word to be added to the sentence.
            spk_trans_dict (Dict[str, str]): The hypothesis sentence for each speaker.
            speaker_list (List[str]): The list of speakers.
            max_word (int, optional): The window of words to be considered for the realignment. Defaults to 20.

        Returns:
            lm_spk_probs (List[float]): The speaker-wise probabilities for the given word.
        """
        if last_char is not None: 
            last_spk = int(last_char.split('_')[-1])
        else:
            last_spk = None 
        if next_char is not None: 
            next_spk = int(next_char.split('_')[-1])
        else:
            next_spk = None 
            
        speaker_list = _get_speaker_list(spk_trans_dict)
        num_spks = len(speaker_list)
        hyp_individ_dict = {spk: {tgt_spk: '' for tgt_spk in speaker_list} for spk in speaker_list}
        hyp_allspks_dict = {spk: '' for spk in speaker_list}
        hyp_prompt_dict = {spk: '' for spk in speaker_list}
        hyp_probs = {spk: 0 for spk in speaker_list}
        hyp_base_individ_probs = {spk: {tgt_spk: 0 for tgt_spk in speaker_list} for spk in speaker_list}
        hyp_base_allspks_probs = {spk: 0 for spk in speaker_list}
        
        for spk in speaker_list:
            PROMPT_STT, PROMPT_END = f'[speaker{spk}]:', ""
            # if len(spk_trans_dict['prompt'].split()) < 100:
            #     allspks_word_list = spk_trans_dict['all_spks'].split()[-(num_spks*max_word):]
            #     prompt_word_list = spk_trans_dict['prompt'].split()[-(num_spks*max_word):]
            # else:
            allspks_word_list = self._get_windowed_context_ngram(spk_trans_dict['all_spks'], num_spks, max_word) 
            prompt_word_list = self._get_windowed_context_llm(spk_trans_dict['prompt'], num_spks, max_word)  
            
            truncated_allspks_words = " ".join(allspks_word_list)
            truncated_prompt_words = " ".join(prompt_word_list)
            hyp_base_allspks_probs[spk] = self.realigning_lm.score(truncated_allspks_words)
            if not spk_trans_dict['all_spks'] == '' and len(truncated_allspks_words.split()) > 0 and truncated_allspks_words.split()[-1] == ARPA_END:
                truncated_allspks_words = re.sub(f' {ARPA_END}$', '', truncated_allspks_words)
            
            if last_spk is not None and last_spk == spk:
                if truncated_allspks_words == '':
                    truncated_allspks_words += f"{ARPA_STT}"
                    truncated_prompt_words += f"{PROMPT_STT}"
                hyp_allspks_dict[spk] = truncated_allspks_words + f" {next_word} {ARPA_END}"
                hyp_prompt_dict[spk] = truncated_prompt_words + f" {next_word} {PROMPT_END}"
            
            elif last_spk != spk:
                if truncated_allspks_words == '':
                    hyp_allspks_dict[spk] = truncated_allspks_words + f"{ARPA_STT} {next_word} {ARPA_END}"
                    hyp_prompt_dict[spk] = truncated_prompt_words + f"{PROMPT_STT} {next_word} {PROMPT_END}"
                else:
                    hyp_allspks_dict[spk] = truncated_allspks_words + f" {ARPA_END} {ARPA_STT} {next_word} {ARPA_END}"
                    hyp_prompt_dict[spk] = truncated_prompt_words + f" {PROMPT_END}{PROMPT_STT} {next_word} {PROMPT_END}"
            
            hyp_prompt_dict[spk] = hyp_prompt_dict[spk].replace(f"[speaker", f"\n[speaker")
                
            for tgt_spk in speaker_list:
                individ_word_list = spk_trans_dict[tgt_spk].split()[-max_word:]
                truncated_individ_words = " ".join(individ_word_list)
                hyp_base_individ_probs[spk][tgt_spk] = self.realigning_lm.score(truncated_individ_words)
                if not spk_trans_dict[tgt_spk] == '' and len(truncated_individ_words.split()) > 0 and truncated_individ_words.split()[-1] == ARPA_END:
                    truncated_individ_words = re.sub(f' {ARPA_END}$', '', truncated_individ_words)
                if tgt_spk == spk:
                    if truncated_individ_words == '':
                        truncated_individ_words += f"{ARPA_STT}"
                    hyp_individ_dict[spk][tgt_spk] = truncated_individ_words + f" {next_word} {ARPA_END}"
                elif tgt_spk != spk and truncated_individ_words != '':
                    hyp_individ_dict[spk][tgt_spk] = truncated_individ_words + f" {ARPA_END}"
            
            indiv_spk_probs = []            
            for tgt_spk in speaker_list:
                sentence = hyp_individ_dict[spk][tgt_spk]
                prob = self.realigning_lm.score(sentence) - hyp_base_individ_probs[spk][tgt_spk]
                indiv_spk_probs.append(prob)
            
            all_spks_sentence = hyp_allspks_dict[spk]
            all_spks_prob = self.realigning_lm.score(all_spks_sentence) - hyp_base_allspks_probs[spk]
            hyp_probs[spk] += sum(indiv_spk_probs) 
            
        lm_spk_logits = [hyp_probs[spk] for spk in sorted(hyp_probs)]
        lm_spk_probs = np.power(10, lm_spk_logits)
        lm_spk_probs_sm = lm_spk_probs/np.sum(lm_spk_probs)
        # confidence = np.power(np.max(lm_spk_probs), alpha)
        confidence = np.power(lm_spk_probs.sum(), alpha)
        return lm_spk_probs_sm, confidence, hyp_prompt_dict
    
    def update_transcript_status(self, spk_trans_dict, spk_label, prev_spk, word):
        if spk_trans_dict[spk_label] == '': # If the sentence is empty
            spk_trans_dict[spk_label] = f" {ARPA_STT} {word}"
        elif spk_label == prev_spk:
            spk_trans_dict[spk_label] += f" {word}"
        elif spk_label != prev_spk and prev_spk is not None:
            if spk_trans_dict[prev_spk].split()[-1] != ARPA_END:
                spk_trans_dict[prev_spk] += f" {ARPA_END}"
            
            if spk_trans_dict[spk_label].split()[-1] == ARPA_END:
                spk_trans_dict[spk_label] += f" {ARPA_STT} {word}"
            else:
                spk_trans_dict[spk_label] += f" {word}"
        return spk_trans_dict

def _get_speaker_list(spk_trans_dict: Dict[str, str]) -> List[str]:
    """ 
    Get the list of speakers from the transcript dictionary.
    
    Args:
        spk_trans_dict: The transcript dictionary for each speaker and merged transcript.
        
    Returns:
        speaker_list: The sorted version of list of speakers.
    """
    speaker_list = []
    for spk in spk_trans_dict.keys():
        if type(spk) == int:
            speaker_list.append(spk)
    return sorted(speaker_list)
    

def _get_valid_pool(pool: Optional[Pool]) -> Optional[Pool]:
    """Return the pool if the pool is appropriate for multiprocessing."""
    if pool is not None and isinstance(
        pool._ctx, mp.context.SpawnContext  # type: ignore [attr-defined] # pylint: disable=W0212
    ):
        logger.warning(
            "Specified pool object has a spawn context, which is not currently supported. "
            "See https://github.com/kensho-technologies/pyctcdecode/issues/65."
            "\nFalling back to sequential decoding."
        )
        return None
    return pool


def _normalize_whitespace(text: str) -> str:
    """Efficiently normalize whitespace."""
    return " ".join(text.split())


def _sort_and_trim_beams(beams: List[LMBeam], beam_width: int) -> List[LMBeam]:
    """Take top N beams by score."""
    return heapq.nlargest(beam_width, beams, key=lambda x: x[-1])


def _sum_log_scores(s1: float, s2: float) -> float:
    """Sum log odds in a numerically stable way."""
    # this is slightly faster than using max
    if s1 >= s2:
        log_sum = s1 + math.log(1 + math.exp(s2 - s1))
    else:
        log_sum = s2 + math.log(1 + math.exp(s1 - s2))
    return log_sum


def _log_softmax(
    x: np.ndarray[Shape, np.dtype[FloatVar]],
    axis: Optional[int] = None,
) -> np.ndarray[Shape, np.dtype[FloatVar]]:
    """Logarithm of softmax function, following implementation of scipy.special."""
    x_max = np.amax(x, axis=axis, keepdims=True)
    if x_max.ndim > 0:
        x_max[~np.isfinite(x_max)] = 0
    elif not np.isfinite(x_max):
        x_max = 0  # pylint: disable=R0204
    tmp = x - x_max
    exp_tmp: np.ndarray[Shape, np.dtype[FloatVar]] = np.exp(tmp)
    # suppress warnings about log of zero
    with np.errstate(divide="ignore"):
        s = np.sum(exp_tmp, axis=axis, keepdims=True)
        out: np.ndarray[Shape, np.dtype[FloatVar]] = np.log(s)
    out = tmp - out
    return out


def _merge_tokens(token_1: str, token_2: str) -> str:
    """Fast, whitespace safe merging of tokens."""
    if len(token_2) == 0:
        text = token_1
    elif len(token_1) == 0:
        text = token_2
    else:
        text = token_1 + " " + token_2
    return text

def get_concat_transcripts(spk_trans_dict, word_window: int=25):
    concat_text = ''
    # Concatenate all the transcripts for all speakers
    speakers_list = []
    for spk in spk_trans_dict.keys():
        if type(spk) == int:
            speakers_list.append(spk)
    speakers_list = sorted(speakers_list)
    for spk in speakers_list:
        # concat_text += spk_trans_dict[spk].replace(f"{ARPA_STT}", '').replace(f" {ARPA_STT}", '').replace(f" {ARPA_END}", '')
        spk_trans = " ".join(spk_trans_dict[spk].split()[-word_window:])
        spk_trans = re.sub(' +', ' ', spk_trans)
        concat_text += f" [SPK{spk}] {spk_trans} "
    concat_text = re.sub(' +', ' ', concat_text).strip()
    return concat_text

def _merge_speaker_beams(beams: List[Beam], word_window) -> List[Beam]:
    """Merge beams with same prefix together."""
    beam_dict = {}
    # print(f"At _merge_speaker_beams - Incoming Beam Size: {len(beams)}")
    # for spk_trans_dict, next_word, word_part, last_char, text_frames, part_frames, logit_score in beams:
    for spk_trans_dict, next_word, text_frames, last_char, next_char, logit_score in beams:
        concat_text = get_concat_transcripts(spk_trans_dict=spk_trans_dict, word_window=word_window)
        # concat_text_hash = self._get_transcript_hash(concat_text=concat_text)
        concat_text_hash = hashlib.md5(concat_text.encode()).hexdigest()
        hash_idx = (concat_text_hash, next_word, last_char, next_char)
        # hash_idx = (concat_text_hash, next_word, next_char)
        if hash_idx not in beam_dict:
            beam_dict[hash_idx] = (
                spk_trans_dict,
                next_word,
                text_frames,
                last_char,
                next_char,
                logit_score,
            )
        else:
            beam_dict[hash_idx] = (
                spk_trans_dict,
                next_word,
                text_frames,
                last_char,
                next_char,
                _sum_log_scores(beam_dict[hash_idx][-1], logit_score),
            )
    return list(beam_dict.values())


def _prune_history(beams: List[LMBeam], lm_order: int, word_window: int) -> List[Beam]:
    """Filter out beams that are the same over max_ngram history.

    Since n-gram language models have a finite history when scoring a new token, we can use that
    fact to prune beams that only differ early on (more than n tokens in the past) and keep only the
    higher scoring ones. Note that this helps speed up the decoding process but comes at the cost of
    some amount of beam diversity. If more than the top beam is used in the output it should
    potentially be disabled.
    """
    # let's keep at least 1 word of history
    min_n_history = max(1, lm_order - 1)
    seen_hashes = set()
    filtered_beams = []
    # for each beam after this, check if we need to add it
    # for spk_trans_dict, next_word, word_part, last_char, text_frames, part_frames, logit_score_before, logit_score in beams:
    for spk_trans_dict, next_word, text_frames, last_char, next_char, logit_score_before, logit_score in beams:
        concat_text = get_concat_transcripts(spk_trans_dict=spk_trans_dict, word_window=word_window)
        # concat_text_hash = self._get_transcript_hash(concat_text=concat_text)
        concat_text_hash = hashlib.md5(concat_text.encode()).hexdigest()
        hash_idx = (concat_text_hash, next_word, next_char)
        if hash_idx not in seen_hashes:
            filtered_beams.append(
                (
                spk_trans_dict,
                next_word,
                text_frames,
                last_char,
                next_char,
                logit_score
                )
            )
            seen_hashes.add(hash_idx)
    return filtered_beams

def find_prev_spk_int(spk_trans_dict, all_spks):
    """
    Find the previous speaker integer for the given speaker.

    Args:
        spk_trans_dict (_type_): 
        all_spks (_type_): _description_

    Returns:
        _type_: _description_
    """
    rfind = spk_trans_dict[all_spks].rfind(ARPA_STT[:2])
    if rfind != -1:
        prev_spk_int = spk_trans_dict[all_spks][rfind:].split()[0].replace("<s", "").replace(">", "")
    else:
        prev_spk_int = None
    return prev_spk_int

class BeamSearchDecoderCTC:
    # Note that we store the language model (large object) as a class variable.
    # The advantage of this is that during multiprocessing they won't cause and overhead in time.
    # This somewhat breaks conventional garbage collection which is why there are
    # specific functions for cleaning up the class variables manually if space needs to be freed up.
    # Specifically we create a random dictionary key during object instantiation which becomes the
    # storage key for the class variable model_container. This allows for multiple model instances
    # to be loaded at the same time.
    model_container: Dict[bytes, Optional[AbstractLanguageModel]] = {}

    # serialization filenames
    _ALPHABET_SERIALIZED_FILENAME = "alphabet.json"
    _LANGUAGE_MODEL_SERIALIZED_DIRECTORY = "language_model"

    def __init__(
        self,
        alphabet: Alphabet,
        alpha: float = 0.5,
        beta: float = 0.1,
        word_window: int = 25,
        use_ngram: bool = True,
        build_up_len: int = 50,
        language_model: Optional[AbstractLanguageModel] = None,
        port: int = None,
    ) -> None:
        """CTC beam search decoder for token logit matrix.

        Args:
            alphabet: class containing the labels for input logit matrices
            language_model: convenience class to store language model functionality
        """
        self._alphabet = alphabet
        # self._idx2speaker = {n: c for n, c in enumerate(self._alphabet.labels)}
        self.max_num_speakers = 4
        self._idx2speaker = {k : f"speaker_{k}" for k in range(self.max_num_speakers)}
        # self._is_bpe = alphabet.is_bpe
        self._model_key = os.urandom(16)
        self.alpha = alpha
        self.beta = beta
        self.use_ngram = use_ngram
        self.build_up_len=build_up_len
        self.word_window = word_window
        self.gamma = 0.9
        self.port = port
        self.spk_lm_decoder = SpeakerToWordAlignerLM(realigning_lm=language_model, beta=self.beta)
        BeamSearchDecoderCTC.model_container[self._model_key] = language_model

    def reset_params(
        self,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        word_window: Optional[int] = None,
        unk_score_offset: Optional[float] = None,
        lm_score_boundary: Optional[bool] = None,
    ) -> None:
        """Reset parameters that don't require re-instantiating the model."""
        # todo: make more generic to accomodate other language models
        language_model = self._language_model
        if language_model is None:
            return
        params: Dict[str, Any] = {}
        if alpha is not None:
            params["alpha"] = alpha
        if word_window is not None:
            params["word_window"] = word_window
        if beta is not None:
            params["beta"] = beta
        if unk_score_offset is not None:
            params["unk_score_offset"] = unk_score_offset
        if lm_score_boundary is not None:
            params["score_boundary"] = lm_score_boundary
        language_model.reset_params(**params)

    @classmethod
    def clear_class_models(cls) -> None:
        """Clear all models from class variable."""
        cls.model_container = {}

    def cleanup(self) -> None:
        """Manual cleanup of models in class variable."""
        if self._model_key in BeamSearchDecoderCTC.model_container:
            del BeamSearchDecoderCTC.model_container[self._model_key]

    @property
    def _language_model(self) -> Optional[AbstractLanguageModel]:
        """Retrieve the language model."""
        return BeamSearchDecoderCTC.model_container[self._model_key]
    
    def _infer_spk_probs_from_lm(self, batch_input):
        # print(f"self._infer_spk_probs_from_lm(): batch_input length: {len(batch_input)}")
        if len(batch_input) > 10:
            pass
        return None

    # def _get_prev_info(self, spk_int: int, target_text: str, prev_trans_dict: Dict[str, str]) -> Tuple[Dict[str, str], str]:
    def _get_prev_info(self, spk_int: int, prev_trans_dict: Dict[str, str]) -> Tuple[Dict[str, str], str]:
        """
        Get the previous word and transcript for the given speaker.

        Args:
            spk_int (int): 
            target_text (str): 
            prev_trans_dict (Dict[str, str]): 

        Returns:
            prev_trans_dict (Dict[str, str]):

            prev_word (str):

        """
        speaker_list = []
        for curr_int in prev_trans_dict.keys(): 
            if type(curr_int) == str:
                continue
            speaker_list.append(curr_int)
        concat_prev_text = ""
        new_prev_trans_dict = {}
        for curr_int in sorted(speaker_list):
            target_text = prev_trans_dict[curr_int]
            if curr_int == spk_int:
                if target_text != '' and len(target_text.split()) > 1:
                    new_prev_trans_dict[spk_int] = " ".join(target_text.split()[:-1])
                    prev_word = target_text.split()[-1]
                elif len(target_text.split()) == 1:
                    new_prev_trans_dict[spk_int] = ''
                    prev_word = target_text.split()[0]
                elif target_text == '':
                    new_prev_trans_dict[spk_int] = ''
                    prev_word = ''
            else:
                new_prev_trans_dict[curr_int] = target_text
            concat_prev_text += f"[SPK{curr_int}] {new_prev_trans_dict[curr_int]} "
        concat_prev_text = re.sub(' +', ' ', concat_prev_text).strip()
        return concat_prev_text, prev_word
        
    def _get_transcript_hash(self, text: str):
        """
        Get the hash of the given transcript.
        
        Args:
            text (str): The transcript to be hashed.
            
        Returns:
            hash (str): The hash of the transcript.
        """
        return hashlib.md5(text.encode()).hexdigest()
    

    def get_prompt(self, spk_trans_dict, next_word, speaker_list, spk_int, prompt_dict, num_spks, max_word):
        EOD = "[end]"
        ASSI = "Answer:[speaker"
        next_word = f"({next_word})"
        speaker_list_str = " or ".join([f"[{spk_id.replace('_','')}]" for spk_id in speaker_list ])
        QUESTION_STR = f"Question: The next word is {next_word}. Who spoke {next_word} ?"
        prompt_word_list = self.spk_lm_decoder._get_windowed_context_llm(spk_trans_dict['prompt'], num_spks, max_word)  
        transcript_prompt = " ".join(prompt_word_list)
        spk_prompt_str = f"{transcript_prompt} {EOD} {QUESTION_STR} \n{ASSI}"
        word_prompt_str = f"{prompt_dict[spk_int]}"
        return spk_prompt_str, word_prompt_str
    
    @timeit 
    def _get_speaker_lm_beams_nvllm(
        self,
        speaker_list,
        beams: List[Beam],
        hotword_scorer: HotwordScorer,
        cached_lm_scores: Dict[LMScoreCacheKey, LMScoreCacheValue],
        cached_partial_token_scores: Dict[str, float],
    ) -> List[LMBeam]:
        """Update score by averaging logit_score and lm_score."""
        # get language model and see if exists
        language_model = self._language_model
        new_beams, batch_input = [], []
        spk_prompts = []
        word_prompts = []
        spk_probs_ngram = []
        word_probs_ngram = []
        for spk_trans_dict, next_word, text_frames, last_char, next_char, logit_score in beams:
            # fast token merge
            concat_text = get_concat_transcripts(spk_trans_dict=spk_trans_dict)
            transcript_hash =  self._get_transcript_hash(text=concat_text) 
            cache_key = (transcript_hash, next_word) 
            next_spk_int = int(next_char.split('_')[-1])
            # if cache_key not in cached_lm_scores:
            # Remove all non-word tokens from the transcript
            concat_prev_text, prev_word = self._get_prev_info(spk_int=next_spk_int, 
                                                                prev_trans_dict=spk_trans_dict)
            transcript_hash =  self._get_transcript_hash(text=concat_prev_text) 
            batch_input.append((cache_key, next_word, last_char, next_char, next_spk_int, spk_trans_dict))
            num_spks = len(_get_speaker_list(spk_trans_dict))
            spk_wise_probs, confidence, prompt_dict = self.spk_lm_decoder.get_spk_wise_probs(next_word=next_word, 
                                                                                            last_char=last_char, 
                                                                                            next_char=next_char,
                                                                                            spk_trans_dict=spk_trans_dict,
                                                                                            max_word=self.word_window,
                                                                                            alpha=self.alpha,
                                                                                            )
            spk_probs_ngram.append(spk_wise_probs)
            word_probs_ngram.append(confidence)
            
            spk_prompt_str, word_prompt_str = self.get_prompt(spk_trans_dict, next_word, speaker_list, next_spk_int, prompt_dict, num_spks, max_word=self.word_window)
                
            spk_prompts.append(spk_prompt_str)
            word_prompts.append(word_prompt_str)

        
        # cached_lm_scores = self.run_batched_inference(batch_input=batch_input)
        use_one_infer = False
        try:
            if use_one_infer:
                resp_dict = send_chat(spk_prompts, tokens_to_generate=4, port=self.port)
                spk_probs, word_probs = resp_dict['spk_probs'], resp_dict['word_probs']
            else:
                resp_dict_spk = send_chat(spk_prompts, tokens_to_generate=4, port=self.port)
                resp_dict_word = send_chat(word_prompts, tokens_to_generate=1, port=self.port)
                spk_probs, word_probs = resp_dict_spk['spk_probs'], resp_dict_word['word_probs']
        
            # print(f"Time taken for spk_probs: {(time.time() - stt1):.4f} s, len(spk_prompts[0].split()): {len(spk_prompts[0].split())}")
            # stt2 = time.time()
            # word_probs = send_chat(word_prompts, tokens_to_generate=1, port=self.port)
            # print(f"Time taken for word_probs: {(time.time() - stt2):.4f} s, len(word_prompts[0].split()): {len(word_prompts[0].split())}")
        except:
            logging.info("[Error] Failed sending chat to nvllm server. Falling back to ngram LM.")
            spk_probs, word_probs = spk_probs_ngram, word_probs_ngram
            # raise ValueError("Failed sending chat to nvllm server. Falling back to ngram LM.")
            
                
        if len(spk_trans_dict['all_spks'].split()) < self.word_window:
            self._beta_flag = 1.0
        else:
            self._beta_flag = 1.0
                
        for idx, (spk_trans_dict, next_word, text_frames, last_char, next_char, logit_score) in enumerate(beams):
            # _, _, spk_wise_probs, confidence  = cached_lm_scores[cache_key]
            spk_wise_probs, confidence = spk_probs[idx], word_probs[idx]
            # spk_wise_probs, confidence = spk_probs_ngram[idx], word_probs_ngram[idx]

            lm_score = np.log(max(confidence * spk_wise_probs[next_spk_int], LOG_MIN_VAL))
            spk_idx_char = int(last_char.split('_')[-1])
            new_beams.append(
                (
                    spk_trans_dict,
                    next_word,
                    text_frames,
                    last_char,
                    next_char,
                    logit_score,
                    logit_score + self._beta_flag * self.beta * lm_score,
                )
            )
        return new_beams 
    
    @timeit 
    def _get_speaker_lm_beams(
        self,
        speaker_list,
        beams: List[Beam],
        hotword_scorer: HotwordScorer,
        cached_lm_scores: Dict[LMScoreCacheKey, LMScoreCacheValue],
        cached_partial_token_scores: Dict[str, float],
    ) -> List[LMBeam]:
        """Update score by averaging logit_score and lm_score."""
        # get language model and see if exists
        language_model = self._language_model
        new_beams, batch_input = [], []
        # for spk_trans_dict, next_word, word_part, last_char, frame_list, frames, logit_score in beams:
        for spk_trans_dict, next_word, text_frames, last_char, next_char, logit_score in beams:
            # fast token merge
            concat_text = get_concat_transcripts(spk_trans_dict=spk_trans_dict)
            transcript_hash =  self._get_transcript_hash(text=concat_text) 
            cache_key = (transcript_hash, next_word) 
            next_spk_int = int(next_char.split('_')[-1])
            if cache_key not in cached_lm_scores:
                # Remove all non-word tokens from the transcript
                concat_prev_text, prev_word = self._get_prev_info(spk_int=next_spk_int, 
                                                                  prev_trans_dict=spk_trans_dict)
                transcript_hash =  self._get_transcript_hash(text=concat_prev_text) 
                batch_input.append((cache_key, next_word, last_char, next_spk_int, spk_trans_dict))
                spk_wise_probs, confidence, prompt_dict = self.spk_lm_decoder.get_spk_wise_probs(next_word=next_word, 
                                                                                    last_char=last_char, 
                                                                                    next_char=next_char,
                                                                                    spk_trans_dict=spk_trans_dict,
                                                                                    max_word=self.word_window,
                                                                                    alpha=self.alpha,
                                                                                    )
                spk_lm_score = np.log(confidence * spk_wise_probs[next_spk_int])
                cached_lm_scores[cache_key] = (spk_lm_score, spk_lm_score, spk_wise_probs, confidence)
                
            if len(spk_trans_dict['all_spks'].split()) < self.word_window:
                self._beta_flag = 1.0
            else:
                self._beta_flag = 1.0
                
            _, _, spk_wise_probs, confidence  = cached_lm_scores[cache_key]
            lm_score = np.log(confidence * spk_wise_probs[next_spk_int])
            spk_idx_char = int(next_char.split('_')[-1])
            new_beams.append(
                (
                    spk_trans_dict,
                    next_word,
                    text_frames,
                    last_char,
                    next_char,
                    logit_score,
                    logit_score + self._beta_flag * self.beta * lm_score,
                )
            )
        # batch_spk_wise_probs = self._infer_spk_probs_from_lm(batch_input=batch_input)
        return new_beams

    def _add_word_to_spk_transcript(
        self, 
        spk_trans_dict, 
        spk_label, 
        prev_spk, 
        word, 
        all_spks = 'all_spks',
        prompt = 'prompt'
        ):
        if type(prev_spk) == str:
            prev_spk = int(prev_spk.split('_')[-1])
            
        if word == '' or word is None or len(word) == 0:
           return spk_trans_dict 
        
        PROMPT_STT = f"[speaker{spk_label}]:"
        PROMPT_END = ""
           
        if spk_label == prev_spk or prev_spk is None:
            # Current speaker transcript
            if spk_trans_dict[spk_label] == '':
                spk_trans_dict[spk_label] = f"{ARPA_STT} {word}"
            else:
                spk_trans_dict[spk_label] += f" {word}"
            
            # All speakers transcript 
            if spk_trans_dict[all_spks] == '':
                spk_trans_dict[all_spks] = f"{ARPA_STT} {word}"
                spk_trans_dict[prompt] = f"{PROMPT_STT} {word}"
            else:
                spk_trans_dict[all_spks] += f" {word}"
                spk_trans_dict[prompt] += f" {word}"
                
        elif spk_label != prev_spk: 
            # Previous speaker transcript
            if prev_spk is not None:
                if spk_trans_dict[prev_spk] == '':
                    pass
                elif spk_trans_dict[prev_spk].split()[-1] == ARPA_END:
                    pass
                elif spk_trans_dict[prev_spk].split()[-1] != ARPA_END:
                    spk_trans_dict[prev_spk] += f" {ARPA_END}"
                
                if spk_trans_dict[spk_label] == '':
                    pass
                elif spk_trans_dict[spk_label].split()[-1] == ARPA_END:
                    pass
                elif spk_trans_dict[spk_label].split()[-1] != ARPA_END:
                    spk_trans_dict[spk_label] += f" {ARPA_END}"
                spk_trans_dict[spk_label] = spk_trans_dict[spk_label].strip()
            
            # New speaker transcript  
            if spk_trans_dict[spk_label] == '':
                spk_trans_dict[spk_label] += f" {ARPA_STT} {word}" 
            elif spk_trans_dict[spk_label].split()[-1] == ARPA_END:
                spk_trans_dict[spk_label] += f" {ARPA_STT} {word}"
            elif spk_trans_dict[spk_label].split()[-1] != ARPA_END:
                spk_trans_dict[spk_label] += f" {ARPA_END} {ARPA_STT} {word}"
            spk_trans_dict[spk_label] = spk_trans_dict[spk_label].strip()
            
            # All speakers transcript 
            ARPA_STT_SPK = f"<s>"
            if spk_trans_dict[all_spks] == '':
                spk_trans_dict[all_spks] += f" {ARPA_STT_SPK} {word}" 
                spk_trans_dict[prompt] += f"{PROMPT_STT} {word}" 
            elif ARPA_END[:-1] in spk_trans_dict[all_spks].split()[-1]:
                spk_trans_dict[all_spks] += f" {ARPA_STT_SPK} {word}"
                spk_trans_dict[prompt] += f"{PROMPT_STT} {word}"
            elif ARPA_END[:-1] not in spk_trans_dict[all_spks].split()[-1]:
                prev_spk_label = find_prev_spk_int(spk_trans_dict, all_spks)
                if prev_spk_label is None:
                    raise ValueError("Previous speaker label is None")
                ARPA_END_SPK = f"</s>"
                spk_trans_dict[all_spks] += f" {ARPA_END_SPK} {ARPA_STT_SPK} {word}"
                spk_trans_dict[prompt] += f" {PROMPT_END}{PROMPT_STT} {word}"
            spk_trans_dict[all_spks] = spk_trans_dict[all_spks].strip()
            # spk_trans_dict[prompt] = spk_trans_dict[prompt]
        return spk_trans_dict

    def _decode_logits(
        self,
        logits: NDArray[NpFloat],
        speaker_list: List[str],
        word_seq: List[Dict[str, float]],
        beam_width: int,
        beam_prune_logp: float,
        token_min_logp: float,
        prune_history: bool,
        hotword_scorer: HotwordScorer,
        lm_start_state: LMState = None,
    ) -> List[OutputBeam]:
        """Perform beam search decoding."""
        # local dictionaries to cache scores during decoding
        # we can pass in an input start state to keep the decoder stateful and working on realtime
        language_model = self._language_model
        cached_lm_scores = {("", False): (0.0, 0.0, lm_start_state)}
        cached_p_lm_scores: Dict[str, float] = {}
        # start with single beam to expand on
        beams = [list(EMPTY_START_BEAM)]
        # bpe we can also have trailing word boundaries ▁⁇▁ so we may need to remember breaks
        # for word_idx, logit_col in enumerate(logits):
        spk_trans_dict = {int(spk_str.split('_')[-1]): '' for spk_str in speaker_list}
        speaker_idx_list = [int(spk_str.split('_')[-1])  for spk_str in speaker_list]
        spk_trans_dict.update({'all_spks': '', 'prompt': ''})
        prev_spk, next_spk = None, None
        # beams[0] = (spk_trans_dict, word_seq[0]['word'], [], prev_spk, next_spk, 0) 
        beams= [(spk_trans_dict, word_seq[0]['word'], [], prev_spk, self._idx2speaker[next_spk], 0) for next_spk in speaker_idx_list]
        # for word_idx, word_dict in enumerate(tqdm(word_seq[:-1])):
        for word_idx, word_dict in enumerate(tqdm(word_seq)):
            logit_col_raw = word_dict['speaker_softmax']
            # logit_col_raw = word_seq[word_idx+1]['speaker_softmax'] if word_idx < len(word_seq)-1 else  word_seq[-1]['speaker_softmax'] 
            logit_col = np.log(np.clip(np.array(logit_col_raw), MIN_TOKEN_CLIP_P, 1))
            max_idx = logit_col.argmax().item()
            # speaker_idx_list = set(np.where(logit_col > token_min_logp)[0]) | {max_idx}
            # speaker_idx_list = speaker_list
            new_beams: List[Beam] = []
            for spk_idx_char in speaker_idx_list:
                next_char = self._idx2speaker[spk_idx_char]
                for (spk_trans_dict,
                     curr_word,
                     text_frames,
                     last_char,
                     curr_char,
                     logit_score,
                    ) in beams:
                    # In the previous beam, next_word was "next word", now it is "current word" curr_word
                    next_word = word_seq[word_idx+1]['word'] if word_idx < len(word_seq)-1 else None
                    curr_spk_int = int(curr_char.split('_')[-1])
                    
                    # Add the word that we calculated LM score for in the previous beam
                    added_spk_trans_dict = self._add_word_to_spk_transcript(spk_trans_dict=deepcopy(spk_trans_dict), 
                                                                            spk_label=curr_spk_int, 
                                                                            prev_spk=last_char, 
                                                                            word=curr_word)
                        
                    spk_logit_char = logit_col[curr_spk_int].item()
                    new_word_dict = deepcopy(word_dict) 
                    new_word_dict['speaker'] = f"speaker_{curr_spk_int}"
                    added_text_frames = text_frames + [new_word_dict]
                    org_last_char = deepcopy(last_char)
                    last_char = curr_char # Update last_char to current char
                    new_beams.append(
                        (
                            added_spk_trans_dict,
                            next_word,
                            added_text_frames,
                            last_char,
                            next_char,
                            logit_score + spk_logit_char,
                        )
                    )

            # lm scoring and beam pruning
            new_beams = _merge_speaker_beams(new_beams, word_window=self.word_window)
            
            # if self.use_ngram or word_idx < 100: 
            if self.use_ngram:
                scored_beams = self._get_speaker_lm_beams(
                    speaker_list,
                    new_beams,
                    hotword_scorer,
                    cached_lm_scores,
                    cached_p_lm_scores,
                )
            else:
                scored_beams = self._get_speaker_lm_beams_nvllm(
                    speaker_list,
                    new_beams,
                    hotword_scorer,
                    cached_lm_scores,
                    cached_p_lm_scores,
                )
            # remove beam outliers
            max_score = max([b[-1] for b in scored_beams])
            min_score = min([b[-1] for b in scored_beams])
            scored_beams_list = [b[-1] for b in scored_beams]
            scored_beams = [b for b in scored_beams if b[-1] >= max_score + beam_prune_logp]
            # beam pruning by taking highest N prefixes and then filtering down
            trimmed_beams = _sort_and_trim_beams(scored_beams, beam_width)
            # prune history and remove lm score from beams
            if prune_history:
                lm_order = 1 if language_model is None else language_model.order
                beams = _prune_history(trimmed_beams, lm_order=lm_order, word_window=self.word_window)
            else:
                beams = [b[:-1] for b in trimmed_beams]
            
        new_beams = _merge_speaker_beams(new_beams, word_window=self.word_window)
        scored_beams = self._get_speaker_lm_beams(
            speaker_list,
            new_beams,
            hotword_scorer,
            cached_lm_scores,
            cached_p_lm_scores,
        )
        # remove beam outliers
        max_score = max([b[-1] for b in scored_beams])
        scored_beams = [b for b in scored_beams if b[-1] >= max_score + beam_prune_logp]
        trimmed_beams = _sort_and_trim_beams(scored_beams, beam_width)
        # remove unnecessary information from beams
        output_beams = []
        for spk_trans_dict, next_word, text_frames, last_char, next_char, logit_score, combined_score in trimmed_beams:
            # cached_lm_scores[(text, True)][-1] if (text, True) in cached_lm_scores else None,
            out_entry = (
                spk_trans_dict,
                next_word,
                text_frames,
                last_char,
                next_char,
                logit_score,
                combined_score,  # same as logit_score if lm is missing
            )
            output_beams.append(out_entry)
        return output_beams

    def decode_beams(
        self,
        logits: NDArray[NpFloat],
        speaker_list: List[List[str]],
        word_dict_seq_list: List[List[Dict[str, float]]],
        beam_width: int = DEFAULT_BEAM_WIDTH,
        beam_prune_logp: float = DEFAULT_PRUNE_LOGP,
        token_min_logp: float = DEFAULT_MIN_TOKEN_LOGP,
        prune_history: bool = DEFAULT_PRUNE_BEAMS,
        hotwords: Optional[Iterable[str]] = None,
        port_num: int = None,
        hotword_weight: float = DEFAULT_HOTWORD_WEIGHT,
        lm_start_state: LMState = None,
    ) -> List[OutputBeam]:
        """Convert input token logit matrix to decoded beams including meta information.

        Args:
            logits: logit matrix of token log probabilities
            beam_width: maximum number of beams at each step in decoding
            beam_prune_logp: beams that are much worse than best beam will be pruned
            token_min_logp: tokens below this logp are skipped unless they are argmax of frame
            prune_history: prune beams based on shared recent history at the cost of beam diversity
            hotwords: list of words with extra importance, can be OOV for LM
            hotword_weight: weight factor for hotword importance
            lm_start_state: language model start state for stateful predictions

        Returns:
            List of beams of type OUTPUT_BEAM with various meta information
        """
        # self._check_logits_dimension(logits)
        # prepare hotword input
        hotword_scorer = HotwordScorer.build_scorer(hotwords, weight=hotword_weight)
        logits = np.log(np.clip(logits, MIN_TOKEN_CLIP_P, 1))
        token_min_logp=np.log(np.clip(0, MIN_TOKEN_CLIP_P, 1))
        self.port = port_num
        decoded_beams = self._decode_logits(
            logits,
            speaker_list=speaker_list,
            word_seq=word_dict_seq_list,
            beam_width=beam_width,
            beam_prune_logp=beam_prune_logp,
            token_min_logp=token_min_logp,
            prune_history=prune_history,
            hotword_scorer=hotword_scorer,
            lm_start_state=lm_start_state,
        )
        return decoded_beams

    def _decode_beams_mp_safe(
        self,
        logits: NDArray[NpFloat],
        speaker_list,
        word_dict_seq_list,
        beam_width: int,
        beam_prune_logp: float,
        token_min_logp: float,
        prune_history: bool,
        hotwords: Optional[Iterable[str]],
        hotword_weight: float,
    ) -> List[OutputBeamMPSafe]:
        """Thing wrapper around self.decode_beams to allow for multiprocessing."""
        decoded_beams = self.decode_beams(
            logits=logits,
            speaker_list=speaker_list,
            word_dict_seq_list=word_dict_seq_list,
            beam_width=beam_width,
            beam_prune_logp=beam_prune_logp,
            token_min_logp=token_min_logp,
            prune_history=prune_history,
            hotwords=hotwords,
            hotword_weight=hotword_weight,
        )
        # remove kenlm state to allow multiprocessing
        decoded_beams_mp_safe = [
            (text, frames_list, logit_score, lm_score)
            for text, _, frames_list, logit_score, lm_score in decoded_beams
        ]
        return decoded_beams_mp_safe

    def decode_beams_batch(
        self,
        pool: Optional[Pool],
        logits_list: NDArray[NpFloat],
        speaker_list_batch: List[List[str]],
        word_dict_seq_batch: List[List[Dict[str, float]]],
        beam_width: int = DEFAULT_BEAM_WIDTH,
        beam_prune_logp: float = DEFAULT_PRUNE_LOGP,
        token_min_logp: float = DEFAULT_MIN_TOKEN_LOGP,
        prune_history: bool = DEFAULT_PRUNE_BEAMS,
        hotwords: Optional[Iterable[str]] = None,
        hotword_weight: float = DEFAULT_HOTWORD_WEIGHT,
    ) -> List[List[OutputBeamMPSafe]]:
        """Use multiprocessing pool to batch decode input logits.

        Note that multiprocessing here does not work for a spawn context, so in that case
        or with no pool, just runs a loop in a single process.

        Args:
            pool: multiprocessing pool for parallel execution
            logits_list: list of logit matrices of token log probabilities
            beam_width: maximum number of beams at each step in decoding
            beam_prune_logp: beams that are much worse than best beam will be pruned
            token_min_logp: tokens below this logp are skipped unless they are argmax of frame
            prune_history: prune beams based on shared recent history at the cost of beam diversity
            hotwords: list of words with extra importance, can be OOV for LM
            hotword_weight: weight factor for hotword importance

        Returns:
            List of list of beams of type OUTPUT_BEAM_MP_SAFE with various meta information
        """
        valid_pool = _get_valid_pool(pool)
        if valid_pool is None:
            return [
                self._decode_beams_mp_safe(
                    logits,
                    speaker_list,
                    word_dict_seq_list,
                    beam_width=beam_width,
                    beam_prune_logp=beam_prune_logp,
                    token_min_logp=token_min_logp,
                    hotwords=hotwords,
                    prune_history=prune_history,
                    hotword_weight=hotword_weight,
                )
                for (logits, speaker_list, word_dict_seq_list) in zip(logits_list, speaker_list_batch, word_dict_seq_batch)
            ]

        # for logits in logits_list:
        #     self._check_logits_dimension(logits)
        p_decode = functools.partial(
            self._decode_beams_mp_safe,
            logits_list,
            speaker_list_batch,
            word_dict_seq_batch,
            beam_width=beam_width,
            beam_prune_logp=beam_prune_logp,
            token_min_logp=token_min_logp,
            hotwords=hotwords,
            prune_history=prune_history,
            hotword_weight=hotword_weight,
        )
        decoded_beams_list: List[List[OutputBeamMPSafe]] = valid_pool.map(p_decode, logits_list, speaker_list_batch, word_dict_seq_batch)
        return decoded_beams_list

    def decode(
        self,
        logits: NDArray[NpFloat],
        beam_width: int = DEFAULT_BEAM_WIDTH,
        beam_prune_logp: float = DEFAULT_PRUNE_LOGP,
        token_min_logp: float = DEFAULT_MIN_TOKEN_LOGP,
        hotwords: Optional[Iterable[str]] = None,
        hotword_weight: float = DEFAULT_HOTWORD_WEIGHT,
        lm_start_state: LMState = None,
    ) -> str:
        """Convert input token logit matrix to decoded text.

        Args:
            logits: logit matrix of token log probabilities
            beam_width: maximum number of beams at each step in decoding
            beam_prune_logp: beams that are much worse than best beam will be pruned
            token_min_logp: tokens below this logp are skipped unless they are argmax of frame
            hotwords: list of words with extra importance, can be OOV for LM
            hotword_weight: weight factor for hotword importance
            lm_start_state: language model start state for stateful predictions

        Returns:
            The decoded text (str)
        """
        decoded_beams = self.decode_beams(
            logits,
            beam_width=beam_width,
            beam_prune_logp=beam_prune_logp,
            token_min_logp=token_min_logp,
            prune_history=True,  # we can set this to True since we only care about top 1 beam
            hotwords=hotwords,
            hotword_weight=hotword_weight,
            lm_start_state=lm_start_state,
        )
        return decoded_beams[0][0]

    def decode_batch(
        self,
        pool: Optional[Pool],
        logits_list: NDArray[NpFloat],
        beam_width: int = DEFAULT_BEAM_WIDTH,
        beam_prune_logp: float = DEFAULT_PRUNE_LOGP,
        token_min_logp: float = DEFAULT_MIN_TOKEN_LOGP,
        hotwords: Optional[Iterable[str]] = None,
        hotword_weight: float = DEFAULT_HOTWORD_WEIGHT,
    ) -> List[str]:
        """Use multiprocessing pool to batch decode input logits.

        Note that multiprocessing here does not work for a spawn context, so in that case
        or with no pool, just runs a loop in a single process.

        Args:
            pool: multiprocessing pool for parallel execution
            logits_list: list of logit matrices of token log probabilities
            beam_width: maximum number of beams at each step in decoding
            beam_prune_logp: beams that are much worse than best beam will be pruned
            token_min_logp: tokens below this logp are skipped unless they are argmax of frame
            hotwords: list of words with extra importance, can be OOV for LM
            hotword_weight: weight factor for hotword importance

        Returns:
            The decoded texts (list of str)
        """
        valid_pool = _get_valid_pool(pool)
        if valid_pool is None:
            return [
                self.decode(
                    logits,
                    beam_width=beam_width,
                    beam_prune_logp=beam_prune_logp,
                    token_min_logp=token_min_logp,
                    hotwords=hotwords,
                    hotword_weight=hotword_weight,
                )
                for logits in logits_list
            ]

        p_decode = functools.partial(
            self.decode,
            beam_width=beam_width,
            beam_prune_logp=beam_prune_logp,
            token_min_logp=token_min_logp,
            hotwords=hotwords,
            hotword_weight=hotword_weight,
        )
        decoded_text_list: List[str] = valid_pool.map(p_decode, logits_list)
        return decoded_text_list

    def save_to_dir(self, filepath: str) -> None:
        """Save a decoder to a directory."""
        alphabet_path = os.path.join(filepath, self._ALPHABET_SERIALIZED_FILENAME)
        with open(alphabet_path, "w") as fi:
            fi.write(self._alphabet.dumps())

        lm = self._language_model
        if lm is None:
            logger.info("decoder has no language model.")
        else:
            lm_path = os.path.join(filepath, self._LANGUAGE_MODEL_SERIALIZED_DIRECTORY)
            os.makedirs(lm_path)
            logger.info("Saving language model to %s", lm_path)
            lm.save_to_dir(lm_path)

    @staticmethod
    def parse_directory_contents(filepath: str) -> Dict[str, Union[str, None]]:
        """Check contents of a directory for correct BeamSearchDecoderCTC files."""
        contents = os.listdir(filepath)
        # filter out hidden files
        contents = [c for c in contents if not c.startswith(".") and not c.startswith("__")]
        if BeamSearchDecoderCTC._ALPHABET_SERIALIZED_FILENAME not in contents:
            raise ValueError(
                f"Could not find alphabet file "
                f"{BeamSearchDecoderCTC._ALPHABET_SERIALIZED_FILENAME}. Found {contents}"
            )
        alphabet_filepath = os.path.join(
            filepath, BeamSearchDecoderCTC._ALPHABET_SERIALIZED_FILENAME
        )
        contents.remove(BeamSearchDecoderCTC._ALPHABET_SERIALIZED_FILENAME)
        lm_directory: Optional[str]
        if contents:
            if BeamSearchDecoderCTC._LANGUAGE_MODEL_SERIALIZED_DIRECTORY not in contents:
                raise ValueError(
                    f"Count not find language model directory. Looking for "
                    f"{BeamSearchDecoderCTC._LANGUAGE_MODEL_SERIALIZED_DIRECTORY}, found {contents}"
                )
            lm_directory = os.path.join(
                filepath, BeamSearchDecoderCTC._LANGUAGE_MODEL_SERIALIZED_DIRECTORY
            )
        else:
            lm_directory = None
        return {"alphabet": alphabet_filepath, "language_model": lm_directory}

    @classmethod
    def load_from_dir(
        cls, filepath: str, unigram_encoding: Optional[str] = None
    ) -> "BeamSearchDecoderCTC":
        """Load a decoder from a directory."""
        filenames = cls.parse_directory_contents(filepath)
        with open(filenames["alphabet"], "r") as fi:  # type: ignore
            alphabet = Alphabet.loads(fi.read())
        if filenames["language_model"] is None:
            language_model = None
        else:
            language_model = LanguageModel.load_from_dir(
                filenames["language_model"], unigram_encoding=unigram_encoding
            )
        return cls(alphabet, language_model=language_model)

    @classmethod
    def load_from_hf_hub(  # type: ignore
        cls, model_id: str, cache_dir: Optional[str] = None, **kwargs: Any
    ) -> "BeamSearchDecoderCTC":
        """Class method to load model from https://huggingface.co/ .

        Args:
            model_id: string, the `model id` of a pretrained model hosted inside a model
                repo on https://huggingface.co/. Valid model ids can be namespaced under a user or
                organization name, like ``kensho/5gram-spanish-kenLM``. For more information, please
                take a look at https://huggingface.co/docs/hub/main.
            cache_dir: path to where the language model should be downloaded and cached.

        Returns:
            instance of BeamSearchDecoderCTC
        """
        if sys.version_info >= (3, 8):
            from importlib.metadata import metadata
        else:
            from importlib_metadata import metadata

        library_name = metadata("pyctcdecode")["Name"]
        cache_dir = cache_dir or os.path.join(Path.home(), ".cache", library_name)

        try:
            from huggingface_hub import snapshot_download  # type: ignore
        except ImportError:
            raise ImportError(
                "You need to install huggingface_hub to use `load_from_hf_hub`. "
                "See https://pypi.org/project/huggingface-hub/ for installation."
            )

        cached_directory = snapshot_download(  # pylint: disable=not-callable
            model_id, cache_dir=cache_dir, **kwargs
        )

        return cls.load_from_dir(cached_directory)


##########################################################################################
# Main entry point and convenience function to create BeamSearchDecoderCTC object ########
##########################################################################################
    # labels: List[str],
def build_diardecoder(
    kenlm_model_path: Optional[str] = None,
    unigrams: Optional[Collection[str]] = None,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    word_window: int = 25,
    use_ngram: bool = True,
    port: int = None,
    build_up_len: int = 25,
    unk_score_offset: float = DEFAULT_UNK_LOGP_OFFSET,
    lm_score_boundary: bool = DEFAULT_SCORE_LM_BOUNDARY,
) -> BeamSearchDecoderCTC:
    """Build a BeamSearchDecoderCTC instance with main functionality.

    Args:
        labels: class containing the labels for input logit matrices
        kenlm_model_path: path to kenlm n-gram language model
        unigrams: list of known word unigrams
        alpha: weight for language model during shallow fusion
        beta: weight for length score adjustment of during scoring
        unk_score_offset: amount of log score offset for unknown tokens
        lm_score_boundary: whether to have kenlm respect boundaries when scoring

    Returns:
        instance of BeamSearchDecoderCTC
    """
    kenlm_model = None if kenlm_model_path is None else kenlm.Model(kenlm_model_path)
    if kenlm_model_path is not None and kenlm_model_path.endswith(".arpa"):
        logger.info("Using arpa instead of binary LM file, decoder instantiation might be slow.")
    if unigrams is None and kenlm_model_path is not None:
        if kenlm_model_path.endswith(".arpa"):
            unigrams = load_unigram_set_from_arpa(kenlm_model_path)
        else:
            logger.warning(
                "Unigrams not provided and cannot be automatically determined from LM file (only "
                "arpa format). Decoding accuracy might be reduced."
            )

    # alphabet = Alphabet.build_alphabet(labels)
    # if unigrams is not None:
    #     verify_alphabet_coverage(alphabet, unigrams)
    if kenlm_model is not None:
        # language_model: Optional[AbstractLanguageModel] = LanguageModel(
        #     kenlm_model,
        #     unigrams,
        #     # alpha=alpha,
        #     # beta=beta,
        #     unk_score_offset=unk_score_offset,
        #     score_boundary=lm_score_boundary,
        # )
        language_model = kenlm_model
    else:
        language_model = None
    alphabet = None
    return BeamSearchDecoderCTC(alphabet, alpha, beta, word_window, use_ngram, build_up_len, language_model, port=port)