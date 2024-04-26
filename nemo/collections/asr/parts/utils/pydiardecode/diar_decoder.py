# Copyright 2021-present Kensho Technologies, LLC.
from __future__ import annotations, division

import functools
import heapq
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
    
    def get_spk_wise_probs(
        self, 
        next_word: str, 
        last_char: str,
        spk_trans_dict: Dict[str, str],
        max_word: int = 25,
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
        speaker_list = _get_speaker_list(spk_trans_dict)
        num_spks = len(speaker_list)
        hyp_individ_dict = {spk: {tgt_spk: '' for tgt_spk in speaker_list} for spk in speaker_list}
        # hyp_allspks_dict = {spk: {tgt_spk: '' for tgt_spk in speaker_list} for spk in speaker_list}
        hyp_allspks_dict = {spk: '' for spk in speaker_list}
        hyp_probs = {spk: 0 for spk in speaker_list}
        hyp_base_individ_probs = {spk: {tgt_spk: 0 for tgt_spk in speaker_list} for spk in speaker_list}
        hyp_base_allspks_probs = {spk: 0 for spk in speaker_list}
        for spk in speaker_list:
            allspks_word_list = spk_trans_dict['all_spks'].split()[-(num_spks*max_word):]
            truncated_allspks_words = " ".join(allspks_word_list)
            hyp_base_allspks_probs[spk] = self.realigning_lm.score(truncated_allspks_words)
            if not spk_trans_dict['all_spks'] == '' and len(truncated_allspks_words.split()) > 0 and truncated_allspks_words.split()[-1] == ARPA_END:
                truncated_allspks_words = re.sub(f' {ARPA_END}$', '', truncated_allspks_words)
            if truncated_allspks_words == '':
                truncated_allspks_words += f"{ARPA_STT}"
            if last_spk is not None and last_spk == spk:
                if truncated_allspks_words == '':
                    truncated_allspks_words += f"{ARPA_STT}"
                hyp_allspks_dict[spk] = truncated_allspks_words + f" {next_word} {ARPA_END}"
            elif last_spk != spk:
                hyp_allspks_dict[spk] = truncated_allspks_words + f" {ARPA_END} {ARPA_STT} {next_word} {ARPA_END}"
            
                
            for tgt_spk in speaker_list:
                individ_word_list = spk_trans_dict[tgt_spk].split()[-max_word:]
                # truncated_individ_words = " ".join(spk_trans_dict[tgt_spk].split()[-max_word:])
                # truncated_allspks_words = " ".join(spk_trans_dict['all_spks'].split()[-(num_spks*max_word):])
                truncated_individ_words = " ".join(individ_word_list)
                # hyp_base_individ_probs[spk][tgt_spk] = self.realigning_lm.score(word=truncated_individ_words, prev_state=None)[0]
                # hyp_base_allspks_probs[spk][tgt_spk] = self.realigning_lm.score(word=truncated_allspks_words, prev_state=None)[0]
                hyp_base_individ_probs[spk][tgt_spk] = self.realigning_lm.score(truncated_individ_words)
                # hyp_base_individ_probs[spk][tgt_spk] = self.realigning_lm.score(truncated_individ_words)
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
                # sentence = re.sub(' +', ' ', sentence) 
                prob = self.realigning_lm.score(sentence) - hyp_base_individ_probs[spk][tgt_spk]
                # print(f"tgt_spk {tgt_spk} Indiv sentence: {sentence}")
                # prob = self.realigning_lm.score(word=sentence.strip(), prev_state=None)[0] 
                indiv_spk_probs.append(prob)
            
            # for tgt_spk in speaker_list:
            all_spks_sentence = hyp_allspks_dict[spk]
            all_spks_prob = self.realigning_lm.score(all_spks_sentence) - hyp_base_allspks_probs[spk]
            # print(f"spk {spk} All spks sentence: {all_spks_sentence}")
            # prob = self.realigning_lm.score(word=all_spks_sentence.strip(), prev_state=None)[0] 
            # hyp_probs[spk] += (1 - self.beta) * sum(indiv_spk_probs) + self.beta * all_spks_prob
            hyp_probs[spk] += sum(indiv_spk_probs) 
            # print(f"For spk: {spk} Individual spk probs: {sum(indiv_spk_probs)}, All spks probs: {all_spks_prob}")
            
        lm_spk_logits = [hyp_probs[spk] for spk in sorted(hyp_probs)]
        lm_spk_probs = softmax(lm_spk_logits)
        return lm_spk_probs
    
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
    
    def simulate_decode_run(
        self, 
        speaker_list: List[str],
        word_dict_seq_list: List[Dict[str, float]],
        ):
        """
        Calculate speaker-wise probabilities for each word in the word_dict_seq_list.

        Args:
            speaker_list (_type_): _description_
            word_dict_seq_list (_type_): _description_

        Returns:
            _type_: _description_
        """
        speaker_list = sorted(speaker_list)
        spk_trans_dict = {spk: '' for spk in speaker_list}
        prev_spk = None
        # speaker_lm_probs = []
        correct_count = 0
        new_word_dict_seq_list = deepcopy(word_dict_seq_list)
        for wi, word_dict in enumerate(word_dict_seq_list):
            word, _spk_label = word_dict['word'], word_dict['speaker']
            
            spk_wise_probs = self.get_spk_wise_probs(next_word=word, spk_trans_dict=spk_trans_dict)
            # speaker_lm_probs.append(spk_wise_probs)
            est_spk_label = speaker_list[np.argmax(spk_wise_probs)]
            if est_spk_label == _spk_label:
                correct_count += 1
            spk_label = _spk_label
            spk_trans_dict = self.update_transcript_status(spk_trans_dict=spk_trans_dict, 
                                                           spk_label=spk_label, 
                                                           prev_spk=prev_spk, 
                                                           word=word)
            
            new_word_dict_seq_list[wi]['speaker'] = spk_label
            prev_spk = spk_label 
        # speaker_lm_probs = np.array(speaker_lm_probs) 
        # print(f"Acc. {correct_count/len(word_dict_seq_list):.4f} Correct count {correct_count}, total {len(word_dict_seq_list)}")
        
        return new_word_dict_seq_list

    def build_arpa_diar_lm(
        self, 
        speaker_list: List[str],
        word_dict_seq_list: List[Dict[str, float]],
        ):
        """
        Calculate speaker-wise probabilities for each word in the word_dict_seq_list.

        Args:
            speaker_list (_type_): _description_
            word_dict_seq_list (_type_): _description_

        Returns:
            _type_: _description_
        """
        spk_trans_dict = {spk: '' for spk in speaker_list}
        prev_spk = None
        speaker_lm_probs = []
        for wi, word_dict in enumerate(word_dict_seq_list):
            word, spk_label = word_dict['word'], word_dict['speaker']
            
            spk_wise_probs = self.get_spk_wise_probs(next_word=word, spk_trans_dict=spk_trans_dict)
            speaker_lm_probs.append(spk_wise_probs)
            if spk_trans_dict[spk_label] == '':
                spk_trans_dict[spk_label] = f" {ARPA_STT} {word}"
            elif spk_label == prev_spk:
                spk_trans_dict[spk_label] += f" {word}"
            elif spk_label != prev_spk and wi > 0 and prev_spk is not None:
                spk_trans_dict[prev_spk] += f" {ARPA_END}"
                spk_trans_dict[spk_label] += f" {ARPA_STT} {word}"
            elif  spk_label != prev_spk and spk_trans_dict[spk_label] != '':
                spk_trans_dict[spk_label] += f" {word}"
            prev_spk = spk_label 
        speaker_lm_probs = np.array(speaker_lm_probs) 
        return speaker_lm_probs, spk_trans_dict

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

def get_concat_transcripts(spk_trans_dict):
    concat_text = ''
    # Concatenate all the transcripts for all speakers
    speakers_list = []
    for spk in spk_trans_dict.keys():
        if type(spk) == int:
            speakers_list.append(spk)
    speakers_list = sorted(speakers_list)
    for spk in speakers_list:
        concat_text += spk_trans_dict[spk].replace(f"{ARPA_STT} ", '').replace(f" {ARPA_STT}", '').replace(f" {ARPA_END}", '')
    return concat_text

def _merge_speaker_beams(beams: List[Beam]) -> List[Beam]:
    """Merge beams with same prefix together."""
    beam_dict = {}
    for spk_trans_dict, next_word, word_part, last_char, text_frames, part_frames, logit_score in beams:
        # new_text = _merge_tokens(text, next_word)
        # print(f"in _merge_speaker_beams(): spk_trans_dict: {spk_trans_dict}")
        concat_text = get_concat_transcripts(spk_trans_dict=deepcopy(spk_trans_dict))
        hash_idx = (concat_text, next_word, last_char)
        if hash_idx not in beam_dict:
            beam_dict[hash_idx] = (
                spk_trans_dict,
                next_word,
                word_part,
                last_char,
                text_frames,
                part_frames,
                logit_score,
            )
        else:
            beam_dict[hash_idx] = (
                spk_trans_dict,
                next_word,
                word_part,
                last_char,
                text_frames,
                part_frames,
                _sum_log_scores(beam_dict[hash_idx][-1], logit_score),
            )
    # if len(beams) > len(beam_dict):
    #     print(f"Merge: {len(beams)} TO {len(beam_dict)}")
    return list(beam_dict.values())


def _merge_beams(beams: List[Beam]) -> List[Beam]:
    """Merge beams with same prefix together."""
    beam_dict = {}
    for text, next_word, word_part, last_char, text_frames, part_frames, logit_score in beams:
        new_text = _merge_tokens(text, next_word)
        hash_idx = (new_text, word_part, last_char)
        if hash_idx not in beam_dict:
            beam_dict[hash_idx] = (
                text,
                next_word,
                word_part,
                last_char,
                text_frames,
                part_frames,
                logit_score,
            )
        else:
            beam_dict[hash_idx] = (
                text,
                next_word,
                word_part,
                last_char,
                text_frames,
                part_frames,
                _sum_log_scores(beam_dict[hash_idx][-1], logit_score),
            )
    # if len(beams) > len(beam_dict):
    #     print(f"Merge: {len(beams)} TO {len(beam_dict)}")
    return list(beam_dict.values())


def _prune_history(beams: List[LMBeam], lm_order: int) -> List[Beam]:
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
    for (text, next_word, word_part, last_char, text_frames, part_frames, logit_score, _) in beams:
        # hash based on history that can still affect lm scoring going forward
        hash_idx = (tuple(text.split()[-min_n_history:]), word_part, last_char)
        if hash_idx not in seen_hashes:
            filtered_beams.append(
                (
                    text,
                    next_word,
                    word_part,
                    last_char,
                    text_frames,
                    part_frames,
                    logit_score,
                )
            )
            seen_hashes.add(hash_idx)
    return filtered_beams


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
        language_model: Optional[AbstractLanguageModel] = None,
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
        self.spk_lm_decoder = SpeakerToWordAlignerLM(realigning_lm=language_model, beta=self.beta)
        BeamSearchDecoderCTC.model_container[self._model_key] = language_model

    def reset_params(
        self,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
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

    # def _check_logits_dimension(
    #     self,
    #     logits: NDArray[NpFloat],
    # ) -> None:
    #     """Verify correct shape and dimensions for input logits."""
    #     if len(logits.shape) != 2:
    #         raise ValueError(
    #             "Input logits have %s dimensions, but need 2: (time, vocabulary)"
    #             % len(logits.shape)
    #         )
    #     if logits.shape[-1] != len(self._idx2speaker):
    #         raise ValueError(
    #             "Input logits shape is %s, but vocabulary is size %s. "
    #             "Need logits of shape: (time, vocabulary)" % (logits.shape, len(self._idx2speaker))
    #         )
    
    def _get_speaker_lm_beams(
        self,
        speaker_list,
        beams: List[Beam],
        hotword_scorer: HotwordScorer,
        cached_lm_scores: Dict[LMScoreCacheKey, LMScoreCacheValue],
        cached_partial_token_scores: Dict[str, float],
        is_eos: bool = False,
    ) -> List[LMBeam]:
        """Update score by averaging logit_score and lm_score."""
        # get language model and see if exists
        language_model = self._language_model
        new_beams = []
        for spk_trans_dict, next_word, word_part, last_char, frame_list, frames, logit_score in beams:
            # fast token merge
            concat_text = get_concat_transcripts(spk_trans_dict=deepcopy(spk_trans_dict))
            cache_key = (concat_text, next_word) 
            # cache_key = (concat_text, next_word) 
            
            # cache_key = (new_text, is_eos)
            if cache_key not in cached_lm_scores:
                spk_int = int(last_char.split('_')[-1])
                # Remove all non-word tokens from the transcript
                target_text = spk_trans_dict[spk_int].replace(f"{ARPA_STT} ", '').replace(f" {ARPA_STT}", '').replace(f" {ARPA_END}", '')
                prev_trans_dict = deepcopy(spk_trans_dict)
                if target_text != '' and len(target_text.split()) > 1:
                    prev_trans_dict[spk_int] = " ".join(target_text.split()[:-1])
                    prev_word = target_text.split()[-1]
                elif len(target_text.split()) == 1:
                    prev_trans_dict[spk_int] = ''
                    prev_word = target_text.split()[0]
                elif target_text == '':
                    prev_trans_dict[spk_int] = ''
                    prev_word = ''
                prev_concat_text = get_concat_transcripts(spk_trans_dict=deepcopy(prev_trans_dict))
                
                prev_cache_key = (prev_concat_text, prev_word) 
                if prev_cache_key not in cached_lm_scores:
                    _, prev_raw_lm_score, start_state = None, 0.0, None
                else:
                    _, prev_raw_lm_score, start_state = cached_lm_scores[prev_cache_key]
                spk_wise_probs = self.spk_lm_decoder.get_spk_wise_probs(next_word=next_word, last_char=last_char, spk_trans_dict=spk_trans_dict)
                spk_lm_score = spk_wise_probs[spk_int]
                # score, end_state = language_model.score(start_state, next_word, is_last_word=is_eos)
                
                raw_lm_score = prev_raw_lm_score + spk_lm_score
                # lm_hw_score = raw_lm_score + hotword_scorer.score(new_text)
                # lm_hw_score = raw_lm_score + 0.0
                # lm_hw_score = 0.0
                end_state = None
                cached_lm_scores[cache_key] = (raw_lm_score, spk_lm_score, end_state)
            lm_score, _, _ = cached_lm_scores[cache_key]

            new_beams.append(
                (
                    spk_trans_dict,
                    "",
                    word_part,
                    last_char,
                    frame_list,
                    frames,
                    logit_score,
                    logit_score + lm_score,
                )
            )
        return new_beams

    def _get_lm_beams(
        self,
        beams: List[Beam],
        hotword_scorer: HotwordScorer,
        cached_lm_scores: Dict[LMScoreCacheKey, LMScoreCacheValue],
        cached_partial_token_scores: Dict[str, float],
        is_eos: bool = False,
    ) -> List[LMBeam]:
        """Update score by averaging logit_score and lm_score."""
        # get language model and see if exists
        language_model = self._language_model
        # if no language model available then return raw score + hotwords as lm score
        if language_model is None:
            new_beams = []
            for text, next_word, word_part, last_char, frame_list, frames, logit_score in beams:
                new_text = _merge_tokens(text, next_word)
                # note that usually this gets scaled with alpha
                lm_hw_score = (
                    logit_score
                    + hotword_scorer.score(new_text)
                    + hotword_scorer.score_partial_token(word_part)
                )

                new_beams.append(
                    (
                        new_text,
                        "",
                        word_part,
                        last_char,
                        frame_list,
                        frames,
                        logit_score,
                        lm_hw_score,
                    )
                )
            return new_beams

        new_beams = []
        for text, next_word, word_part, last_char, frame_list, frames, logit_score in beams:
            # fast token merge
            new_text = _merge_tokens(text, next_word)
            cache_key = (new_text, is_eos)
            if cache_key not in cached_lm_scores:
                _, prev_raw_lm_score, start_state = cached_lm_scores[(text, False)]
                score, end_state = language_model.score(start_state, next_word, is_last_word=is_eos)
                raw_lm_score = prev_raw_lm_score + score
                lm_hw_score = raw_lm_score + hotword_scorer.score(new_text)
                cached_lm_scores[cache_key] = (lm_hw_score, raw_lm_score, end_state)
            lm_score, _, _ = cached_lm_scores[cache_key]

            if len(word_part) > 0:
                if word_part not in cached_partial_token_scores:
                    # if prefix available in hotword trie use that, otherwise default to char trie
                    if word_part in hotword_scorer:
                        cached_partial_token_scores[word_part] = hotword_scorer.score_partial_token(
                            word_part
                        )
                    else:
                        cached_partial_token_scores[word_part] = language_model.score_partial_token(
                            word_part
                        )
                lm_score += cached_partial_token_scores[word_part]

            new_beams.append(
                (
                    new_text,
                    "",
                    word_part,
                    last_char,
                    frame_list,
                    frames,
                    logit_score,
                    logit_score + lm_score,
                )
            )

        return new_beams
    
    def _add_word_to_spk_transcript(self, spk_trans_dict, spk_label, prev_spk, word, all_spks = 'all_spks'):
        if type(prev_spk) == str:
            prev_spk = int(prev_spk.split('_')[-1])
            
        if word == '' or len(word) == 0:
           return spk_trans_dict 
            
        if spk_label == prev_spk:
            # Current speaker transcript
            if spk_trans_dict[spk_label] == '':
                spk_trans_dict[spk_label] = f"{ARPA_STT} {word}"
            else:
                spk_trans_dict[spk_label] += f" {word}"
            
            # All speakers transcript 
            if spk_trans_dict[all_spks] == '':
                spk_trans_dict[all_spks] = f"{ARPA_STT} {word}"
            else:
                spk_trans_dict[all_spks] += f" {word}"
                
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
            
            # New speaker transcript  
            if spk_trans_dict[spk_label] == '':
                spk_trans_dict[spk_label] += f" {ARPA_STT} {word}" 
            elif spk_trans_dict[spk_label].split()[-1] == ARPA_END:
                spk_trans_dict[spk_label] += f" {ARPA_STT} {word}"
            elif spk_trans_dict[spk_label].split()[-1] != ARPA_END:
                spk_trans_dict[spk_label] += f" {ARPA_END} {ARPA_STT} {word}"
            
            # All speakers transcript 
            if spk_trans_dict[all_spks] == '':
                spk_trans_dict[all_spks] += f" {ARPA_STT} {word}" 
            elif spk_trans_dict[all_spks].split()[-1] == ARPA_END:
                spk_trans_dict[all_spks] += f" {ARPA_STT} {word}"
            elif spk_trans_dict[all_spks].split()[-1] != ARPA_END:
                spk_trans_dict[all_spks] += f" {ARPA_END} {ARPA_STT} {word}"
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
        spk_trans_dict['all_spks'] = ''
        prev_spk = None
        beams[0][0] = spk_trans_dict
        beams[0][1] = ''
        beams[0][3] = prev_spk
        beams[0][4] = [word_seq[0]]
        for word_idx, word_dict in enumerate(tqdm(word_seq[:-1])):
            logit_col_raw = word_dict['speaker_softmax']
            logit_col = np.log(np.clip(np.array(logit_col_raw), MIN_TOKEN_CLIP_P, 1))
            max_idx = logit_col.argmax().item()
            speaker_idx_list = set(np.where(logit_col > token_min_logp)[0]) | {max_idx}
            new_beams: List[Beam] = []
            if speaker_idx_list == set():
                import ipdb; ipdb.set_trace()
            for spk_idx_char in speaker_idx_list:
                spk_logit_char = logit_col[spk_idx_char].item()
                char = self._idx2speaker[spk_idx_char]
                for (
                    spk_trans_dict,
                    next_word,
                    word_part,
                    last_char,
                    text_frames,
                    part_frames,
                    logit_score,
                ) in beams:
                    # if only blank token or same token
                    new_part_frames = (
                        (word_idx, word_idx + 1)
                        if part_frames[0] < 0
                        else (part_frames[0], word_idx + 1)
                    )
                    # next_word = word_dict['word']
                    curr_word = word_dict['word']
                    next_word = word_seq[word_idx+1]['word']
                    # print(f"Word Index: {word_idx} Added word: --->[  {next_word}  ] to speaker: {spk_idx_char}")
                    # print(f"Before spk trans dict: {spk_trans_dict}")
                    added_spk_trans_dict = self._add_word_to_spk_transcript(spk_trans_dict=deepcopy(spk_trans_dict), 
                                                                            spk_label=spk_idx_char, 
                                                                            prev_spk=last_char, 
                                                                            word=curr_word)
                    # if word_idx > 0 and word_idx % 30 == 0:
                    #     import ipdb; ipdb.set_trace()
                    # print(f"Added  spk trans dict: {added_spk_trans_dict}")
                    
                    new_word_dict = deepcopy(word_dict) 
                    new_word_dict['speaker'] = f"speaker_{spk_idx_char}"
                    if word_idx == 0:
                        added_text_frames = [new_word_dict]
                    else:
                        added_text_frames = text_frames + [new_word_dict]
                    new_beams.append(
                        (
                            added_spk_trans_dict,
                            next_word,
                            word_part + char,
                            char,
                            added_text_frames,
                            new_part_frames,
                            self.alpha * logit_score + (1-self.alpha) * spk_logit_char,
                        )
                    )
                    if new_beams == []:
                        import ipdb; ipdb.set_trace()

            # lm scoring and beam pruning
            if new_beams == []:
                import ipdb; ipdb.set_trace()
            new_beams = _merge_speaker_beams(new_beams)
            if new_beams == []:
                import ipdb; ipdb.set_trace()
            
            scored_beams = self._get_speaker_lm_beams(
                speaker_list,
                new_beams,
                hotword_scorer,
                cached_lm_scores,
                cached_p_lm_scores,
            )
            if scored_beams == []:
                import ipdb; ipdb.set_trace()
            # remove beam outliers
            max_score = max([b[-1] for b in scored_beams])
            scored_beams = [b for b in scored_beams if b[-1] >= max_score + beam_prune_logp]
            # beam pruning by taking highest N prefixes and then filtering down
            trimmed_beams = _sort_and_trim_beams(scored_beams, beam_width)
            # if word_idx % 15 == 0:
            #     import ipdb; ipdb.set_trace()
            # prune history and remove lm score from beams
            if prune_history:
                lm_order = 1 if language_model is None else language_model.order
                beams = _prune_history(trimmed_beams, lm_order=lm_order)
            else:
                beams = [b[:-1] for b in trimmed_beams]

        new_beams = _merge_speaker_beams(new_beams)
        scored_beams = self._get_speaker_lm_beams(
            speaker_list,
            new_beams,
            hotword_scorer,
            cached_lm_scores,
            cached_p_lm_scores,
            is_eos=True,
        )
        # remove beam outliers
        max_score = max([b[-1] for b in scored_beams])
        scored_beams = [b for b in scored_beams if b[-1] >= max_score + beam_prune_logp]
        trimmed_beams = _sort_and_trim_beams(scored_beams, beam_width)
        # remove unnecessary information from beams
        output_beams = []
        for spk_trans_dict, _, _, _, text_frames, _, logit_score, combined_score in trimmed_beams:
            # cached_lm_scores[(text, True)][-1] if (text, True) in cached_lm_scores else None,
            out_entry = (
                spk_trans_dict,
                None,
                text_frames,
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
        # import ipdb; ipdb.set_trace()
    else:
        language_model = None
    alphabet = None
    return BeamSearchDecoderCTC(alphabet, alpha, beta, language_model)