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

import os
from collections import OrderedDict
from statistics import mode
from typing import Dict, List, Tuple, Optional
import pickle

from torch import nn
import torch
import numpy as np
import time


from nemo.collections.asr.parts.utils.vad_utils import (
    get_channel_averaging_matrix
)

from nemo.collections.asr.parts.utils.offline_clustering import get_argmin_mat
from nemo.collections.asr.parts.utils.speaker_utils import convert_rttm_line, prepare_split_data, get_subsegments
from nemo.collections.common.parts.preprocessing.collections import DiarizationSpeechLabel
from nemo.core.classes import Dataset
from nemo.core.neural_types import AudioSignal, EncodedRepresentation, LengthsType, NeuralType, ProbsType

from nemo.collections.asr.parts.utils.channel_clustering import channel_cluster_from_coherence

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def get_subsegments_to_scale_timestamps(subsegments: List[Tuple[float, float]], feat_per_sec, decimals=2):
    """
    Convert subsegment timestamps to scale timestamps.

    Args:
        subsegments (List[Tuple[float, float]]):
            List of subsegment timestamps.

    Returns:
        scale_ts (torch.tensor):
            Tensor containing scale timestamps.
    """
    # scale_ts = (torch.tensor(subsegments) * feat_per_sec).long()
    seg_ts = (torch.tensor(subsegments) * feat_per_sec).float()
    scale_ts_round = torch.round(seg_ts, decimals=decimals)
    scale_ts = scale_ts_round.long()
    scale_ts[:, 1] = scale_ts[:, 0] + scale_ts[:, 1]
    return scale_ts 

def get_ms_seg_timestamps(
    duration: float, 
    feat_per_sec: int, 
    scale_n: int,
    multiscale_args_dict: Dict,
    dtype,
    min_subsegment_duration: float,
    ):
    """
    Get start and end time of segments in each scale.

    Args:
        sample:
            `DiarizationSpeechLabel` instance from preprocessing.collections
    Returns:
        ms_seg_timestamps (torch.tensor):
            Tensor containing Multiscale segment timestamps.
        ms_seg_counts (torch.tensor):
            Number of segments for each scale. This information is used for reshaping embedding batch
            during forward propagation.
    """
    ms_seg_timestamps_list = []
    total_steps = None
    ms_seg_counts = [0 for _ in range(scale_n)]
    for scale_idx in reversed(range(scale_n)):
        subsegments = get_subsegments(offset=0, 
                                        window=multiscale_args_dict['scale_dict'][scale_idx][0],
                                        shift=multiscale_args_dict['scale_dict'][scale_idx][1],
                                        duration=duration, 
                                        min_subsegment_duration=min_subsegment_duration)
        scale_ts_tensor = get_subsegments_to_scale_timestamps(subsegments, feat_per_sec, decimals=2)
        if scale_idx == scale_n - 1:
            total_steps = scale_ts_tensor.shape[0]
        ms_seg_counts[scale_idx] = scale_ts_tensor.shape[0]
        scale_ts_padded = torch.cat([scale_ts_tensor, torch.zeros(total_steps - scale_ts_tensor.shape[0], 2, dtype=scale_ts_tensor.dtype)], dim=0)
        ms_seg_timestamps_list.append(scale_ts_padded.detach())
    ms_seg_timestamps_list = ms_seg_timestamps_list[::-1]
    ms_seg_timestamps = torch.stack(ms_seg_timestamps_list).type(dtype)
    ms_seg_counts = torch.tensor(ms_seg_counts)
    return ms_seg_timestamps, ms_seg_counts

def get_frame_targets_from_rttm(
    rttm_timestamps: list, 
    offset: float, 
    duration: float, 
    round_digits: int, 
    feat_per_sec: int, 
    max_spks: int,
    ):
    """
    Create a multi-dimensional vector sequence containing speaker timestamp information in RTTM.
    The unit-length is the frame shift length of the acoustic feature. The feature-level annotations
    `feat_level_target` will later be converted to base-segment level diarization label.

    Args:
        rttm_timestamps (list):
            List containing start and end time for each speaker segment label.
            stt_list, end_list and speaker_list are contained.
        feat_per_sec (int):
            Number of feature frames per second. This quantity is determined by window_stride variable in preprocessing module.
        target_spks (tuple):
            Speaker indices that are generated from combinations. If there are only one or two speakers,
            only a single target_spks variable is generated.

    Returns:
        feat_level_target (torch.tensor):
            Tensor containing label for each feature level frame.
    """
    stt_list, end_list, speaker_list = rttm_timestamps
    sorted_speakers = sorted(list(set(speaker_list)))
    total_fr_len = int(duration * feat_per_sec)
    if len(sorted_speakers) > max_spks:
        raise ValueError(f"Number of speakers in RTTM file {len(sorted_speakers)} exceeds the maximum number of speakers: {max_spks}")
    feat_level_target = torch.zeros(total_fr_len, max_spks) 
    for count, (stt, end, spk_rttm_key) in enumerate(zip(stt_list, end_list, speaker_list)):
        if end < offset or stt > offset + duration:
            continue
        stt, end = max(offset, stt), min(offset + duration, end)
        spk = spk_rttm_key
        stt_fr, end_fr = int((stt - offset) * feat_per_sec), int((end - offset)* feat_per_sec)
        feat_level_target[stt_fr:end_fr, spk] = 1
    return feat_level_target

    
def get_scale_mapping_list(uniq_timestamps):
    """
    Call get_argmin_mat function to find the index of the non-base-scale segment that is closest to the
    given base-scale segment. For each scale and each segment, a base-scale segment is assigned.

    Args:
        uniq_timestamps: (dict)
            The dictionary containing embeddings, timestamps and multiscale weights.
            If uniq_timestamps contains only one scale, single scale diarization is performed.

    Returns:
        scale_mapping_argmat (torch.tensor):

            The element at the m-th row and the n-th column of the scale mapping matrix indicates the (m+1)-th scale
            segment index which has the closest center distance with (n+1)-th segment in the base scale.

            - Example:
                `scale_mapping_argmat[2][101] = 85`

            In the above example, the code snippet means that 86-th segment in the 3rd scale (python index is 2) is
            mapped to the 102-th segment in the base scale. Thus, the longer segments bound to have more repeating
            numbers since multiple base scale segments (since the base scale has the shortest length) fall into the
            range of the longer segments. At the same time, each row contains N numbers of indices where N is number
            of segments in the base-scale (i.e., the finest scale).
    """
    timestamps_in_scales = []
    for key, val in uniq_timestamps['scale_dict'].items():
        timestamps_in_scales.append(torch.tensor(val['time_stamps']))
    session_scale_mapping_list = get_argmin_mat(timestamps_in_scales)
    scale_mapping_argmat = [[] for _ in range(len(uniq_timestamps['scale_dict'].keys()))]
    for scale_idx in range(len(session_scale_mapping_list)):
        scale_mapping_argmat[scale_idx] = session_scale_mapping_list[scale_idx]
    scale_mapping_argmat = torch.stack(scale_mapping_argmat)
    return scale_mapping_argmat


def extract_seg_info_from_rttm(uniq_id, offset, duration, rttm_lines, mapping_dict=None, target_spks=None, round_digits=3):
    """
    Get RTTM lines containing speaker labels, start time and end time. target_spks contains two targeted
    speaker indices for creating groundtruth label files. Only speakers in target_spks variable will be
    included in the output lists.
    """
    rttm_stt, rttm_end = offset, offset + duration
    stt_list, end_list, speaker_list, pairwise_infer_spks = [], [], [], []

    speaker_set = []
    sess_to_global_spkids = dict()
    for rttm_line in rttm_lines:
        start, end, speaker = convert_rttm_line(rttm_line)
        if start > end:
            continue
        if (end > rttm_stt and start < rttm_end) or (start < rttm_end and end > rttm_stt):
            start, end = max(start, rttm_stt), min(end, rttm_end)
        else:
            continue
        # if target_spks is None or speaker in pairwise_infer_spks:
        end_list.append(round(end, round_digits))
        stt_list.append(round(start, round_digits))
        if speaker not in speaker_set:
            speaker_set.append(speaker)
        speaker_list.append(speaker_set.index(speaker))
        sess_to_global_spkids.update({speaker_set.index(speaker):speaker})
    rttm_mat = (stt_list, end_list, speaker_list)
    return rttm_mat, sess_to_global_spkids

## copied from trainer dataloader
def get_soft_label_vectors(
    feat_level_target, 
    duration, 
    ms_seg_counts,
    feat_per_sec,
    seg_stride,
    feat_per_segment,
    max_spks,
    ):
    """
    Generate the final targets for the actual diarization step.
    """
    soft_label_vec_list = []
    stride = int(feat_per_sec * seg_stride)
    for index in range(torch.max(ms_seg_counts)):
        seg_stt_feat, seg_end_feat = (stride * index), (stride * index + feat_per_segment)
        if seg_stt_feat < feat_level_target.shape[0]:
            range_label_sum = torch.sum(feat_level_target[seg_stt_feat:seg_end_feat, :], axis=0)
            soft_label_vec_list.append(range_label_sum)
        else:
            range_label_sum = torch.zeros(max_spks)
            range_label_sum[0] = int(feat_per_segment) # Silence label should exist always
            soft_label_vec_list.append(range_label_sum)
    return soft_label_vec_list

def get_step_level_targets(
    soft_label_vec_list,
    feat_per_segment,
    randomize_overlap_labels,
    div_n,
    soft_label_thres,
    sess_to_global_spkids,
    ): 
    soft_label_sum = torch.stack(soft_label_vec_list)
    total_steps = soft_label_sum.shape[0]
    label_total = soft_label_sum.sum(dim=1) # Only sum speaker labels, not silence at dim 0
    label_total = torch.clamp(label_total, max=feat_per_segment) # Clamp the maximum value to make max vector value 1
    label_total[label_total == 0] = 1 # Avoid divide by zero by assigning 1
    if randomize_overlap_labels:
        # Randomize the overlap labels to shuffle argmax function results
        soft_label_sum = (torch.rand_like(soft_label_sum)/div_n + (1- 1/div_n)) * soft_label_sum
    soft_label_vec = (soft_label_sum.t()/label_total).t()
    step_target = (soft_label_vec >= soft_label_thres).float()
    base_clus_label = soft_label_vec.argmax(dim=1) 
    base_clus_label[soft_label_vec.sum(dim=1)== 0] = -1 # If there is no existing label, put -1 
    if base_clus_label.shape[0] != total_steps:
        raise ValueError(f"base_clus_label.shape[0] != total_steps, {base_clus_label.shape[0]} != {total_steps}")
    return step_target, base_clus_label

def get_global_seg_spk_labels(sess_to_global_spkids, base_clus_label, global_speaker_label_table):
    if sess_to_global_spkids is not None: 
        global_seg_int_labels =[]
        for _, global_str_id in sess_to_global_spkids.items():
            global_int_label = global_speaker_label_table[global_str_id]
            global_seg_int_labels.append(global_int_label)
        global_seg_int_labels.append(0) # This is for silence (-1), silence gets 0 global int speaker label
        global_seg_int_labels = torch.tensor(global_seg_int_labels).int()
    global_seg_spk_labels = global_seg_int_labels[base_clus_label]
    return global_seg_spk_labels

def parse_rttm_for_ms_targets(
    uniq_id, 
    rttm_file, 
    offset, 
    duration, 
    ms_seg_counts,
    round_digits,
    feat_per_sec,
    seg_stride,
    max_spks,
    feat_per_segment,
    randomize_overlap_labels,
    div_n,
    soft_label_thres,
    global_speaker_label_table,
    ):
    """
    Generate target tensor variable by extracting groundtruth diarization labels from an RTTM file.
    """
    rttm_lines = open(rttm_file).readlines()
    # uniq_id = get_uniq_id_with_range(sample)
    rttm_timestamps, sess_to_global_spkids = extract_seg_info_from_rttm(uniq_id, offset, duration, rttm_lines)
    fr_level_target = get_frame_targets_from_rttm(rttm_timestamps=rttm_timestamps, 
                                                    offset=offset,
                                                    duration=duration,
                                                    round_digits=round_digits, 
                                                    feat_per_sec=feat_per_sec, 
                                                    max_spks=max_spks)
    
    soft_label_vectors = get_soft_label_vectors(feat_level_target=fr_level_target, 
                                                duration=duration, 
                                                ms_seg_counts=ms_seg_counts,
                                                feat_per_sec=feat_per_sec,
                                                seg_stride=seg_stride,
                                                feat_per_segment=feat_per_segment,
                                                max_spks=max_spks)

    seg_target, base_clus_label = get_step_level_targets(soft_label_vec_list=soft_label_vectors,
                                                        feat_per_segment=feat_per_segment,
                                                        randomize_overlap_labels=randomize_overlap_labels,
                                                        div_n=div_n,
                                                        soft_label_thres=soft_label_thres,
                                                        sess_to_global_spkids=sess_to_global_spkids) 
    # global_seg_spk_labels = get_global_seg_spk_labels(sess_to_global_spkids=sess_to_global_spkids,
    #                                                     base_clus_label=base_clus_label,
    #                                                     global_speaker_label_table=global_speaker_label_table)
    
    return seg_target

def get_speaker_labels_from_diar_rttms(collection):
    global_speaker_set = set()
    for diar_label_entity in collection:
        spk_id_list = list(diar_label_entity.sess_spk_dict.values())
        global_speaker_set.update(set(spk_id_list))
       
    global_speaker_register_dict = {'[sil]': 0}
    for global_int_spk_label, spk_id_str in enumerate(global_speaker_set):
        global_speaker_register_dict[spk_id_str] = global_int_spk_label + 1
        
    return global_speaker_register_dict

class _AudioMSDDTrainDataset(Dataset):
    """
    Dataset class that loads a json file containing paths to audio files,
    RTTM files and number of speakers. This Dataset class is designed for
    training or fine-tuning speaker embedding extractor and diarization decoder
    at the same time.

    Example:
    {"audio_filepath": "/path/to/audio_0.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_0.rttm}
    ...
    {"audio_filepath": "/path/to/audio_n.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_n.rttm}

    Args:
        manifest_filepath (str):
            Path to input manifest json files.
        multiscale_args_dict (dict):
            Dictionary containing the parameters for multiscale segmentation and clustering.
        emb_dir (str):
            Path to a temporary folder where segmentation information for embedding extraction is saved.
        soft_label_thres (float):
            Threshold that determines the label of each segment based on RTTM file information.
        featurizer:
            Featurizer instance for generating audio_signal from the raw waveform.
        window_stride (float):
            Window stride for acoustic feature. This value is used for calculating the numbers of feature-level frames.
        emb_batch_size (int):
            Number of embedding vectors that are trained with attached computational graphs.
        pairwise_infer (bool):
            This variable should be True if dataloader is created for an inference task.
        random_flip (bool):
            If True, the two labels and input signals are randomly flipped per every epoch while training.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports."""
        output_types = {
            "audio_signal": NeuralType(('B', 'T'), AudioSignal()),
            "feature_length": NeuralType(('B'), LengthsType()),
            "ms_seg_timestamps": NeuralType(('B', 'C', 'T', 'D'), LengthsType()),
            "ms_seg_counts": NeuralType(('B', 'C'), LengthsType()),
            "clus_label_index": NeuralType(('B', 'T'), LengthsType()),
            "scale_mapping": NeuralType(('B', 'C', 'T'), LengthsType()),
            "ch_clus_mat": NeuralType(('B', 'C', 'C'), LengthsType()),
            "targets": NeuralType(('B', 'T', 'C'), ProbsType()),
            "global_spk_labels": NeuralType(('B', 'T'), LengthsType()),
        }

        return output_types

    def __init__(
        self,
        *,
        manifest_filepath: str,
        multiscale_args_dict: str,
        emb_dir: str,
        soft_label_thres: float,
        session_len_sec: float,
        num_spks: int,
        featurizer,
        window_stride,
        emb_batch_size,
        pairwise_infer: bool,
        min_subsegment_duration: float = 0.03,
        random_flip: bool = True,
        global_rank: int = 0,
        num_workers: int = 30,
        dtype=torch.float32,
        randomize_overlap_labels: bool = True,
        randomize_offset: bool = True,
        encoder_infer_mode: bool = False,
        perm_mock_embs: bool = False,
        mock_emb_noise_std: float = 0.05,
        mock_emb_degree_of_freedom: int = 16,
        validation_mode: bool = False,
    ):
        super().__init__()
        self.collection = DiarizationSpeechLabel(
            manifests_files=manifest_filepath.split(','),
            # emb_dict=None,
            clus_label_dict=None,
            pairwise_infer=pairwise_infer,
        )
        self.emb_dim = 192
        self.mock_emb_degree_of_freedom = mock_emb_degree_of_freedom
        self.validation_mode = validation_mode
        self.perm_mock_embs = perm_mock_embs
        self.mock_emb_noise_std = mock_emb_noise_std
        self.featurizer = featurizer
        self.multiscale_args_dict = multiscale_args_dict
        self.session_len_sec = session_len_sec
        self.scale_n = len(self.multiscale_args_dict['scale_dict'])
        self.scale_dict = {int(k): v for k, v in multiscale_args_dict['scale_dict'].items()}
        self.feat_per_sec = int(1 / window_stride)
        self.feat_per_segment = int(self.scale_dict[self.scale_n-1][0] / window_stride)

        self.seg_stride = self.scale_dict[self.scale_n-1][1]
        self.max_raw_feat_len = int(self.multiscale_args_dict['scale_dict'][0][0] * self.feat_per_sec)
        self.random_flip = random_flip
        if random_flip:
            self.randomize_overlap_labels = True
            self.randomize_offset = False
        else:
            self.randomize_overlap_labels = False
            self.randomize_offset = False

        self.div_n = 20
        self.emb_dir = emb_dir
        self.round_digits = 2
        self.decim = 10 ** self.round_digits
        self.soft_label_thres = soft_label_thres
        self.pairwise_infer = pairwise_infer
        self.max_spks = num_spks
        self.emb_batch_size = emb_batch_size
        self.global_rank = global_rank
        self.manifest_filepath = manifest_filepath
        self.min_subsegment_duration = min_subsegment_duration
        self.num_workers = num_workers
        self.dtype = dtype
        self.encoder_infer_mode = encoder_infer_mode
        self.global_speaker_label_table = get_speaker_labels_from_diar_rttms(self.collection)
        self.ch_clus_mat_dict = {}
        self.channel_cluster_dict = {}
        # self.use_1ch_from_ch_clus = True
        self.use_1ch_from_ch_clus = False
        self.ms_seg_timestamps, self.ms_seg_counts = self.get_ms_seg_timestamps(duration=self.session_len_sec, min_subsegment_duration=self.scale_dict[self.scale_n-1][0]) 
        self.scale_mapping = torch.stack(get_argmin_mat(self.ms_seg_timestamps))
        self.min_noise_std = 0.005
  
    def __len__(self):
        return len(self.collection)

    def get_soft_label_vectors(self, feat_level_target, duration, ms_seg_counts):
        """
        Generate the final targets for the actual diarization step.
        Here, frame level means step level which is also referred to as segments.
        We follow the original paper and refer to the step level as "frames".

        Args:
            feat_level_target (torch.tensor):
                Tensor variable containing hard-labels of speaker activity in each feature-level segment.
            duration (float):
                Duration of the audio file in seconds.

        Returns:
            step_target (torch.tensor):
                Tensor variable containing hard-labels of speaker activity in each step-level segment.
        """
        soft_label_vec_list = []
        stride = int(self.feat_per_sec * self.seg_stride)
        for index in range(torch.max(ms_seg_counts)):
            seg_stt_feat, seg_end_feat = (stride * index), (stride * index + self.feat_per_segment)
            if seg_stt_feat < feat_level_target.shape[0]:
                range_label_sum = torch.sum(feat_level_target[seg_stt_feat:seg_end_feat, :], axis=0)
                soft_label_vec_list.append(range_label_sum)
            else:
                range_label_sum = torch.zeros(self.max_spks)
                range_label_sum[0] = int(self.feat_per_segment) # Silence label should exist always
                soft_label_vec_list.append(range_label_sum)
        return soft_label_vec_list

    def get_step_level_targets(self, soft_label_vec_list, sess_to_global_spkids): 
        soft_label_sum = torch.stack(soft_label_vec_list)
        total_steps = soft_label_sum.shape[0]
        label_total = soft_label_sum.sum(dim=1) # Only sum speaker labels, not silence at dim 0
        label_total = torch.clamp(label_total, max=self.feat_per_segment) # Clamp the maximum value to make max vector value 1
        label_total[label_total == 0] = 1 # Avoid divide by zero by assigning 1
        if self.randomize_overlap_labels:
            # Randomize the overlap labels to shuffle argmax function results
            soft_label_sum = (torch.rand_like(soft_label_sum)/self.div_n + (1- 1/self.div_n)) * soft_label_sum
        soft_label_vec = (soft_label_sum.t()/label_total).t()
        step_target = (soft_label_vec >= self.soft_label_thres).float()

        base_clus_label = soft_label_vec.argmax(dim=1) 
        base_clus_label[soft_label_vec.sum(dim=1)== 0] = -1 # If there is no existing label, put -1 
        if base_clus_label.shape[0] != total_steps:
            raise ValueError(f"base_clus_label.shape[0] != total_steps, {base_clus_label.shape[0]} != {total_steps}")
        return step_target, base_clus_label

    def parse_rttm_for_ms_targets(self, uniq_id, rttm_file, offset, duration, ms_seg_counts):
        """
        Generate target tensor variable by extracting groundtruth diarization labels from an RTTM file.
        This function converts (start, end, speaker_id) format into base-scale (the finest scale) segment level
        diarization label in a matrix form.

        Example of seg_target:
            [[0., 1.], [0., 1.], [1., 1.], [1., 0.], [1., 0.], ..., [0., 1.]]

        Args:
            sample:
                `DiarizationSpeechLabel` instance containing sample information such as audio filepath and RTTM filepath.
            target_spks (tuple):
                Speaker indices that are generated from combinations. If there are only one or two speakers,
                only a single target_spks tuple is generated.

        Returns:
            clus_label_index (torch.tensor):
                Groundtruth clustering label (cluster index for each segment) from RTTM files for training purpose.
            seg_target  (torch.tensor):
                Tensor variable containing hard-labels of speaker activity in each base-scale segment.
            scale_mapping (torch.tensor):
                Matrix containing the segment indices of each scale. scale_mapping is necessary for reshaping the
                multiscale embeddings to form an input matrix for the MSDD model.
        """
        rttm_lines = open(rttm_file).readlines()
        # uniq_id = self.get_uniq_id_with_range(sample)
        rttm_timestamps, sess_to_global_spkids = extract_seg_info_from_rttm(uniq_id, offset, duration, rttm_lines)

        fr_level_target = get_frame_targets_from_rttm(rttm_timestamps=rttm_timestamps, 
                                                      offset=offset,
                                                      duration=duration,
                                                      round_digits=self.round_digits, 
                                                      feat_per_sec=self.feat_per_sec, 
                                                      max_spks=self.max_spks) 
        soft_label_vectors = self.get_soft_label_vectors(feat_level_target=fr_level_target, 
                                                         duration=duration, 
                                                         ms_seg_counts=ms_seg_counts)
        seg_target, base_clus_label = self.get_step_level_targets(soft_label_vec_list=soft_label_vectors, sess_to_global_spkids=sess_to_global_spkids)
        return seg_target

    def get_uniq_id_with_range(self, sample, deci=3):
        """
        Generate unique training sample ID from unique file ID, offset and duration. The start-end time added
        unique ID is required for identifying the sample since multiple short audio samples are generated from a single
        audio file. The start time and end time of the audio stream uses millisecond units if `deci=3`.

        Args:
            sample:
                `DiarizationSpeechLabel` instance from collections.

        Returns:
            uniq_id (str):
                Unique sample ID which includes start and end time of the audio stream.
                Example: abc1001_3122_6458
        """
        bare_uniq_id = os.path.splitext(os.path.basename(sample.rttm_file))[0]
        offset = str(int(round(sample.offset, deci) * pow(10, deci)))
        endtime = str(int(round(sample.offset + sample.duration, deci) * pow(10, deci)))
        uniq_id = f"{bare_uniq_id}_{offset}_{endtime}"
        return uniq_id

    def get_subsegments_to_scale_timestamps(self, subsegments: List[Tuple[float, float]], decimals=2):
        """
        Convert subsegment timestamps to scale timestamps.

        Args:
            subsegments (List[Tuple[float, float]]):
                List of subsegment timestamps.

        Returns:
            scale_ts (torch.tensor):
                Tensor containing scale timestamps.
        """
        # scale_ts = (torch.tensor(subsegments) * self.feat_per_sec).long()
        seg_ts = (torch.tensor(subsegments) * self.feat_per_sec).float()
        scale_ts_round = torch.round(seg_ts, decimals=decimals)
        scale_ts = scale_ts_round.long()
        scale_ts[:, 1] = scale_ts[:, 0] + scale_ts[:, 1]
        return scale_ts 

    def get_ms_seg_timestamps(
        self, 
        duration: float, 
        min_subsegment_duration: float=0.03
        ):
        """
        Get start and end time of segments in each scale.

        Args:
            sample:
                `DiarizationSpeechLabel` instance from preprocessing.collections
        Returns:
            ms_seg_timestamps (torch.tensor):
                Tensor containing Multiscale segment timestamps.
            ms_seg_counts (torch.tensor):
                Number of segments for each scale. This information is used for reshaping embedding batch
                during forward propagation.
        """
        if duration < 0:
            raise ValueError(f"duration {duration} cannot be negative")
        ms_seg_timestamps_list = []
        total_steps = None
        ms_seg_counts = [0 for _ in range(self.scale_n)]
        for scale_idx in reversed(range(self.scale_n)):
            subsegments = get_subsegments(offset=0, 
                                          window=self.multiscale_args_dict['scale_dict'][scale_idx][0],
                                          shift=self.multiscale_args_dict['scale_dict'][scale_idx][1],
                                          duration=duration, 
                                          min_subsegment_duration=min_subsegment_duration)
            scale_ts_tensor = self.get_subsegments_to_scale_timestamps(subsegments)
            if scale_idx == self.scale_n - 1:
                total_steps = scale_ts_tensor.shape[0]
            ms_seg_counts[scale_idx] = scale_ts_tensor.shape[0]
            scale_ts_padded = torch.cat([scale_ts_tensor, torch.zeros(total_steps - scale_ts_tensor.shape[0], 2, dtype=scale_ts_tensor.dtype)], dim=0)
            ms_seg_timestamps_list.append(scale_ts_padded.detach())
        ms_seg_timestamps_list = ms_seg_timestamps_list[::-1]
        ms_seg_timestamps = torch.stack(ms_seg_timestamps_list).type(self.dtype)
        ms_seg_counts = torch.tensor(ms_seg_counts)
        return ms_seg_timestamps, ms_seg_counts
    
    def channel_cluster(self, mc_audio_signal, sample_rate):
        clusters, mag_coherence = channel_cluster_from_coherence(
                                    audio_signal=mc_audio_signal.t(),
                                    sample_rate=sample_rate,
                                    output_coherence=True,
                                    ) 
        ch_clus_mat = get_channel_averaging_matrix(clusters)
        # if self.use_1ch_from_ch_clus:
        #     ch_inds = torch.max(ch_clus_mat, dim=1)[1]
        #     ch_clus_mat = torch.zeros_like(ch_clus_mat)
        #     ch_clus_mat[torch.arange(ch_inds.shape[0]), ch_inds] = 1
        return ch_clus_mat
    
    def get_random_cov_mat(self, emb_dim: int = 192, max_spks: int = 4, mock_emb_degree_of_freedom: int = 16):
        source_vecs = torch.rand(max_spks, emb_dim, mock_emb_degree_of_freedom)
        covs = torch.bmm(source_vecs, source_vecs.transpose(1,2)) 
        covs = covs / torch.max(covs)
        covs.add_(torch.eye(emb_dim).unsqueeze(0)*0.001)
        covs = torch.block_diag(*covs)
        return covs
        
    def __getitem__(self, index):
        # print(f"Starting dataloader item idx: {index}")
        start_time = time.time()
        sample = self.collection[index]
        if sample.offset is None:
            sample.offset = 0
        offset = sample.offset
        duration = min(sample.duration, self.session_len_sec)
        # duration = self.session_len_sec

        uniq_id = self.get_uniq_id_with_range(sample)
        # ms_seg_timestamps, ms_seg_counts = self.get_ms_seg_timestamps(uniq_id=uniq_id, 
        #                                                               offset=offset,
        #                                                               duration=self.session_len_sec,
        #                                                               feat_per_sec=self.feat_per_sec, 
        #                                                               min_subsegment_duration=self.scale_dict[self.scale_n-1][0])
        targets= self.parse_rttm_for_ms_targets(uniq_id=uniq_id, 
                                                rttm_file=sample.rttm_file,
                                                offset=offset,
                                                duration=duration,
                                                ms_seg_counts=self.ms_seg_counts)

        # audio_signal = torch.nn.functional.pad(torch.tensor([]), (0, 0, 0, self.session_len_sec*self.featurizer.sample_rate), mode='constant', value=0) 
        audio_signal = torch.zeros(self.session_len_sec*self.featurizer.sample_rate)
        torch.manual_seed(targets.sum().long() + index)
        base_dim = self.max_spks * self.mock_emb_degree_of_freedom
        if base_dim > self.emb_dim:
            raise ValueError(f"max_spks * mock_emb_degree_of_freedom {base_dim} cannot be larger than emb_dim {self.emb_dim}")
        seq_len, n_spks = targets.shape
        rep_count = int(self.emb_dim//self.mock_emb_degree_of_freedom)
        
        # Generate `self.max_spks` worth of random embeddings
        mvec = torch.randn(self.max_spks * self.mock_emb_degree_of_freedom)
        zero_mean = torch.zeros_like(mvec)
        covs = self.get_random_cov_mat(emb_dim=self.mock_emb_degree_of_freedom, 
                                       max_spks=self.max_spks, 
                                       mock_emb_degree_of_freedom=self.mock_emb_degree_of_freedom)
            
        # Create a correlated multivariate normal noise 
        multivariate_normal = torch.distributions.MultivariateNormal(zero_mean, covs)
        zero_mean_mvgn_emb_randomness =  multivariate_normal.sample(sample_shape=(seq_len,))
        mvgn_emb_seed = mvec.unsqueeze(0).repeat(seq_len, 1) +  self.mock_emb_noise_std * zero_mean_mvgn_emb_randomness
        
        # Mix the embeddings according to the speaker tracks
        spk_wise_mvgn_emb_seed =  mvgn_emb_seed.reshape(seq_len, self.max_spks, self.mock_emb_degree_of_freedom)
        spk_wise_mvgn_emb_seed = spk_wise_mvgn_emb_seed.repeat_interleave(rep_count, dim=-1)
        mixture_emb_seed_clean = torch.bmm(targets.unsqueeze(1), spk_wise_mvgn_emb_seed).squeeze(1)
        
        # Add noise to the embedding seed
        mixture_emb_seed_dirty = mixture_emb_seed_clean + torch.randn_like(mixture_emb_seed_clean) * (self.mock_emb_noise_std + self.min_noise_std)
        feature_length =  torch.tensor(mixture_emb_seed_dirty.shape[0]).long()
        return mixture_emb_seed_dirty, feature_length, targets



def _msdd_train_collate_fn(self, batch):
    packed_batch = list(zip(*batch))
    speaker_seed, feature_length, targets = packed_batch
    speaker_seed_list, feature_length_list = [], []
    
    ms_seg_timestamps_list, ms_seg_counts_list, scale_clus_label_list, scale_mapping_list, targets_list, global_spk_labels_list = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    max_raw_feat_len = max([x.shape[0] for x in speaker_seed])
    max_target_len = max([x.shape[0] for x in targets])
    # max_total_seg_len = max([x.shape[0] for x in clus_label_index])
    if max([len(feat.shape) for feat in speaker_seed]) > 1:
        max_ch = max([feat.shape[1] for feat in speaker_seed])
    else:
        max_ch = 1

    for feat, feat_len, tgt in batch:
        seq_len = tgt.shape[0]
        if len(feat.shape) > 1:
            pad_feat = (0, 0, 0, max_raw_feat_len - feat.shape[0])
        else:
            pad_feat = (0, max_raw_feat_len - feat.shape[0])
        if feat.shape[0] < feat_len:
            feat_len_pad = feat_len - feat.shape[0]
            feat = torch.nn.functional.pad(feat, (0, feat_len_pad))
        pad_tgt = (0, 0, 0, max_target_len - seq_len)
        padded_feat = torch.nn.functional.pad(feat, pad_feat)
        padded_tgt = torch.nn.functional.pad(tgt, pad_tgt)

        
        if max_ch > 1 and padded_feat.shape[1] < max_ch:
            feat_ch_pad = max_ch - padded_feat.shape[1]
            padded_feat = torch.nn.functional.pad(padded_feat, (0, feat_ch_pad))

        speaker_seed_list.append(padded_feat)
        feature_length_list.append(feat_len.clone().detach())
        targets_list.append(padded_tgt)
        speaker_seed = torch.stack(speaker_seed_list)
    feature_length = torch.stack(feature_length_list)
    targets = torch.stack(targets_list)
    return speaker_seed, feature_length, targets 

def XX_msdd_train_collate_fn(self, batch):
    """
    Collate batch of variables that are needed for raw waveform to diarization label training.
    The following variables are included in training/validation batch:

    Args:
        batch (tuple:
            Batch tuple containing the variables for the diarization training.
    Returns:
        audio_signal (torch.tensor):
            Raw waveform samples (time series) loaded from the audio_filepath in the input manifest file.
        feature lengths (time series sample length):
            A list of lengths of the raw waveform samples.
        ms_seg_timestamps (torch.tensor):
            Matrix containing the start time and end time (timestamps) for each segment and each scale.
            ms_seg_timestamps is needed for extracting acoustic features from raw waveforms.
        ms_seg_counts (torch.tensor):
            Matrix containing The number of segments for each scale. ms_seg_counts is necessary for reshaping
            the input matrix for the MSDD model.
        clus_label_index (torch.tensor):
            Groundtruth Clustering label (cluster index for each segment) from RTTM files for training purpose.
            clus_label_index is necessary for calculating cluster-average embedding.
        scale_mapping (torch.tensor):
            Matrix containing the segment indices of each scale. scale_mapping is necessary for reshaping the
            multiscale embeddings to form an input matrix for the MSDD model.
        targets (torch.tensor):
            Groundtruth Speaker label for the given input embedding sequence.
    """
    packed_batch = list(zip(*batch))
    # audio_signal, feature_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, ch_clus_arrays, targets, global_spk_labels = packed_batch
    audio_signal, feature_length, targets = packed_batch
    audio_signal_list, feature_length_list = [], []
    
    ms_seg_timestamps_list, ms_seg_counts_list, scale_clus_label_list, scale_mapping_list, targets_list, global_spk_labels_list = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    max_raw_feat_len = max([x.shape[0] for x in audio_signal])
    max_target_len = max([x.shape[0] for x in targets])
    # max_total_seg_len = max([x.shape[0] for x in clus_label_index])
    if max([len(feat.shape) for feat in audio_signal]) > 1:
        max_ch = max([feat.shape[1] for feat in audio_signal])
    else:
        max_ch = 1
    # max_clus_ch = max([clus_arr.shape[0] for clus_arr in ch_clus_arrays])
    # arg_max_idx = torch.argmax(torch.tensor([ x.shape[1] for x in ms_seg_timestamps]))
    # ms_seg_ts = ms_seg_timestamps[arg_max_idx]
    # ms_seg_ct = ms_seg_counts[arg_max_idx]

    # for feat, feat_len, _, _, scale_clus, scl_map, ch_clus_m, tgt, glb_lbl in batch:
    for feat, feat_len, tgt in batch:
        seq_len = tgt.shape[0]
        if len(feat.shape) > 1:
            pad_feat = (0, 0, 0, max_raw_feat_len - feat.shape[0])
        else:
            pad_feat = (0, max_raw_feat_len - feat.shape[0])
        if feat.shape[0] < feat_len:
            feat_len_pad = feat_len - feat.shape[0]
            feat = torch.nn.functional.pad(feat, (0, feat_len_pad))
        pad_tgt = (0, 0, 0, max_target_len - seq_len)
        # pad_sm = (0, max_target_len - seq_len)
        # pad_sc = (0, max_total_seg_len - scale_clus.shape[0])
        # pad_glb_lbl = (0, max_total_seg_len - glb_lbl.shape[0])
        # pad_chclus = (0, max_ch - ch_clus_m.shape[1], 0, max_clus_ch - ch_clus_m.shape[0])
        
        padded_feat = torch.nn.functional.pad(feat, pad_feat)
        padded_tgt = torch.nn.functional.pad(tgt, pad_tgt)
        # padded_sm = torch.nn.functional.pad(scl_map, pad_sm)
        # padded_scale_clus = torch.nn.functional.pad(scale_clus, pad_sc, value=-1)
        # padded_global_label = torch.nn.functional.pad(glb_lbl, pad_glb_lbl, value=-1)
        # padded_chclus = torch.nn.functional.pad(ch_clus_m, pad_chclus)
        
        if max_ch > 1 and padded_feat.shape[1] < max_ch:
            feat_ch_pad = max_ch - padded_feat.shape[1]
            padded_feat = torch.nn.functional.pad(padded_feat, (0, feat_ch_pad))

        audio_signal_list.append(padded_feat)
        feature_length_list.append(feat_len.clone().detach())
        # ms_seg_timestamps_list.append(ms_seg_ts)
        # ms_seg_counts_list.append(ms_seg_ct.clone().detach())
        # scale_clus_label_list.append(padded_scale_clus)
        # global_spk_labels_list.append(padded_global_label)
        # scale_mapping_list.append(padded_sm)
        targets_list.append(padded_tgt)
        # ch_clus_list.append(padded_chclus)
        audio_signal = torch.stack(audio_signal_list)
    feature_length = torch.stack(feature_length_list)
        
    # ms_seg_timestamps = torch.stack(ms_seg_timestamps_list)
    # clus_label_index = torch.stack(scale_clus_label_list)
    # ms_seg_counts = torch.stack(ms_seg_counts_list)
    # scale_mapping = torch.stack(scale_mapping_list)
    targets = torch.stack(targets_list)
    # ch_clus_mat = torch.stack(ch_clus_list)
    # global_spk_labels = torch.stack(global_spk_labels_list)
    # return audio_signal, feature_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, ch_clus_mat, targets, global_spk_labels
    return audio_signal, feature_length, targets



class AudioToSpeechMSDDTrainDataset(_AudioMSDDTrainDataset):
    """
    Dataset class that loads a json file containing paths to audio files,
    rttm files and number of speakers. This Dataset class is designed for
    training or fine-tuning speaker embedding extractor and diarization decoder
    at the same time.

    Example:
    {"audio_filepath": "/path/to/audio_0.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_0.rttm}
    ...
    {"audio_filepath": "/path/to/audio_n.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_n.rttm}

    Args:
        manifest_filepath (str):
            Path to input manifest json files.
        multiscale_args_dict (dict):
            Dictionary containing the parameters for multiscale segmentation and clustering.
        emb_dir (str):
            Path to a temporary folder where segmentation information for embedding extraction is saved.
        soft_label_thres (float):
            A threshold that determines the label of each segment based on RTTM file information.
        featurizer:
            Featurizer instance for generating features from the raw waveform.
        window_stride (float):
            Window stride for acoustic feature. This value is used for calculating the numbers of feature-level frames.
        emb_batch_size (int):
            Number of embedding vectors that are trained with attached computational graphs.
        pairwise_infer (bool):
            This variable should be True if dataloader is created for an inference task.
    """

    def __init__(
        self,
        *,
        manifest_filepath: str,
        multiscale_args_dict: Dict,
        emb_dir: str,
        soft_label_thres: float,
        session_len_sec: float,
        random_flip: bool,
        num_spks: int,
        featurizer,
        window_stride,
        emb_batch_size,
        pairwise_infer: bool,
        global_rank: int,
        encoder_infer_mode: bool,        
        perm_mock_embs: bool = False,
        mock_emb_noise_std: float = 0.05,
        mock_emb_degree_of_freedom: int = 16,
        validation_mode: bool = False,
    ):
        super().__init__(
            manifest_filepath=manifest_filepath,
            multiscale_args_dict=multiscale_args_dict,
            emb_dir=emb_dir,
            soft_label_thres=soft_label_thres,
            session_len_sec=session_len_sec,
            random_flip=random_flip,
            num_spks=num_spks,
            featurizer=featurizer,
            window_stride=window_stride,
            emb_batch_size=emb_batch_size,
            pairwise_infer=pairwise_infer,
            global_rank=global_rank,
            encoder_infer_mode=encoder_infer_mode,
            perm_mock_embs=perm_mock_embs,
            mock_emb_noise_std=mock_emb_noise_std,
            mock_emb_degree_of_freedom=mock_emb_degree_of_freedom,
            validation_mode=validation_mode,
        )

    def msdd_train_collate_fn(self, batch):
        return _msdd_train_collate_fn(self, batch)