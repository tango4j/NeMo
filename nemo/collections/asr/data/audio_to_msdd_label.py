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

import torch
import numpy as np

from nemo.collections.asr.parts.utils.offline_clustering import get_argmin_mat
from nemo.collections.asr.parts.utils.speaker_utils import convert_rttm_line, prepare_split_data, get_subsegments
from nemo.collections.common.parts.preprocessing.collections import DiarizationSpeechLabel
from nemo.core.classes import Dataset
from nemo.core.neural_types import AudioSignal, EncodedRepresentation, LengthsType, NeuralType, ProbsType

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

    Args:
        uniq_id (str):
            Unique file ID that refers to an input audio file and corresponding RTTM (Annotation) file.
        rttm_lines (list):
            List containing RTTM lines in str format.
        mapping_dict (dict):
            Mapping between the estimated speakers and the speakers in the ground-truth annotation.
            `mapping_dict` variable is only provided when the inference mode is running in sequence-eval mode.
            Sequence eval mode uses the mapping between the estimated speakers and the speakers in ground-truth annotation.
    Returns:
        rttm_tup (tuple):
            Tuple containing lists of start time, end time and speaker labels.

    """
    rttm_stt, rttm_end = offset, offset + duration
    stt_list, end_list, speaker_list, pairwise_infer_spks = [], [], [], []
    if target_spks:
        inv_map = {v: k for k, v in mapping_dict.items()}
        for spk_idx in target_spks:
            spk_str = f'speaker_{spk_idx}'
            if spk_str in inv_map:
                pairwise_infer_spks.append(inv_map[spk_str])
    speaker_set = []
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
    rttm_mat = (stt_list, end_list, speaker_list)
    return rttm_mat


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
            Featurizer instance for generating features from the raw waveform.
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
            "features": NeuralType(('B', 'T'), AudioSignal()),
            "feature_length": NeuralType(('B'), LengthsType()),
            "ms_seg_timestamps": NeuralType(('B', 'C', 'T', 'D'), LengthsType()),
            "ms_seg_counts": NeuralType(('B', 'C'), LengthsType()),
            "clus_label_index": NeuralType(('B', 'T'), LengthsType()),
            "scale_mapping": NeuralType(('B', 'C', 'T'), LengthsType()),
            "targets": NeuralType(('B', 'T', 'C'), ProbsType()),
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
    ):
        super().__init__()
        self.collection = DiarizationSpeechLabel(
            manifests_files=manifest_filepath.split(','),
            emb_dict=None,
            clus_label_dict=None,
            pairwise_infer=pairwise_infer,
        )
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
            self.randomize_overlap_labels = randomize_overlap_labels
            self.randomize_offset = randomize_offset
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

    def get_step_level_targets(self, soft_label_vec_list): 
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
        rttm_timestamps = extract_seg_info_from_rttm(uniq_id, offset, duration, rttm_lines)
        fr_level_target = get_frame_targets_from_rttm(rttm_timestamps=rttm_timestamps, 
                                                      offset=offset,
                                                      duration=duration,
                                                      round_digits=self.round_digits, 
                                                      feat_per_sec=self.feat_per_sec, 
                                                      max_spks=self.max_spks) 
        soft_label_vectors = self.get_soft_label_vectors(feat_level_target=fr_level_target, 
                                                         duration=duration, 
                                                         ms_seg_counts=ms_seg_counts)
        seg_target, base_clus_label = self.get_step_level_targets(soft_label_vec_list=soft_label_vectors)
        return seg_target, base_clus_label

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

    def get_subsegments_to_scale_timestamps(self, subsegments: List[Tuple[float, float]]):
        """
        Convert subsegment timestamps to scale timestamps.

        Args:
            subsegments (List[Tuple[float, float]]):
                List of subsegment timestamps.

        Returns:
            scale_ts (torch.tensor):
                Tensor containing scale timestamps.
        """
        scale_ts = (torch.tensor(subsegments) * self.feat_per_sec).long()
        scale_ts[:, 1] = scale_ts[:, 0] + scale_ts[:, 1]
        return scale_ts 

    def get_ms_seg_timestamps(
        self, 
        uniq_id: str, 
        offset: float, 
        duration: float, 
        feat_per_sec: int, 
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
    
    def __getitem__(self, index):
        sample = self.collection[index]
        if sample.offset is None:
            sample.offset = 0
        offset = sample.offset
        if self.randomize_offset and self.session_len_sec < sample.duration:
            n_durs = max(int( (sample.duration-self.session_len_sec) // self.session_len_sec) - 1, 0)
            offset = round(torch.randint(0, n_durs, (1,)).item() * self.session_len_sec, self.round_digits)
            duration = round(self.session_len_sec, self.round_digits)
        else:
            duration = min(sample.duration, self.session_len_sec)
            offset = 0
        uniq_id = self.get_uniq_id_with_range(sample)
        ms_seg_timestamps, ms_seg_counts = self.get_ms_seg_timestamps(uniq_id=uniq_id, 
                                                                      offset=offset,
                                                                      duration=duration,
                                                                      feat_per_sec=self.feat_per_sec)
        
        scale_mapping = torch.stack(get_argmin_mat(ms_seg_timestamps))
        targets, clus_label_index = self.parse_rttm_for_ms_targets(uniq_id=uniq_id, 
                                                                   rttm_file=sample.rttm_file,
                                                                   offset=offset,
                                                                   duration=duration,
                                                                   ms_seg_counts=ms_seg_counts)

        features = self.featurizer.process(sample.audio_file, offset=offset, duration=duration)
        feature_length = torch.tensor(features.shape[0]).long()
        if self.random_flip:
            flip = torch.cat([torch.randperm(self.max_spks), torch.tensor(-1).unsqueeze(0)])
            clus_label_index, targets = flip[clus_label_index], targets[:, flip[:self.max_spks]]
        if torch.max(ms_seg_counts) != clus_label_index.shape[0]:
            raise ValueError(f"ms_seg_counts: {ms_seg_counts}, clus_label_index.shape[0]: {clus_label_index.shape[0]}")
        return features, feature_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, targets


class _AudioMSDDInferDataset(Dataset):
    """
    Dataset class that loads a json file containing paths to audio files,
    RTTM files and number of speakers. This Dataset class is built for diarization inference and
    evaluation. Speaker embedding sequences, segment timestamps, cluster-average speaker embeddings
    are loaded from memory and fed into the dataloader.

    Example:
    {"audio_filepath": "/path/to/audio_0.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_0.rttm}
    ...
    {"audio_filepath": "/path/to/audio_n.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_n.rttm}

    Args:
        manifest_filepath (str):
             Path to input manifest json files.
        emb_dict (dict):
            Dictionary containing cluster-average embeddings and speaker mapping information.
        emb_seq (dict):
            Dictionary containing multiscale speaker embedding sequence, scale mapping and corresponding segment timestamps.
        clus_label_dict (dict):
            Subsegment-level (from base-scale) speaker labels from clustering results.
        soft_label_thres (float):
            A threshold that determines the label of each segment based on RTTM file information.
        featurizer:
            Featurizer instance for generating features from raw waveform.
        seq_eval_mode (bool):
            If True, F1 score will be calculated for each speaker pair during inference mode.
        window_stride (float):
            Window stride for acoustic feature. This value is used for calculating the numbers of feature-level frames.
        use_single_scale_clus (bool):
            Use only one scale for clustering instead of using multiple scales of embeddings for clustering.
        pairwise_infer (bool):
            This variable should be True if dataloader is created for an inference task.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports."""
        output_types = OrderedDict(
            {
                "ms_emb_seq": NeuralType(('B', 'T', 'C', 'D'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
                "ms_avg_embs": NeuralType(('B', 'C', 'D', 'C'), EncodedRepresentation()),
                "targets": NeuralType(('B', 'T', 'C'), ProbsType()),
            }
        )
        return output_types

    def __init__(
        self,
        *,
        manifest_filepath: str,
        emb_dict: Dict,
        emb_seq: Dict,
        clus_label_dict: Dict,
        soft_label_thres: float,
        seq_eval_mode: bool,
        window_stride: float,
        use_single_scale_clus: bool,
        pairwise_infer: bool,
    ):
        super().__init__()
        self.collection = DiarizationSpeechLabel(
            manifests_files=manifest_filepath.split(','),
            emb_dict=emb_dict,
            clus_label_dict=clus_label_dict,
            seq_eval_mode=seq_eval_mode,
            pairwise_infer=pairwise_infer,
        )
        self.emb_dict = emb_dict
        self.emb_seq = emb_seq
        self.clus_label_dict = clus_label_dict
        self.round_digits = 2
        self.decim = 10 ** self.round_digits
        self.feat_per_sec = int(1 / window_stride)
        self.soft_label_thres = soft_label_thres
        self.pairwise_infer = pairwise_infer
        self.max_spks = 2
        self.use_single_scale_clus = use_single_scale_clus
        self.seq_eval_mode = seq_eval_mode

    def __len__(self):
        return len(self.collection)

    def parse_rttm_multiscale(self, sample):
        """
        Generate target tensor variable by extracting groundtruth diarization labels from an RTTM file.
        This function is only used when ``self.seq_eval_mode=True`` and RTTM files are provided. This function converts
        (start, end, speaker_id) format into base-scale (the finest scale) segment level diarization label in a matrix
        form to create target matrix.

        Args:
            sample:
                DiarizationSpeechLabel instance containing sample information such as audio filepath and RTTM filepath.
            target_spks (tuple):
                Two Indices of targeted speakers for evaluation.
                Example of target_spks: (2, 3)
        Returns:
            seg_target (torch.tensor):
                Tensor variable containing hard-labels of speaker activity in each base-scale segment.
        """
        if sample.rttm_file is None:
            raise ValueError(f"RTTM file is not provided for this sample {sample}")
        rttm_lines = open(sample.rttm_file).readlines()
        uniq_id = os.path.splitext(os.path.basename(sample.rttm_file))[0]
        mapping_dict = self.emb_dict[max(self.emb_dict.keys())][uniq_id]['mapping']
        rttm_timestamps = extract_seg_info_from_rttm(uniq_id, offset, duration, rttm_lines, mapping_dict, sample.target_spks)
        fr_level_target = assign_frame_level_spk_vector(rttm_timestamps, 
                                                        offset, 
                                                        duration, 
                                                        self.round_digits, 
                                                        self.feat_per_sec, 
                                                        sample.target_spks
        )
        seg_target = self.get_diar_target_labels_from_fr_target(uniq_id, fr_level_target)
        return seg_target

    def get_diar_target_labels_from_fr_target(self, uniq_id, fr_level_target):
        """
        Generate base-scale level binary diarization label from frame-level target matrix. For the given frame-level
        speaker target matrix fr_level_target, we count the number of frames that belong to each speaker and calculate
        ratios for each speaker into the `soft_label_vec` variable. Finally, `soft_label_vec` variable is compared with `soft_label_thres`
        to determine whether a label vector should contain 0 or 1 for each speaker bin. Note that seg_target variable has
        dimension of (number of base-scale segments x 2) dimension.

        Example of seg_target:
            [[0., 1.], [0., 1.], [1., 1.], [1., 0.], [1., 0.], ..., [0., 1.]]

        Args:
            uniq_id (str):
                Unique file ID that refers to an input audio file and corresponding RTTM (Annotation) file.
            fr_level_target (torch.tensor):
                frame-level binary speaker annotation (1: exist 0: non-exist) generated from RTTM file.

        Returns:
            seg_target (torch.tensor):
                Tensor variable containing binary hard-labels of speaker activity in each base-scale segment.

        """
        if fr_level_target is None:
            return None
        else:
            seg_target_list = []
            for (seg_stt, seg_end, label_int) in self.clus_label_dict[uniq_id]:
                seg_stt_fr, seg_end_fr = int(seg_stt * self.feat_per_sec), int(seg_end * self.feat_per_sec)
                soft_label_vec = torch.sum(fr_level_target[seg_stt_fr:seg_end_fr, :], axis=0) / (
                    seg_end_fr - seg_stt_fr
                )
                label_vec = (soft_label_vec > self.soft_label_thres).int()
                seg_target_list.append(label_vec)
            seg_target = torch.stack(seg_target_list)
            return seg_target

    def __getitem__(self, index):
        sample = self.collection[index]
        if sample.offset is None:
            sample.offset = 0

        uniq_id = os.path.splitext(os.path.basename(sample.audio_file))[0]
        scale_n = len(self.emb_dict.keys())
        _avg_embs = torch.stack([self.emb_dict[scale_index][uniq_id]['avg_embs'] for scale_index in range(scale_n)])

        if self.pairwise_infer:
            avg_embs = _avg_embs[:, :, self.collection[index].target_spks]
        else:
            avg_embs = _avg_embs

        if avg_embs.shape[2] > self.max_spks:
            raise ValueError(
                f" avg_embs.shape[2] {avg_embs.shape[2]} should be less than or equal to self.max_num_speakers {self.max_spks}"
            )

        feats = []
        for scale_index in range(scale_n):
            repeat_mat = self.emb_seq["session_scale_mapping"][uniq_id][scale_index]
            feats.append(self.emb_seq[scale_index][uniq_id][repeat_mat, :])
        feats_out = torch.stack(feats).permute(1, 0, 2)
        feats_len = feats_out.shape[0]

        if self.seq_eval_mode:
            targets = self.parse_rttm_multiscale(sample)
        else:
            targets = torch.zeros(feats_len, 2).float()

        return feats_out, feats_len, targets, avg_embs


def _msdd_train_collate_fn(self, batch):
    """
    Collate batch of variables that are needed for raw waveform to diarization label training.
    The following variables are included in training/validation batch:

    Args:
        batch (tuple):
            Batch tuple containing the variables for the diarization training.
    Returns:
        features (torch.tensor):
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
    features, feature_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, targets = packed_batch
    features_list, feature_length_list = [], []
    ms_seg_timestamps_list, ms_seg_counts_list, scale_clus_label_list, scale_mapping_list, targets_list = (
        [],
        [],
        [],
        [],
        [],
    )

    max_raw_feat_len = max([x.shape[0] for x in features])
    max_target_len = max([x.shape[0] for x in targets])
    max_total_seg_len = max([x.shape[0] for x in clus_label_index])

    for feat, feat_len, ms_seg_ts, ms_seg_ct, scale_clus, scl_map, tgt in batch:
        seq_len = tgt.shape[0]
        pad_feat = (0, max_raw_feat_len - feat_len)
        pad_tgt = (0, 0, 0, max_target_len - seq_len)
        pad_sm = (0, max_target_len - seq_len)
        pad_ts = (0, 0, 0, max_target_len - seq_len)
        pad_sc = (0, max_total_seg_len - scale_clus.shape[0])
        padded_feat = torch.nn.functional.pad(feat, pad_feat)
        padded_tgt = torch.nn.functional.pad(tgt, pad_tgt)
        padded_sm = torch.nn.functional.pad(scl_map, pad_sm)
        padded_ms_seg_ts = torch.nn.functional.pad(ms_seg_ts, pad_ts)
        padded_scale_clus = torch.nn.functional.pad(scale_clus, pad_sc, value=-1)

        features_list.append(padded_feat)
        feature_length_list.append(feat_len.clone().detach())
        ms_seg_timestamps_list.append(padded_ms_seg_ts)
        ms_seg_counts_list.append(ms_seg_ct.clone().detach())
        scale_clus_label_list.append(padded_scale_clus)
        scale_mapping_list.append(padded_sm)
        targets_list.append(padded_tgt)

    features = torch.stack(features_list)
    feature_length = torch.stack(feature_length_list)
    ms_seg_timestamps = torch.stack(ms_seg_timestamps_list)
    clus_label_index = torch.stack(scale_clus_label_list)
    ms_seg_counts = torch.stack(ms_seg_counts_list)
    scale_mapping = torch.stack(scale_mapping_list)
    targets = torch.stack(targets_list)
    return features, feature_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, targets


def _msdd_infer_collate_fn(self, batch):
    """
    Collate batch of feats (speaker embeddings), feature lengths, target label sequences and cluster-average embeddings.

    Args:
        batch (tuple):
            Batch tuple containing feats, feats_len, targets and ms_avg_embs.
    Returns:
        feats (torch.tensor):
            Collated speaker embedding with unified length.
        feats_len (torch.tensor):
            The actual length of each embedding sequence without zero padding.
        targets (torch.tensor):
            Groundtruth Speaker label for the given input embedding sequence.
        ms_avg_embs (torch.tensor):
            Cluster-average speaker embedding vectors.
    """

    packed_batch = list(zip(*batch))
    feats, feats_len, targets, ms_avg_embs = packed_batch
    feats_list, flen_list, targets_list, ms_avg_embs_list = [], [], [], []
    max_audio_len = max(feats_len)
    max_target_len = max([x.shape[0] for x in targets])

    for feature, feat_len, target, ivector in batch:
        flen_list.append(feat_len)
        ms_avg_embs_list.append(ivector)
        if feat_len < max_audio_len:
            pad_a = (0, 0, 0, 0, 0, max_audio_len - feat_len)
            pad_t = (0, 0, 0, max_target_len - target.shape[0])
            padded_feature = torch.nn.functional.pad(feature, pad_a)
            padded_target = torch.nn.functional.pad(target, pad_t)
            feats_list.append(padded_feature)
            targets_list.append(padded_target)
        else:
            targets_list.append(target.clone().detach())
            feats_list.append(feature.clone().detach())

    feats = torch.stack(feats_list)
    feats_len = torch.tensor(flen_list)
    targets = torch.stack(targets_list)
    ms_avg_embs = torch.stack(ms_avg_embs_list)
    return feats, feats_len, targets, ms_avg_embs


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
        )

    def msdd_train_collate_fn(self, batch):
        return _msdd_train_collate_fn(self, batch)


class AudioToSpeechMSDDInferDataset(_AudioMSDDInferDataset):
    """
    Dataset class that loads a json file containing paths to audio files,
    rttm files and number of speakers. The created labels are used for diarization inference.

    Example:
    {"audio_filepath": "/path/to/audio_0.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_0.rttm}
    ...
    {"audio_filepath": "/path/to/audio_n.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_n.rttm}

    Args:
        manifest_filepath (str):
            Path to input manifest json files.
        emb_dict (dict):
            Dictionary containing cluster-average embeddings and speaker mapping information.
        emb_seq (dict):
            Dictionary containing multiscale speaker embedding sequence, scale mapping and corresponding segment timestamps.
        clus_label_dict (dict):
            Subsegment-level (from base-scale) speaker labels from clustering results.
        soft_label_thres (float):
            Threshold that determines speaker labels of segments depending on the overlap with groundtruth speaker timestamps.
        featurizer:
            Featurizer instance for generating features from raw waveform.
        use_single_scale_clus (bool):
            Use only one scale for clustering instead of using multiple scales of embeddings for clustering.
        seq_eval_mode (bool):
            If True, F1 score will be calculated for each speaker pair during inference mode.
        window_stride (float):
            Window stride for acoustic feature. This value is used for calculating the numbers of feature-level frames.
        pairwise_infer (bool):
            If True, this Dataset class operates in inference mode. In inference mode, a set of speakers in the input audio
            is split into multiple pairs of speakers and speaker tuples (e.g. 3 speakers: [(0,1), (1,2), (0,2)]) and then
            fed into the MSDD to merge the individual results.
    """

    def __init__(
        self,
        *,
        manifest_filepath: str,
        emb_dict: Dict,
        emb_seq: Dict,
        clus_label_dict: Dict,
        soft_label_thres: float,
        session_len_sec: float,
        random_flip: bool,
        num_spks: int,
        use_single_scale_clus: bool,
        seq_eval_mode: bool,
        window_stride: float,
        pairwise_infer: bool,
    ):
        super().__init__(
            manifest_filepath=manifest_filepath,
            emb_dict=emb_dict,
            emb_seq=emb_seq,
            clus_label_dict=clus_label_dict,
            soft_label_thres=soft_label_thres,
            session_len_sec=session_len_sec,
            random_flip=random_flip,
            num_spks=num_spks,
            use_single_scale_clus=use_single_scale_clus,
            window_stride=window_stride,
            seq_eval_mode=seq_eval_mode,
            pairwise_infer=pairwise_infer,
        )

    def msdd_infer_collate_fn(self, batch):
        return _msdd_infer_collate_fn(self, batch)