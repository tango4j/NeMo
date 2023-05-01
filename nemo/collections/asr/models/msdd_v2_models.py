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

import copy
import json
import sox
import os
import pickle as pkl
import tempfile
from collections import OrderedDict
from pathlib import Path
from statistics import mode
from typing import Any, Dict, List, Optional, Tuple, Union
from operator import attrgetter

import numpy as np
import time
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from pyannote.core import Annotation
from pyannote.metrics.diarization import DiarizationErrorRate
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm
import torch.nn.functional as F

# from nemo.collections.asr.data.audio_to_diar_label import AudioToSpeechMSDDInferDataset, AudioToSpeechMSDDTrainDataset
from nemo.collections.asr.data.audio_to_msdd_label import AudioToSpeechMSDDInferDataset, AudioToSpeechMSDDTrainDataset
# from nemo.collections.asr.models.multi_classification_models import EncDecMultiClassificationModel
from nemo.collections.asr.metrics.der import score_labels
from nemo.collections.asr.metrics.multi_binary_acc import MultiBinaryAccuracy
from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.models.asr_model import ExportableEncDecModel
from nemo.collections.asr.models.clustering_diarizer import (
    _MODEL_CONFIG_YAML,
    _SPEAKER_MODEL,
    _VAD_MODEL,
    get_available_model_names,
)
from nemo.collections.asr.models.configs.diarizer_config import NeuralDiarizerInferenceConfig
from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.utils.offline_clustering import (
    getCosAffinityMatrix,
    cos_similarity,
    cos_similarity_batch,
)

from nemo.collections.asr.parts.utils.speaker_utils import (
    audio_rttm_map,
    get_embs_and_timestamps,
    get_id_tup_dict,
    get_scale_mapping_argmat,
    get_uniq_id_list_from_manifest,
    get_uniqname_from_filepath,
    parse_scale_configs,
    get_timestamps,
    get_subsegments,
    labels_to_pyannote_object,
    make_rttm_with_overlap,
    rttm_to_labels,
)
from nemo.collections.asr.parts.utils.manifest_utils import (
read_manifest,
write_manifest,
)
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType
from nemo.core.neural_types.elements import ProbsType
from nemo.utils import logging

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


__all__ = ['EncDecDiarLabelModel', 'ClusterEmbedding', 'NeuralDiarizer']

from nemo.core.classes import Loss, Typing, typecheck

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    pp_string = str(round(pp / float(10**6), 2)) + "M"
    print(f"size: {pp_string}")
    return pp

def check_multiscale_data(method):
    def wrapper(self, *args, **kwargs):
        for var_str in ['ms_seg_timestamps', 'ms_seg_counts', 'scale_mapping']:
            if torch.std(kwargs[var_str].float(), dim=0).sum() != 0:
                raise ValueError(f"Multi-scale variables should have the same values for all samples in a batch.")
            check_batch_unity(kwargs[var_str])
        return method(self, *args, **kwargs)
    return wrapper


def check_batch_unity(batch_tensor):
    if torch.std(batch_tensor.float(), dim=0).sum() != 0:
        raise ValueError(f"Multi-scale variables should have the same values for all samples in a batch.")

class AffinityLoss(Loss, Typing):
    """
    Computes Binary Cross Entropy (BCE) loss. The BCELoss class expects output from Sigmoid function.
    """

    @property
    def input_types(self):
        """Input types definitions for AnguarLoss.
        """
        return {
            "probs": NeuralType(('B', 'T', 'C'), ProbsType()),
            'labels': NeuralType(('B', 'T', 'C'), LabelsType()),
            "signal_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """
        Output types definitions for binary cross entropy loss. Weights for labels can be set using weight variables.
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, reduction='sum', gamma=0.0, negative_margin=0.5, positive_margin=0.05):
        super().__init__()
        self.reduction = reduction
        self.gamma = gamma # weights for loss values of [different, same] speaker segments
        self.negative_margin = negative_margin
        self.positive_margin = positive_margin

    def forward(self, batch_affinity_mat, targets):
        """
        Calculate binary cross entropy loss based on probs, labels and signal_lengths variables.

        Args:
            probs (torch.tensor)
                Predicted probability value which ranges from 0 to 1. Sigmoid output is expected.
            labels (torch.tensor)
                Groundtruth label for the predicted samples.
            signal_lengths (torch.tensor):
                The actual length of the sequence without zero-padding.

        Returns:
            loss (NeuralType)
                Binary cross entropy loss value.
        """
        gt_affinity = cos_similarity_batch(targets, targets)
        gt_affinity_margin = gt_affinity.clamp_(self.negative_margin, 1.0 - self.positive_margin)
        batch_affinity_mat_margin = batch_affinity_mat.clamp_(self.negative_margin, 1.0 - self.positive_margin)
        elem_affinity_loss = torch.abs(gt_affinity_margin - batch_affinity_mat_margin)
        positive_samples = gt_affinity * elem_affinity_loss
        negative_samples = (1-gt_affinity) * elem_affinity_loss
        affinity_loss = (1-self.gamma) * positive_samples.sum() + self.gamma * negative_samples.sum()
        return affinity_loss


def get_scale_dist_mat_t(source_scale_idx, ms_seg_timestamps, msdd_scale_n, deci=1):
    """
    Get distance matrix between anchors of the source scale and base scale.

    Args:
        source_scale_idx (int): Source scale index
        ms_seg_timestamps (Tensor): Multi-scale segment timestamps

    Returns:
        abs_dist_mat (Tensor): Distance matrix between anchors of the source scale and base scale
    """
    source_scale_anchor_zeros = torch.mean(ms_seg_timestamps[source_scale_idx, :, :], dim=1) / deci
    base_scale_anchor_zeros = torch.mean(ms_seg_timestamps[msdd_scale_n-1, :, :], dim=1) / deci
    # Get only non-zero timestamps (= Remove zero-padding)
    source_scale_anchor = source_scale_anchor_zeros[source_scale_anchor_zeros.nonzero()].t()
    base_scale_anchor = base_scale_anchor_zeros[base_scale_anchor_zeros.nonzero()].t()
    # Calculate absolute distance matrix
    curr_mat = torch.tile(source_scale_anchor, (base_scale_anchor.shape[0], 1))
    base_mat = torch.tile(base_scale_anchor, (source_scale_anchor.shape[0], 1)).t()
    abs_dist_mat = torch.abs(curr_mat - base_mat)
    return abs_dist_mat

def get_padded_timestamps_t(ms_ts_dict, base_scale):
    """
    Get zero-padded timestamps of all scales.

    """
    ms_seg_timestamps_list = []
    max_length = len(ms_ts_dict[base_scale]['time_stamps'])
    for data_dict in ms_ts_dict.values():
        ms_seg_timestamps = data_dict['time_stamps']
        ms_seg_timestamps = torch.tensor(ms_seg_timestamps)
        padded_ms_seg_ts = F.pad(input=ms_seg_timestamps, pad=(0, 0, 0, max_length - len(ms_seg_timestamps)), mode='constant', value=0)
        ms_seg_timestamps_list.append(padded_ms_seg_ts)
    ms_seg_timestamps = torch.stack(ms_seg_timestamps_list)
    return ms_seg_timestamps

def get_interpolate_weights(
    ms_seg_timestamps: torch.Tensor, 
    base_seq_len: int, 
    msdd_multiscale_args_dict: dict, 
    emb_scale_n: int, 
    msdd_scale_n: int, 
    is_integer_ts=False
    ):
    """
    Interpolate embeddings to a finer scale.

    Args:
        emb_fix (torch.Tensor): embeddings of the base scale
        ms_seg_timestamps (torch.Tensor): timestamps of the base scale
        base_seq_len (int): length of the base scale
    
    Returns:
        emb_fix (torch.Tensor): interpolated embeddings
    """
    deci = 100.0 if is_integer_ts else 1.0
    half_scale = msdd_multiscale_args_dict['scale_dict'][emb_scale_n-1][1]
    session_scale_dist_mat = get_scale_dist_mat_t(source_scale_idx=emb_scale_n-1, 
                                                  ms_seg_timestamps=ms_seg_timestamps[:, :base_seq_len, :], 
                                                  msdd_scale_n=msdd_scale_n, deci=deci)
    target_bool = (session_scale_dist_mat < half_scale)
    session_scale_dist_mat.flatten()[target_bool.flatten() == False] = half_scale
    dist_delta = (half_scale - session_scale_dist_mat.flatten()).reshape(base_seq_len, target_bool.shape[1])
    interpolated_weights = ((dist_delta ** 2).t() / torch.sum(dist_delta ** 2, dim=1).t()).t()  
    return interpolated_weights 

def interpolate_embs_t(
    emb_t: torch.Tensor, 
    ms_seg_timestamps: torch.Tensor, 
    base_seq_len: int, 
    msdd_multiscale_args_dict: dict, 
    emb_scale_n: int, 
    msdd_scale_n: int, 
    is_integer_ts=False
    ):
    """
    Interpolate embeddings to a finer scale.

    Args:
        emb_fix (torch.Tensor): embeddings of the base scale
        ms_seg_timestamps (torch.Tensor): timestamps of the base scale
        base_seq_len (int): length of the base scale
    
    Returns:
        emb_fix (torch.Tensor): interpolated embeddings
    """
    deci = 100.0 if is_integer_ts else 1.0
    half_scale = msdd_multiscale_args_dict['scale_dict'][emb_scale_n-1][1]
    session_scale_dist_mat = get_scale_dist_mat_t(source_scale_idx=emb_scale_n-1, 
                                                  ms_seg_timestamps=ms_seg_timestamps[:, :base_seq_len, :], 
                                                  msdd_scale_n=msdd_scale_n, deci=deci)
    max_length = ms_seg_timestamps[msdd_scale_n-1].shape[0]
    target_bool = (session_scale_dist_mat < half_scale)
    session_scale_dist_mat.flatten()[target_bool.flatten() == False] = half_scale
    dist_delta = (half_scale - session_scale_dist_mat.flatten()).reshape(base_seq_len, target_bool.shape[1])
    interpolated_weights = ((dist_delta ** 2).t() / torch.sum(dist_delta ** 2, dim=1).t()).t()  
    interpolated_embs = interpolated_weights @ emb_t
    # rep_interpolated_embs = F.pad(input=interpolated_embs, pad=(0, 0, max_length - base_seq_len, 0), mode='constant', value=0) # Wrong zero padding
    rep_interpolated_embs = F.pad(input=interpolated_embs, pad=(0, 0, 0, max_length - base_seq_len), mode='constant', value=0)
    return rep_interpolated_embs, interpolated_weights 


def get_batch_cosine_sim(ms_emb_seq: torch.Tensor) -> torch.Tensor:
    """
    Calculate cosine similarity in batch mode.

    Args:
        ms_emb_seq (Tensor):
            Multi-scale embedding sequence.

    Returns:
        batch_cos_sim (Tensor):
            Cosine similarity values
    """
    batch_embs = ms_emb_seq.reshape(-1, ms_emb_seq.shape[-3], ms_emb_seq.shape[-1])
    batch_cos_sim = cos_similarity_batch(batch_embs, batch_embs).reshape(ms_emb_seq.shape[0], ms_emb_seq.shape[2], ms_emb_seq.shape[1], ms_emb_seq.shape[1])
    batch_cos_sim = torch.mean(batch_cos_sim, dim=1) # Average over scales: (batch_size, base scale timesteps, base scale timesteps)
    return batch_cos_sim 

def load_subsegment_to_dict(manifest_file: str):
    """
    Load subsegment information from manifest file.

    Args:
        manifest_file (str): path to manifest file

    Returns:
        time_stamps (dict): dictionary of subsegment information indexed by uniq_ids
    """
    time_stamps = {}
    with open(manifest_file, 'r', encoding='utf-8') as manifest:
        for i, line in enumerate(manifest.readlines()):
            line = line.strip()
            dic = json.loads(line)
            uniq_name = get_uniqname_from_filepath(dic['audio_filepath'])
            if uniq_name not in time_stamps:
                time_stamps[uniq_name] = []
            start = dic['offset']
            end = start + dic['duration']
            time_stamps[uniq_name].append([start, end])
    return time_stamps

class EncDecDiarLabelModel(ModelPT, ExportableEncDecModel):
    """
    Encoder decoder class for multiscale diarization decoder (MSDD). Model class creates training, validation methods for setting
    up data performing model forward pass.

    This model class expects config dict for:
        * preprocessor
        * msdd_model
        * speaker_model
    """

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []

        model = PretrainedModelInfo(
            pretrained_model_name="diar_msdd_telephonic",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/diar_msdd_telephonic/versions/1.0.1/files/diar_msdd_telephonic.nemo",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:diar_msdd_telephonic",
        )
        result.append(model)
        return result

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """
        Initialize an MSDD model and the specified speaker embedding model. In this init function, training and validation datasets are prepared.
        """
        self._trainer = trainer if trainer else None
        self.cfg_msdd_model = cfg
        
        self._init_segmentation_info()
        if self._trainer:
            self.world_size = trainer.num_nodes * trainer.num_devices
            self.emb_batch_size = self.cfg_msdd_model.emb_batch_size
            self.pairwise_infer = False
        else:
            self.world_size = 1
            self.pairwise_infer = True

        self._init_msdd_scales() 

        super().__init__(cfg=self.cfg_msdd_model, trainer=trainer)
        window_length_in_sec = self.cfg_msdd_model.diarizer.speaker_embeddings.parameters.window_length_in_sec
        if isinstance(window_length_in_sec, int) or len(window_length_in_sec) <= 1:
            raise ValueError("window_length_in_sec should be a list containing multiple segment (window) lengths")
        else:
            self.cfg_msdd_model.msdd_module.scale_n = self.cfg_msdd_model.scale_n

        self.preprocessor = EncDecSpeakerLabelModel.from_config_dict(self.cfg_msdd_model.preprocessor)
        self.frame_per_sec = int(1 / self.preprocessor._cfg.window_stride)
        self.feat_dim = self.preprocessor._cfg.features
        self.max_feat_frame_count = int(self.msdd_multiscale_args_dict["scale_dict"][0][0] * self.frame_per_sec) # 0-th scale, window length
        self.msdd = EncDecDiarLabelModel.from_config_dict(self.cfg_msdd_model.msdd_module)
        self.msdd_classifier = EncDecDiarLabelModel.from_config_dict(self.cfg_msdd_model.msdd_classifier)
        self.encoder_infer_mode = False

        if trainer is not None:
            # self._init_vad_model()
            self._init_speaker_model()
            self.add_speaker_model_config(cfg)
            self.loss = instantiate(self.cfg_msdd_model.loss)
            self.affinity_loss = AffinityLoss()
            self.alpha = self.cfg_msdd_model.loss.alpha
        else:
            self.msdd._speaker_model = EncDecSpeakerLabelModel.from_config_dict(cfg.speaker_model_cfg)

        # Call `self.save_hyperparameters` in modelPT.py again since cfg should contain speaker model's config.
        self.save_hyperparameters("cfg")

        self._accuracy_test = MultiBinaryAccuracy()
        self._accuracy_train = MultiBinaryAccuracy()
        self._accuracy_valid = MultiBinaryAccuracy()

        self.time_flag = 0.0
        self.time_flag_end = 0.0

    def _init_msdd_scales(self,):
        window_length_in_sec = self.cfg_msdd_model.diarizer.speaker_embeddings.parameters.window_length_in_sec
        self.msdd_multiscale_args_dict = self.multiscale_args_dict
        self.model_spk_num = self.cfg_msdd_model.max_num_of_spks
        if self.cfg_msdd_model.get('interpolated_scale', None) is not None:
            if self.cfg_msdd_model.interpolated_scale > 0.1:
                raise ValueError("Interpolated scale must be smaller than 0.1")
            # Use an interpolated scale so add another scale 
            self.cfg_msdd_model.scale_n = len(window_length_in_sec) + 1 # Adding one interpolated scale
            self.emb_scale_n = len(window_length_in_sec) # Scales that are extracted from the audio
            self.msdd_multiscale_args_dict['scale_dict'][self.emb_scale_n] = (self.cfg_msdd_model.interpolated_scale, self.cfg_msdd_model.interpolated_scale/2)
            self.msdd_multiscale_args_dict['multiscale_weights'] = [1.0] * (self.emb_scale_n+1)
            self.msdd_scale_n = int(self.emb_scale_n+1) if self.cfg_msdd_model.interpolated_scale is not None else int(self.emb_scale_n)
        else:
            # Only use the scales in window_length_in_sec
            self.cfg_msdd_model.scale_n = len(window_length_in_sec)
            self.emb_scale_n = self.cfg_msdd_model.scale_n
            self.msdd_scale_n = self.cfg_msdd_model.scale_n

    def setup_optimizer_param_groups(self):
        """
        Override function in ModelPT to allow for different parameter groups for the speaker model and the MSDD model.


        """
        if not hasattr(self, "parameters"):
            self._optimizer_param_groups = None
            return

        known_groups = []
        param_groups = []
        if "optim_param_groups" in self.cfg:
            param_groups_cfg = self.cfg.optim_param_groups
            for group_levels, group_cfg_levels in param_groups_cfg.items():
                retriever = attrgetter(group_levels)
                module = retriever(self)
                if module is None:
                    raise ValueError(f"{group_levels} not found in model.")
                elif hasattr(module, "parameters"):
                    known_groups.append(group_levels)
                    new_group = {"params": module.parameters()}
                    for k, v in group_cfg_levels.items():
                        new_group[k] = v
                    param_groups.append(new_group)
                else:
                    raise ValueError(f"{group} does not have parameters.")

            other_params = []
            for n, p in self.named_parameters():
                is_unknown = True
                for group in known_groups:
                    if group in n :
                        is_unknown = False
                if is_unknown:
                    other_params.append(p)

            if len(other_params):
                param_groups = [{"params": other_params}] + param_groups
        else:
            param_groups = [{"params": self.parameters()}]

        self._optimizer_param_groups = param_groups

    def add_speaker_model_config(self, cfg):
        """
        Add config dictionary of the speaker model to the model's config dictionary. This is required to
        save and load speaker model with MSDD model.

        Args:
            cfg (DictConfig): DictConfig type variable that conatains hyperparameters of MSDD model.
        """
        with open_dict(cfg):
            cfg_cp = copy.copy(self.msdd._speaker_model.cfg)
            cfg.speaker_model_cfg = cfg_cp
            del cfg.speaker_model_cfg.train_ds
            del cfg.speaker_model_cfg.validation_ds

    def _init_segmentation_info(self):
        """Initialize segmentation settings: window, shift and multiscale weights.
        """
        self._diarizer_params = self.cfg_msdd_model.diarizer
        self.multiscale_args_dict = parse_scale_configs(
            self._diarizer_params.speaker_embeddings.parameters.window_length_in_sec,
            self._diarizer_params.speaker_embeddings.parameters.shift_length_in_sec,
            self._diarizer_params.speaker_embeddings.parameters.multiscale_weights,
        )

    # def _init_vad_model(self):
    #     model_path = self.cfg_msdd_model.diarizer.vad.model_path
    #     self._diarizer_params = self.cfg_msdd_model.diarizer

    #     if not torch.cuda.is_available():
    #         rank_id = torch.device('cpu')
    #     elif self._trainer:
    #         rank_id = torch.device(self._trainer.global_rank)
    #     else:
    #         rank_id = None
    #     self._vad_model = EncDecMultiClassificationModel.restore_from(restore_path=model_path, map_location=self._cfg.device)

    def _init_speaker_model(self):
        """
        Initialize speaker embedding model with model name or path passed through config. Note that speaker embedding model is loaded to
        `self.msdd` to enable multi-gpu and multi-node training. In addition, speaker embedding model is also saved with msdd model when
        `.ckpt` files are saved.
        """
        model_path = self.cfg_msdd_model.diarizer.speaker_embeddings.model_path
        self._diarizer_params = self.cfg_msdd_model.diarizer

        if not torch.cuda.is_available():
            rank_id = torch.device('cpu')
        elif self._trainer:
            rank_id = torch.device(self._trainer.global_rank)
        else:
            rank_id = None

        if model_path is not None and model_path.endswith('.nemo'):
            self.msdd._speaker_model = EncDecSpeakerLabelModel.restore_from(model_path, map_location=rank_id)
            logging.info("Speaker Model restored locally from {}".format(model_path))
        elif model_path.endswith('.ckpt'):
            self._speaker_model = EncDecSpeakerLabelModel.load_from_checkpoint(model_path, map_location=rank_id)
            logging.info("Speaker Model restored locally from {}".format(model_path))
        else:
            if model_path not in get_available_model_names(EncDecSpeakerLabelModel):
                logging.warning(
                    "requested {} model name not available in pretrained models, instead".format(model_path)
                )
                model_path = "titanet_large"
            logging.info("Loading pretrained {} model from NGC".format(model_path))
            self.msdd._speaker_model = EncDecSpeakerLabelModel.from_pretrained(
                model_name=model_path, map_location=rank_id
            )
        if self._cfg.freeze_speaker_model:
            self.msdd._speaker_model.eval()

        self._speaker_params = self.cfg_msdd_model.diarizer.speaker_embeddings.parameters

    def __setup_dataloader_from_config(self, config):
        featurizer = WaveformFeaturizer(
            sample_rate=config['sample_rate'], int_values=config.get('int_values', False), augmentor=None
        )

        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None

        logging.info(f"Loading dataset from {config.manifest_filepath}")

        if self._trainer is not None:
            global_rank = self._trainer.global_rank
        else:
            global_rank = 0

        dataset = AudioToSpeechMSDDTrainDataset(
            manifest_filepath=config.manifest_filepath,
            emb_dir=config.emb_dir,
            multiscale_args_dict=self.msdd_multiscale_args_dict,
            soft_label_thres=config.soft_label_thres,
            random_flip=config.random_flip,
            session_len_sec=config.session_len_sec,
            num_spks=config.num_spks,
            featurizer=featurizer,
            window_stride=self.cfg_msdd_model.preprocessor.window_stride,
            emb_batch_size=100,
            pairwise_infer=False,
            global_rank=global_rank,
            encoder_infer_mode=self.encoder_infer_mode,
        )

        self.data_collection = dataset.collection
        collate_ds = dataset
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            collate_fn=collate_ds.msdd_train_collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=False,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def __setup_dataloader_from_config_infer(
        self, config: DictConfig, emb_seq: dict, clus_label_dict: dict, pairwise_infer=False
    ):
        shuffle = config.get('shuffle', False)

        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None

        dataset = AudioToSpeechMSDDInferDataset(
            manifest_filepath=config['manifest_filepath'],
            clus_label_dict=clus_label_dict,
            emb_seq=emb_seq,
            multiscale_args_dict=self.msdd_multiscale_args_dict,
            soft_label_thres=config.soft_label_thres,
            seq_eval_mode=config.seq_eval_mode,
            window_stride=self._cfg.preprocessor.window_stride,
            use_single_scale_clus=False,
            pairwise_infer=pairwise_infer,
            session_len_sec=config.session_len_sec,
            max_spks=config.num_spks,
        )

        self.data_collection = dataset.collection
        collate_ds = dataset
        collate_fn = collate_ds.msdd_infer_collate_fn
        batch_size = config['batch_size']
        
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        self._train_dl = self.__setup_dataloader_from_config(config=train_data_config,)

    def setup_validation_data(self, val_data_layer_config: Optional[Union[DictConfig, Dict]]):
        self._validation_dl = self.__setup_dataloader_from_config(config=val_data_layer_config,)

    def segment_diar_window_test_dataset(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        test_manifest_jsons = read_manifest(test_data_config['manifest_filepath'])
        longest_scale_idx = 0
        self.segmented_manifest_list = []
        self.diar_window = self.cfg.session_len_sec
        self.diar_window_shift = self.cfg.session_len_sec - self.multiscale_args_dict['scale_dict'][longest_scale_idx][1]
        for manifest_json in test_manifest_jsons:
            if manifest_json['duration'] is None:
                manifest_json['duration'] = sox.file_info.duration(manifest_json['audio_filepath'])
            total_offset, total_duration = manifest_json['offset'], manifest_json['duration']
            subsegments = get_subsegments(offset=total_offset, 
                                          window=self.diar_window, 
                                          shift=self.diar_window_shift, 
                                          duration=total_duration)
                
            for (stt, dur) in subsegments:
                segment_manifest_json = copy.deepcopy(manifest_json)
                segment_manifest_json['offset'], segment_manifest_json['duration'] = stt, dur
                self.segmented_manifest_list.append(segment_manifest_json)

        segmented_manifest_path = test_data_config.manifest_filepath.replace('.json', '.segmented.json') 
        self.segmented_manifest_path = segmented_manifest_path
        write_manifest(output_path=segmented_manifest_path, target_manifest=self.segmented_manifest_list)
        test_data_config.manifest_filepath = segmented_manifest_path
        return test_data_config

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        test_data_config = self.segment_diar_window_test_dataset(test_data_config)
        self._test_dl = self.__setup_dataloader_from_config_infer(
            config=test_data_config,
            emb_seq=self.emb_seq_test,
            clus_label_dict=self.clus_test_label_dict,
            pairwise_infer=self.pairwise_infer,
        )
    
    def setup_encoder_infer_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        val_data_config = self.segment_diar_window_test_dataset(val_data_config)
        self.encoder_infer_mode = True
        self._validation_dl = self.__setup_dataloader_from_config(config=val_data_config,)


    def setup_multiple_test_data(self, test_data_config):
        """
        MSDD does not use multiple_test_data template. This function is a placeholder for preventing error.
        """
        return None

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            audio_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            audio_eltype = AudioSignal()
        return {
            "features": NeuralType(('B', 'T'), audio_eltype),
            "feature_length": NeuralType(('B',), LengthsType()),
            "ms_seg_timestamps": NeuralType(('B', 'C', 'T', 'D'), LengthsType()),
            "ms_seg_counts": NeuralType(('B', 'C'), LengthsType()),
            "clus_label_index": NeuralType(('B', 'T'), LengthsType()),
            "scale_mapping": NeuralType(('B', 'C', 'T'), LengthsType()),
            "targets": NeuralType(('B', 'T', 'C'), ProbsType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        return OrderedDict(
            {
                "probs": NeuralType(('B', 'T', 'C'), ProbsType()),
                "scale_weights": NeuralType(('B', 'T', 'C', 'D'), ProbsType()),
                "batch_affinity_mat": NeuralType(('B', 'T', 'T'), ProbsType()),
            }
        )

    @check_multiscale_data
    def get_ms_emb_fixed(
        self, 
        embs: torch.Tensor, 
        scale_mapping: torch.Tensor, 
        ms_seg_counts: torch.Tensor, 
        ms_seg_timestamps: torch.Tensor,
    ):
        # check_ms_data(ms_seg_timestamps, ms_seg_counts, scale_mapping)
        batch_size = scale_mapping.shape[0]
        split_emb_tup = torch.split(embs, ms_seg_counts[:, :self.emb_scale_n].flatten().tolist(), dim=0)
        ms_ts = ms_seg_timestamps[:, :, :ms_seg_counts[0][self.emb_scale_n-1],:]

        base_seq_len = ms_seg_counts[0][self.msdd_scale_n-1].item()
        target_embs = torch.vstack(split_emb_tup).reshape(batch_size, -1, embs.shape[-1])
        target_timestamps = torch.vstack(split_emb_tup).reshape(batch_size, -1, embs.shape[-1])
        intp_w = get_interpolate_weights(ms_seg_timestamps[0], 
                                         base_seq_len, 
                                         self.msdd_multiscale_args_dict, 
                                         self.emb_scale_n, 
                                         self.msdd_scale_n, 
                                         is_integer_ts=True)

        repeat_mats_ext = scale_mapping[0][:self.emb_scale_n].to(embs.device)
        ext_max_seq_len = ms_seg_counts[0][self.emb_scale_n-1].to(embs.device)
        all_seq_len = ms_seg_counts[0][-1].to(embs.device)
        
        # To make offset values such as, [10, 20, 60, x] -> [0, 10, 30, 90]
        scale_count_offsets = torch.tensor([0] + torch.cumsum(ms_seg_counts[0][:self.emb_scale_n-1], dim=0).tolist())
        repeat_mats_ext = repeat_mats_ext + (scale_count_offsets.to(embs.device)).unsqueeze(1).repeat(1, all_seq_len).to(embs.device)
        extracted_embs = target_embs[:, repeat_mats_ext.flatten(), :].reshape(batch_size, self.emb_scale_n, -1, embs.shape[-1])
        finest_extracted_start = ms_seg_counts[0][:self.emb_scale_n-1].sum()
        interpolated_embs = torch.bmm(intp_w.repeat(batch_size, 1, 1), target_embs[:, finest_extracted_start:, :]).unsqueeze(1)
        ms_emb_seq = torch.cat((extracted_embs, interpolated_embs), dim=1).transpose(2, 1)

        # Repeat and align timestamps
        repeat_mats_all = scale_mapping[0].to(embs.device)
        ms_ts = ms_seg_timestamps.reshape(batch_size, -1, 2)
        repeat_mats_all = repeat_mats_all + (all_seq_len * torch.arange(self.msdd_scale_n).to(embs.device)).unsqueeze(1).repeat(1, all_seq_len).to(embs.device)
        ms_ts_rep = ms_ts[:, repeat_mats_all.flatten(), :].reshape(batch_size, self.msdd_scale_n, -1, 2)
        return ms_emb_seq, ms_ts_rep

    @torch.no_grad()
    def get_cluster_avg_embs_model(
        self,  
        ms_emb_seq:torch.Tensor, 
        clus_label_index: torch.Tensor, 
        seq_lengths: torch.Tensor, 
    ) -> torch.Tensor:
        """
        `max_base_seq_len` is the longest base scale sequence length can be used for batch processing.

        Args:
            ms_emb_seq (Tensor):
                Multi-scale embedding sequence that is mapped, matched and repeated. 
            clus_label_index (Tensor):
                Cluster label index for each segment in the batch.
            ms_seg_counts (Tensor):
                Cumulative sum of the number of segments in each scale.

        Returns:
            ms_avg_embs (Tensor):
                Cluster average embeddings for each segment in the batch.
        """
        batch_size = ms_emb_seq.shape[0]
        # max_base_seq_len = torch.max(ms_seg_counts[:, -1])
        max_base_seq_len = torch.max(seq_lengths)

        # Create 0 and 1 to mask embedding vectors with speaker labels 
        spk_label_mask = torch.stack([ (clus_label_index == spk).float() for spk  in range(self.model_spk_num) ]).permute(1, 2, 0)
        spk_label_mask_sum = spk_label_mask.sum(dim=1)
        spk_label_mask_sum[(spk_label_mask_sum == 0)] = 1 # avoid divide by zero for empty clusters
        
        # ms_emb_seq should be matched with spk_label_mask's length so truncate it 
        ms_emb_seq_trunc = ms_emb_seq[:, :max_base_seq_len, :, :].reshape(batch_size, max_base_seq_len, -1).type(torch.float32)
        ms_weighted_sum_raw = torch.bmm(spk_label_mask.permute(0, 2, 1), ms_emb_seq_trunc)
        ms_weighted_sum = ms_weighted_sum_raw.permute(0, 2, 1).reshape(batch_size, self.msdd_scale_n, -1, self.model_spk_num)
        denom_label_count = torch.tile((1/spk_label_mask_sum).unsqueeze(1).unsqueeze(1), (1, self.msdd_scale_n, ms_emb_seq.shape[3], 1))
        ms_avg_embs = ms_weighted_sum * denom_label_count
        return ms_avg_embs
    
    def get_feat_range_matirx(self, processed_signal, feature_count_range, target_timestamps, device):
        feat_index_range = torch.arange(0, processed_signal.shape[2]).to(device) 
        feature_frame_offsets = torch.repeat_interleave(target_timestamps[:, 0], feature_count_range)
        feature_frame_interval_range = torch.concat([feat_index_range[stt:end] for (stt, end) in target_timestamps]).to(device)
        feature_frame_length_range = feature_frame_interval_range - feature_frame_offsets
        return feature_frame_length_range, feature_frame_interval_range, 

    def get_feature_index_map(
        self, 
        emb_scale_n,
        processed_signal, 
        processed_signal_len, 
        ms_seg_timestamps, 
        ms_seg_counts,
        device: torch.device,
        ):
        batch_size = processed_signal.shape[0]
        ms_seg_counts_embs = ms_seg_counts[:, :emb_scale_n] 

        total_seg_count = torch.sum(ms_seg_counts_embs)
        ms_seg_counts_embs_flatten =  ms_seg_counts_embs.flatten()

        # The following index-tensors are needed for matrix reshaping without nested for-loops.
        batch_index_range = torch.repeat_interleave(torch.arange(batch_size).to(device), ms_seg_counts_embs.sum(dim=1), dim=0)
        scale_index_range = torch.repeat_interleave(torch.arange(emb_scale_n).repeat(batch_size).to(device) , ms_seg_counts_embs_flatten)

        # Pre-compute sequence indices for faster assigning: 
        seq_index_range = torch.arange(ms_seg_counts_embs_flatten.max())
        segment_index_range = torch.concat([seq_index_range[:seq_len] for seq_len in ms_seg_counts_embs_flatten]).to(device)
        target_timestamps = ms_seg_timestamps[batch_index_range, scale_index_range, segment_index_range, :].to(torch.int64)
        feature_count_range = target_timestamps[:, 1] - target_timestamps[:, 0]
        
        # Pre-compute feature indices for faster assigning:
        feature_frame_length_range, feature_frame_interval_range= self.get_feat_range_matirx(processed_signal, feature_count_range, target_timestamps, processed_signal.device)
        
        # Assign frame-by-frame indices for one-pass assignment without nested for-loops
        ms_seg_count_frame_range = torch.repeat_interleave(torch.arange(total_seg_count).to(device), feature_count_range)       
        batch_frame_range = torch.repeat_interleave(batch_index_range, feature_count_range)
        return total_seg_count, ms_seg_count_frame_range, feature_frame_length_range, batch_frame_range, feature_frame_interval_range, feature_count_range

    def forward_encode(
        self,
        processed_signal, 
        processed_signal_len, 
        total_seg_count,
        ms_seg_count_frame_range, 
        feature_frame_length_range, 
        batch_frame_range, 
        feature_frame_interval_range,
        feature_count_range,
        device,
        ):
        # Assign the acoustic feature values in processed_signal at once
        encoded, _ = self.msdd._speaker_model.encoder(audio_signal=processed_signal, length=processed_signal_len)
        encoded_segments = torch.zeros(total_seg_count, encoded.shape[1], self.max_feat_frame_count).to(torch.float32).to(device)
        encoded_segments[ms_seg_count_frame_range, :, feature_frame_length_range] = encoded[batch_frame_range, :, feature_frame_interval_range]
        _, embs = self.msdd._speaker_model.decoder(encoder_output=encoded_segments, length=feature_count_range) 
        return embs

    def forward_multiscale(
        self, 
        processed_signal, 
        processed_signal_len, 
        ms_seg_timestamps, 
        ms_seg_counts,
        vad_probs_frame=None,
        ):
        tsc, mscfr, fflr, bfr, ffir, fcr = self.get_feature_index_map(
                                                            emb_scale_n=self.emb_scale_n,
                                                            processed_signal=processed_signal, 
                                                            processed_signal_len=processed_signal_len, 
                                                            ms_seg_timestamps=ms_seg_timestamps, 
                                                            ms_seg_counts=ms_seg_counts, 
                                                            device=processed_signal.device)

        embs = self.forward_encode(processed_signal=processed_signal, 
                                    processed_signal_len=processed_signal_len, 
                                    total_seg_count=tsc,
                                    ms_seg_count_frame_range=mscfr, 
                                    feature_frame_length_range=fflr, 
                                    batch_frame_range=bfr, 
                                    feature_frame_interval_range=ffir,
                                    feature_count_range=fcr,
                                    device=processed_signal.device,
                                    )
        if vad_probs_frame is not None:
            target_timestamps = ms_seg_timestamps[0, -1].to(torch.int64)
            feature_count_range = target_timestamps[:, 1] - target_timestamps[:, 0]
            fflr, ffir = self.get_feat_range_matirx(processed_signal, feature_count_range, target_timestamps, device=processed_signal.device)
            vad_probs_steps = vad_probs_frame[:, ffir].reshape(vad_probs_frame.shape[0], ms_seg_counts.max(), -1).mean(dim=2)
        else:
            vad_probs_steps = None 
        return embs, vad_probs_steps

    def length_to_mask(self, context_embs):
        """
        Convert length values to encoder mask input tensor.

        Args:
            lengths (torch.Tensor): tensor containing lengths of sequences
            max_len (int): maximum sequence length

        Returns:
            mask (torch.Tensor): tensor of shape (batch_size, max_len) containing 0's
                                in the padded region and 1's elsewhere
        """
        lengths=torch.tensor([context_embs.shape[1]] * context_embs.shape[0]) 
        batch_size = context_embs.shape[0]
        max_len=context_embs.shape[1]
        # create a tensor with the shape (batch_size, 1) filled with ones
        row_vector = torch.arange(max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
        # create a tensor with the shape (batch_size, max_len) filled with lengths
        length_matrix = lengths.unsqueeze(1).expand(-1, max_len).to(lengths.device)
        # create a mask by comparing the row vector and length matrix
        mask = row_vector < length_matrix
        return mask.float().to(context_embs.device)

    def forward_infer(
        self, 
        ms_emb_seq,
        seq_lengths,
        ms_avg_embs,
        ):
        """

        Args:
            ms_emb_seq (torch.Tensor): tensor containing embeddings of multiscale embedding vectors
                Dimension: (batch_size, max_seg_count, msdd_scale_n, emb_dim)
            length (torch.Tensor): tensor containing lengths of multiscale segments
                Dimension: (batch_size, max_seg_count)
            ms_avg_embs (torch.Tensor): tensor containing average embeddings of multiscale segments
                Dimension: (batch_size, msdd_scale_n, emb_dim)

        """
        context_embs, scale_weights = self.msdd.forward_context(ms_emb_seq=ms_emb_seq, 
                                                                length=seq_lengths,
                                                                ms_avg_embs=ms_avg_embs)
        encoder_mask = self.length_to_mask(context_embs)
        classifier_logits = self.msdd_classifier(encoder_states=context_embs, 
                                                 encoder_mask=encoder_mask)
        preds, scale_weights = self.msdd.forward_speaker_logits(classifier_logits, scale_weights)
        return preds, scale_weights 
    
    def forward_encoder(
        self, 
        features, 
        feature_length, 
        ms_seg_timestamps, 
        ms_seg_counts, 
        scale_mapping, 
        vad_probs_frame,
    ):
        """
        Encoder part for end-to-end diarisaiton model.

        """
        processed_signal, processed_signal_length = self.msdd._speaker_model.preprocessor(
            input_signal=features, length=feature_length
        )

        embs, vad_probs = self.forward_multiscale(
            processed_signal=processed_signal, 
            processed_signal_len=processed_signal_length, 
            ms_seg_timestamps=ms_seg_timestamps, 
            ms_seg_counts=ms_seg_counts,
            vad_probs_frame=vad_probs_frame,
        )
        embs = embs.detach()

        # Reshape the embedding vectors into multi-scale inputs
        ms_emb_seq, ms_ts_rep = self.get_ms_emb_fixed(embs=embs, scale_mapping=scale_mapping, ms_seg_counts=ms_seg_counts, ms_seg_timestamps=ms_seg_timestamps)
        return ms_emb_seq, vad_probs, ms_ts_rep

    @typecheck()
    def forward(
        self, 
        features, 
        feature_length, 
        ms_seg_timestamps, 
        ms_seg_counts, 
        clus_label_index, 
        scale_mapping, 
    ):
        # Step 1: Multiscale Encoder 
        processed_signal, processed_signal_length = self.msdd._speaker_model.preprocessor(
            input_signal=features, length=feature_length
        )

        embs, vad_probs = self.forward_multiscale(
            processed_signal=processed_signal, 
            processed_signal_len=processed_signal_length, 
            ms_seg_timestamps=ms_seg_timestamps, 
            ms_seg_counts=ms_seg_counts, 
        )
        if self._cfg.freeze_speaker_model:
            embs = embs.detach()

        # Reshape the embedding vectors into multi-scale inputs
        ms_emb_seq = self.get_ms_emb_fixed(embs=embs, 
                                           scale_mapping=scale_mapping, 
                                           ms_seg_counts=ms_seg_counts, 
                                           ms_seg_timestamps=ms_seg_timestamps
                                           )


        # Step 2: Clustering for initialization
        # Compute the cosine similarity between the input and the cluster average embeddings
        batch_cos_sim = get_batch_cosine_sim(ms_emb_seq)
        ms_avg_embs = self.get_cluster_avg_embs_model(ms_emb_seq, clus_label_index, ms_seg_counts[:,-1])

        # Step 3: MSDD Inference
        preds, scale_weights = self.msdd.forward_infer(ms_emb_seq=ms_emb_seq, seq_lengths=ms_seg_counts[:, -1], ms_avg_embs=ms_avg_embs)
        return preds, scale_weights, batch_cos_sim

    def training_step(self, batch: list, batch_idx: int):
        features, feature_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, targets = batch
        sequence_lengths = torch.tensor([x[-1] for x in ms_seg_counts.detach()])
        preds, _, batch_affinity_mat = self.forward(
            features=features,
            feature_length=feature_length,
            ms_seg_timestamps=ms_seg_timestamps,
            ms_seg_counts=ms_seg_counts,
            clus_label_index=clus_label_index,
            scale_mapping=scale_mapping,
        )
        loss_1 = self.loss(probs=preds, labels=targets)
        loss_2 = self.affinity_loss.forward(batch_affinity_mat=batch_affinity_mat, targets=targets)
        # print(f"loss_1: {(1-self.alpha)*loss_1}, loss_2: {self.alpha*loss_2}")
        loss = (1-self.alpha) * loss_1 + self.alpha * loss_2
            # probs=preds, labels=targets, signal_lengths=sequence_lengths)
        self._accuracy_train(preds, targets, sequence_lengths)
        torch.cuda.empty_cache()
        f1_acc = self._accuracy_train.compute()
        self.log('loss', loss, sync_dist=True)
        self.log('loss_bce', (1-self.alpha) * loss_1, sync_dist=True)
        self.log('loss_aff', self.alpha * loss_2, sync_dist=True)
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'], sync_dist=True)
        self.log('train_f1_acc', f1_acc, sync_dist=True)
        self._accuracy_train.reset()
        return {'loss': loss}

    def validation_step(self, batch: list, batch_idx: int, dataloader_idx: int = 0):
        features, feature_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, targets = batch
        sequence_lengths = torch.tensor([x[-1] for x in ms_seg_counts])

        preds, _, batch_affinity_mat = self.forward(
            features=features,
            feature_length=feature_length,
            ms_seg_timestamps=ms_seg_timestamps,
            ms_seg_counts=ms_seg_counts,
            clus_label_index=clus_label_index,
            scale_mapping=scale_mapping,
            targets=targets,
        )

        loss_1 = self.loss(probs=preds, labels=targets)
        loss_2 = self.affinity_loss.forward(batch_affinity_mat=batch_affinity_mat, targets=targets)
        loss = (1-self.alpha) * loss_1 + self.alpha * loss_2
        self._accuracy_valid(preds, targets, sequence_lengths)
        f1_acc = self._accuracy_valid.compute()
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_f1_acc', f1_acc, sync_dist=True)
        self.log('val_loss_bce', (1-self.alpha) * loss_1, sync_dist=True)
        self.log('val_loss_aff',  self.alpha * loss_2, sync_dist=True)
        return {
            'val_loss': loss,
            'val_f1_acc': f1_acc,
        }

    def multi_validation_epoch_end(self, outputs: list, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        f1_acc = self._accuracy_valid.compute()
        self._accuracy_valid.reset()

        self.log('val_loss', val_loss_mean, sync_dist=True)
        self.log('val_f1_acc', f1_acc, sync_dist=True)
        return {
            'val_loss': val_loss_mean,
            'val_f1_acc': f1_acc,
        }

    def multi_test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]], dataloader_idx: int = 0):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        f1_acc = self._accuracy_test.compute()
        self._accuracy_test.reset()
        self.log('test_f1_acc', f1_acc, sync_dist=True)
        return {
            'test_loss': test_loss_mean,
            'test_f1_acc': f1_acc,
        }

    def compute_accuracies(self):
        """
        Calculate F1 score and accuracy of the predicted sigmoid values.

        Returns:
            f1_score (float):
                F1 score of the estimated diarized speaker label sequences.
            simple_acc (float):
                Accuracy of predicted speaker labels: (total # of correct labels)/(total # of sigmoid values)
        """
        f1_score = self._accuracy_test.compute()
        num_correct = torch.sum(self._accuracy_test.true.bool())
        total_count = torch.prod(torch.tensor(self._accuracy_test.targets.shape))
        simple_acc = num_correct / total_count
        return f1_score, simple_acc


class ClusterEmbedding(torch.nn.Module):
    """
    This class is built for calculating cluster-average embeddings, segmentation and load/save of the estimated cluster labels.
    The methods in this class is used for the inference of MSDD models.

    Args:
        cfg_diar_infer (DictConfig):
            Config dictionary from diarization inference YAML file
        cfg_msdd_model (DictConfig):
            Config dictionary from MSDD model checkpoint file

    Class Variables:
        self.cfg_diar_infer (DictConfig):
            Config dictionary from diarization inference YAML file
        cfg_msdd_model (DictConfig):
            Config dictionary from MSDD model checkpoint file
        self._speaker_model (class `EncDecSpeakerLabelModel`):
            This is a placeholder for class instance of `EncDecSpeakerLabelModel`
        self.scale_window_length_list (list):
            List containing the window lengths (i.e., scale length) of each scale.
        self.emb_scale_n (int):
            Number of scales for multi-scale clustering diarizer
        self.base_scale_index (int):
            The index of the base-scale which is the shortest scale among the given multiple scales
    """

    def __init__(
        self, cfg_diar_infer: DictConfig, cfg_msdd_model: DictConfig, speaker_model: Optional[EncDecSpeakerLabelModel]
    ):
        super().__init__()
        self.cfg_diar_infer = cfg_diar_infer
        self._cfg_msdd = cfg_msdd_model
        self.clus_diar_model = None
        self._speaker_model = None
        self.model_spk_num = cfg_msdd_model.max_num_of_spks
        self.scale_window_length_list = list(
            self.cfg_diar_infer.diarizer.speaker_embeddings.parameters.window_length_in_sec
        )
        self.emb_scale_n = len(self.scale_window_length_list)
        self.base_scale_index = len(self.scale_window_length_list) - 1
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.msdd_model = self._init_msdd_model(cfg=cfg_diar_infer, device=cfg_diar_infer.device)
        self.transfer_diar_params_to_model_params(cfg_diar_infer)
        self.clus_diar_model = ClusteringDiarizer(cfg=self.cfg_diar_infer, speaker_model=self.msdd_model, is_modular=False)

    def prepare_cluster_embs_infer(self):
        """
        Launch clustering diarizer to prepare embedding vectors and clustering results.
        """
        self.max_num_speakers = self.cfg_diar_infer.diarizer.clustering.parameters.max_num_speakers
        self.msdd_model._cfg.diarizer.speaker_embeddings.parameters = self.cfg_diar_infer.diarizer.speaker_embeddings.parameters
        emb_seq_test, ms_time_stamps, clus_test_label_dict = self.run_clustering_diarizer(
            self._cfg_msdd.test_ds.manifest_filepath, self._cfg_msdd.test_ds.emb_dir
        )
        return emb_seq_test, ms_time_stamps, clus_test_label_dict

    def assign_labels_to_longer_segs(self, base_clus_label_dict: Dict, session_scale_mapping_dict: Dict):
        """
        In multi-scale speaker diarization system, clustering result is solely based on the base-scale (the shortest scale).
        To calculate cluster-average speaker embeddings for each scale that are longer than the base-scale, this function assigns
        clustering results for the base-scale to the longer scales by measuring the distance between subsegment timestamps in the
        base-scale and non-base-scales.

        Args:
            base_clus_label_dict (dict):
                Dictionary containing clustering results for base-scale segments. Indexed by `uniq_id` string.
            session_scale_mapping_dict (dict):
                Dictionary containing multiscale mapping information for each session. Indexed by `uniq_id` string.

        Returns:
            all_scale_clus_label_dict (dict):
                Dictionary containing clustering labels of all scales. Indexed by scale_index in integer format.

        """
        all_scale_clus_label_dict = {scale_index: {} for scale_index in range(self.msdd_scale_n)}
        for uniq_id, uniq_scale_mapping_dict in session_scale_mapping_dict.items():
            base_scale_clus_label = np.array([x[-1] for x in base_clus_label_dict[uniq_id]])
            all_scale_clus_label_dict[self.msdd_scale_n-1][uniq_id] = base_scale_clus_label.tolist()
            for scale_index in range(self.msdd_scale_n - 1):
                new_clus_label = []
                max_index = max(uniq_scale_mapping_dict[scale_index])
                for seg_idx in range(max_index + 1):
                    if seg_idx in uniq_scale_mapping_dict[scale_index]:
                        seg_clus_label = mode(base_scale_clus_label[uniq_scale_mapping_dict[scale_index] == seg_idx])
                    else:
                        seg_clus_label = 0 if len(new_clus_label) == 0 else new_clus_label[-1]
                    new_clus_label.append(seg_clus_label)
                all_scale_clus_label_dict[scale_index][uniq_id] = new_clus_label
        return all_scale_clus_label_dict

    def get_cluster_avg_embs(
        self, emb_scale_seq_dict: Dict, session_clus_labels: List, speaker_mapping_dict: Dict, session_scale_mapping_dict: Dict, 
    ):
        """
        MSDD requires cluster-average speaker embedding vectors for each scale. This function calculates an average embedding vector for each cluster (speaker)
        and each scale.

        Args:
            emb_scale_seq_dict (dict):
                Dictionary containing embedding sequence for each scale. Keys are scale index in integer.
            session_clus_labels (list):
                Clustering results from clustering diarizer including all the sessions provided in input manifest files.
            speaker_mapping_dict (dict):
                Speaker mapping dictionary in case RTTM files are provided. This is mapping between integer based speaker index and
                speaker ID tokens in RTTM files.
                Example:
                    {'en_0638': {'speaker_0': 'en_0638_A', 'speaker_1': 'en_0638_B'},
                     'en_4065': {'speaker_0': 'en_4065_B', 'speaker_1': 'en_4065_A'}, ...,}
            session_scale_mapping_dict (dict):
                Dictionary containing multiscale mapping information for each session. Indexed by `uniq_id` string.

        Returns:
            emb_sess_avg_dict (dict):
                Dictionary containing speaker mapping information and cluster-average speaker embedding vector.
                Each session-level dictionary is indexed by scale index in integer.
            output_clus_label_dict (dict):
                Subegmentation timestamps in float type and Clustering result in integer type. Indexed by `uniq_id` keys.
        """
        emb_sess_avg_dict = {
            scale_index: {key: [] for key in emb_scale_seq_dict[self.emb_scale_n - 1].keys()}
            for scale_index in emb_scale_seq_dict.keys()
        }
        all_scale_clus_label_dict = self.assign_labels_to_longer_segs(
            session_clus_labels, session_scale_mapping_dict
        )
        for scale_index in emb_scale_seq_dict.keys():
            for uniq_id, _emb_tensor in emb_scale_seq_dict[scale_index].items():
                if type(_emb_tensor) == list:
                    emb_tensor = torch.tensor(np.array(_emb_tensor))
                else:
                    emb_tensor = _emb_tensor
                clus_label_list = all_scale_clus_label_dict[scale_index][uniq_id]
                spk_set = set(clus_label_list)

                # Create a label array which identifies clustering result for each segment.
                label_array = torch.Tensor(clus_label_list)
                avg_embs = torch.zeros(emb_tensor.shape[1], self.model_spk_num)
                for spk_idx in spk_set:
                    selected_embs = emb_tensor[label_array == spk_idx]
                    avg_embs[:, spk_idx] = torch.mean(selected_embs, dim=0)

                if speaker_mapping_dict is not None:
                    inv_map = {clus_key: rttm_key for rttm_key, clus_key in speaker_mapping_dict[uniq_id].items()}
                else:
                    inv_map = None

                emb_sess_avg_dict[scale_index][uniq_id] = {'mapping': inv_map, 'avg_embs': avg_embs}

        # Replace base scale clus label
        if len(emb_sess_avg_dict.keys()) < self.msdd_scale_n:
            emb_sess_avg_dict[self.msdd_scale_n-1] = emb_sess_avg_dict[self.emb_scale_n-1]
        return emb_sess_avg_dict, session_clus_labels
    
    def generate_interpolated_segments(self):
        multiscale_timestamps_by_scale, embeddings_by_scale = {}, {}
        for scale_idx in range(self.msdd_scale_n):
            if scale_idx < self.msdd_scale_n - 1:
                multiscale_timestamps_by_scale[scale_idx] = self.clus_diar_model.multiscale_embeddings_and_timestamps[scale_idx][1]
            elif scale_idx == self.msdd_scale_n - 1:
                self.clus_diar_model._run_segmentation(window=self.msdd_multiscale_args_dict['scale_dict'][self.msdd_scale_n-1][0],
                                                        shift=self.msdd_multiscale_args_dict['scale_dict'][self.msdd_scale_n-1][1],
                                                        scale_tag=f'_scale{self.msdd_scale_n-1}')
                time_stamps_by_scale = load_subsegment_to_dict(self.clus_diar_model.subsegments_manifest_path)
                multiscale_timestamps_by_scale[scale_idx] = time_stamps_by_scale
                self.clus_diar_model.multiscale_embeddings_and_timestamps.update({scale_idx: [embeddings_by_scale, time_stamps_by_scale]})
        return multiscale_timestamps_by_scale

    def generate_interpolated_embs(self, emb_scale_seq_dict):
        self._set_msdd_scale_args()
        emb_scale_seq_dict[self.msdd_scale_n - 1] = {}
        multiscale_timestamps_by_scale = self.generate_interpolated_segments()
        timestamps_by_sessions = get_timestamps(multiscale_timestamps_by_scale, self.msdd_multiscale_args_dict)
        for uniq_id, msdd_ts_dict in timestamps_by_sessions.items():
            # Prepare padded time-stamps
            ms_seg_timestamps = get_padded_timestamps_t(msdd_ts_dict['scale_dict'], self.msdd_scale_n-1)
            finest_extracted_embs = self.clus_diar_model.multiscale_embeddings_and_timestamps[self.emb_scale_n-1][0][uniq_id]

            # Get interpolated embeddings for inference 
            interpolated_embs, _ = interpolate_embs_t(emb_t=finest_extracted_embs,
                                                   ms_seg_timestamps=ms_seg_timestamps,
                                                   base_seq_len=ms_seg_timestamps.shape[1],
                                                   msdd_multiscale_args_dict=self.msdd_multiscale_args_dict, 
                                                   emb_scale_n=self.emb_scale_n, 
                                                   msdd_scale_n=self.emb_scale_n+1)
            # Add interpolated embedding vectors
            self.clus_diar_model.multiscale_embeddings_and_timestamps[self.msdd_scale_n-1][0][uniq_id] = interpolated_embs
            emb_scale_seq_dict[self.msdd_scale_n-1].update({uniq_id : interpolated_embs})
        return emb_scale_seq_dict, timestamps_by_sessions
   
    def _init_msdd_model(self, cfg: Union[DictConfig, NeuralDiarizerInferenceConfig], device: str):
        """
        Initialized MSDD model with the provided config. Load either from `.nemo` file or `.ckpt` checkpoint files.
        """
        model_path = cfg.diarizer.msdd_model.model_path
        # model_path = cfg_msdd_model.model_path
        if model_path.endswith('.nemo'):
            logging.info(f"Using local nemo file from {model_path}")
            msdd_model = EncDecDiarLabelModel.restore_from(restore_path=model_path, map_location=device)
        elif model_path.endswith('.ckpt'):
            logging.info(f"Using local checkpoint from {model_path}")
            msdd_model = EncDecDiarLabelModel.load_from_checkpoint(
                checkpoint_path=model_path, map_location=device
            )
        else:
            if model_path not in get_available_model_names(EncDecDiarLabelModel):
                logging.warning(f"requested {model_path} model name not available in pretrained models, instead")
            logging.info("Loading pretrained {} model from NGC".format(model_path))
            msdd_model = EncDecDiarLabelModel.from_pretrained(model_name=model_path, map_location=device)
        return msdd_model

    
    def generate_interpolated_labels(self, session_scale_mapping_dict, session_clus_labels, timestamps_by_sessions):
        for uniq_id, uniq_scale_mapping_dict in session_scale_mapping_dict.items():
            base_scale_clus_label = np.array([x[-1] for x in session_clus_labels[uniq_id]])
            interpolated_labels = []
            itp_ts = timestamps_by_sessions[uniq_id]['scale_dict'][self.msdd_scale_n-1]['time_stamps']
            itp_label = base_scale_clus_label[uniq_scale_mapping_dict[self.emb_scale_n-1]]
            for ts, label in zip(itp_ts, itp_label):
                interpolated_labels.append([ts[0], ts[1], label])
            session_clus_labels[uniq_id] = interpolated_labels
        return session_clus_labels
    
    def _init_clus_diarizer(self, manifest_filepath, emb_dir):
        self.cfg_diar_infer.diarizer.manifest_filepath = manifest_filepath
        self.cfg_diar_infer.diarizer.out_dir = emb_dir

        # Run ClusteringDiarizer which includes system VAD or oracle VAD.
        self._out_dir = self.clus_diar_model._diarizer_params.out_dir
        self.out_rttm_dir = os.path.join(self._out_dir, 'pred_rttms')
        os.makedirs(self.out_rttm_dir, exist_ok=True)

        self.clus_diar_model._cluster_params = self.cfg_diar_infer.diarizer.clustering.parameters
        self.clus_diar_model.multiscale_args_dict[
            "multiscale_weights"
        ] = self.cfg_diar_infer.diarizer.speaker_embeddings.parameters.multiscale_weights
        self.clus_diar_model._diarizer_params.speaker_embeddings.parameters = (
            self.cfg_diar_infer.diarizer.speaker_embeddings.parameters
        )
        self.clus_diar_model = ClusteringDiarizer(cfg=self.cfg_diar_infer, speaker_model=self.msdd_model, is_modular=False)
        
    def run_clustering_diarizer(self, manifest_filepath: str, emb_dir: str):
        """
        If no pre-existing data is provided, run clustering diarizer from scratch. This will create scale-wise speaker embedding
        sequence, cluster-average embeddings, scale mapping and base scale clustering labels. Note that speaker embedding `state_dict`
        is loaded from the `state_dict` in the provided MSDD checkpoint.

        Args:
            manifest_filepath (str):
                Input manifest file for creating audio-to-RTTM mapping.
            emb_dir (str):
                Output directory where embedding files and timestamp files are saved.

        Returns:
            emb_sess_avg_dict (dict):
                Dictionary containing cluster-average embeddings for each session.
            emb_scale_seq_dict (dict):
                Dictionary containing embedding tensors which are indexed by scale numbers.
            base_clus_label_dict (dict):
                Dictionary containing clustering results. Clustering results are cluster labels for the base scale segments.
        """
        self._init_clus_diarizer(manifest_filepath, emb_dir)
        cluster_params = self.clus_diar_model._cluster_params
        cluster_params = dict(cluster_params) if isinstance(cluster_params, DictConfig) else cluster_params.dict()
        logging.info(f"Multiscale Weights: {self.clus_diar_model.multiscale_args_dict['multiscale_weights']}")
        logging.info(f"Clustering Parameters: {json.dumps(cluster_params, indent=4)}")
        self.embeddings, self.time_stamps, self.vad_probs, self.session_clus_labels = self.clus_diar_model.forward(batch_size=self.cfg_diar_infer.batch_size)
        return (self.embeddings, self.time_stamps, self.session_clus_labels,)

    def _set_msdd_scale_args(self):
        self.msdd_multiscale_args_dict = self.clus_diar_model.multiscale_args_dict
        self.emb_scale_n = len(self.cfg_diar_infer.diarizer.speaker_embeddings.parameters.window_length_in_sec) # Scales that are extracted from the audio
        self.msdd_multiscale_args_dict['scale_dict'][self.emb_scale_n] = (self._cfg_msdd.interpolated_scale, self._cfg_msdd.interpolated_scale/2)
        self.msdd_multiscale_args_dict['multiscale_weights'] = [1.0] * (self.emb_scale_n+1)
        self.msdd_scale_n = int(self.emb_scale_n+1) if self._cfg_msdd.interpolated_scale is not None else int(self.emb_scale_n)
        self.base_scale_index = self.msdd_scale_n - 1

    def transfer_diar_params_to_model_params(self, cfg):
        """
        Transfer the parameters that are needed for MSDD inference from the diarization inference config files
        to MSDD model config `msdd_model.cfg`.
        """
        self.msdd_model.cfg_msdd_model.diarizer.out_dir = cfg.diarizer.out_dir
        
        self.msdd_model.cfg_msdd_model.test_ds.manifest_filepath = cfg.diarizer.manifest_filepath
        self.msdd_model.cfg_msdd_model.test_ds.emb_dir = cfg.diarizer.out_dir
        self.msdd_model.cfg_msdd_model.test_ds.batch_size = cfg.diarizer.msdd_model.parameters.infer_batch_size
        
        self.msdd_model.cfg.test_ds.manifest_filepath = cfg.diarizer.manifest_filepath
        self.msdd_model.cfg.test_ds.emb_dir = cfg.diarizer.out_dir
        self.msdd_model.cfg.test_ds.batch_size = cfg.diarizer.msdd_model.parameters.infer_batch_size
        
        self.msdd_model.cfg_msdd_model.validation_ds.manifest_filepath = cfg.diarizer.manifest_filepath
        self.msdd_model.cfg_msdd_model.validation_ds.emb_dir = cfg.diarizer.out_dir
        # self.msdd_model.cfg_msdd_model.validation_ds.batch_size = cfg.batch_size
        self.msdd_model.cfg_msdd_model.validation_ds.batch_size = cfg.diarizer.msdd_model.parameters.infer_batch_size

        self.msdd_model.cfg_msdd_model.max_num_of_spks = cfg.diarizer.clustering.parameters.max_num_speakers
        self.msdd_model.cfg_msdd_model.diarizer.speaker_embeddings.parameters = cfg.diarizer.speaker_embeddings.parameters
        self.msdd_model.interpolated_scale = cfg.diarizer.speaker_embeddings.parameters.interpolate_scale
        
    def get_scale_map(self, embs_and_timestamps):
        """
        Save multiscale mapping data into dictionary format.

        Args:
            embs_and_timestamps (dict):
                Dictionary containing embedding tensors and timestamp tensors. Indexed by `uniq_id` string.
        Returns:
            session_scale_mapping_dict (dict):
                Dictionary containing multiscale mapping information for each session. Indexed by `uniq_id` string.
        """
        session_scale_mapping_dict = {}
        for uniq_id, uniq_embs_and_timestamps in embs_and_timestamps.items():
            scale_mapping_dict = get_scale_mapping_argmat(uniq_embs_and_timestamps)
            session_scale_mapping_dict[uniq_id] = scale_mapping_dict
        return session_scale_mapping_dict

    def check_clustering_labels(self, out_dir):
        """
        Check whether the laoded clustering label file is including clustering results for all sessions.
        This function is used for inference mode of MSDD.

        Args:
            out_dir (str):
                Path to the directory where clustering result files are saved.
        Returns:
            file_exists (bool):
                Boolean that indicates whether clustering result file exists.
            clus_label_path (str):
                Path to the clustering label output file.
        """
        clus_label_path = os.path.join(
            out_dir, 'speaker_outputs', f'subsegments_scale{self.base_scale_index}_cluster.label'
        )
        if not os.path.exists(clus_label_path):
            logging.info(f"Clustering label file {clus_label_path} does not exist.")
        return clus_label_path

    def load_clustering_labels(self, out_dir):
        """
        Load clustering labels generated by clustering diarizer. This function is used for inference mode of MSDD.

        Args:
            out_dir (str):
                Path to the directory where clustering result files are saved.
        Returns:
            emb_scale_seq_dict (dict):
                List containing clustering results in string format.
        """
        clus_label_path = self.check_clustering_labels(out_dir)
        logging.info(f"Loading cluster label file from {clus_label_path}")
        base_clus_label_dict = {}
        with open(clus_label_path) as f:
            clus_labels = f.readlines()
        for line in clus_labels:
            uniq_id = line.split()[0]
            if uniq_id not in base_clus_label_dict:
                base_clus_label_dict[uniq_id] = []
            label = int(line.split()[-1].split('_')[-1])
            stt, end = [round(float(x), 2) for x in line.split()[1:3]]
            base_clus_label_dict[uniq_id].append([stt, end, label])
        return base_clus_label_dict

    def load_emb_scale_seq_dict(self, out_dir):
        """
        Load saved embeddings generated by clustering diarizer. This function is used for inference mode of MSDD.

        Args:
            out_dir (str):
                Path to the directory where embedding pickle files are saved.
        Returns:
            emb_scale_seq_dict (dict):
                Dictionary containing embedding tensors which are indexed by scale numbers.
        """
        window_len_list = list(self.cfg_diar_infer.diarizer.speaker_embeddings.parameters.window_length_in_sec)
        emb_scale_seq_dict = {scale_index: None for scale_index in range(len(window_len_list))}
        for scale_index in range(len(window_len_list)):
            pickle_path = os.path.join(
                out_dir, 'speaker_outputs', 'embeddings', f'subsegments_scale{scale_index}_embeddings.pkl'
            )
            logging.info(f"Loading embedding pickle file of scale:{scale_index} at {pickle_path}")
            with open(pickle_path, "rb") as input_file:
                emb_dict = pkl.load(input_file)
            for key, val in emb_dict.items():
                emb_dict[key] = val
            emb_scale_seq_dict[scale_index] = emb_dict
        return emb_scale_seq_dict

class NeuralDiarizer(LightningModule):
    """
    Class for inference based on multiscale diarization decoder (MSDD). MSDD requires initializing clustering results from
    clustering diarizer. Overlap-aware diarizer requires separate RTTM generation and evaluation modules to check the effect of
    overlap detection in speaker diarization.
    """

    def __init__(self, cfg: Union[DictConfig, NeuralDiarizerInferenceConfig]):
        super().__init__()
        self._cfg = cfg
        # Parameter settings for MSDD model
        self.use_speaker_model_from_ckpt = cfg.diarizer.msdd_model.parameters.get('use_speaker_model_from_ckpt', True)
        self.use_clus_as_main = cfg.diarizer.msdd_model.parameters.get('use_clus_as_main', False)
        self.max_overlap_spks = cfg.diarizer.msdd_model.parameters.get('max_overlap_spks', 2)
        self.num_spks_per_model = cfg.diarizer.msdd_model.parameters.get('num_spks_per_model', 2)
        self.use_adaptive_thres = cfg.diarizer.msdd_model.parameters.get('use_adaptive_thres', True)
        self.max_pred_length = cfg.diarizer.msdd_model.parameters.get('max_pred_length', 0)
        self.diar_eval_settings = cfg.diarizer.msdd_model.parameters.get(
            'diar_eval_settings', [(0.25, False)]
        )
            # 'diar_eval_settings', [(0.25, True), (0.25, False), (0.0, False)]

        self.msdd_model = self._init_msdd_model(cfg)
        # self.clus_diar_model = self._init_clus_diarizer(cfg)
        self.diar_window_length = cfg.diarizer.msdd_model.parameters.diar_window_length
        
        self.transfer_diar_params_to_model_params(cfg)
        self.msdd_model._init_segmentation_info()

        # Initialize clustering and embedding preparation instance (as a diarization encoder).
        self.clustering_embedding = ClusterEmbedding(
            cfg_diar_infer=cfg, cfg_msdd_model=self.msdd_model.cfg, speaker_model=self._speaker_model
        )

        # Parameters for creating diarization results from MSDD outputs.
        self.clustering_max_spks = self.msdd_model._cfg.max_num_of_spks
        self.overlap_infer_spk_limit = cfg.diarizer.msdd_model.parameters.get(
            'overlap_infer_spk_limit', self.clustering_max_spks
        )
    def transfer_diar_params_to_model_params(self, cfg):
        """
        Transfer the parameters that are needed for MSDD inference from the diarization inference config files
        to MSDD model config `msdd_model.cfg`.
        """
        self.msdd_model.cfg_msdd_model.diarizer.out_dir = cfg.diarizer.out_dir
        self.msdd_model.cfg_msdd_model.test_ds.manifest_filepath = cfg.diarizer.manifest_filepath
        self.msdd_model.cfg_msdd_model.test_ds.emb_dir = cfg.diarizer.out_dir
        self.msdd_model.cfg_msdd_model.test_ds.batch_size = cfg.diarizer.msdd_model.parameters.infer_batch_size
        
        self.msdd_model.cfg.test_ds.manifest_filepath = cfg.diarizer.manifest_filepath
        self.msdd_model.cfg.test_ds.emb_dir = cfg.diarizer.out_dir
        
        self.msdd_model.cfg.test_ds.batch_size = cfg.diarizer.msdd_model.parameters.infer_batch_size
        
        self.msdd_model.cfg_msdd_model.validation_ds.manifest_filepath = cfg.diarizer.manifest_filepath
        self.msdd_model.cfg_msdd_model.validation_ds.emb_dir = cfg.diarizer.out_dir
        self.msdd_model.cfg_msdd_model.validation_ds.batch_size = cfg.batch_size

        self.msdd_model.cfg_msdd_model.max_num_of_spks = cfg.diarizer.clustering.parameters.max_num_speakers
        self.msdd_model.cfg_msdd_model.diarizer.speaker_embeddings.parameters = cfg.diarizer.speaker_embeddings.parameters
        self.msdd_model.interpolated_scale = cfg.diarizer.speaker_embeddings.parameters.interpolate_scale

    @rank_zero_only
    def save_to(self, save_path: str):
        """
        Saves model instances (weights and configuration) into EFF archive.
        You can use "restore_from" method to fully restore instance from .nemo file.

        .nemo file is an archive (tar.gz) with the following:
            model_config.yaml - model configuration in .yaml format. You can deserialize this into cfg argument for model's constructor
            model_wights.chpt - model checkpoint

        Args:
            save_path: Path to .nemo file where model instance should be saved
        """
        self.clus_diar = self.clustering_embedding.clus_diar_model
        _NEURAL_DIAR_MODEL = "msdd_model.nemo"

        with tempfile.TemporaryDirectory() as tmpdir:
            config_yaml = os.path.join(tmpdir, _MODEL_CONFIG_YAML)
            spkr_model = os.path.join(tmpdir, _SPEAKER_MODEL)
            neural_diar_model = os.path.join(tmpdir, _NEURAL_DIAR_MODEL)

            self.clus_diar.to_config_file(path2yaml_file=config_yaml)
            if self.clus_diar.has_vad_model:
                vad_model = os.path.join(tmpdir, _VAD_MODEL)
                self.clus_diar._vad_model.save_to(vad_model)
            self.clus_diar._speaker_model.save_to(spkr_model)
            self.msdd_model.save_to(neural_diar_model)
            self.clus_diar.__make_nemo_file_from_folder(filename=save_path, source_dir=tmpdir)

    def extract_standalone_speaker_model(self, prefix: str = 'msdd._speaker_model.') -> EncDecSpeakerLabelModel:
        """
        MSDD model file contains speaker embedding model and MSDD model. This function extracts standalone speaker model and save it to
        `self.spk_emb_state_dict` to be loaded separately for clustering diarizer.

        Args:
            ext (str):
                File-name extension of the provided model path.
        Returns:
            standalone_model_path (str):
                Path to the extracted standalone model without speaker embedding extractor model.
        """
        model_state_dict = self.msdd_model.state_dict()
        spk_emb_module_names = []
        for name in model_state_dict.keys():
            if prefix in name:
                spk_emb_module_names.append(name)

        spk_emb_state_dict = {}
        for name in spk_emb_module_names:
            org_name = name.replace(prefix, '')
            spk_emb_state_dict[org_name] = model_state_dict[name]

        _speaker_model = EncDecSpeakerLabelModel.from_config_dict(self.msdd_model.cfg.speaker_model_cfg)
        _speaker_model.load_state_dict(spk_emb_state_dict)
        return _speaker_model

    def _init_msdd_model(self, cfg: Union[DictConfig, NeuralDiarizerInferenceConfig]):
        """
        Initialized MSDD model with the provided config. 
        Load either from `.nemo` file or `.ckpt` checkpoint files.
        """
        model_path = cfg.diarizer.msdd_model.model_path
        if model_path.endswith('.nemo'):
            logging.info(f"Using local nemo file from {model_path}")
            self.msdd_model = EncDecDiarLabelModel.restore_from(restore_path=model_path, map_location=cfg.device)
        elif model_path.endswith('.ckpt'):
            logging.info(f"Using local checkpoint from {model_path}")
            self.msdd_model = EncDecDiarLabelModel.load_from_checkpoint(
                checkpoint_path=model_path, map_location=cfg.device
            )
        else:
            if model_path not in get_available_model_names(EncDecDiarLabelModel):
                logging.warning(f"requested {model_path} model name not available in pretrained models, instead")
            logging.info("Loading pretrained {} model from NGC".format(model_path))
            self.msdd_model = EncDecDiarLabelModel.from_pretrained(model_name=model_path, map_location=cfg.device)
        # Load speaker embedding model state_dict which is loaded from the MSDD checkpoint.
        if self.use_speaker_model_from_ckpt:
            self._speaker_model = self.extract_standalone_speaker_model()
        else:
            self._speaker_model = None
        return self.msdd_model

    def get_pred_mat(self, data_list: List[Union[Tuple[int], List[torch.Tensor]]]) -> torch.Tensor:
        """
        This module puts together the pairwise, two-speaker, predicted results to form a finalized matrix that has dimension of
        `(total_len, n_est_spks)`. The pairwise results are evenutally averaged. For example, in 4 speaker case (speaker 1, 2, 3, 4),
        the sum of the pairwise results (1, 2), (1, 3), (1, 4) are then divided by 3 to take average of the sigmoid values.

        Args:
            data_list (list):
                List containing data points from `test_data_collection` variable. `data_list` has sublists `data` as follows:
                data[0]: `target_spks` tuple
                    Examples: (0, 1, 2)
                data[1]: Tensor containing estimaged sigmoid values.
                   [[0.0264, 0.9995],
                    [0.0112, 1.0000],
                    ...,
                    [1.0000, 0.0512]]

        Returns:
            sum_pred (Tensor):
                Tensor containing the averaged sigmoid values for each speaker.
        """
        all_tups = tuple()
        for data in data_list:
            all_tups += data[0]
        n_est_spks = len(set(all_tups))
        digit_map = dict(zip(sorted(set(all_tups)), range(n_est_spks)))
        total_len = max([sess[1].shape[1] for sess in data_list])
        sum_pred = torch.zeros(total_len, n_est_spks)
        for (_dim_tup, pred_mat) in data_list:
            dim_tup = [digit_map[x] for x in _dim_tup]
            if len(pred_mat.shape) == 3:
                pred_mat = pred_mat.squeeze(0)
            if n_est_spks <= self.num_spks_per_model:
                sum_pred = pred_mat
            else:
                _end = pred_mat.shape[0]
                sum_pred[:_end, dim_tup] += pred_mat.cpu().float()
        sum_pred = sum_pred / (n_est_spks - 1)
        return sum_pred

    def get_integrated_preds_list(
        self, uniq_id_list: List[str], test_data_collection: List[Any], preds_list: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Merge multiple sequence inference outputs into a session level result.

        Args:
            uniq_id_list (list):
                List containing `uniq_id` values.
            test_data_collection (collections.DiarizationLabelEntity):
                Class instance that is containing session information such as targeted speaker indices, audio filepaths and RTTM filepaths.
            preds_list (list):
                List containing tensors filled with sigmoid values.

        Returns:
            output_list (list):
                List containing session-level estimated prediction matrix.
        """
        session_dict = get_id_tup_dict(uniq_id_list, test_data_collection, preds_list)
        output_dict = {uniq_id: [] for uniq_id in uniq_id_list}
        for uniq_id, data_list in session_dict.items():
            sum_pred = self.get_pred_mat(data_list)
            output_dict[uniq_id] = sum_pred.unsqueeze(0)
        output_list = [output_dict[uniq_id] for uniq_id in uniq_id_list]
        return output_list

    def get_emb_clus_infer(self, cluster_embeddings):
        """Assign dictionaries containing the clustering results from the class instance `cluster_embeddings`.
        """
        self.msdd_model.emb_sess_test_dict = cluster_embeddings.emb_sess_test_dict
        self.msdd_model.clus_test_label_dict = cluster_embeddings.clus_test_label_dict
        self.msdd_model.emb_seq_test = cluster_embeddings.emb_seq_test

    @torch.no_grad()
    def diarize(self) -> Optional[List[Optional[List[Tuple[DiarizationErrorRate, Dict]]]]]:
        """
        Launch diarization pipeline which starts from VAD (or a oracle VAD stamp generation), 
        initialization clustering and multiscale diarization decoder (MSDD).
        Note that the result of MSDD can include multiple speakers at the same time. 
        Therefore, RTTM output of MSDD needs to be based on `make_rttm_with_overlap()`
        function that can generate overlapping timestamps. 
        `self.run_overlap_aware_eval()` function performs DER evaluation.
        """
        self.transfer_diar_params_to_model_params(self._cfg)
        self.msdd_model.emb_seq_test, self.msdd_model.ms_time_stamps, self.msdd_model.clus_test_label_dict = self.clustering_embedding.prepare_cluster_embs_infer()
        self.msdd_model.pairwise_infer = True
        # self.get_emb_clus_infer(self.clustering_embedding)
        # preds_list, targets_list, signal_lengths_list 
        preds, targets, ms_ts = self.run_multiscale_decoder()
        thresholds = list(self._cfg.diarizer.msdd_model.parameters.sigmoid_threshold)
        for threshold in thresholds:
            self.run_overlap_aware_eval(preds, ms_ts, threshold) 

    def get_range_average(
        self, signals: torch.Tensor, emb_vectors: torch.Tensor, diar_window_index: int, test_data_collection: List[Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        This function is only used when `split_infer=True`. This module calculates cluster-average embeddings for the given short range.
        The range length is set by `self.diar_window_length`, and each cluster-average is only calculated for the specified range.
        Note that if the specified range does not contain some speakers (e.g. the range contains speaker 1, 3) compared to the global speaker sets
        (e.g. speaker 1, 2, 3, 4) then the missing speakers (e.g. speakers 2, 4) are assigned with zero-filled cluster-average speaker embedding.

        Args:
            signals (Tensor):
                Zero-padded Input multi-scale embedding sequences.
                Shape: (length, scale_n, emb_vectors, emb_dim)
            emb_vectors (Tensor):
                Cluster-average multi-scale embedding vectors.
                Shape: (length, scale_n, emb_vectors, emb_dim)
            diar_window_index (int):
                Index of split diarization wondows.
            test_data_collection (collections.DiarizationLabelEntity)
                Class instance that is containing session information such as targeted speaker indices, audio filepath and RTTM filepath.

        Returns:
            return emb_vectors_split (Tensor):
                Cluster-average speaker embedding vectors for each scale.
            emb_seq (Tensor):
                Zero-padded multi-scale embedding sequences.
            seq_len (int):
                Length of the sequence determined by `self.diar_window_length` variable.
        """
        emb_vectors_split = torch.zeros_like(emb_vectors)
        uniq_id = os.path.splitext(os.path.basename(test_data_collection.audio_file))[0]
        clus_label_tensor = torch.tensor([x[-1] for x in self.msdd_model.clus_test_label_dict[uniq_id]])
        for spk_idx in range(len(test_data_collection.target_spks)):
            stt, end = (
                diar_window_index * self.diar_window_length,
                min((diar_window_index + 1) * self.diar_window_length, clus_label_tensor.shape[0]),
            )
            seq_len = end - stt
            if stt < clus_label_tensor.shape[0]:
                target_clus_label_tensor = clus_label_tensor[stt:end]
                emb_seq, seg_length = (
                    signals[stt:end, :, :],
                    min(
                        self.diar_window_length,
                        clus_label_tensor.shape[0] - diar_window_index * self.diar_window_length,
                    ),
                )
                target_clus_label_bool = target_clus_label_tensor == test_data_collection.target_spks[spk_idx]

                # There are cases where there is no corresponding speaker in split range, so any(target_clus_label_bool) could be False.
                if any(target_clus_label_bool):
                    emb_vectors_split[:, :, spk_idx] = torch.mean(emb_seq[target_clus_label_bool], dim=0)

                # In case when the loop reaches the end of the sequence
                if seq_len < self.diar_window_length:
                    emb_seq = torch.cat(
                        [
                            emb_seq,
                            torch.zeros(self.diar_window_length - seq_len, emb_seq.shape[1], emb_seq.shape[2]).to(
                                signals.device
                            ),
                        ],
                        dim=0,
                    )
            else:
                emb_seq = torch.zeros(self.diar_window_length, emb_vectors.shape[0], emb_vectors.shape[1]).to(
                    signals.device
                )
                seq_len = 0
        return emb_vectors_split, emb_seq, seq_len

    def get_range_clus_avg_emb(
        self, test_batch: List[torch.Tensor], _test_data_collection: List[Any], device: torch.device('cpu')
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This function is only used when `get_range_average` function is called. This module calculates cluster-average embeddings for
        the given short range. The range length is set by `self.diar_window_length`, and each cluster-average is only calculated for the specified range.

        Args:
            test_batch: (list)
                List containing embedding sequences, length of embedding sequences, ground truth labels (if exists) and initializing embedding vectors.
            test_data_collection: (list)
                List containing test-set dataloader contents. test_data_collection includes wav file path, RTTM file path, clustered speaker indices.

        Returns:
            sess_emb_vectors (Tensor):
                Tensor of cluster-average speaker embedding vectors.
                Shape: (batch_size, scale_n, emb_dim, 2*num_of_spks)
            sess_emb_seq (Tensor):
                Tensor of input multi-scale embedding sequences.
                Shape: (batch_size, length, scale_n, emb_dim)
            sess_sig_lengths (Tensor):
                Tensor of the actucal sequence length without zero-padding.
                Shape: (batch_size)
        """
        _signals, signal_lengths, _targets, _emb_vectors = test_batch
        sess_emb_vectors, sess_emb_seq, sess_sig_lengths = [], [], []
        split_count = torch.ceil(torch.tensor(_signals.shape[1] / self.diar_window_length)).int()
        self.max_pred_length = max(self.max_pred_length, self.diar_window_length * split_count)
        for k in range(_signals.shape[0]):
            signals, emb_vectors, test_data_collection = _signals[k], _emb_vectors[k], _test_data_collection[k]
            for diar_window_index in range(split_count):
                emb_vectors_split, emb_seq, seq_len = self.get_range_average(
                    signals, emb_vectors, diar_window_index, test_data_collection
                )
                sess_emb_vectors.append(emb_vectors_split)
                sess_emb_seq.append(emb_seq)
                sess_sig_lengths.append(seq_len)
        sess_emb_vectors = torch.stack(sess_emb_vectors).to(device)
        sess_emb_seq = torch.stack(sess_emb_seq).to(device)
        sess_sig_lengths = torch.tensor(sess_sig_lengths).to(device)
        return sess_emb_vectors, sess_emb_seq, sess_sig_lengths

    def run_multiscale_decoder(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Setup the parameters needed for batch inference and run batch inference. Note that each sample is pairwise speaker input.
        The pairwise inference results are reconstructed to make session-wise prediction results.

        Returns:
            integrated_preds_list: (list)
                List containing the session-wise speaker predictions in torch.tensor format.
            targets_list: (list)
                List containing the ground-truth labels in matrix format filled with  0 or 1.
            signal_lengths_list: (list)
                List containing the actual length of each sequence in session.
        """
        self.out_rttm_dir = self.clustering_embedding.out_rttm_dir
        self.msdd_model.setup_test_data(self.msdd_model.cfg.test_ds)
        self.msdd_model.eval()
        timestamp_res = 0.01

        cumul_sample_count = [0]
        preds_list, targets_list, timestamps_list = [], [], []
        uniq_id_list = get_uniq_id_list_from_manifest(self.msdd_model.cfg.test_ds.manifest_filepath)
        test_data_collection = [d for d in self.msdd_model.data_collection]
        device = self.msdd_model.device
        for test_batch_idx, test_batch in enumerate(tqdm(self.msdd_model.test_dataloader())):
            ms_emb_seq, ms_timestamps, seq_lengths, clus_label_index, targets = test_batch
            ms_emb_seq, seq_lengths, clus_label_index, targets = ms_emb_seq.to(device), seq_lengths.to(device), clus_label_index.to(device), targets.to(device)
            cumul_sample_count.append(cumul_sample_count[-1] + seq_lengths.shape[0])
            ms_avg_embs = self.msdd_model.get_cluster_avg_embs_model(ms_emb_seq, clus_label_index, seq_lengths).to(self.msdd_model.device)
            
            ms_emb_seq = ms_emb_seq.type(ms_avg_embs.dtype)
            preds, scale_weights = self.msdd_model.forward_infer(ms_emb_seq=ms_emb_seq, seq_lengths=seq_lengths, ms_avg_embs=ms_avg_embs)

            if self._cfg.diarizer.msdd_model.parameters.seq_eval_mode:
                self.msdd_model._accuracy_test(preds.type(torch.float), _targets.type(torch.float), signal_lengths)
            preds_list.append(preds.detach().cpu())
            targets_list.append(targets.detach().cpu())
            timestamps_list.append(timestamp_res*ms_timestamps)

        if self._cfg.diarizer.msdd_model.parameters.seq_eval_mode:
            f1_score, simple_acc = self.msdd_model.compute_accuracies()
            logging.info(f"Test Inference F1 score. {f1_score:.4f}, simple Acc. {simple_acc:.4f}")
        test_manifest_file = self.msdd_model.cfg.test_ds.manifest_filepath
        all_preds, all_targets, all_time_stamps = self._stitch_and_save(test_manifest_file, preds_list, targets_list, timestamps_list)
        return all_preds, all_targets, all_time_stamps


    def _stitch_and_save(
        self, 
        manifest_file, 
        all_preds_list, 
        all_targets_list, 
        all_ts_list,
        ):
        diar_manifest = read_manifest(manifest_file)
        batch_size = self.msdd_model.cfg.test_ds.batch_size
        decimals = 2
        feat_per_sec = 100
        feat_len = float(1/feat_per_sec)

        overlap_sec = self.msdd_model.diar_window - self.msdd_model.cfg.diarizer.speaker_embeddings.parameters.window_length_in_sec[0]
        self.preds, self.targets, self.time_stamps = {}, {}, {}
        
        base_seg_per_longest_seg = int((self.msdd_model.diar_window - self.msdd_model.diar_window_shift) / (self.msdd_model.cfg.interpolated_scale/2))-1
        all_manifest_uniq_ids = get_uniq_id_list_from_manifest(self.msdd_model.segmented_manifest_path)
        
        for batch_idx, (preds, targets, ms_seg_ts) in enumerate(zip(all_preds_list, all_targets_list, all_ts_list)):
            batch_len = preds.shape[0]
            batch_uniq_ids = all_manifest_uniq_ids[batch_idx * batch_size: (batch_idx+1) * batch_size ]
            batch_manifest = self.msdd_model.segmented_manifest_list[batch_idx * batch_size: (batch_idx+1) * batch_size ]
            for sample_id, uniq_id in enumerate(batch_uniq_ids):
                if uniq_id in self.preds:
                    
                    offset_sec = int(batch_manifest[sample_id]['offset'])
                    ts_trunc = self.time_stamps[uniq_id][:, :-base_seg_per_longest_seg, :] 
                    ts_add = all_ts_list[batch_idx][sample_id][:] + offset_sec
                    self.time_stamps[uniq_id]= torch.cat((ts_trunc, ts_add), dim=1)
                    
                    preds_trunc = self.preds[uniq_id][:-base_seg_per_longest_seg, :]
                    preds_add = all_preds_list[batch_idx][sample_id]
                    self.preds[uniq_id] = torch.cat((preds_trunc, preds_add), dim=0)
                    
                    targets_trunc = self.targets[uniq_id][:-base_seg_per_longest_seg, :]
                    targets_add = all_targets_list[batch_idx][sample_id]
                    self.targets[uniq_id] = torch.cat((targets_trunc, targets_add), dim=0)
                else:
                    self.time_stamps[uniq_id] = all_ts_list[batch_idx][sample_id]
                    self.preds[uniq_id] = all_preds_list[batch_idx][sample_id]
                    self.targets[uniq_id] = all_targets_list[batch_idx][sample_id]
        
        return self.preds, self.targets, self.time_stamps

    def run_overlap_aware_eval(
        self, preds_dict, ms_ts, threshold: float
    ) -> List[Optional[Tuple[DiarizationErrorRate, Dict]]]:
        """
        Based on the predicted sigmoid values, render RTTM files then evaluate the overlap-aware diarization results.

        Args:
            preds_list: (list)
                List containing predicted pairwise speaker labels.
            threshold: (float)
                A floating-point threshold value that determines overlapped speech detection.
                    - If threshold is 1.0, no overlap speech is detected and only detect major speaker.
                    - If threshold is 0.0, all speakers are considered active at any time step.
        """
        logging.info(
            f"     [Threshold: {threshold:.4f}] [use_clus_as_main={self.use_clus_as_main}] [diar_window={self.diar_window_length}]"
        )
        outputs = []
        # manifest_filepath = self.msdd_model.cfg.test_ds.manifest_filepath
        manifest_filepath = self._cfg.diarizer.manifest_filepath
        rttm_map = audio_rttm_map(manifest_filepath)
        # for k, (collar, ignore_overlap) in enumerate(self.diar_eval_settings):
        collar = 0.25
        ignore_overlap = True

        all_reference, all_hypothesis = make_rttm_with_overlap(
            manifest_filepath,
            self.msdd_model.clus_test_label_dict,
            preds_dict=preds_dict,
            ms_ts=ms_ts,
            threshold=threshold,
            infer_overlap=True,
            use_clus_as_main=self.use_clus_as_main,
            overlap_infer_spk_limit=self.overlap_infer_spk_limit,
            use_adaptive_thres=self.use_adaptive_thres,
            max_overlap_spks=self.max_overlap_spks,
            out_rttm_dir=self.out_rttm_dir,
        )

        output = score_labels(
            rttm_map,
            all_reference,
            all_hypothesis,
            collar=collar,
            ignore_overlap=ignore_overlap,
            verbose=self._cfg.verbose,
        )
        outputs.append(output)
        logging.info(f"  \n")
        return outputs

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        vad_model_name: str = 'vad_multilingual_marblenet',
        map_location: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Instantiate a `NeuralDiarizer` to run Speaker Diarization.

        Args:
            model_name (str): Path/Name of the neural diarization model to load.
            vad_model_name (str): Path/Name of the voice activity detection (VAD) model to load.
            map_location (str): Optional str to map the instantiated model to a device (cpu, cuda).
                By default, (None), it will select a GPU if available, falling back to CPU otherwise.
            verbose (bool): Enable verbose logging when loading models/running diarization.
        Returns:
            `NeuralDiarizer`
        """
        logging.setLevel(logging.INFO if verbose else logging.WARNING)
        cfg = NeuralDiarizerInferenceConfig.init_config(
            diar_model_path=model_name, vad_model_path=vad_model_name, map_location=map_location, verbose=verbose,
        )
        return cls(cfg)

    def __call__(
        self,
        audio_filepath: str,
        batch_size: int = 64,
        num_workers: int = 1,
        max_speakers: Optional[int] = None,
        num_speakers: Optional[int] = None,
        out_dir: Optional[str] = None,
        verbose: bool = False,
    ) -> Union[Annotation, List[Annotation]]:
        """
        Run the `NeuralDiarizer` inference pipeline.

        Args:
            audio_filepath (str, list): Audio path to run speaker diarization on.
            max_speakers (int): If known, the max number of speakers in the file(s).
            num_speakers (int): If known, the exact number of speakers in the file(s).
            batch_size (int): Batch size when running inference.
            num_workers (int): Number of workers to use in data-loading.
            out_dir (str): Path to store intermediate files during inference (default temp directory).
        Returns:
            `pyannote.Annotation` for each audio path, containing speaker labels and segment timestamps.
        """
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out_dir) as tmpdir:
            manifest_path = os.path.join(tmpdir, 'manifest.json')
            meta = [
                {
                    'audio_filepath': audio_filepath,
                    'offset': 0,
                    'duration': None,
                    'label': 'infer',
                    'text': '-',
                    'num_speakers': num_speakers,
                    'rttm_filepath': None,
                    'uem_filepath': None,
                }
            ]

            with open(manifest_path, 'w') as f:
                f.write('\n'.join(json.dumps(x) for x in meta))

            self._initialize_configs(
                manifest_path=manifest_path,
                max_speakers=max_speakers,
                num_speakers=num_speakers,
                tmpdir=tmpdir,
                batch_size=batch_size,
                num_workers=num_workers,
                verbose=verbose,
            )

            self.msdd_model.cfg.test_ds.manifest_filepath = manifest_path
            self.diarize()

            pred_labels_clus = rttm_to_labels(f'{tmpdir}/pred_rttms/{Path(audio_filepath).stem}.rttm')
        return labels_to_pyannote_object(pred_labels_clus)

    def _initialize_configs(
        self,
        manifest_path: str,
        max_speakers: Optional[int],
        num_speakers: Optional[int],
        tmpdir: tempfile.TemporaryDirectory,
        batch_size: int,
        num_workers: int,
        verbose: bool,
    ) -> None:
        self._cfg.batch_size = batch_size
        self._cfg.num_workers = num_workers
        self._cfg.diarizer.manifest_filepath = manifest_path
        self._cfg.diarizer.out_dir = tmpdir
        self._cfg.verbose = verbose
        self._cfg.diarizer.clustering.parameters.oracle_num_speakers = num_speakers is not None
        if max_speakers:
            self._cfg.diarizer.clustering.parameters.max_num_speakers = max_speakers
        self.transfer_diar_params_to_model_params(self._cfg)

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        return EncDecDiarLabelModel.list_available_models()