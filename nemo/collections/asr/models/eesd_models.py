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
import pickle as pkl
from collections import OrderedDict
from pathlib import Path
from statistics import mode
from typing import Any, Dict, List, Optional, Tuple, Union
from operator import attrgetter

import time
import torch
from torch import nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Optional, Union, Dict 
import itertools

from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations

from nemo.collections.asr.data.audio_to_msdd_label import AudioToSpeechMSDDTrainDataset
from nemo.collections.asr.data.audio_to_msdd_mock_label import AudioToSpeechMSDDTrainDataset as AudioToSpeechMSDDTrainMockEmbDataset
from nemo.collections.asr.models.multi_classification_models import EncDecMultiClassificationModel
from nemo.collections.asr.metrics.der import score_labels
from nemo.collections.asr.metrics.multi_binary_acc import MultiBinaryAccuracy
from nemo.collections.asr.models.asr_model import ExportableEncDecModel
from nemo.collections.asr.models.clustering_diarizer import (
    _MODEL_CONFIG_YAML,
    _SPEAKER_MODEL,
    get_available_model_names,
)


from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.utils.offline_clustering import (
    cos_similarity_batch,
)

from nemo.collections.asr.parts.utils.speaker_utils import parse_scale_configs
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType
from nemo.core.neural_types.elements import ProbsType, LabelsType, LossType
from nemo.utils import logging

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


__all__ = ['EncDecDiarLabelModel']

from nemo.core.classes import Loss, Typing, typecheck

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

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

    def __init__(self, gamma=0.0, negative_margin=0.5, positive_margin=0.05):
        super().__init__()
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

class SortformerEncLabelModel(ModelPT, ExportableEncDecModel):
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
        self.cfg_e2e_diarizer_model = cfg
        self.encoder_infer_mode = False
        
        self._init_segmentation_info()
        if self._trainer:
            self.world_size = trainer.num_nodes * trainer.num_devices
            self.pairwise_infer = False
        else:
            self.world_size = 1
            self.pairwise_infer = True

        self._init_msdd_scales() 
        if self._trainer is not None and self.cfg_e2e_diarizer_model.get('augmentor', None) is not None:
            self.augmentor = process_augmentations(self.cfg_e2e_diarizer_model.augmentor)
        else:
            self.augmentor = None
        super().__init__(cfg=self.cfg_e2e_diarizer_model, trainer=trainer)
        window_length_in_sec = self.cfg_e2e_diarizer_model.diarizer.speaker_embeddings.parameters.window_length_in_sec
        if isinstance(window_length_in_sec, int) or len(window_length_in_sec) <= self.cfg_e2e_diarizer_model.interpolated_scale:
            raise ValueError("window_length_in_sec should be a list containing multiple segment (window) lengths")
        else:
            self.cfg_e2e_diarizer_model.diarizer_module.scale_n = self.cfg_e2e_diarizer_model.scale_n

        self.preprocessor = EncDecSpeakerLabelModel.from_config_dict(self.cfg_e2e_diarizer_model.preprocessor)
        self.frame_per_sec = int(1 / self.preprocessor._cfg.window_stride)
        self.feat_dim = self.preprocessor._cfg.features
        self.max_feat_frame_count = int(self.msdd_multiscale_args_dict["scale_dict"][0][0] * self.frame_per_sec) # 0-th scale, window length
        self.sortformer_diarizer = SortformerEncLabelModel.from_config_dict(self.cfg_e2e_diarizer_model.diarizer_module)
        self.sortformer_encoder = SortformerEncLabelModel.from_config_dict(self.cfg_e2e_diarizer_model.sortformer_encoder)
        self.transformer_encoder = SortformerEncLabelModel.from_config_dict(self.cfg_e2e_diarizer_model.transformer_encoder)
        self.global_loss_ratio = self.cfg_e2e_diarizer_model.get('global_loss_ratio', 300)
   
        self.original_audio_offsets = {}
        self.train_f1_acc_history = []
        self.train_f1_acc_window_length = self.cfg_e2e_diarizer_model.get('train_f1_acc_window_length', 5)
        self.train_f1_acc_thres_pil_shift = self.cfg_e2e_diarizer_model.get('train_f1_acc_thres_pil_shift', 0.55)

        self.eps = 1e-3
        self.emb_dim = self.cfg_e2e_diarizer_model.diarizer_module.emb_dim
        self.train_non_linear_transform_layer = self.non_linear_transform_layer(layer_n=1, input_size=self.emb_dim, hidden_size=2*self.emb_dim, output_size=self.emb_dim, seed=100)
        self.valid_non_linear_transform_layer = self.non_linear_transform_layer(layer_n=4, input_size=self.emb_dim, hidden_size=2*self.emb_dim, output_size=self.emb_dim, seed=200)
        
        # MSDD v2 parameters
        self.encoder_infer_mode = False

        if trainer is not None:
            self._init_speaker_model()
            self.add_speaker_model_config(cfg)
            self.loss = instantiate(self.cfg_e2e_diarizer_model.loss)
            self.affinity_loss = instantiate(self.cfg_e2e_diarizer_model.affinity_loss) 
            self.power_p_aff = self.cfg_e2e_diarizer_model.get('power_p_aff', 3)
            self.thres_aff = self.cfg_e2e_diarizer_model.get('thres_aff', 0.25)
            self.mc_audio_normalize = self.cfg_e2e_diarizer_model.get('mc_audio_normalize', True)
            self.power_p = self.cfg_e2e_diarizer_model.get('power_p', 2)
            self.mix_count = self.cfg_e2e_diarizer_model.get('mix_count', 3)
            self.multichannel_mixing = self.cfg_e2e_diarizer_model.get('multichannel_mixing', True)
        else:
            self._init_speaker_model()
            self.loss = instantiate(self.cfg_e2e_diarizer_model.loss)
            self.multichannel_mixing = self.cfg_e2e_diarizer_model.get('multichannel_mixing', True)
        self.alpha = self.cfg_e2e_diarizer_model.alpha
        self.affinity_weighting = self.cfg_e2e_diarizer_model.get('affinity_weighting', True)
        self.msdd_overlap_add = self.cfg_e2e_diarizer_model.get("msdd_overlap_add", True)
        self.use_1ch_from_ch_clus = self.cfg_e2e_diarizer_model.get("use_1ch_from_ch_clus", True)
        if self.cfg_e2e_diarizer_model.get("multichannel", None) is not None:
            self.power_p=self.cfg_e2e_diarizer_model.multichannel.parameters.get("power_p", 4)
            self.mix_count=self.cfg_e2e_diarizer_model.multichannel.parameters.get("mix_count", 2) 
        else:
            self.power_p=4
            self.mix_count=2
        
        # Call `self.save_hyperparameters` in modelPT.py again since cfg should contain speaker model's config.
        self.save_hyperparameters("cfg")

        self._accuracy_test = MultiBinaryAccuracy()
        self._accuracy_train = MultiBinaryAccuracy()
        self._accuracy_valid = MultiBinaryAccuracy()
        self._accuracy_valid_toplyr = MultiBinaryAccuracy()
        self._accuracy_valid_prdmean = MultiBinaryAccuracy()
        self._accuracy_valid = MultiBinaryAccuracy()
        self._accuracy_test_toplyr = MultiBinaryAccuracy()
        self._accuracy_test_prdmean = MultiBinaryAccuracy()
        self._accuracy_train_vad= MultiBinaryAccuracy()
        self._accuracy_valid_vad= MultiBinaryAccuracy()
        self._accuracy_test_vad= MultiBinaryAccuracy()
        self._accuracy_train_ovl= MultiBinaryAccuracy()
        self._accuracy_valid_ovl= MultiBinaryAccuracy()
        self._accuracy_test_ovl= MultiBinaryAccuracy()
        self.max_f1_acc = 0.0

        self.time_flag = 0.0
        self.time_flag_end = 0.0

        speaker_inds = list(range(self.cfg_e2e_diarizer_model.max_num_of_spks))
        # Get all permutations
        self.spk_perm = torch.tensor(list(itertools.permutations(speaker_inds)))
        
    def _init_msdd_scales(self,):
        window_length_in_sec = self.cfg_e2e_diarizer_model.diarizer.speaker_embeddings.parameters.window_length_in_sec
        self.msdd_multiscale_args_dict = self.multiscale_args_dict
        self.model_spk_num = self.cfg_e2e_diarizer_model.max_num_of_spks
        if self.cfg_e2e_diarizer_model.get('interpolated_scale', None) is not None:
            # if self.cfg_e2e_diarizer_model.interpolated_scale > 0.1:
            #     raise ValueError("Interpolated scale must be smaller than 0.1")
            # Use an interpolated scale so add another scale 
            self.cfg_e2e_diarizer_model.scale_n = len(window_length_in_sec) + 1 # Adding one interpolated scale
            self.emb_scale_n = len(window_length_in_sec) # Scales that are extracted from the audio
            self.msdd_multiscale_args_dict['scale_dict'][self.emb_scale_n] = (self.cfg_e2e_diarizer_model.interpolated_scale, self.cfg_e2e_diarizer_model.interpolated_scale/2)
            self.msdd_multiscale_args_dict['multiscale_weights'] = [1.0] * (self.emb_scale_n+1)
            self.msdd_scale_n = int(self.emb_scale_n+1) if self.cfg_e2e_diarizer_model.interpolated_scale is not None else int(self.emb_scale_n)
        else:
            # Only use the scales in window_length_in_sec
            self.cfg_e2e_diarizer_model.scale_n = len(window_length_in_sec)
            self.emb_scale_n = self.cfg_e2e_diarizer_model.scale_n
            self.msdd_scale_n = self.cfg_e2e_diarizer_model.scale_n

    def non_linear_transform_layer(self, layer_n, input_size, hidden_size, output_size, seed):
        torch.manual_seed(seed)
        layers = []

        # First layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.Sigmoid())

        # Additional hidden layers
        for _ in range(1, layer_n):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Sigmoid())

        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))

        # Create the sequential model
        model = nn.Sequential(*layers)
        model.apply(init_weights)
        model.eval()
        return model
    
    def setup_optimizer_param_groups(self):
        """
        Override function in ModelPT to allow for different parameter groups for the speaker model and the MSDD model.
        """
        if not hasattr(self, "parameters"):
            self._optimizer_param_groups = None
            return

        param_groups, known_groups = [], []
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
            cfg_cp = copy.copy(self.sortformer_diarizer._speaker_model.cfg)
            cfg.speaker_model_cfg = cfg_cp
            del cfg.speaker_model_cfg.train_ds
            del cfg.speaker_model_cfg.validation_ds
    
    def _init_segmentation_info(self):
        """Initialize segmentation settings: window, shift and multiscale weights.
        """
        self._diarizer_params = self.cfg_e2e_diarizer_model.diarizer
        self.multiscale_args_dict = parse_scale_configs(
            self._diarizer_params.speaker_embeddings.parameters.window_length_in_sec,
            self._diarizer_params.speaker_embeddings.parameters.shift_length_in_sec,
            self._diarizer_params.speaker_embeddings.parameters.multiscale_weights,
        )

    def _init_speaker_model(self):
        """
        Initialize speaker embedding model with model name or path passed through config. Note that speaker embedding model is loaded to
        `self.msdd` to enable multi-gpu and multi-node training. In addition, speaker embedding model is also saved with msdd model when
        `.ckpt` files are saved.
        """
        model_path = self.cfg_e2e_diarizer_model.diarizer.speaker_embeddings.model_path
        self._diarizer_params = self.cfg_e2e_diarizer_model.diarizer

        if not torch.cuda.is_available():
            rank_id = torch.device('cpu')
        elif self._trainer:
            if self._trainer.global_rank > torch.cuda.device_count() - 1:
                rank_id = torch.device(self._trainer.global_rank % torch.cuda.device_count())
            else:
                rank_id = torch.device(self._trainer.global_rank)
        else:
            rank_id = None
        
        if model_path is not None and model_path.endswith('.nemo'):
            self.sortformer_diarizer._speaker_model = EncDecSpeakerLabelModel.restore_from(model_path, map_location=rank_id)
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
            self.sortformer_diarizer._speaker_model = EncDecSpeakerLabelModel.from_pretrained(
                model_name=model_path, map_location=rank_id
            )
        
        if self.cfg_e2e_diarizer_model.get("speaker_decoder", None) is not None:
            self.sortformer_diarizer._speaker_model_decoder = EncDecSpeakerLabelModel.from_config_dict(self.cfg_e2e_diarizer_model.speaker_decoder)
            self.sortformer_diarizer._speaker_model.decoder.angular = True
            self.sortformer_diarizer._speaker_model.decoder.final = self.sortformer_diarizer._speaker_model_decoder.final
            
        if self._cfg.freeze_speaker_model:
            self.sortformer_diarizer._speaker_model.eval()

        self._speaker_params = self.cfg_e2e_diarizer_model.diarizer.speaker_embeddings.parameters
    
    def __setup_dataloader_from_config(self, config):
        featurizer = WaveformFeaturizer(
            sample_rate=config['sample_rate'], int_values=config.get('int_values', False), augmentor=self.augmentor
        )

        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None

        logging.info(f"Loading dataset from {config.manifest_filepath}")

        if self._trainer is not None:
            global_rank = self._trainer.global_rank
        else:
            global_rank = 0
        time_flag = time.time()
        print(f"AAB: Starting Dataloader Instance loading... Step A")
        
        if self.cfg_e2e_diarizer_model.use_mock_embs:
            AudioToSpeechDiarTrainDataset = AudioToSpeechMSDDTrainMockEmbDataset
        else:
            AudioToSpeechDiarTrainDataset = AudioToSpeechMSDDTrainDataset
        
        dataset = AudioToSpeechDiarTrainDataset(
            manifest_filepath=config.manifest_filepath,
            emb_dir=config.emb_dir,
            multiscale_args_dict=self.msdd_multiscale_args_dict,
            soft_label_thres=config.soft_label_thres,
            random_flip=config.random_flip,
            session_len_sec=config.session_len_sec,
            num_spks=config.num_spks,
            featurizer=featurizer,
            window_stride=self.cfg_e2e_diarizer_model.preprocessor.window_stride,
            emb_batch_size=100,
            pairwise_infer=False,
            global_rank=global_rank,
            encoder_infer_mode=self.encoder_infer_mode,
        )
        logging.info(f"AAB: Dataloader dataset is created, starting torch.utils.data.Dataloader step B: {time.time() - time_flag}")

        self.data_collection = dataset.collection
        self.collate_ds = dataset
        self.ms_seg_timestamps = dataset.ms_seg_timestamps
        self.ms_seg_counts = dataset.ms_seg_counts
        self.scale_mapping = dataset.scale_mapping
         
        dataloader_instance = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            collate_fn=self.collate_ds.msdd_train_collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=False,
            num_workers=config.get('num_workers', 1),
            pin_memory=config.get('pin_memory', False),
        )
        print(f"AAC: Dataloader Instance loading is done ETA Step B done: {time.time() - time_flag}")
        return dataloader_instance
    
    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        self._train_dl = self.__setup_dataloader_from_config(config=train_data_config,)

    def setup_validation_data(self, val_data_layer_config: Optional[Union[DictConfig, Dict]]):
        self._validation_dl = self.__setup_dataloader_from_config(config=val_data_layer_config,)
    
    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        self._test_dl = self.__setup_dataloader_from_config(config=test_data_config,)

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
            "audio_signal": NeuralType(('B', 'T'), audio_eltype),
            "audio_signal_length": NeuralType(('B',), LengthsType()),
            "ms_seg_timestamps": NeuralType(('B', 'C', 'T', 'D'), LengthsType()),
            "ms_seg_counts": NeuralType(('B', 'C'), LengthsType()),
            "scale_mapping": NeuralType(('B', 'C', 'T'), LengthsType()),
            "global_spk_labels": NeuralType(('B', 'T'), LengthsType()),
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

        base_seq_len = ms_seg_counts[0][self.msdd_scale_n-1].item()
        target_embs = torch.vstack(split_emb_tup).reshape(batch_size, -1, embs.shape[-1])
        intp_w = get_interpolate_weights(ms_seg_timestamps[0], 
                                         base_seq_len, 
                                         self.msdd_multiscale_args_dict, 
                                         self.emb_scale_n, 
                                         self.msdd_scale_n, 
                                         is_integer_ts=True)
        
        # To make offset values such as, [10, 20, 60, x] -> [0, 10, 30, 90]
        ms_emb_seq = self.add_interpolated_embs(target_embs=target_embs, 
                                                intp_w=intp_w, 
                                                scale_mapping=scale_mapping,
                                                ms_seg_counts=ms_seg_counts, 
                                                embs=embs, 
        )
        return ms_emb_seq
    
    def repeat_and_align(
        self, 
        ms_seg_timestamps, 
        scale_mapping, 
        all_seq_len, 
        batch_size
        ):
        device = ms_seg_timestamps.device
        repeat_mats_all = scale_mapping[0].to(device)
        ms_ts = ms_seg_timestamps.reshape(batch_size, -1, 2)
        offsets_for_batch = (all_seq_len * torch.arange(self.msdd_scale_n).to(device)).unsqueeze(1).repeat(1, all_seq_len).to(device)
        repeat_mats_all = repeat_mats_all + offsets_for_batch
        ms_ts_rep = ms_ts[:, repeat_mats_all.flatten(), :].reshape(batch_size, self.msdd_scale_n, -1, 2)
        return ms_ts_rep

    def add_interpolated_embs(
        self, 
        target_embs, 
        intp_w,
        scale_mapping,
        ms_seg_counts, 
        embs, 
        ):
        batch_size = scale_mapping.shape[0]
        repeat_mats_ext = scale_mapping[0][:self.emb_scale_n].to(embs.device)
        all_seq_len = ms_seg_counts[0][-1].to(embs.device) 
        
        scale_count_offsets = torch.tensor([0] + torch.cumsum(ms_seg_counts[0][:self.emb_scale_n-1], dim=0).tolist())
        repeat_mats_ext = repeat_mats_ext + (scale_count_offsets.to(embs.device)).unsqueeze(1).repeat(1, all_seq_len).to(embs.device)
        extracted_embs = target_embs[:, repeat_mats_ext.flatten(), :].reshape(batch_size, self.emb_scale_n, -1, embs.shape[-1])
        finest_extracted_start = ms_seg_counts[0][:self.emb_scale_n-1].sum()
        interpolated_embs = torch.bmm(intp_w.repeat(batch_size, 1, 1), target_embs[:, finest_extracted_start:, :]).unsqueeze(1)
        ms_emb_seq = torch.cat((extracted_embs, interpolated_embs), dim=1).transpose(2, 1) 
        return ms_emb_seq
    
    def get_feature_index_map(
        self, 
        emb_scale_n,
        processed_signal, 
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
        feature_frame_length_range, feature_frame_interval_range= self.get_feat_range_matirx(max_feat_len=processed_signal.shape[2], 
                                                                                             feature_count_range=feature_count_range, 
                                                                                             target_timestamps=target_timestamps, 
                                                                                             device=processed_signal.device)
        # Assign frame-by-frame indices for one-pass assignment without nested for-loops
        ms_seg_count_frame_range = torch.repeat_interleave(torch.arange(total_seg_count).to(device), feature_count_range)       
        batch_frame_range = torch.repeat_interleave(batch_index_range, feature_count_range)
        return total_seg_count, ms_seg_count_frame_range, feature_frame_length_range, batch_frame_range, feature_frame_interval_range, feature_count_range

    def forward_multiscale(
        self, 
        processed_signal, 
        processed_signal_len, 
        ms_seg_timestamps, 
        ms_seg_counts,
        ):
        tsc, mscfr, fflr, bfr, ffir, fcr = self.get_feature_index_map(emb_scale_n=self.emb_scale_n,
                                                                      processed_signal=processed_signal, 
                                                                      ms_seg_timestamps=ms_seg_timestamps, 
                                                                      ms_seg_counts=ms_seg_counts, 
                                                                      device=processed_signal.device)

        embs, pools = self.forward_multi_decoder(processed_signal=processed_signal, 
                                            processed_signal_len=processed_signal_len, 
                                            total_seg_count=tsc,
                                            ms_seg_count_frame_range=mscfr, 
                                            feature_frame_length_range=fflr, 
                                            batch_frame_range=bfr, 
                                            feature_frame_interval_range=ffir,
                                            feature_count_range=fcr,
                                            device=processed_signal.device,
                                            )

        return embs, pools


    def forward_multi_decoder(
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
        encoded, _ = self.sortformer_diarizer._speaker_model.encoder(audio_signal=processed_signal, length=processed_signal_len)
        encoded_segments = torch.zeros(total_seg_count, encoded.shape[1], self.max_feat_frame_count).to(torch.float32).to(device)
        encoded_segments[ms_seg_count_frame_range, :, feature_frame_length_range] = encoded[batch_frame_range, :, feature_frame_interval_range]
        pools, embs = self.sortformer_diarizer._speaker_model.decoder(encoder_output=encoded_segments, length=feature_count_range) 
        return embs, pools
    
    def get_feat_range_matirx(self, max_feat_len, feature_count_range, target_timestamps, device):
        """ 
        """
        feat_index_range = torch.arange(0, max_feat_len).to(device) 
        feature_frame_offsets = torch.repeat_interleave(target_timestamps[:, 0], feature_count_range)
        feature_frame_interval_range = torch.concat([feat_index_range[stt:end] for (stt, end) in target_timestamps]).to(device)
        feature_frame_length_range = feature_frame_interval_range - feature_frame_offsets
        return feature_frame_length_range, feature_frame_interval_range
    
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

    def forward_infer(self, emb_seq):
        """

        Args:
            ms_emb_seq (torch.Tensor): tensor containing embeddings of multiscale embedding vectors
                Dimension: (batch_size, max_seg_count, msdd_scale_n, emb_dim)
            length (torch.Tensor): tensor containing lengths of multiscale segments
                Dimension: (batch_size, max_seg_count)
            ms_avg_embs (torch.Tensor): tensor containing average embeddings of multiscale segments
                Dimension: (batch_size, msdd_scale_n, emb_dim)

        """
        attn_score_list, preds_list, attn_score_stack, encoder_states_list = [], [], None, []
        encoder_mask = self.length_to_mask(emb_seq)
        if self._cfg.sortformer_encoder.num_layers > 0 and self._cfg.sortformer_encoder.sort_layer_on == 'pre':
            emb_seq, attn_score_list, preds_list, preds_mean, encoder_states_list = self.sortformer_encoder(encoder_states=emb_seq, encoder_mask=encoder_mask)
            attn_score_stack = torch.hstack(attn_score_list)
            
        emb_seq = self.transformer_encoder(encoder_states=emb_seq, encoder_mask=encoder_mask)
        
        if self._cfg.sortformer_encoder.num_layers > 0 and self._cfg.sortformer_encoder.sort_layer_on == 'post':
            emb_seq, attn_score_list, preds_list, preds_mean, encoder_states_list = self.sortformer_encoder(encoder_states=emb_seq, encoder_mask=encoder_mask)
            attn_score_stack = torch.hstack(attn_score_list)
        _preds = self.sortformer_diarizer.forward_speaker_sigmoids(emb_seq)
        _preds = self.sort_probs_and_labels(_preds, discrete=False)
        
        if self.sortformer_encoder.sort_bin_order and self._cfg.sortformer_encoder.num_layers > 0:
            preds = self.alpha * _preds + (1 - self.alpha) * preds_mean
            preds = self.sort_probs_and_labels(preds, discrete=False)
        else:
            preds = _preds
        return preds, _preds, attn_score_stack, preds_list, encoder_states_list
    
    def forward_encoder(
        self, 
        audio_signal, 
        audio_signal_length, 
        ms_seg_timestamps, 
        ms_seg_counts, 
        scale_mapping, 
    ):
        """
        Encoder part for end-to-end diarizaiton model.

        """
        audio_signal = audio_signal.to(self.device)
        self.sortformer_diarizer._speaker_model = self.sortformer_diarizer._speaker_model.to(self.device)
        self.sortformer_encoder = self.sortformer_encoder.to(self.device)
        audio_signal = (1/(audio_signal.max()+self.eps)) * audio_signal 
            
        processed_signal, processed_signal_length = self.sortformer_diarizer._speaker_model.preprocessor(
            input_signal=audio_signal, length=audio_signal_length
        )

        embs, pools = self.forward_multiscale(
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
                                           ms_seg_timestamps=ms_seg_timestamps)
        return ms_emb_seq
    
    def forward(
        self, 
        audio_signal, 
        audio_signal_length, 
        ms_seg_timestamps, 
        ms_seg_counts, 
        scale_mapping, 
    ):
        """
        Forward pass for training.
        
        if self.cfg_e2e_diarizer_model.use_mock_embs is True, then audio_signal is used as emb_seed.
            audio_signal dimension is [batch, emb_seed_dim*max_num_of_spks]
        if self.cfg_e2e_diarizer_model.use_mock_embs is False, then audio_signal is actual time series audio signal.
        """        
        if self.cfg_e2e_diarizer_model.use_mock_embs:
            emb_seq = self.train_non_linear_transform_layer(audio_signal)
        else:
            attn_weights = None
            ms_emb_seq = self.forward_encoder(
                audio_signal=audio_signal, 
                audio_signal_length=audio_signal_length,
                ms_seg_timestamps=ms_seg_timestamps,
                ms_seg_counts=ms_seg_counts,
                scale_mapping=scale_mapping,
            ) # (batch_size, max_seg_count, msdd_scale_n, emb_dim)
            # Step 2: Clustering for initialization
            # Compute the cosine similarity between the input and the cluster average embeddings
            if self.cfg_e2e_diarizer_model.get("multi_scale_method", None) == "mean":
                emb_seq = ms_emb_seq.mean(dim=2)
            elif self.cfg_e2e_diarizer_model.get("multi_scale_method", None) == "attention":
                emb_seq, attn_score_stack = self.sortformer_diarizer.apply_attention_weight(ms_emb_seq=ms_emb_seq)
                # raise NotImplementedError
            elif self.cfg_e2e_diarizer_model.get("multi_scale_method", None) == "only_interpolate":
                emb_seq = ms_emb_seq[:, :, -1, :] # Original shape: (batch_size, max_seg_count, scale_index, emb_dim)
            else:
                raise ValueError(f"Unknown multi-scale method: {self.cfg_e2e_diarizer_model.get('multi_scale_method', None)}")
            
        # Step 3: SortFormer Diarization Inference
        preds, _preds, attn_score_stack, preds_list, encoder_states_list = self.forward_infer(emb_seq)
        return preds, _preds, attn_score_stack, preds_list, encoder_states_list
    
    def find_first_nonzero(self, mat, max_cap_val=-1):
        # non zero values mask
        non_zero_mask = mat != 0

        # operations on the mask to find first nonzero values in the rows
        mask_max_values, mask_max_indices = torch.max(non_zero_mask, dim=1)

        # if the max-mask is zero, there is no nonzero value in the row
        mask_max_indices[mask_max_values == 0] = max_cap_val
        return mask_max_indices

    def sort_probs_and_labels(self, labels, discrete=True, thres=0.5):
        """
        Sorts probs and labels in descending order of signal_lengths.
        """
        max_cap_val = labels.shape[1] + 1 
        if not discrete:
            labels_discrete = torch.zeros_like(labels).to(labels.device)
            dropped_labels = labels.clone()
            dropped_labels[labels <= thres] = 0
            max_inds = torch.argmax(dropped_labels, dim=2)
            labels_discrete_flatten = dropped_labels.reshape(-1, labels.shape[-1])
            ax1 = torch.arange(labels_discrete_flatten.shape[0])
            labels_discrete_flatten[ax1, max_inds.reshape(-1)[ax1]] = 1
            labels_discrete = labels_discrete_flatten.reshape(labels.shape)
            labels_discrete[labels <= thres] = 0
        else:
            labels_discrete = labels
        label_fz = self.find_first_nonzero(labels_discrete, max_cap_val)
        label_fz[label_fz == -1] = max_cap_val 
        sorted_inds = torch.sort(label_fz)[1]
        sorted_labels = labels.transpose(0,1)[:, torch.arange(labels.shape[0]).unsqueeze(1), sorted_inds].transpose(0, 1)
        return sorted_labels 
        
    def sort_targets_with_preds(self, labels, preds, discrete=True, thres=0.5, add_pil_loss=False, pil_loss_thres=0.1):
        """
        Sorts probs and labels in descending order of signal_lengths.
        """
        max_cap_val = labels.shape[1] + 1 
        perm_size = self.spk_perm.shape[0] 
        permed_labels = labels[:, :, self.spk_perm]
        preds_rep = torch.unsqueeze(preds, 2).repeat(1,1, self.spk_perm.shape[0],1)
        match_score = torch.sum(permed_labels * preds_rep, axis=1).sum(axis=2)
        batch_best_perm = torch.argmax(match_score, axis=1)
        rep_spk_perm = self.spk_perm.repeat(batch_best_perm.shape[0],1) # (batch_size * perm_size, max_num_of_spks)
        global_inds_vec = torch.arange(0, perm_size*batch_best_perm.shape[0], perm_size).to(batch_best_perm.device) + batch_best_perm 
        batch_perm_inds = rep_spk_perm[global_inds_vec.to(rep_spk_perm.device), :] # (batch_size, max_num_of_spks)
        max_score_permed_labels = torch.vstack([ labels[k, :, batch_perm_inds[k]].unsqueeze(0) for k in range(batch_perm_inds.shape[0])]) 
        return max_score_permed_labels
    
    def compute_aux_f1(self, preds, targets):
        preds_bin = (preds > 0.5).to(torch.int64).detach()
        targets_ovl_mask = (targets.sum(dim=2) > 2)
        preds_vad_mask = (preds_bin.sum(dim=2) > 0)
        targets_vad_mask = (targets.sum(dim=2) > 0)
        preds_ovl = preds[targets_ovl_mask, :].unsqueeze(0)
        targets_ovl = targets[targets_ovl_mask, :].unsqueeze(0)
        preds_vad_mask_ = preds_vad_mask.int().unsqueeze(0)
        targets_vad_mask_ = targets_vad_mask.int().unsqueeze(0) 
        return preds_vad_mask_, preds_ovl, targets_vad_mask_, targets_ovl
    
    def _reset_train_f1_accs(self):
        self._accuracy_train.reset() 
        self._accuracy_train_vad.reset()
        self._accuracy_train_ovl.reset()

    def _is_pil_shift(self):
        """
        Check if the mean F1 score is above the threshold for pil shift.

        Returns:
            (bool): True if the mean F1 score is above the threshold for pil shift, False otherwise.
        """
        if len(self.train_f1_acc_history) >= self.train_f1_acc_window_length:
            mean_f1 = torch.mean(torch.tensor(self.train_f1_acc_history), dim=0)
            print(f"Mean F1 score: {mean_f1}, thres: {self.train_f1_acc_thres_pil_shift}")
            if mean_f1 > self.train_f1_acc_thres_pil_shift:
                return True
            else:
                return False
        else:
            return False
        
    def training_step(self, batch: list, batch_idx: int):
        start = time.time()
        if self.cfg_e2e_diarizer_model.use_mock_embs:
            audio_signal, audio_signal_length, targets = batch 
        else: # In this case, audio_signal is emb_seed
            audio_signal, audio_signal_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, ch_clus_mat, targets, global_spk_labels = batch
        
        batch_size = audio_signal.shape[0]
        ms_seg_counts = self.ms_seg_counts.unsqueeze(0).repeat(batch_size, 1).to(audio_signal.device)
        ms_seg_timestamps = self.ms_seg_timestamps.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(audio_signal.device)
        scale_mapping = self.scale_mapping.unsqueeze(0).repeat(batch_size, 1, 1) 
        
        sequence_lengths = torch.tensor([x[-1] for x in ms_seg_counts.detach()])
        self.validation_mode = False
        preds, _preds, attn_score_stack, preds_list, encoder_states_list = self.forward(
            audio_signal=audio_signal,
            audio_signal_length=audio_signal_length,
            ms_seg_timestamps=ms_seg_timestamps,
            ms_seg_counts=ms_seg_counts,
            scale_mapping=scale_mapping,
        )

        if self.loss.sorted_loss:
            # Perform arrival-time sorting (ATS)
            targets_ats = self.sort_probs_and_labels(targets.clone(), discrete=True)
            # `targets_pil` should not be used for training purpose.
            targets_pil = self.sort_targets_with_preds(targets.clone(), 
                                                       preds, 
                                                       discrete=True, 
                                                       add_pil_loss=self.cfg_e2e_diarizer_model.add_pil_loss, 
                                                       pil_loss_thres=self.cfg_e2e_diarizer_model.pil_loss_thres)
            if self.cfg_e2e_diarizer_model.get('use_pil_f1_score', True):
                targets_f1_score = targets_pil 
            else:
                targets_f1_score = targets_ats
                
            if self.cfg_e2e_diarizer_model.get('use_pil_train', False):
                targets_tr_loss = targets_pil 
            else:
                targets_tr_loss = targets_ats
        else:
            targets_f1_score = targets
            targets_tr_loss = targets
            
        if self._is_pil_shift():
            # print(f"PIL shift detectedl, mean f1 acc {torch.mean(torch.tensor(self.train_f1_acc_history), dim=0)}")
            targets_tr_loss = targets_pil 
            targets_f1_score = targets_pil  
                                          
        mid_layer_count = len(preds_list)
        if mid_layer_count > 0:
            torch.cat(preds_list).reshape(-1, *preds.shape)
            # All mid-layer outputs + final layer output
            preds_list.append(_preds)
            preds_all = torch.cat(preds_list)
            targets_rep = targets_tr_loss.repeat(mid_layer_count+1,1,1)
            sequence_lengths_rep = sequence_lengths.repeat(mid_layer_count+1)
            spk_loss = self.loss(probs=preds_all, labels=targets_rep, signal_lengths=sequence_lengths_rep)/(mid_layer_count+1)
        else:
            spk_loss = self.loss(probs=preds, labels=targets_tr_loss, signal_lengths=sequence_lengths)
            preds_mean = preds
        self._reset_train_f1_accs()
        preds_vad, preds_ovl, targets_vad, targets_ovl = self.compute_aux_f1(preds, targets_f1_score)
        self._accuracy_train_vad(preds_vad, targets_vad, sequence_lengths)
        self._accuracy_train_ovl(preds_ovl, targets_ovl, sequence_lengths)
        train_f1_vad = self._accuracy_train_vad.compute()
        train_f1_ovl = self._accuracy_train_ovl.compute()
        loss = spk_loss
        self._accuracy_train(preds, targets_f1_score, sequence_lengths)
        f1_acc = self._accuracy_train.compute()
        
        # Add F1 score to history
        if len(self.train_f1_acc_history) > self.train_f1_acc_window_length:
            del self.train_f1_acc_history[0]
        self.train_f1_acc_history.append(f1_acc.item())
        # print(f"self.train_f1_acc_history: {self.train_f1_acc_history}")
        # print(f'Train F1 score: {f1_acc:.4f}')
        self.log('loss', loss, sync_dist=True)
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'], sync_dist=True)
        self.log('train_f1_acc', f1_acc, sync_dist=True)
        self.log('train_f1_vad_acc', train_f1_vad, sync_dist=True)
        self.log('train_f1_ovl_acc', train_f1_ovl, sync_dist=True)
        self._accuracy_train.reset()
        return {'loss': loss}
    
    def _reset_valid_f1_accs(self):
        self._accuracy_valid.reset() 
        self._accuracy_valid_vad.reset()
        self._accuracy_valid_ovl.reset()
        self._accuracy_valid_toplyr.reset()
        self._accuracy_valid_prdmean.reset()
    
    def _reset_test_f1_accs(self):
        self._accuracy_valid.reset() 
        self._accuracy_test_vad.reset()
        self._accuracy_test_ovl.reset()
        self._accuracy_test_toplyr.reset()
        self._accuracy_test_prdmean.reset()
        
    def _cumulative_test_set_eval(self, score_dict: Dict[str, float], batch_idx: int, sample_count: int):
        if batch_idx == 0:
            self._reset_test_f1_accs()
            self.total_sample_counts = 0
            self.cumulative_f1_acc_sum = 0
            self.cumulative_f1_toplyr_acc_sum = 0
            self.cumulative_f1_prdmean_acc_sum = 0
            self.cumulative_f1_vad_acc_sum = 0
            self.cumulative_f1_ovl_acc_sum = 0
            
        self.total_sample_counts += sample_count
        self.cumulative_f1_acc_sum += score_dict['f1_acc'] * sample_count
        self.cumulative_f1_toplyr_acc_sum += score_dict['f1_toplyr_acc'] * sample_count
        self.cumulative_f1_prdmean_acc_sum += score_dict['f1_prdmean_acc'] * sample_count
        self.cumulative_f1_vad_acc_sum += score_dict['f1_vad_acc'] * sample_count
        self.cumulative_f1_ovl_acc_sum += score_dict['f1_ovl_acc'] * sample_count
        
        cumulative_f1_acc = self.cumulative_f1_acc_sum / self.total_sample_counts
        cumulative_f1_toplyr_acc = self.cumulative_f1_toplyr_acc_sum / self.total_sample_counts
        cumulative_f1_prdmean_acc = self.cumulative_f1_prdmean_acc_sum / self.total_sample_counts
        cumulative_f1_vad_acc = self.cumulative_f1_vad_acc_sum / self.total_sample_counts
        cumulative_f1_ovl_acc = self.cumulative_f1_ovl_acc_sum / self.total_sample_counts
        
        return {"cum_test_f1_acc": cumulative_f1_acc,
                "cum_test_f1_toplyr_acc": cumulative_f1_toplyr_acc,
                "cum_test_f1_prdmean_acc": cumulative_f1_prdmean_acc,
                "cum_test_f1_vad_acc": cumulative_f1_vad_acc,
                "cum_test_f1_ovl_acc": cumulative_f1_ovl_acc,
        }
        

    def validation_step(self, batch: list, batch_idx: int, dataloader_idx: int = 0):
        if self.cfg_e2e_diarizer_model.use_mock_embs:
            audio_signal, audio_signal_length, targets = batch 
        else: # In this case, audio_signal is emb_seed
            audio_signal, audio_signal_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, ch_clus_mat, targets, global_spk_labels = batch
        
        batch_size = audio_signal.shape[0]
        ms_seg_counts = self.ms_seg_counts.unsqueeze(0).repeat(batch_size, 1).to(audio_signal.device)
        ms_seg_timestamps = self.ms_seg_timestamps.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(audio_signal.device)
        scale_mapping = self.scale_mapping.unsqueeze(0).repeat(batch_size, 1, 1)
        sequence_lengths = torch.tensor([x[-1] for x in ms_seg_counts])
        self.validation_mode = True
        preds, _preds, attn_score_stack, preds_list, encoder_states_list = self.forward(
            audio_signal=audio_signal,
            audio_signal_length=audio_signal_length,
            ms_seg_timestamps=ms_seg_timestamps,
            ms_seg_counts=ms_seg_counts,
            scale_mapping=scale_mapping,
        )
        if self.loss.sorted_loss:
            targets_ats = self.sort_probs_and_labels(targets, discrete=True)
            targets_pil = self.sort_targets_with_preds(targets.clone(), 
                                                   preds, 
                                                   discrete=True, 
                                                   add_pil_loss=self.cfg_e2e_diarizer_model.add_pil_loss, 
                                                   pil_loss_thres=self.cfg_e2e_diarizer_model.pil_loss_thres)
            if self.loss.sorted_loss:
                # Perform arrival-time sorting (ATS)
                targets_ats = self.sort_probs_and_labels(targets.clone(), discrete=True)
                # `targets_pil` should not be used for training purpose.
                targets_pil = self.sort_targets_with_preds(targets.clone(), 
                                                            preds, 
                                                            discrete=True, 
                                                            add_pil_loss=self.cfg_e2e_diarizer_model.add_pil_loss, 
                                                            pil_loss_thres=self.cfg_e2e_diarizer_model.pil_loss_thres)
                if self.cfg_e2e_diarizer_model.get('use_pil_f1_score', True):
                    targets_f1_score = targets_pil 
                else:
                    targets_f1_score = targets_ats
                    
                if self.cfg_e2e_diarizer_model.get('use_pil_train', False):
                    targets_tr_loss = targets_pil 
                else:
                    targets_tr_loss = targets_ats
        else:
            targets_f1_score = targets
            targets_tr_loss = targets 
        
        if self._is_pil_shift():
            targets_f1_score = targets_pil  
            targets_tr_loss = targets_pil 
 
        # spk_loss = self.loss(probs=preds, labels=targets_tr_loss, signal_lengths=sequence_lengths)
        mid_layer_count = len(preds_list)
        if mid_layer_count > 0:
            # Only mid-layer outputs 
            preds_mid_all = torch.cat(preds_list).reshape(-1, *preds.shape)
            torch.cat(preds_list).reshape(-1, *preds.shape)
            preds_mean = preds_mid_all.mean(dim=0)
            # All mid-layer outputs + final layer output
            preds_list.append(_preds)
            preds_all = torch.cat(preds_list)
            # `targets_tr_loss` is the target tensor for calculating loss and backprop.
            targets_rep = targets_tr_loss.repeat(mid_layer_count+1,1,1)
            sequence_lengths_rep = sequence_lengths.repeat(mid_layer_count+1)
            loss = self.loss(probs=preds_all, labels=targets_rep, signal_lengths=sequence_lengths_rep)/(mid_layer_count+1)
        else:
            loss = self.loss(probs=preds, labels=targets_tr_loss, signal_lengths=sequence_lengths)  
            preds_mean = preds
        self._reset_valid_f1_accs()
        preds_vad, preds_ovl, targets_vad, targets_ovl = self.compute_aux_f1(preds, targets_f1_score)
        self._accuracy_valid_vad(preds_vad, targets_vad, sequence_lengths)
        valid_f1_vad = self._accuracy_valid_vad.compute()
        self._accuracy_valid_ovl(preds_ovl, targets_ovl, sequence_lengths)
        valid_f1_ovl = self._accuracy_valid_ovl.compute()
        self._accuracy_valid(preds, targets_f1_score, sequence_lengths)
        f1_acc = self._accuracy_valid.compute()
        self._accuracy_valid_toplyr.update(_preds, targets_f1_score, sequence_lengths)
        f1_acc_toplyr = self._accuracy_valid_toplyr.compute()
        self._accuracy_valid_prdmean.update(preds_mean, targets_f1_score, sequence_lengths)
        f1_acc_prdmean = self._accuracy_valid_prdmean.compute()

        self.log('val_loss', loss, sync_dist=True)
        self.log('val_f1_acc', f1_acc, sync_dist=True)
        self.log('val_f1_toplyr_acc', f1_acc_toplyr, sync_dist=True)
        self.log('val_f1_prdmean_acc', f1_acc_prdmean, sync_dist=True)
        self.log('val_f1_vad_acc', valid_f1_vad, sync_dist=True)
        self.log('val_f1_ovl_acc', valid_f1_ovl, sync_dist=True)
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
    def predict_step(self, batch: list, batch_idx: int, dataloader_idx: int = 0):
        # for batch in tqdm(self._test_dl): 
        if self.cfg_e2e_diarizer_model.use_mock_embs:
            audio_signal, audio_signal_length, targets = batch 
        else: # In this case, audio_signal is emb_seed
            audio_signal, audio_signal_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, ch_clus_mat, targets, global_spk_labels = batch
        
        batch_size = audio_signal.shape[0]
        ms_seg_counts = self.ms_seg_counts.unsqueeze(0).repeat(batch_size, 1).to(audio_signal.device)
        ms_seg_timestamps = self.ms_seg_timestamps.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(audio_signal.device)
        scale_mapping = self.scale_mapping.unsqueeze(0).repeat(batch_size, 1, 1)
        sequence_lengths = torch.tensor([x[-1] for x in ms_seg_counts])
        self.validation_mode = True
        preds, _preds, attn_score_stack, preds_list, encoder_states_list = self.forward(
            audio_signal=audio_signal,
            audio_signal_length=audio_signal_length,
            ms_seg_timestamps=ms_seg_timestamps,
            ms_seg_counts=ms_seg_counts,
            scale_mapping=scale_mapping,
        )
        if self.loss.sorted_loss:
            targets_sort_order = self.sort_probs_and_labels(targets, discrete=True)
            targets = self.sort_targets_with_preds(targets_sort_order, 
                                                   preds, 
                                                   discrete=True, 
                                                   add_pil_loss=self.cfg_e2e_diarizer_model.add_pil_loss, 
                                                   pil_loss_thres=self.cfg_e2e_diarizer_model.pil_loss_thres)
        spk_loss = self.loss(probs=preds, labels=targets, signal_lengths=sequence_lengths)
        mid_layer_count = len(preds_list)
        if mid_layer_count > 0:
            # Only mid-layer outputs 
            preds_mid_all = torch.cat(preds_list).reshape(-1, *preds.shape)
            torch.cat(preds_list).reshape(-1, *preds.shape)
            preds_mean = preds_mid_all.mean(dim=0)
            # All mid-layer outputs + final layer output
            preds_list.append(_preds)
            preds_all = torch.cat(preds_list)
            targets_rep = targets.repeat(mid_layer_count+1,1,1)
            sequence_lengths_rep = sequence_lengths.repeat(mid_layer_count+1)
            loss = self.loss(probs=preds_all, labels=targets_rep, signal_lengths=sequence_lengths_rep)/(mid_layer_count+1)
        else:
            loss = self.loss(probs=preds, labels=targets, signal_lengths=sequence_lengths)  
            preds_mean = preds
        # self._reset_test_f1_accs()
        preds_vad, preds_ovl, targets_vad, targets_ovl = self.compute_aux_f1(preds, targets)
        self._accuracy_test_vad(preds_vad, targets_vad, sequence_lengths)
        test_f1_vad = self._accuracy_test_vad.compute()
        self._accuracy_test_ovl(preds_ovl, targets_ovl, sequence_lengths)
        test_f1_ovl = self._accuracy_test_ovl.compute()
        self._accuracy_test(preds, targets, sequence_lengths)
        f1_acc = self._accuracy_test.compute()
        self._accuracy_test_toplyr(_preds, targets, sequence_lengths)
        f1_acc_toplyr = self._accuracy_test_toplyr.compute()
        self._accuracy_test_prdmean(preds_mean, targets, sequence_lengths)
        f1_acc_prdmean = self._accuracy_test_prdmean.compute()
        return preds_all 
   
    def test_step(self, batch: list, batch_idx: int, dataloader_idx: int = 0):
        if self.cfg_e2e_diarizer_model.use_mock_embs:
            audio_signal, audio_signal_length, targets = batch 
        else: # In this case, audio_signal is emb_seed
            audio_signal, audio_signal_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, ch_clus_mat, targets, global_spk_labels = batch
        
        batch_size = audio_signal.shape[0]
        ms_seg_counts = self.ms_seg_counts.unsqueeze(0).repeat(batch_size, 1).to(audio_signal.device)
        ms_seg_timestamps = self.ms_seg_timestamps.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(audio_signal.device)
        scale_mapping = self.scale_mapping.unsqueeze(0).repeat(batch_size, 1, 1)
        sequence_lengths = torch.tensor([x[-1] for x in ms_seg_counts])
        preds, _preds, attn_score_stack, preds_list, encoder_states_list = self.forward(
            audio_signal=audio_signal,
            audio_signal_length=audio_signal_length,
            ms_seg_timestamps=ms_seg_timestamps,
            ms_seg_counts=ms_seg_counts,
            scale_mapping=scale_mapping,
        )
        mid_layer_count = len(preds_list)
        if self.loss.sorted_loss:
                # Perform arrival-time sorting (ATS)
                targets_ats = self.sort_probs_and_labels(targets.clone(), discrete=True)
                # `targets_pil` should not be used for training purpose.
                targets_pil = self.sort_targets_with_preds(targets.clone(), 
                                                            preds, 
                                                            discrete=True, 
                                                            add_pil_loss=self.cfg_e2e_diarizer_model.get('add_pil_loss', True),
                                                            pil_loss_thres=self.cfg_e2e_diarizer_model.get('pil_loss_thres', 0.0)
                )
                if self.cfg_e2e_diarizer_model.get('use_pil_f1_score', True):
                    targets_f1_score = targets_pil 
                else:
                    targets_f1_score = targets_ats
                    
                if self.cfg_e2e_diarizer_model.get('use_pil_train', False):
                    targets_tr_loss = targets_pil 
                else:
                    targets_tr_loss = targets_ats
                    
        if mid_layer_count > 0:
            # Only mid-layer outputs 
            preds_mid_all = torch.cat(preds_list).reshape(-1, *preds.shape)
            torch.cat(preds_list).reshape(-1, *preds.shape)
            preds_mean = preds_mid_all.mean(dim=0)
            # All mid-layer outputs + final layer output
            preds_list.append(_preds)
            preds_all = torch.cat(preds_list)
            targets_rep = targets_tr_loss.repeat(mid_layer_count+1,1,1)
            sequence_lengths_rep = sequence_lengths.repeat(mid_layer_count+1)
        else:
            preds_mean = preds
        # import ipdb; ipdb.set_trace()
        preds_vad, preds_ovl, targets_vad, targets_ovl = self.compute_aux_f1(preds, targets_f1_score)
        self._accuracy_test_vad(preds_vad, targets_vad, sequence_lengths, cumulative=True)
        test_f1_vad = self._accuracy_test_vad.compute()
        self._accuracy_test_ovl(preds_ovl, targets_ovl, sequence_lengths, cumulative=True)
        test_f1_ovl = self._accuracy_test_ovl.compute()
        self._accuracy_test(preds, targets_f1_score, sequence_lengths, cumulative=True)
        f1_acc = self._accuracy_test.compute()
        self._accuracy_test_toplyr(_preds, targets_f1_score, sequence_lengths, cumulative=True)
        f1_acc_toplyr = self._accuracy_test_toplyr.compute()
        self._accuracy_test_prdmean(preds_mean, targets_f1_score, sequence_lengths, cumulative=True)
        f1_acc_prdmean = self._accuracy_test_prdmean.compute()
        # if self.cfg_e2e_diarizer_model.get('save_tensor_images', False):
        if True:
            tags = f"f1acc{f1_acc:.4f}_bidx{batch_idx}"
            tags = tags.replace('0.', '0p')
            print(f"Saving tensor images with tags: {tags}")
            # directory = '/home/taejinp/projects/sortformer_script/tensor_image/'
            # directory = '/home/taejinp/projects/sortformer_script/tensor_image_v2/'
            # directory = '/home/taejinp/projects/sortformer_script/tensor_image_model_v501/'
            # directory_ex = '/home/taejinp/projects/sortformer_script/tensor_image_ch109/'
            # directory_ex = None
            # directory = self.cfg_e2e_diarizer_model.get('tensor_image_dir', directory_ex) 
            directory = self._cfg.diarizer.get('out_dir', None)
            if directory is None:
                raise ValueError(f"No output directory specified for tensor image saving. Please set the `out_dir` in the config file.")
            else:
                print(f"Saving tensor images to directory: {directory}")
            torch.save(preds, f'{directory}/preds_{tags}.pt')
            torch.save(targets_f1_score, f'{directory}/targets_{tags}.pt')
        self.max_f1_acc = max(self.max_f1_acc, f1_acc)
        batch_score_dict = {"f1_acc": f1_acc, "f1_toplyr_acc": f1_acc_toplyr, "f1_prdmean_acc": f1_acc_prdmean, "f1_vad_acc": test_f1_vad, "f1_ovl_acc": test_f1_ovl}
        cum_score_dict = self._cumulative_test_set_eval(score_dict=batch_score_dict, batch_idx=batch_idx, sample_count=len(sequence_lengths))
        print(cum_score_dict)
        return preds_all
    
    def test_batch(self,):
        for batch in tqdm(self._test_dl): 
            if self.cfg_e2e_diarizer_model.use_mock_embs:
                audio_signal, audio_signal_length, targets = batch 
            else: # In this case, audio_signal is emb_seed
                audio_signal, audio_signal_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, ch_clus_mat, targets, global_spk_labels = batch
            batch_size = audio_signal.shape[0]
            ms_seg_counts = self.ms_seg_counts.unsqueeze(0).repeat(batch_size, 1).to(audio_signal.device)
            ms_seg_timestamps = self.ms_seg_timestamps.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(audio_signal.device)
            scale_mapping = self.scale_mapping.unsqueeze(0).repeat(batch_size, 1, 1)
            sequence_lengths = torch.tensor([x[-1] for x in ms_seg_counts])
            self.validation_mode = True
            preds, _preds, attn_score_stack, preds_list, encoder_states_list = self.forward(
                audio_signal=audio_signal,
                audio_signal_length=audio_signal_length,
                ms_seg_timestamps=ms_seg_timestamps,
                ms_seg_counts=ms_seg_counts,
                scale_mapping=scale_mapping,
            )
            if self.loss.sorted_loss:
                # Perform arrival-time sorting (ATS)
                targets_ats = self.sort_probs_and_labels(targets.clone(), discrete=True)
                # `targets_pil` should not be used for training purpose.
                targets_pil = self.sort_targets_with_preds(targets.clone(), 
                                                            preds, 
                                                            discrete=True, 
                                                            add_pil_loss=self.cfg_e2e_diarizer_model.add_pil_loss, 
                                                            pil_loss_thres=self.cfg_e2e_diarizer_model.pil_loss_thres)
                if self.cfg_e2e_diarizer_model.get('use_pil_f1_score', True):
                    targets_f1_score = targets_pil 
                else:
                    targets_f1_score = targets_ats
                    
                if self.cfg_e2e_diarizer_model.get('use_pil_train', False):
                    targets_tr_loss = targets_pil 
                else:
                    targets_tr_loss = targets_ats
            # spk_loss = self.loss(probs=preds, labels=targets, signal_lengths=sequence_lengths)
            mid_layer_count = len(preds_list)
            if mid_layer_count > 0:
                # Only mid-layer outputs 
                preds_mid_all = torch.cat(preds_list).reshape(-1, *preds.shape)
                torch.cat(preds_list).reshape(-1, *preds.shape)
                preds_mean = preds_mid_all.mean(dim=0)
                # All mid-layer outputs + final layer output
                preds_list.append(_preds)
                preds_all = torch.cat(preds_list)
                targets_rep = targets_tr_loss.repeat(mid_layer_count+1,1,1)
                sequence_lengths_rep = sequence_lengths.repeat(mid_layer_count+1)
                loss = self.loss(probs=preds_all, labels=targets_rep, signal_lengths=sequence_lengths_rep)/(mid_layer_count+1)
            else:
                loss = self.loss(probs=preds, labels=targets_tr_loss, signal_lengths=sequence_lengths)  
                preds_mean = preds
            # self._reset_test_f1_accs()
            preds_vad, preds_ovl, targets_vad, targets_ovl = self.compute_aux_f1(preds, targets_f1_score)
            self._accuracy_test_vad(preds_vad, targets_vad, sequence_lengths)
            test_f1_vad = self._accuracy_test_vad.compute()
            self._accuracy_test_ovl(preds_ovl, targets_ovl, sequence_lengths)
            test_f1_ovl = self._accuracy_test_ovl.compute()
            self._accuracy_valid(preds, targets_f1_score, sequence_lengths)
            f1_acc = self._accuracy_valid.compute()
            self._accuracy_test_toplyr(_preds, targets_f1_score, sequence_lengths)
            f1_acc_toplyr = self._accuracy_test_toplyr.compute()
            self._accuracy_test_prdmean(preds_mean, targets_f1_score, sequence_lengths)
            f1_acc_prdmean = self._accuracy_test_prdmean.compute()

        
    def diarize(self,):
        pass

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

