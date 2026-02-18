# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import math
import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text import _AudioTextDataset
from nemo.collections.asr.data.audio_to_text_dali import AudioToBPEDALIDataset, DALIOutputs
from nemo.collections.asr.data.audio_to_text_lhotse import LhotseSpeechToTextBpeDataset
from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
from nemo.collections.asr.parts.mixins import ASRBPEMixin
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecoding, RNNTBPEDecodingConfig
from nemo.collections.asr.parts.utils.asr_batching import get_semi_sorted_batch_sampler
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.mixins import AccessMixin
from nemo.core.neural_types import AcousticEncodedRepresentation, AudioSignal, LengthsType, NeuralType, SpectrogramType
from nemo.utils import logging, model_utils


class EncDecRNNTBPEModel(EncDecRNNTModel, ASRBPEMixin):
    """Base class for encoder decoder RNNT-based models with subword tokenization."""

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_contextnet_256",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_contextnet_256",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_contextnet_256/versions/1.6.0/files/stt_en_contextnet_256.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_contextnet_512",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_contextnet_512",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_contextnet_512/versions/1.6.0/files/stt_en_contextnet_512.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_contextnet_1024",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_contextnet_1024",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_contextnet_1024/versions/1.9.0/files/stt_en_contextnet_1024.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_contextnet_256_mls",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_contextnet_256_mls",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_contextnet_256_mls/versions/1.0.0/files/stt_en_contextnet_256_mls.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_contextnet_512_mls",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_contextnet_512_mls",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_contextnet_512_mls/versions/1.0.0/files/stt_en_contextnet_512_mls.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_contextnet_1024_mls",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_contextnet_1024_mls",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_contextnet_1024_mls/versions/1.0.0/files/stt_en_contextnet_1024_mls.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_transducer_small",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_transducer_small",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_transducer_small/versions/1.6.0/files/stt_en_conformer_transducer_small.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_transducer_medium",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_transducer_medium",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_transducer_medium/versions/1.6.0/files/stt_en_conformer_transducer_medium.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_transducer_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_transducer_large/versions/1.10.0/files/stt_en_conformer_transducer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_transducer_large_ls",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_transducer_large_ls",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_transducer_large_ls/versions/1.8.0/files/stt_en_conformer_transducer_large_ls.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_transducer_xlarge",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_transducer_xlarge",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_transducer_xlarge/versions/1.10.0/files/stt_en_conformer_transducer_xlarge.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_transducer_xxlarge",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_transducer_xxlarge",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_transducer_xxlarge/versions/1.8.0/files/stt_en_conformer_transducer_xxlarge.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_de_contextnet_1024",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_de_contextnet_1024",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_de_contextnet_1024/versions/1.4.0/files/stt_de_contextnet_1024.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_fr_contextnet_1024",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_fr_contextnet_1024",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_fr_contextnet_1024/versions/1.5/files/stt_fr_contextnet_1024.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_es_contextnet_1024",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_es_contextnet_1024",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_es_contextnet_1024/versions/1.8.0/files/stt_es_contextnet_1024.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_de_conformer_transducer_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_de_conformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_de_conformer_transducer_large/versions/1.5.0/files/stt_de_conformer_transducer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_fr_conformer_transducer_large",
            description="For details about this model, please visit https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_fr_conformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_fr_conformer_transducer_large/versions/1.5/files/stt_fr_conformer_transducer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_es_conformer_transducer_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_es_conformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_es_conformer_transducer_large/versions/1.8.0/files/stt_es_conformer_transducer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_enes_conformer_transducer_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_enes_conformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_enes_conformer_transducer_large/versions/1.0.0/files/stt_enes_conformer_transducer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_enes_contextnet_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_enes_contextnet_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_enes_contextnet_large/versions/1.0.0/files/stt_enes_contextnet_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_ca_conformer_transducer_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_ca_conformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_ca_conformer_transducer_large/versions/1.11.0/files/stt_ca_conformer_transducer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_rw_conformer_transducer_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_rw_conformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_rw_conformer_transducer_large/versions/1.11.0/files/stt_rw_conformer_transducer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_enes_conformer_transducer_large_codesw",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_enes_conformer_transducer_large_codesw",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_enes_conformer_transducer_large_codesw/versions/1.0.0/files/stt_enes_conformer_transducer_large_codesw.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_kab_conformer_transducer_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_kab_conformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_kab_conformer_transducer_large/versions/1.12.0/files/stt_kab_conformer_transducer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_be_conformer_transducer_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_be_conformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_be_conformer_transducer_large/versions/1.12.0/files/stt_be_conformer_transducer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_hr_conformer_transducer_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_hr_conformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_hr_conformer_transducer_large/versions/1.11.0/files/stt_hr_conformer_transducer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_it_conformer_transducer_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_it_conformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_it_conformer_transducer_large/versions/1.13.0/files/stt_it_conformer_transducer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_ru_conformer_transducer_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_ru_conformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_ru_conformer_transducer_large/versions/1.13.0/files/stt_ru_conformer_transducer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_eo_conformer_transducer_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_eo_conformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_eo_conformer_transducer_large/versions/1.14.0/files/stt_eo_conformer_transducer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_fastconformer_transducer_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_transducer_large/versions/1.0.0/files/stt_en_fastconformer_transducer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_fastconformer_transducer_large_ls",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_transducer_large_ls",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_transducer_large_ls/versions/1.0.0/files/stt_en_fastconformer_transducer_large_ls.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_fastconformer_transducer_xlarge",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_transducer_xlarge",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_transducer_xlarge/versions/1.20.1/files/stt_en_fastconformer_transducer_xlarge.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_fastconformer_transducer_xxlarge",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_transducer_xxlarge",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_transducer_xxlarge/versions/1.20.1/files/stt_en_fastconformer_transducer_xxlarge.nemo",
        )
        results.append(model)

        return results

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        # Tokenizer is necessary for this model
        if 'tokenizer' not in cfg:
            raise ValueError("`cfg` must have `tokenizer` config to create a tokenizer !")

        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)

        # Setup the tokenizer
        self._setup_tokenizer(cfg.tokenizer)

        # Initialize a dummy vocabulary
        vocabulary = self.tokenizer.tokenizer.get_vocab()

        # Set the new vocabulary
        with open_dict(cfg):
            cfg.labels = ListConfig(list(vocabulary))

        with open_dict(cfg.decoder):
            cfg.decoder.vocab_size = len(vocabulary)

        with open_dict(cfg.joint):
            cfg.joint.num_classes = len(vocabulary)
            cfg.joint.vocabulary = ListConfig(list(vocabulary))
            cfg.joint.jointnet.encoder_hidden = cfg.model_defaults.enc_hidden
            cfg.joint.jointnet.pred_hidden = cfg.model_defaults.pred_hidden

        super().__init__(cfg=cfg, trainer=trainer)

        self.cfg.decoding = self.set_decoding_type_according_to_loss(self.cfg.decoding)
        # Setup decoding object
        self.decoding = RNNTBPEDecoding(
            decoding_cfg=self.cfg.decoding,
            decoder=self.decoder,
            joint=self.joint,
            tokenizer=self.tokenizer,
        )

        # Setup wer object
        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=0,
            use_cer=self._cfg.get('use_cer', False),
            log_prediction=self._cfg.get('log_prediction', True),
            dist_sync_on_step=True,
        )

        # Setup fused Joint step if flag is set
        if self.joint.fuse_loss_wer:
            self.joint.set_loss(self.loss)
            self.joint.set_wer(self.wer)

    def change_vocabulary(
        self,
        new_tokenizer_dir: Union[str, DictConfig],
        new_tokenizer_type: str,
        decoding_cfg: Optional[DictConfig] = None,
    ):
        """
        Changes vocabulary used during RNNT decoding process. Use this method when fine-tuning
        on from pre-trained model. This method changes only decoder and leaves encoder and pre-processing
        modules unchanged. For example, you would use it if you want to use pretrained encoder when fine-tuning
        on data in another language, or when you'd need model to learn capitalization, punctuation
        and/or special characters.

        Args:
            new_tokenizer_dir: Directory path to tokenizer or a config for a new tokenizer
                (if the tokenizer type is `agg`)
            new_tokenizer_type: Type of tokenizer. Can be either `agg`, `bpe` or `wpe`.
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.

        Returns: None

        """
        if isinstance(new_tokenizer_dir, DictConfig):
            if new_tokenizer_type == 'agg':
                new_tokenizer_cfg = new_tokenizer_dir
            else:
                raise ValueError(
                    f'New tokenizer dir should be a string unless the tokenizer is `agg`, but this tokenizer \
                        type is: {new_tokenizer_type}'
                )
        else:
            new_tokenizer_cfg = None

        if new_tokenizer_cfg is not None:
            tokenizer_cfg = new_tokenizer_cfg
        else:
            if not os.path.isdir(new_tokenizer_dir):
                raise NotADirectoryError(
                    f'New tokenizer dir must be non-empty path to a directory. But I got: {new_tokenizer_dir}'
                )

            if new_tokenizer_type.lower() not in ('bpe', 'wpe'):
                raise ValueError(f'New tokenizer type must be either `bpe` or `wpe`, got {new_tokenizer_type}')

            tokenizer_cfg = OmegaConf.create({'dir': new_tokenizer_dir, 'type': new_tokenizer_type})

        # Setup the tokenizer
        self._setup_tokenizer(tokenizer_cfg)

        # Initialize a dummy vocabulary
        vocabulary = self.tokenizer.tokenizer.get_vocab()

        joint_config = self.joint.to_config_dict()
        new_joint_config = copy.deepcopy(joint_config)
        if self.tokenizer_type == "agg":
            new_joint_config["vocabulary"] = ListConfig(vocabulary)
        else:
            new_joint_config["vocabulary"] = ListConfig(list(vocabulary.keys()))

        new_joint_config['num_classes'] = len(vocabulary)
        del self.joint
        self.joint = EncDecRNNTBPEModel.from_config_dict(new_joint_config)

        decoder_config = self.decoder.to_config_dict()
        new_decoder_config = copy.deepcopy(decoder_config)
        new_decoder_config.vocab_size = len(vocabulary)
        del self.decoder
        self.decoder = EncDecRNNTBPEModel.from_config_dict(new_decoder_config)

        del self.loss
        self.loss = RNNTLoss(num_classes=self.joint.num_classes_with_blank - 1)

        if decoding_cfg is None:
            # Assume same decoding config as before
            decoding_cfg = self.cfg.decoding

        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(RNNTBPEDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)
        decoding_cfg = self.set_decoding_type_according_to_loss(decoding_cfg)

        self.decoding = RNNTBPEDecoding(
            decoding_cfg=decoding_cfg,
            decoder=self.decoder,
            joint=self.joint,
            tokenizer=self.tokenizer,
        )

        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=self.wer.batch_dim_index,
            use_cer=self.wer.use_cer,
            log_prediction=self.wer.log_prediction,
            dist_sync_on_step=True,
        )

        # Setup fused Joint step
        if self.joint.fuse_loss_wer or (
            self.decoding.joint_fused_batch_size is not None and self.decoding.joint_fused_batch_size > 0
        ):
            self.joint.set_loss(self.loss)
            self.joint.set_wer(self.wer)

        # Update config
        with open_dict(self.cfg.joint):
            self.cfg.joint = new_joint_config

        with open_dict(self.cfg.decoder):
            self.cfg.decoder = new_decoder_config

        with open_dict(self.cfg.decoding):
            self.cfg.decoding = decoding_cfg

        logging.info(f"Changed decoder to output to {self.joint.vocabulary} vocabulary.")

    def change_decoding_strategy(self, decoding_cfg: DictConfig, verbose: bool = True):
        """
        Changes decoding strategy used during RNNT decoding process.

        Args:
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
            verbose: A flag to enable/disable logging.
        """
        if decoding_cfg is None:
            # Assume same decoding config as before
            logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
            decoding_cfg = self.cfg.decoding

        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(RNNTBPEDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)
        decoding_cfg = self.set_decoding_type_according_to_loss(decoding_cfg)

        self.decoding = RNNTBPEDecoding(
            decoding_cfg=decoding_cfg,
            decoder=self.decoder,
            joint=self.joint,
            tokenizer=self.tokenizer,
        )

        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=self.wer.batch_dim_index,
            use_cer=self.wer.use_cer,
            log_prediction=self.wer.log_prediction,
            dist_sync_on_step=True,
        )

        # Setup fused Joint step
        if self.joint.fuse_loss_wer or (
            self.decoding.joint_fused_batch_size is not None and self.decoding.joint_fused_batch_size > 0
        ):
            self.joint.set_loss(self.loss)
            self.joint.set_wer(self.wer)

        self.joint.temperature = decoding_cfg.get('temperature', 1.0)

        # Update config
        with open_dict(self.cfg.decoding):
            self.cfg.decoding = decoding_cfg

        if verbose:
            logging.info(f"Changed decoding strategy to \n{OmegaConf.to_yaml(self.cfg.decoding)}")

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if config.get("use_lhotse"):
            return get_lhotse_dataloader_from_config(
                config,
                # During transcription, the model is initially loaded on the CPU.
                # To ensure the correct global_rank and world_size are set,
                # these values must be passed from the configuration.
                global_rank=self.global_rank if not config.get("do_transcribe", False) else config.get("global_rank"),
                world_size=self.world_size if not config.get("do_transcribe", False) else config.get("world_size"),
                dataset=LhotseSpeechToTextBpeDataset(
                    tokenizer=self.tokenizer,
                    return_cuts=config.get("do_transcribe", False),
                ),
                tokenizer=self.tokenizer,
            )

        dataset = audio_to_text_dataset.get_audio_to_text_bpe_dataset_from_config(
            config=config,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            world_size=self.world_size,
            tokenizer=self.tokenizer,
            preprocessor_cfg=self.cfg.get("preprocessor", None),
        )

        if dataset is None:
            return None

        if isinstance(dataset, AudioToBPEDALIDataset):
            # DALI Dataset implements dataloader interface
            return dataset

        shuffle = config['shuffle']
        if isinstance(dataset, torch.utils.data.IterableDataset):
            shuffle = False

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset.datasets[0], 'collate_fn'):
            # support datasets that are lists of entries
            collate_fn = dataset.datasets[0].collate_fn
        else:
            # support datasets that are lists of lists
            collate_fn = dataset.datasets[0].datasets[0].collate_fn

        batch_sampler = None
        if config.get('use_semi_sorted_batching', False):
            if not isinstance(dataset, _AudioTextDataset):
                raise RuntimeError(
                    "Semi Sorted Batch sampler can be used with AudioToCharDataset or AudioToBPEDataset "
                    f"but found dataset of type {type(dataset)}"
                )
            # set batch_size and batch_sampler to None to disable automatic batching
            batch_sampler = get_semi_sorted_batch_sampler(self, dataset, config)
            config['batch_size'] = None
            config['drop_last'] = False
            shuffle = False

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            sampler=batch_sampler,
            batch_sampler=None,
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:
            paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
                Recommended length per file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the audio manifest is temporarily
                stored.

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        if 'manifest_filepath' in config:
            manifest_filepath = config['manifest_filepath']
            batch_size = config['batch_size']
        else:
            manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
            batch_size = min(config['batch_size'], len(config['paths2audio_files']))

        dl_config = {
            'use_lhotse': config.get('use_lhotse', True),
            'manifest_filepath': manifest_filepath,
            'sample_rate': self.preprocessor._sample_rate,
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
            'channel_selector': config.get('channel_selector', None),
            'use_start_end_token': self.cfg.validation_ds.get('use_start_end_token', False),
        }

        if config.get("augmentor"):
            dl_config['augmentor'] = config.get("augmentor")

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer


class MSEncDecRNNTBPEModel(EncDecRNNTBPEModel):
    """
    A Multi-Speaker RNNT BPE model that fuses Sortformer diarization output
    with ASR encoder output before feeding into the RNNT decoder/joint.

    The model extends EncDecRNNTBPEModel by optionally loading a pretrained
    Sortformer diarization model and fusing its frame-level speaker predictions
    with the ASR encoder output. The fusion can be done via:
      - 'projection': concatenation + linear projection (default)
      - 'sinusoidal': sinusoidal position encoding weighted by speaker posteriors
      - 'metacat': outer-product based fusion + linear projection
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        # Setup LSS loss (latent speaker supervision), auxiliary optional loss
        if self.cfg.get('lss_loss', None) is not None:
            with open_dict(self.cfg.lss_loss):
                self.cfg.lss_loss.pad_id = self.tokenizer.pad_id
                self.cfg.lss_loss.is_rnnt = True
                if self.cfg.lss_loss.get('speaker_token_ids', None) is None:
                    self.cfg.lss_loss.speaker_token_ids = [
                        self.tokenizer.token_to_id(f"<|spltoken{i}|>")
                        for i in range(self.cfg.get('max_num_speakers', 4))
                    ]
            self.lss_loss = MSEncDecRNNTBPEModel.from_config_dict(self.cfg.lss_loss)
        else:
            self.lss_loss = None

        # Optionally load pretrained ASR encoder weights
        if 'asr_model_path' in self.cfg and self.cfg.asr_model_path is not None:
            self._init_asr_model()

        if 'diar_model_path' in self.cfg and self.cfg.diar_model_path is not None:
            self.diar = True
            # Initialize the Sortformer diarization model
            self._init_diar_model()

            if 'max_num_speakers' in cfg and cfg.max_num_speakers not in (None, ''):
                self.max_num_speakers = int(cfg.max_num_speakers)
            else:
                self.max_num_speakers = 4

            # Layer normalization: 'ln', 'l2', or None
            if 'norm' in cfg:
                if cfg.norm == 'ln':
                    self.asr_norm = torch.nn.LayerNorm(cfg.model_defaults.enc_hidden)
                    self.diar_norm = torch.nn.LayerNorm(self.max_num_speakers)
                self.norm = cfg.norm
            else:
                self.norm = None

            if 'kernel_norm' in cfg:
                self.kernel_norm = cfg.kernel_norm
            else:
                self.kernel_norm = None

            # Projection layer: diar_preds (num_speakers) + asr_enc (enc_hidden) -> enc_hidden
            proj_in_size = self.max_num_speakers + cfg.model_defaults.enc_hidden
            proj_out_size = cfg.model_defaults.enc_hidden
            self.diar_joint_proj = torch.nn.Sequential(
                torch.nn.Linear(proj_in_size, proj_out_size * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(proj_out_size * 2, proj_out_size),
            )

            # Segment length and shift (for potential speaker embedding segmentation)
            if 'segment_length' in cfg:
                self.segment_length = cfg.segment_length
            else:
                self.segment_length = 16
            if 'segment_shift' in cfg:
                self.segment_shift = cfg.segment_shift
            else:
                self.segment_shift = 8

            if 'diar_kernel_type' in cfg:
                if cfg.diar_kernel_type == 'sinusoidal':
                    self.diar_kernel_type = cfg.diar_kernel_type
                    self.diar_kernel = self.get_sinusoid_position_encoding(
                        self.max_num_speakers, cfg.model_defaults.enc_hidden
                    )
                elif cfg.diar_kernel_type == 'metacat':
                    self.diar_kernel_type = cfg.diar_kernel_type
                    # Outer-product projection layer
                    metacat_proj_in_size = self.max_num_speakers * cfg.model_defaults.enc_hidden
                    metacat_proj_out_size = cfg.model_defaults.enc_hidden
                    self.metacat_diar_joint_proj = torch.nn.Sequential(
                        torch.nn.Linear(metacat_proj_in_size, metacat_proj_out_size * 2),
                        torch.nn.ReLU(),
                        torch.nn.Linear(metacat_proj_out_size * 2, metacat_proj_out_size),
                    )
                    self.diar_kernel = self.metacat_diar_joint_proj
            else:
                self.diar_kernel_type = 'projection'
                self.diar_kernel = self.diar_joint_proj

            # FastConformer layer fusion: inject diar encoder per-layer
            # hidden states into ASR encoder MHA cache at each streaming step.
            self.use_fc_layer_fusion = cfg.get('use_fc_layer_fusion', False)
            if self.use_fc_layer_fusion:
                self._setup_fc_layer_fusion()

            # Pre-encode level diarization fusion: fuse speaker information
            # into ASR pre-encode embeddings before feeding into encoder layers.
            # Both methods are applied in serial:
            #   1. 'sinusoidal' — speaker position encoding weighted by diar preds
            #   2. 'merge' — project + add diar pre-encode to ASR pre-encode
            # The fused embeddings are fed to the encoder with
            # bypass_pre_encode=True so speaker context propagates through
            # all encoder layers and into the MHA cache for future steps.
            self.use_pre_encode_diar_fusion = cfg.get('use_pre_encode_diar_fusion', False)
            if self.use_pre_encode_diar_fusion:
                self._setup_pre_encode_diar_fusion()
                logging.info("[MSEncDecRNNTBPEModel] use_pre_encode_diar_fusion=True")
            else:
                logging.info("[MSEncDecRNNTBPEModel] use_pre_encode_diar_fusion=False")

        else:
            self.diar = False
            self.use_fc_layer_fusion = False
            self.use_pre_encode_diar_fusion = False

    def _init_diar_model(self):
        """
        Initialize the Sortformer diarization model from a pretrained checkpoint.
        """
        # Import here to avoid circular import
        from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel

        model_path = self.cfg.diar_model_path

        if model_path.endswith('.nemo'):
            pretrained_diar_model = SortformerEncLabelModel.restore_from(model_path, map_location="cpu")
            logging.info("Diarization Model restored locally from {}".format(model_path))
        elif model_path.endswith('.ckpt'):
            pretrained_diar_model = SortformerEncLabelModel.load_from_checkpoint(model_path, map_location="cpu")
            logging.info("Diarization Model restored locally from {}".format(model_path))
        else:
            pretrained_diar_model = None
            logging.info("Diarization model path incorrect: {}".format(model_path))

        self.diarization_model = pretrained_diar_model

        if self.cfg.freeze_diar:
            self.diarization_model.eval()

        # Configure diar streaming params if available
        if self.cfg.get('diar_streaming', None) is not None:
            self._configure_diar_streaming()

    def _configure_diar_streaming(self):
        """
        Configure the Sortformer diarization model for streaming mode.

        Reads streaming parameters from ``self.cfg.diar_streaming`` and sets
        them on ``diarization_model.sortformer_modules``, exactly matching the
        inference-time setup in
        speech_to_text_cache_aware_streaming_sot_mtasr_infer.py (lines 531-544).

        The parameters control how the streaming diarizer maintains its speaker
        cache (spkcache), FIFO buffer, and chunk boundaries.  They must be
        synced with the ASR encoder's streaming context:

            SPKCACHE_LEN + FIFO_LEN == LEFT_CONTEXT  (e.g. 210 + 70 = 280)
            CHUNK_LEN == encoder valid_out_len        (e.g. 14)
        """
        diar_scfg = self.cfg.diar_streaming
        sm = self.diarization_model.sortformer_modules
        if self.cfg.get('simulate_streaming', False):
            self.diarization_model.streaming_mode = True
        else:
            self.diarization_model.streaming_mode = False

        sm.spkcache_len = diar_scfg.get('spkcache_len', 210)
        sm.spkcache_refresh_rate = diar_scfg.get('spkcache_refresh_rate', 14)
        sm.fifo_len = diar_scfg.get('fifo_len', 70)
        sm.chunk_len = diar_scfg.get('chunk_len', 14)
        sm.chunk_left_context = diar_scfg.get('chunk_left_context', 0)
        sm.chunk_right_context = diar_scfg.get('chunk_right_context', 0)
        sm.log = False
        self.diarization_model._suppress_streaming_pbar = True

        logging.info(
            f"[diar_streaming] Configured Sortformer streaming params: "
            f"spkcache_len={sm.spkcache_len}, "
            f"spkcache_refresh_rate={sm.spkcache_refresh_rate}, "
            f"fifo_len={sm.fifo_len}, chunk_len={sm.chunk_len}, "
            f"chunk_left_context={sm.chunk_left_context}, "
            f"chunk_right_context={sm.chunk_right_context}"
        )

        # ── Sync checks (all values in encoder frames) ─────────────────
        self.encoder.setup_streaming_params()
        scfg = self.encoder.streaming_cfg

        # 1) FIFO_LEN + SPKCACHE_LEN must equal ASR left context
        #    (streaming_cfg.last_channel_cache_size)
        diar_total_history = sm.spkcache_len + sm.fifo_len
        asr_left_context = scfg.last_channel_cache_size
        if self.cfg.get('simulate_streaming', False) and diar_total_history != asr_left_context:
            raise ValueError(
                f"[diar_streaming] Diar/ASR left-context mismatch! "
                f"SPKCACHE_LEN({sm.spkcache_len}) + FIFO_LEN({sm.fifo_len}) "
                f"= {diar_total_history} != ASR streaming_cfg.last_channel_cache_size"
                f"({asr_left_context}).  These must be equal so that the "
                f"diarizer's total history window matches the ASR encoder's "
                f"left context."
            )
            logging.info(
                f"[diar_streaming] Left-context sync OK: "
                f"spkcache({sm.spkcache_len}) + fifo({sm.fifo_len}) "
                f"= {diar_total_history} == ASR last_channel_cache_size({asr_left_context})"
            )

            # 2) CHUNK_LEN must equal ASR encoder valid_out_len
            if scfg is not None and hasattr(scfg, 'valid_out_len'):
                vol = scfg.valid_out_len
                if isinstance(vol, list):
                    vol = vol[-1]  # use the steady-state (non-first) value
                if sm.chunk_len != vol:
                    raise ValueError(
                        f"[diar_streaming] Diar/ASR chunk-length mismatch! "
                        f"diar chunk_len({sm.chunk_len}) != "
                        f"ASR encoder valid_out_len({vol}).  These must be equal "
                        f"so diar and ASR produce the same number of frames "
                        f"per step."
                    )
                logging.info(
                    f"[diar_streaming] Chunk-length sync OK: "
                    f"diar chunk_len({sm.chunk_len}) == "
                    f"encoder valid_out_len({vol})"
                )

    # ── FastConformer layer fusion ─────────────────────────────────────

    def _setup_fc_layer_fusion(self):
        """
        Set up per-layer fusion between diar and ASR FastConformer encoders.

        The diar encoder's per-layer hidden states are captured via forward
        hooks and projected into the ASR encoder's MHA cache before each
        streaming step.  This lets the ASR attention organically incorporate
        speaker information at every layer.

        Layer mapping (ASR has more layers than diar):
            ASR layers [0 .. n_diar-1]         ← diar layers [0 .. n_diar-1]
            ASR layers [n_diar .. n_asr-1]     ← diar layers [-(n_asr-n_diar):]
        i.e.  [diar_encoders[:]; diar_encoders[-extra:]]
        """
        n_diar = len(self.diarization_model.encoder.layers)
        n_asr = len(self.encoder.layers)
        d_diar = self.diarization_model.encoder.d_model
        d_asr = self.encoder.d_model

        # Build layer mapping
        extra = max(0, n_asr - n_diar)
        if extra > 0:
            mapping = list(range(n_diar)) + list(range(n_diar - extra, n_diar))
        else:
            # n_asr <= n_diar: just use the first n_asr diar layers
            mapping = list(range(n_asr))
        self._fc_fusion_layer_map = mapping

        # Projection layer (learnable): diar d_model → ASR d_model
        if d_diar != d_asr:
            self.fc_fusion_proj = torch.nn.Linear(d_diar, d_asr)
        else:
            self.fc_fusion_proj = torch.nn.Identity()

        # Hook infrastructure for capturing diar encoder per-layer outputs
        self._diar_layer_outputs = {}
        self._capture_diar_layers = False
        self._diar_hooks = []
        for idx, layer in enumerate(self.diarization_model.encoder.layers):
            hook = layer.register_forward_hook(self._make_diar_layer_hook(idx))
            self._diar_hooks.append(hook)

        logging.info(
            f"[fc_layer_fusion] Setup complete: "
            f"n_diar_layers={n_diar}, n_asr_layers={n_asr}, "
            f"d_diar={d_diar}, d_asr={d_asr}, "
            f"projection={'Linear' if d_diar != d_asr else 'Identity'}, "
            f"layer_map(len={len(mapping)})={mapping}"
        )

    def _make_diar_layer_hook(self, layer_idx):
        """Return a forward hook that captures one diar encoder layer's output."""
        def hook(module, input, output):
            if self._capture_diar_layers:
                # ConformerLayer returns audio_signal tensor (no tuple when
                # cache is not used, which is the diar encoder's case).
                out = output[0] if isinstance(output, tuple) else output
                self._diar_layer_outputs[layer_idx] = out.detach()
        return hook

    def _fuse_fc_layer_embeddings(self, cache_last_channel):
        """
        Fuse diar encoder per-layer hidden states into ASR cache_last_channel.

        For each ASR layer *i*, the mapped diar layer *j*'s full output
        (covering spkcache + fifo + chunk) is projected to ASR d_model and
        **added** to the corresponding slice of the MHA cache.  Diar outputs
        are right-aligned to the cache dimension (left-padded with zeros when
        the diar context is shorter than the cache in early steps).

        Args:
            cache_last_channel: Tensor [n_layers, B, cache_size, d_model_asr]

        Returns:
            Modified cache_last_channel with diar information fused in.
        """
        cache_size = cache_last_channel.shape[2]
        device = cache_last_channel.device
        dtype = cache_last_channel.dtype

        fused_layers = []
        for asr_idx, diar_idx in enumerate(self._fc_fusion_layer_map):
            diar_out = self._diar_layer_outputs[diar_idx]      # [B, T_diar, d_diar]
            projected = self.fc_fusion_proj(diar_out)           # [B, T_diar, d_asr]

            T = projected.shape[1]
            if T >= cache_size:
                # Right-align: take the last cache_size frames
                aligned = projected[:, -cache_size:, :]
            else:
                # Left-pad with zeros
                B_sz = projected.shape[0]
                pad = torch.zeros(
                    B_sz, cache_size - T, projected.shape[2],
                    device=device, dtype=dtype,
                )
                aligned = torch.cat([pad, projected], dim=1)

            fused_layers.append(cache_last_channel[asr_idx] + aligned)

        self._diar_layer_outputs.clear()
        return torch.stack(fused_layers, dim=0)

    # ── Pre-encode level diarization fusion ────────────────────────────

    def _setup_pre_encode_diar_fusion(self):
        """
        Set up pre-encode level diarization fusion modules.

        Both methods are always created and applied in serial:

        (1) Sinusoidal: speaker position encoding weighted by diar predictions,
            added to ASR pre-encode embeddings.  Same idea as the post-encode
            sinusoidal kernel but applied earlier, giving every encoder layer
            access to speaker identity.

        (2) Merge: project diar encoder's pre-encode embeddings to ASR
            dimension and add to ASR pre-encode embeddings.  This directly
            injects speaker-aware representations from the diar encoder into
            the ASR encoder's input, letting self-attention in every encoder
            layer consume speaker context.
        """
        d_asr = self.encoder.d_model

        # ── (1) Sinusoidal kernel ─────────────────────────────────────
        self.pre_encode_diar_kernel = self.get_sinusoid_position_encoding(
            self.max_num_speakers, d_asr
        )
        logging.info(
            f"[pre_encode_diar_fusion] Sinusoidal kernel setup: "
            f"max_speakers={self.max_num_speakers}, d_asr={d_asr}"
        )

        # ── (2) Merge projection ──────────────────────────────────────
        d_diar = self.diarization_model.encoder.d_model
        if d_diar != d_asr:
            self.pre_encode_merge_proj = torch.nn.Linear(d_diar, d_asr)
        else:
            self.pre_encode_merge_proj = torch.nn.Identity()
        logging.info(
            f"[pre_encode_diar_fusion] Merge projection setup: "
            f"d_diar={d_diar} -> d_asr={d_asr}, "
            f"projection={'Linear' if d_diar != d_asr else 'Identity'}"
        )

    def _apply_pre_encode_diar_fusion(self, audio_chunk, chunk_lengths, chunk_diar_preds, drop_extra):
        """
        Fuse diarization information at the pre-encode embedding level.

        Runs the ASR encoder's ``pre_encode`` on the audio chunk, then applies
        **both** fusion methods in serial:
            1. Sinusoidal speaker embedding (from diar predictions)
            2. Merge diar pre-encode projection

        Args:
            audio_chunk: (B, D_mel, T) mel features with pre-encode cache prepended.
            chunk_lengths: (B,) per-sample lengths in mel frames.
            chunk_diar_preds: (B, T_chunk, num_speakers) diar predictions for
                this chunk (already detached from diar model graph).
            drop_extra: int, number of extra pre-encoded frames to drop
                (same as streaming_cfg.drop_extra_pre_encoded for non-first steps).

        Returns:
            fused_pre_encode: (B, T_valid, d_asr) fused pre-encode embeddings.
            pre_encode_lengths: (B,) lengths after pre-encode and dropping.
        """
        audio_signal_t = audio_chunk.transpose(1, 2)  # (B, D_mel, T) -> (B, T, D_mel)

        # ── ASR pre-encode (gradient flows if ASR not frozen) ──────────
        asr_pre_encode, pre_encode_lengths = self.encoder.pre_encode(
            x=audio_signal_t, lengths=chunk_lengths
        )
        pre_encode_lengths = pre_encode_lengths.to(torch.int64)

        if drop_extra > 0:
            asr_pre_encode = asr_pre_encode[:, drop_extra:, :]
            pre_encode_lengths = (pre_encode_lengths - drop_extra).clamp(min=0)

        # ── (1) Sinusoidal speaker embedding ───────────────────────────
        dp = chunk_diar_preds
        if dp.shape[1] != asr_pre_encode.shape[1]:
            if dp.shape[1] < asr_pre_encode.shape[1]:
                dp = self.fix_diar_output(dp, asr_pre_encode.shape[1])
            else:
                dp = dp[:, :asr_pre_encode.shape[1], :]

        speaker_emb = torch.matmul(
            dp, self.pre_encode_diar_kernel.to(dp.device)
        )
        if self.kernel_norm == 'l2':
            speaker_emb = torch.nn.functional.normalize(speaker_emb, p=2, dim=-1)

        asr_pre_encode = asr_pre_encode + speaker_emb

        # ── (2) Merge diar pre-encode projection ──────────────────────
        with torch.no_grad():
            diar_pre_encode, _ = self.diarization_model.encoder.pre_encode(
                x=audio_signal_t, lengths=chunk_lengths
            )
            if drop_extra > 0:
                diar_pre_encode = diar_pre_encode[:, drop_extra:, :]
            diar_pre_encode = diar_pre_encode.detach()

        projected_diar_pre_encode = self.pre_encode_merge_proj(diar_pre_encode)

        T_asr = asr_pre_encode.shape[1]
        T_diar = projected_diar_pre_encode.shape[1]
        if T_asr != T_diar:
            min_len = min(T_asr, T_diar)
            asr_pre_encode = asr_pre_encode[:, :min_len, :]
            projected_diar_pre_encode = projected_diar_pre_encode[:, :min_len, :]
            pre_encode_lengths = pre_encode_lengths.clamp(max=min_len)

        fused_pre_encode = asr_pre_encode + projected_diar_pre_encode
        return fused_pre_encode, pre_encode_lengths

    def _init_asr_model(self):
        """
        Initialize the ASR encoder from a pretrained EncDecRNNTBPEModel.
        Only the encoder weights are loaded (decoder and joint are kept as initialized).
        """
        model_path = self.cfg.asr_model_path

        if model_path is not None and model_path.endswith('.nemo'):
            pretrained_asr_model = EncDecRNNTBPEModel.restore_from(model_path, map_location="cpu")
            logging.info("ASR Model restored locally from {}".format(model_path))
        elif model_path.endswith('.ckpt'):
            pretrained_asr_model = EncDecRNNTBPEModel.load_from_checkpoint(model_path, map_location="cpu")
            logging.info("ASR Model restored locally from {}".format(model_path))
        else:
            pretrained_asr_model = None

        if pretrained_asr_model is not None:
            logging.info("Restoring ASR model encoder parameters from pretrained model.")
            self.encoder.load_state_dict(pretrained_asr_model.encoder.state_dict(), strict=True)

        if self.cfg.freeze_asr:
            self.encoder.eval()

    def maybe_init_from_pretrained_checkpoint(self, cfg, map_location='cpu'):
        """
        Override to handle init_from_nemo_model for MSEncDecRNNTBPEModel.

        The pretrained ASR model is an EncDecRNNTBPEModel, not an MSEncDecRNNTBPEModel,
        so we must use EncDecRNNTBPEModel.restore_from() instead of self.restore_from()
        to avoid a class/config mismatch when restoring the checkpoint.

        Supports both string and dict forms of init_from_nemo_model:
            # String: load all matching weights
            +init_from_nemo_model="<path to .nemo>"

            # Dict: selective loading with include/exclude
            +init_from_nemo_model.asr_model.path="<path to .nemo>"
            +init_from_nemo_model.asr_model.include="[encoder,preprocessor]"
        """
        if 'init_from_nemo_model' in cfg and cfg.init_from_nemo_model is not None:
            with open_dict(cfg):
                if isinstance(cfg.init_from_nemo_model, str):
                    model_path = cfg.init_from_nemo_model
                    restored_model = EncDecRNNTBPEModel.restore_from(
                        model_path, map_location=map_location, strict=cfg.get("init_strict", True)
                    )
                    self.load_state_dict(restored_model.state_dict(), strict=False)
                    logging.info(f'Model checkpoint restored from nemo file with path : `{model_path}`')
                    del restored_model

                elif isinstance(cfg.init_from_nemo_model, (DictConfig, dict)):
                    model_load_dict = cfg.init_from_nemo_model
                    for model_load_cfg in model_load_dict.values():
                        model_path = model_load_cfg.path
                        restored_model = EncDecRNNTBPEModel.restore_from(
                            model_path, map_location=map_location, strict=cfg.get("init_strict", True)
                        )

                        include = model_load_cfg.pop('include', [""])
                        exclude = model_load_cfg.pop('exclude', [])

                        self.load_part_of_state_dict(
                            restored_model.state_dict(),
                            include,
                            exclude,
                            f'nemo file with path `{model_path}`',
                        )
                        logging.info(
                            f'Selectively restored from nemo file `{model_path}` '
                            f'(include={include}, exclude={exclude})'
                        )
                        del restored_model
                else:
                    raise TypeError("Invalid type: init_from_nemo_model is not a string or a dict!")

            # Apply freeze_asr after loading pretrained weights
            if self.cfg.get('freeze_asr', False):
                self.encoder.eval()
                logging.info("ASR encoder set to eval mode (freeze_asr=True)")
        else:
            # Fall back to the base class implementation for other init methods
            # (e.g., init_from_pretrained_model, init_from_ptl_ckpt)
            super().maybe_init_from_pretrained_checkpoint(cfg, map_location)

    def forward_diar(self, input_signal=None, input_signal_length=None):
        """
        Forward pass through the Sortformer diarization model.

        Args:
            input_signal: [B, T] raw audio signal
            input_signal_length: [B] lengths

        Returns:
            preds: [B, T_diar, num_speakers] frame-level speaker predictions
        """
        preds = self.diarization_model.forward(
            audio_signal=input_signal, audio_signal_length=input_signal_length
        )
        return preds

    def fix_diar_output(self, diar_pred, asr_frame_count):
        """
        Extend diarization predictions to match the ASR encoder frame count
        by repeating the last frame.

        Args:
            diar_pred: [B, T_diar, num_speakers]
            asr_frame_count: int, target number of frames

        Returns:
            extended_diar_preds: [B, asr_frame_count, num_speakers]
        """
        last_emb = diar_pred[:, -1, :].unsqueeze(1)
        additional_frames = asr_frame_count - diar_pred.shape[1]
        last_repeats = last_emb.repeat(1, additional_frames, 1)
        extended_diar_preds = torch.cat((diar_pred, last_repeats), dim=1)
        return extended_diar_preds

    def get_sinusoid_position_encoding(self, max_position, embedding_dim):
        """
        Generates a sinusoid position encoding matrix.

        Args:
            max_position (int): The maximum position (number of speakers).
            embedding_dim (int): The dimension of the embeddings.

        Returns:
            torch.Tensor: Shape (max_position, embedding_dim) sinusoid position encodings.
        """
        position = np.arange(max_position)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embedding_dim, 2) * -(np.log(10000.0) / embedding_dim))

        position_encoding = np.zeros((max_position, embedding_dim))
        position_encoding[:, 0::2] = np.sin(position * div_term)
        position_encoding[:, 1::2] = np.cos(position * div_term)

        position_encoding_tensor = torch.tensor(position_encoding, dtype=torch.float32)
        return position_encoding_tensor

    def fuse_diar_preds(self, encoded, diar_preds):
        """
        Fuse diarization predictions with ASR encoder states.

        This method is extracted from forward() so it can be reused in both
        the standard forward pass and in streaming inference (via
        conformer_stream_step_with_diarization / cache_aware_stream_step_with_diarization).

        It adds speaker identity information (speaker_infusion_asr) to the
        encoder states, which is critical for multi-talker ASR performance —
        especially in streaming where the cache has limited context (~5 s).

        Args:
            encoded: [B, D, T] encoder output
            diar_preds: [B, T_diar, num_speakers] frame-level speaker predictions

        Returns:
            encoded: [B, D, T] fused encoder output
        """
        # Convert encoder output from (B, D, T) to (B, T, D) for fusion
        asr_enc_states = encoded.permute(0, 2, 1)  # (B, T, D)

        # Fix frame count mismatch between diar and ASR encoder
        if diar_preds.shape[1] != asr_enc_states.shape[1]:
            if diar_preds.shape[1] < asr_enc_states.shape[1]:
                diar_preds = self.fix_diar_output(diar_preds, asr_enc_states.shape[1])
            else:
                diar_preds = diar_preds[:, :asr_enc_states.shape[1], :]

        # Normalize the features
        if self.norm == 'ln':
            diar_preds = self.diar_norm(diar_preds)
            asr_enc_states = self.asr_norm(asr_enc_states)
        elif self.norm == 'l2':
            diar_preds = torch.nn.functional.normalize(diar_preds, p=2, dim=-1)
            asr_enc_states = torch.nn.functional.normalize(asr_enc_states, p=2, dim=-1)

        # Fuse diarization predictions with ASR encoder states
        if self.diar_kernel_type == 'sinusoidal':
            speaker_infusion_asr = torch.matmul(
                diar_preds, self.diar_kernel.to(diar_preds.device)
            )
            if self.kernel_norm == 'l2':
                speaker_infusion_asr = torch.nn.functional.normalize(
                    speaker_infusion_asr, p=2, dim=-1
                )
            enc_states = speaker_infusion_asr + asr_enc_states

        elif self.diar_kernel_type == 'metacat':
            concat_enc_states = asr_enc_states.unsqueeze(2) * diar_preds.unsqueeze(3)
            concat_enc_states = concat_enc_states.flatten(2, 3)
            enc_states = self.metacat_diar_joint_proj(concat_enc_states)

        else:  # 'projection' (default)
            concat_enc_states = torch.cat([asr_enc_states, diar_preds], dim=-1)
            enc_states = self.diar_joint_proj(concat_enc_states)

        # Convert back to (B, D, T) for RNNT decoder/joint
        return enc_states.permute(0, 2, 1)

    @staticmethod
    def _pick_streaming_param(val, is_first):
        """Resolve list-or-scalar streaming config param to a single value."""
        if isinstance(val, list):
            return val[0] if is_first else val[1]
        return val

    def _init_simulated_streaming_state(self, processed_signal, processed_signal_length):
        """
        Initialize all streaming state for the chunk-by-chunk loop.

        Sets up the encoder streaming config, encoder MHA/conv caches,
        diar streaming state (if diar is enabled), and sampling frames.
        Logs the streaming config once on first call.

        Args:
            processed_signal:  Tensor [B, D, T] preprocessed mel features.
            processed_signal_length:  Tensor [B] per-sample lengths.

        Returns:
            Tuple of:
                scfg:                   Encoder streaming config object.
                cache_last_channel:     Tensor [n_layers, B, cache_size, d_model].
                cache_last_time:        Tensor [n_layers, B, d_model, kernel_size].
                cache_last_channel_len: Tensor [B].
                diar_streaming_state:   StreamingSortformerState or None.
                diar_pred_out_stream:   Tensor [B, 0, num_speakers] or None.
                sampling_frames:        Sampling frames from pre_encode, or None.
        """
        batch_size = processed_signal.size(0)
        device = processed_signal.device
        dtype = processed_signal.dtype
        total_mel_frames = processed_signal.size(-1)

        # ── Streaming config ───────────────────────────────────────────
        self.encoder.setup_streaming_params()
        scfg = self.encoder.streaming_cfg
        if not getattr(self, '_logged_streaming_cfg', False):
            logging.info(
                f"[simulate_streaming] encoder.att_context_size={self.encoder.att_context_size}, "
                f"encoder.att_context_style={self.encoder.att_context_style}, "
                f"encoder.subsampling_factor={self.encoder.subsampling_factor}"
            )
            logging.info(
                f"[simulate_streaming] streaming_cfg: chunk_size={scfg.chunk_size}, "
                f"shift_size={scfg.shift_size}, valid_out_len={scfg.valid_out_len}, "
                f"last_channel_cache_size={scfg.last_channel_cache_size}, "
                f"pre_encode_cache_size={scfg.pre_encode_cache_size}, "
                f"drop_extra_pre_encoded={scfg.drop_extra_pre_encoded}"
            )
            logging.info(
                f"[simulate_streaming] total_mel_frames={total_mel_frames}, "
                f"batch_size={batch_size}"
            )
            self._logged_streaming_cfg = True

        # ── Encoder cache ──────────────────────────────────────────────
        cache_last_channel, cache_last_time, cache_last_channel_len = (
            self.encoder.get_initial_cache_state(
                batch_size=batch_size, dtype=dtype, device=device,
            )
        )

        # ── Diar streaming state (frozen) ──────────────────────────────
        diar_streaming_state = None
        diar_pred_out_stream = None
        if self.diar:
            diar_streaming_state = (
                self.diarization_model.sortformer_modules.init_streaming_state(
                    batch_size=batch_size, device=device,
                )
            )
            diar_pred_out_stream = torch.zeros(
                (batch_size, 0, self.max_num_speakers),
                device=device, dtype=dtype,
            )

        # ── Sampling frames (minimum chunk size for subsampling) ───────
        sampling_frames = None
        if (
            hasattr(self.encoder, 'pre_encode')
            and hasattr(self.encoder.pre_encode, 'get_sampling_frames')
        ):
            sampling_frames = self.encoder.pre_encode.get_sampling_frames()

        return (
            scfg, cache_last_channel, cache_last_time,
            cache_last_channel_len, diar_streaming_state,
            diar_pred_out_stream, sampling_frames,
        )

    def forward_simulated_streaming(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
    ):
        """
        Simulated-streaming forward pass for training.

        Processes audio chunk-by-chunk through the encoder with cache, runs the
        frozen Sortformer diarizer in streaming mode with spkcache/FIFO, and
        fuses per-chunk diar predictions with encoder output -- matching the
        inference-time streaming pipeline in perform_streaming().

        This closes the train/serve gap: the model learns to work with the
        same limited-context encoder cache and imperfect per-chunk diar
        predictions that it will see at inference time.

        The output has the same shape as forward(), so the RNNT decoder/joint/
        loss in training_step remains completely unchanged.

        Args:
            input_signal:  Tensor [B, T] of raw audio signals.
            input_signal_length:  Vector [B] of audio lengths.
            processed_signal:  Tensor [B, D, T] of preprocessed audio features.
            processed_signal_length:  Vector [B] of processed audio lengths.

        Returns:
            encoded:     Tensor [B, D, T_total] fused encoder output.
            encoded_len: Tensor [B] per-sample number of valid encoder frames.
        """
        # ── Preprocessing (identical to forward()) ─────────────────────
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(
                input_spec=processed_signal, length=processed_signal_length
            )

        # ── Initialize streaming state ─────────────────────────────────
        (
            scfg, cache_last_channel, cache_last_time,
            cache_last_channel_len, diar_streaming_state,
            diar_pred_out_stream, sampling_frames,
        ) = self._init_simulated_streaming_state(processed_signal, processed_signal_length)

        batch_size = processed_signal.size(0)
        device = processed_signal.device
        dtype = processed_signal.dtype
        total_mel_frames = processed_signal.size(-1)
        _pick = self._pick_streaming_param

        # ── Chunk-by-chunk loop ────────────────────────────────────────
        all_encoded = []
        all_encoded_len = []
        buffer_idx = 0
        step_num = 0

        # Optional tqdm progress bar (disabled by default)
        _pbar = None
        if not getattr(self, '_suppress_streaming_pbar', True):
            _first_shift = _pick(scfg.shift_size, True)
            _rest_shift = _pick(scfg.shift_size, False)
            if _rest_shift > 0 and total_mel_frames > _first_shift:
                _est_steps = 1 + math.ceil((total_mel_frames - _first_shift) / _rest_shift)
            else:
                _est_steps = max(1, total_mel_frames // max(_first_shift, 1))
            try:
                from tqdm import tqdm
                _pbar = tqdm(
                    total=_est_steps, desc="sim-stream chunks",
                    leave=False, dynamic_ncols=True,
                )
            except ImportError:
                pass

        while buffer_idx < total_mel_frames:
            is_first = (step_num == 0)

            cur_chunk_size = _pick(scfg.chunk_size, is_first)
            cur_shift_size = _pick(scfg.shift_size, is_first)
            cur_pre_encode_cache_size = _pick(scfg.pre_encode_cache_size, is_first)

            # ── Extract audio chunk from the mel-feature buffer ────────
            audio_chunk = processed_signal[:, :, buffer_idx : buffer_idx + cur_chunk_size]

            # Skip if chunk too small for encoder subsampling
            if sampling_frames is not None:
                cur_sf = _pick(sampling_frames, is_first)
                if audio_chunk.size(-1) < cur_sf:
                    break

            # ── Prepend pre-encode cache (matches CacheAwareStreamingAudioBuffer) ──
            if is_first:
                # First chunk: zero-pad for pre-encode cache
                cache_pre_encode = torch.zeros(
                    (batch_size, processed_signal.size(1), cur_pre_encode_cache_size),
                    device=device, dtype=dtype,
                )
            else:
                # Subsequent chunks: use previous mel frames as cache
                start = max(0, buffer_idx - cur_pre_encode_cache_size)
                cache_pre_encode = processed_signal[:, :, start : buffer_idx]
                if cache_pre_encode.size(-1) < cur_pre_encode_cache_size:
                    pad_size = cur_pre_encode_cache_size - cache_pre_encode.size(-1)
                    pad = torch.zeros(
                        (batch_size, processed_signal.size(1), pad_size),
                        device=device, dtype=dtype,
                    )
                    cache_pre_encode = torch.cat([pad, cache_pre_encode], dim=-1)

            added_len = cache_pre_encode.size(-1)
            audio_chunk = torch.cat([cache_pre_encode, audio_chunk], dim=-1)

            # ── Per-sample chunk lengths ───────────────────────────────
            chunk_lengths = torch.clamp(
                processed_signal_length - buffer_idx + added_len,
                min=0,
                max=audio_chunk.size(-1),
            )

            # ── Is this the last chunk? ────────────────────────────────
            next_buffer_idx = buffer_idx + cur_shift_size
            is_last = (next_buffer_idx >= total_mel_frames)

            # ── drop_extra_pre_encoded ─────────────────────────────────
            drop_extra = 0 if is_first else scfg.drop_extra_pre_encoded

            # ── Frozen diarization streaming step ──────────────────────
            chunk_diar_preds = None
            if self.diar:
                # Enable per-layer capture for FC layer fusion
                if self.use_fc_layer_fusion:
                    self._capture_diar_layers = True

                with torch.no_grad():
                    prev_diar_len = diar_pred_out_stream.shape[1]
                    diar_streaming_state, diar_pred_out_stream = (
                        self.diarization_model.forward_streaming_step(
                            processed_signal=audio_chunk.transpose(1, 2),
                            processed_signal_length=chunk_lengths,
                            streaming_state=diar_streaming_state,
                            total_preds=diar_pred_out_stream,
                            drop_extra_pre_encoded=drop_extra,
                        )
                    )
                    # Per-chunk diar preds (not cumulative) for correct
                    # alignment with this chunk's encoder output
                    chunk_diar_preds = diar_pred_out_stream[:, prev_diar_len:, :].detach()

                # Stop capturing and disable flag
                if self.use_fc_layer_fusion:
                    self._capture_diar_layers = False

            # ── FC layer fusion: diar per-layer states → ASR cache ─────
            if self.use_fc_layer_fusion and getattr(self, '_diar_layer_outputs', {}):
                cache_last_channel = self._fuse_fc_layer_embeddings(
                    cache_last_channel
                )

            # ── ASR encoder streaming step (with gradients) ────────────
            if self.use_pre_encode_diar_fusion and chunk_diar_preds is not None:
                # Pre-encode level fusion: run ASR pre_encode ourselves,
                # fuse speaker info, then feed to encoder with
                # bypass_pre_encode=True so encoder layers attend to
                # speaker-enriched embeddings from the start.
                with torch.set_grad_enabled(not self.cfg.get('freeze_asr', False)):
                    fused_pre_encode, fused_lengths = self._apply_pre_encode_diar_fusion(
                        audio_chunk, chunk_lengths, chunk_diar_preds, drop_extra,
                    )
                    (
                        encoded,
                        encoded_len,
                        cache_last_channel_next,
                        cache_last_time_next,
                        cache_last_channel_next_len,
                    ) = self.encoder.cache_aware_stream_step(
                        processed_signal=fused_pre_encode,
                        processed_signal_length=fused_lengths,
                        cache_last_channel=cache_last_channel,
                        cache_last_time=cache_last_time,
                        cache_last_channel_len=cache_last_channel_len,
                        keep_all_outputs=is_last,
                        drop_extra_pre_encoded=0,
                        bypass_pre_encode=True,
                    )
            else:
                with torch.set_grad_enabled(not self.cfg.get('freeze_asr', False)):
                    (
                        encoded,
                        encoded_len,
                        cache_last_channel_next,
                        cache_last_time_next,
                        cache_last_channel_next_len,
                    ) = self.encoder.cache_aware_stream_step(
                        processed_signal=audio_chunk,
                        processed_signal_length=chunk_lengths,
                        cache_last_channel=cache_last_channel,
                        cache_last_time=cache_last_time,
                        cache_last_channel_len=cache_last_channel_len,
                        keep_all_outputs=is_last,
                        drop_extra_pre_encoded=drop_extra,
                    )
            # ── Diar fusion at post-encode level (always applied) ──────
            # Sinusoidal/projection/metacat fusion on encoder output,
            # kept alongside pre-encode fusion per user requirement.
            if self.diar and chunk_diar_preds is not None:
                encoded = self.fuse_diar_preds(encoded, chunk_diar_preds)

            # ── Collect this chunk's output ─────────────────────────────
            all_encoded.append(encoded)
            all_encoded_len.append(encoded_len)

            # ── Detach cache to avoid BPTT across all chunks ───────────
            cache_last_channel = cache_last_channel_next.detach()
            cache_last_time = cache_last_time_next.detach()
            cache_last_channel_len = cache_last_channel_next_len

            buffer_idx = next_buffer_idx
            step_num += 1
            if _pbar is not None:
                _pbar.update(1)

        if _pbar is not None:
            _pbar.close()

        # ── Concatenate all chunks ─────────────────────────────────────
        if len(all_encoded) == 0:
            # Audio too short for even one chunk; fall back to standard forward
            logging.warning(
                "forward_simulated_streaming: no chunks produced, "
                "falling back to standard forward()"
            )
            return self.forward(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
            )

        encoded = torch.cat(all_encoded, dim=-1)  # [B, D, T_total]
        encoded_len = torch.stack(all_encoded_len, dim=0).sum(dim=0)  # [B]

        return encoded, encoded_len

    @typecheck()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
    ):
        """
        Forward pass of the model. For RNNT models, forward() only performs the
        encoder forward. This override adds diarization fusion after the encoder
        so that the fused representation is used by the RNNT decoder/joint in
        training_step and validation_pass.

        Args:
            input_signal: Tensor [B, T] of raw audio signals.
            input_signal_length: Vector [B] of audio lengths.
            processed_signal: Tensor [B, D, T] of preprocessed audio (e.g. from DALI).
            processed_signal_length: Vector [B] of processed audio lengths.

        Returns:
            A tuple of 2 elements:
            1) encoded: Tensor [B, D, T] encoder output (with diarization fused if enabled).
            2) encoded_len: Tensor [B] lengths after encoder.
        """
        # --- Preprocessing ---
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(
                input_spec=processed_signal, length=processed_signal_length
            )

        # --- Diarization forward (needed before encoder for pre-encode fusion) ---
        diar_preds = None
        if self.diar and input_signal is not None:
            with torch.set_grad_enabled(not self.cfg.get('freeze_diar', False)):
                diar_preds = self.forward_diar(input_signal, input_signal_length)
                # diar_preds shape: (B, T_diar, num_speakers)

        # --- ASR encoder forward (with optional pre-encode diar fusion) ---
        if self.use_pre_encode_diar_fusion and diar_preds is not None:
            # Pre-encode fusion: run pre_encode ourselves, fuse speaker info,
            # then feed to encoder with bypass_pre_encode=True.
            # drop_extra=0 because there is no streaming cache in the
            # non-streaming forward path (cache_last_channel is None).
            with torch.set_grad_enabled(not self.cfg.get('freeze_asr', False)):
                fused_pre_encode, pre_encode_lengths = self._apply_pre_encode_diar_fusion(
                    processed_signal, processed_signal_length, diar_preds, drop_extra=0,
                )
                encoded, encoded_len = self.encoder(
                    audio_signal=fused_pre_encode,
                    length=pre_encode_lengths,
                    bypass_pre_encode=True,
                )
        else:
            with torch.set_grad_enabled(not self.cfg.get('freeze_asr', False)):
                encoded, encoded_len = self.encoder(
                    audio_signal=processed_signal, length=processed_signal_length
                )
        # encoded shape: (B, D, T)

        # --- Post-encode diar fusion (always applied when diar is available) ---
        if diar_preds is not None:
            encoded = self.fuse_diar_preds(encoded, diar_preds)

        return encoded, encoded_len

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        signal, signal_len, transcript, transcript_len = batch

        # Choose between standard full-audio forward and simulated-streaming forward.
        # simulate_streaming=True trains the encoder+RNNT with the same chunk-by-chunk
        # cache-aware path and per-chunk diar fusion used at inference time, closing
        # the train/serve gap that causes accuracy degradation after ~20-30 s.
        use_simulated_streaming = self.cfg.get('simulate_streaming', False)

        if use_simulated_streaming:
            if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
                encoded, encoded_len = self.forward_simulated_streaming(
                    processed_signal=signal, processed_signal_length=signal_len,
                )
            else:
                encoded, encoded_len = self.forward_simulated_streaming(
                    input_signal=signal, input_signal_length=signal_len,
                )
        else:
            # forward() only performs encoder forward (with diarization fusion if enabled)
            if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
                encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
            else:
                encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            # Compute full joint and loss
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
            loss_value = self.loss(
                log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
            )

            # Latent speaker supervision loss (auxiliary, optional)
            if self.lss_loss is not None:
                lss_loss_value = self.lss_loss(
                    log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
                )
                loss_value = loss_value + lss_loss_value

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if (sample_id + 1) % log_every_n_steps == 0:
                self.wer.update(
                    predictions=encoded,
                    predictions_lengths=encoded_len,
                    targets=transcript,
                    targets_lengths=transcript_len,
                )
                _, scores, words = self.wer.compute()
                self.wer.reset()
                tensorboard_logs.update({'training_batch_wer': scores.float() / words})

        else:
            # If experimental fused Joint-Loss-WER is used
            if (sample_id + 1) % log_every_n_steps == 0:
                compute_wer = True
            else:
                compute_wer = False

            # Fused joint step
            loss_value, wer, _, _ = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=transcript_len,
                compute_wer=compute_wer,
            )

            # Latent speaker supervision loss (auxiliary, optional)
            if self.lss_loss is not None:
                # Use the inner joint() method to compute raw joint output (bypasses fused loss path)
                # Transpose from [B, D, T] to [B, T, D] since inner joint() skips @typecheck transpose
                joint = self.joint.joint(encoded.transpose(1, 2), decoder.transpose(1, 2))
                lss_loss_value = self.lss_loss(
                    log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
                )
                loss_value = loss_value + lss_loss_value

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                tensorboard_logs.update({'training_batch_wer': wer})

        # Log items
        self.log_dict(tensorboard_logs)

        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {'loss': loss_value}

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len = batch

        # Use the same forward path as training so val metrics reflect
        # actual streaming performance (not inflated full-context scores).
        use_simulated_streaming = self.cfg.get('simulate_streaming', False)

        if use_simulated_streaming:
            if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
                encoded, encoded_len = self.forward_simulated_streaming(
                    processed_signal=signal, processed_signal_length=signal_len,
                )
            else:
                encoded, encoded_len = self.forward_simulated_streaming(
                    input_signal=signal, input_signal_length=signal_len,
                )
        else:
            if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
                encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
            else:
                encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        tensorboard_logs = {}

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            if self.compute_eval_loss:
                decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)
                joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)

                loss_value = self.loss(
                    log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
                )

                # Latent speaker supervision loss (auxiliary, optional)
                if self.lss_loss is not None:
                    lss_loss_value = self.lss_loss(
                        log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
                    )
                    loss_value = loss_value + lss_loss_value

                tensorboard_logs['val_loss'] = loss_value

            self.wer.update(
                predictions=encoded,
                predictions_lengths=encoded_len,
                targets=transcript,
                targets_lengths=transcript_len,
            )
            wer, wer_num, wer_denom = self.wer.compute()
            self.wer.reset()

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        else:
            # If experimental fused Joint-Loss-WER is used
            compute_wer = True

            if self.compute_eval_loss:
                decoded, target_len, states = self.decoder(targets=transcript, target_length=transcript_len)
            else:
                decoded = None
                target_len = transcript_len

            # Fused joint step
            loss_value, wer, wer_num, wer_denom = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoded,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=target_len,
                compute_wer=compute_wer,
            )

            # Latent speaker supervision loss (auxiliary, optional)
            if self.lss_loss is not None and loss_value is not None:
                # Compute joint output separately for lss_loss
                if decoded is None:
                    decoded, target_len, states = self.decoder(targets=transcript, target_length=transcript_len)
                # Use the inner joint() method to compute raw joint output (bypasses fused loss path)
                # Transpose from [B, D, T] to [B, T, D] since inner joint() skips @typecheck transpose
                joint = self.joint.joint(encoded.transpose(1, 2), decoded.transpose(1, 2))
                lss_loss_value = self.lss_loss(
                    log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_len
                )
                loss_value = loss_value + lss_loss_value

            if loss_value is not None:
                tensorboard_logs['val_loss'] = loss_value

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        return tensorboard_logs

    def on_validation_epoch_start(self) -> None:
        """Disable CUDA graphs for validation in MSEncDecRNNTBPEModel.
        The parent class force-enables CUDA graphs here, but CUDA graph
        capture is incompatible with the dynamic diarization fusion path.
        """
        self.disable_cuda_graphs()

    def on_test_epoch_start(self) -> None:
        """Disable CUDA graphs for testing in MSEncDecRNNTBPEModel.
        Same reason as on_validation_epoch_start.
        """
        self.disable_cuda_graphs()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.validation_pass(batch, batch_idx, dataloader_idx)
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            self.validation_step_outputs.append(metrics)
        return metrics

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logs = self.validation_pass(batch, batch_idx, dataloader_idx=dataloader_idx)
        test_logs = {name.replace("val_", "test_"): value for name, value in logs.items()}
        if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
            self.test_step_outputs[dataloader_idx].append(test_logs)
        else:
            self.test_step_outputs.append(test_logs)
        return test_logs
