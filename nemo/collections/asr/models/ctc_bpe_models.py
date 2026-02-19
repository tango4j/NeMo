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
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.parts.mixins import ASRBPEMixin
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCBPEDecoding, CTCBPEDecodingConfig
from nemo.collections.asr.parts.utils.asr_batching import get_semi_sorted_batch_sampler
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.classes.mixins import AccessMixin
from nemo.utils import logging, model_utils

__all__ = ['EncDecCTCModelBPE', 'MSEncDecCTCModelBPE']


class EncDecCTCModelBPE(EncDecCTCModel, ASRBPEMixin):
    """Encoder decoder CTC-based models with Byte Pair Encoding."""

    def __init__(self, cfg: DictConfig, trainer=None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg, make_copy=False)

        if 'tokenizer' not in cfg:
            raise ValueError("`cfg` must have `tokenizer` config to create a tokenizer !")

        # Setup the tokenizer
        self._setup_tokenizer(cfg.tokenizer)

        # Initialize a dummy vocabulary
        vocabulary = self.tokenizer.tokenizer.get_vocab()

        # Set the new vocabulary
        with open_dict(cfg):
            # sidestepping the potential overlapping tokens issue in aggregate tokenizers
            if self.tokenizer_type == "agg":
                cfg.decoder.vocabulary = ListConfig(vocabulary)
            else:
                cfg.decoder.vocabulary = ListConfig(list(vocabulary.keys()))

        # Override number of classes if placeholder provided
        num_classes = cfg.decoder["num_classes"]

        if num_classes < 1:
            logging.info(
                "\nReplacing placeholder number of classes ({}) with actual number of classes - {}".format(
                    num_classes, len(vocabulary)
                )
            )
            cfg.decoder["num_classes"] = len(vocabulary)

        super().__init__(cfg=cfg, trainer=trainer)

        # Setup decoding objects
        decoding_cfg = self.cfg.get('decoding', None)

        # In case decoding config not found, use default config
        if decoding_cfg is None:
            decoding_cfg = OmegaConf.structured(CTCBPEDecodingConfig)
            with open_dict(self.cfg):
                self.cfg.decoding = decoding_cfg

        self.decoding = CTCBPEDecoding(self.cfg.decoding, tokenizer=self.tokenizer)

        # Setup metric with decoding strategy
        self.wer = WER(
            decoding=self.decoding,
            use_cer=self._cfg.get('use_cer', False),
            dist_sync_on_step=True,
            log_prediction=self._cfg.get("log_prediction", False),
        )

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
            num_workers: (int) number of workers. Depends of the batch_size and machine. \
                0 - only the main process will load batches, 1 - one worker (not main process)

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

    def change_vocabulary(
        self,
        new_tokenizer_dir: Union[str, DictConfig],
        new_tokenizer_type: str,
        decoding_cfg: Optional[DictConfig] = None,
    ):
        """
        Changes vocabulary of the tokenizer used during CTC decoding process.
        Use this method when fine-tuning on from pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged.
        For example, you would use it if you want to use pretrained encoder when fine-tuning on a
        data in another language, or when you'd need model to learn capitalization, punctuation
        and/or special characters.

        Args:
            new_tokenizer_dir: Directory path to tokenizer or a config for a new tokenizer
                (if the tokenizer type is `agg`)
            new_tokenizer_type: Either `agg`, `bpe` or `wpe`. `bpe` is used for SentencePiece tokenizers,
                whereas `wpe` is used for `BertTokenizer`.
            new_tokenizer_cfg: A config for the new tokenizer. if provided, pre-empts the dir and type

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

        # Set the new vocabulary
        decoder_config = copy.deepcopy(self.decoder.to_config_dict())
        # sidestepping the potential overlapping tokens issue in aggregate tokenizers
        if self.tokenizer_type == "agg":
            decoder_config.vocabulary = ListConfig(vocabulary)
        else:
            decoder_config.vocabulary = ListConfig(list(vocabulary.keys()))

        decoder_num_classes = decoder_config['num_classes']

        # Override number of classes if placeholder provided
        logging.info(
            "\nReplacing old number of classes ({}) with new number of classes - {}".format(
                decoder_num_classes, len(vocabulary)
            )
        )

        decoder_config['num_classes'] = len(vocabulary)

        del self.decoder
        self.decoder = EncDecCTCModelBPE.from_config_dict(decoder_config)
        del self.loss
        self.loss = CTCLoss(
            num_classes=self.decoder.num_classes_with_blank - 1,
            zero_infinity=True,
            reduction=self._cfg.get("ctc_reduction", "mean_batch"),
        )

        if decoding_cfg is None:
            # Assume same decoding config as before
            decoding_cfg = self.cfg.decoding

        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(CTCBPEDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

        self.decoding = CTCBPEDecoding(decoding_cfg=decoding_cfg, tokenizer=self.tokenizer)

        self.wer = WER(
            decoding=self.decoding,
            use_cer=self._cfg.get('use_cer', False),
            log_prediction=self._cfg.get("log_prediction", False),
            dist_sync_on_step=True,
        )

        # Update config
        with open_dict(self.cfg.decoder):
            self._cfg.decoder = decoder_config

        with open_dict(self.cfg.decoding):
            self._cfg.decoding = decoding_cfg

        logging.info(f"Changed tokenizer to {self.decoder.vocabulary} vocabulary.")

    def change_decoding_strategy(self, decoding_cfg: DictConfig, verbose: bool = True):
        """
        Changes decoding strategy used during CTC decoding process.

        Args:
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
            verbose: Whether to print the new config or not.
        """
        if decoding_cfg is None:
            # Assume same decoding config as before
            logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
            decoding_cfg = self.cfg.decoding

        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(CTCBPEDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

        self.decoding = CTCBPEDecoding(
            decoding_cfg=decoding_cfg,
            tokenizer=self.tokenizer,
        )

        self.wer = WER(
            decoding=self.decoding,
            use_cer=self.wer.use_cer,
            log_prediction=self.wer.log_prediction,
            dist_sync_on_step=True,
        )

        self.decoder.temperature = decoding_cfg.get('temperature', 1.0)

        # Update config
        with open_dict(self.cfg.decoding):
            self.cfg.decoding = decoding_cfg

        if verbose:
            logging.info(f"Changed decoding strategy to \n{OmegaConf.to_yaml(self.cfg.decoding)}")

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_citrinet_256",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_citrinet_256",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_citrinet_256/versions/1.0.0rc1/files/stt_en_citrinet_256.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_citrinet_512",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_citrinet_512",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_citrinet_512/versions/1.0.0rc1/files/stt_en_citrinet_512.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_citrinet_1024",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_citrinet_1024",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_citrinet_1024/versions/1.0.0rc1/files/stt_en_citrinet_1024.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_citrinet_256_gamma_0_25",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:\nemo:stt_en_citrinet_256_gamma_0_25",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_citrinet_256_gamma_0_25/versions/1.0.0/files/stt_en_citrinet_256_gamma_0_25.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_citrinet_512_gamma_0_25",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_citrinet_512_gamma_0_25",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_citrinet_512_gamma_0_25/versions/1.0.0/files/stt_en_citrinet_512_gamma_0_25.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_citrinet_1024_gamma_0_25",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_citrinet_1024_gamma_0_25",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_citrinet_1024_gamma_0_25/versions/1.0.0/files/stt_en_citrinet_1024_gamma_0_25.nemo",
        )

        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_es_citrinet_512",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_es_citrinet_512",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_es_citrinet_512/versions/1.0.0/files/stt_es_citrinet_512.nemo",
        )

        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_de_citrinet_1024",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_de_citrinet_1024",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_de_citrinet_1024/versions/1.5.0/files/stt_de_citrinet_1024.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_fr_citrinet_1024_gamma_0_25",
            description="For details about this model, please visit https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_fr_citrinet_1024_gamma_0_25",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_fr_citrinet_1024_gamma_0_25/versions/1.5/files/stt_fr_citrinet_1024_gamma_0_25.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_fr_no_hyphen_citrinet_1024_gamma_0_25",
            description="For details about this model, please visit https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_fr_citrinet_1024_gamma_0_25",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_fr_citrinet_1024_gamma_0_25/versions/1.5/files/stt_fr_no_hyphen_citrinet_1024_gamma_0_25.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_es_citrinet_1024_gamma_0_25",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_es_citrinet_1024_gamma_0_25",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_es_citrinet_1024_gamma_0_25/versions/1.8.0/files/stt_es_citrinet_1024_gamma_0_25.nemo",
        )

        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_ctc_small",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_ctc_small",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_ctc_small/versions/1.6.0/files/stt_en_conformer_ctc_small.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_ctc_medium",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_ctc_medium",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_ctc_medium/versions/1.6.0/files/stt_en_conformer_ctc_medium.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_ctc_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_ctc_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_ctc_large/versions/1.10.0/files/stt_en_conformer_ctc_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_ctc_xlarge",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_ctc_xlarge",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_ctc_xlarge/versions/1.10.0/files/stt_en_conformer_ctc_xlarge.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_squeezeformer_ctc_xsmall_ls",
            description="For details about this model, please visit https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_squeezeformer_ctc_xsmall_ls",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_squeezeformer_ctc_xsmall_ls/versions/1.13.0/files/stt_en_squeezeformer_ctc_xsmall_ls.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_squeezeformer_ctc_small_ls",
            description="For details about this model, please visit https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_squeezeformer_ctc_small_ls",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_squeezeformer_ctc_small_ls/versions/1.13.0/files/stt_en_squeezeformer_ctc_small_ls.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_squeezeformer_ctc_small_medium_ls",
            description="For details about this model, please visit https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_squeezeformer_ctc_small_medium_ls",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_squeezeformer_ctc_small_medium_ls/versions/1.13.0/files/stt_en_squeezeformer_ctc_small_medium_ls.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_squeezeformer_ctc_medium_ls",
            description="For details about this model, please visit https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_squeezeformer_ctc_medium_ls",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_squeezeformer_ctc_medium_ls/versions/1.13.0/files/stt_en_squeezeformer_ctc_medium_ls.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_squeezeformer_ctc_medium_large_ls",
            description="For details about this model, please visit https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_squeezeformer_ctc_medium_large_ls",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_squeezeformer_ctc_medium_large_ls/versions/1.13.0/files/stt_en_squeezeformer_ctc_medium_large_ls.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_squeezeformer_ctc_large_ls",
            description="For details about this model, please visit https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_squeezeformer_ctc_large_ls",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_squeezeformer_ctc_large_ls/versions/1.13.0/files/stt_en_squeezeformer_ctc_large_ls.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_ctc_small_ls",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_ctc_small_ls",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_ctc_small_ls/versions/1.0.0/files/stt_en_conformer_ctc_small_ls.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_ctc_medium_ls",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_ctc_medium_ls",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_ctc_medium_ls/versions/1.0.0/files/stt_en_conformer_ctc_medium_ls.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_ctc_large_ls",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_ctc_large_ls",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_ctc_large_ls/versions/1.0.0/files/stt_en_conformer_ctc_large_ls.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_fr_conformer_ctc_large",
            description="For details about this model, please visit https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_fr_conformer_ctc_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_fr_conformer_ctc_large/versions/1.5.1/files/stt_fr_conformer_ctc_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_fr_no_hyphen_conformer_ctc_large",
            description="For details about this model, please visit https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_fr_conformer_ctc_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_fr_conformer_ctc_large/versions/1.5.1/files/stt_fr_no_hyphen_conformer_ctc_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_de_conformer_ctc_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_de_conformer_ctc_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_de_conformer_ctc_large/versions/1.5.0/files/stt_de_conformer_ctc_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_es_conformer_ctc_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_es_conformer_ctc_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_es_conformer_ctc_large/versions/1.8.0/files/stt_es_conformer_ctc_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_hi_conformer_ctc_medium",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_hi_conformer_ctc_medium",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_hi_conformer_ctc_medium/versions/1.6.0/files/stt_hi_conformer_ctc_medium.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_mr_conformer_ctc_medium",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_mr_conformer_ctc_medium",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_mr_conformer_ctc_medium/versions/1.6.0/files/stt_mr_conformer_ctc_medium.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_enes_conformer_ctc_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_enes_conformer_ctc_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_enes_conformer_ctc_large/versions/1.0.0/files/stt_enes_conformer_ctc_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_ca_conformer_ctc_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_ca_conformer_ctc_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_ca_conformer_ctc_large/versions/1.11.0/files/stt_ca_conformer_ctc_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_rw_conformer_ctc_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_rw_conformer_ctc_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_rw_conformer_ctc_large/versions/1.11.0/files/stt_rw_conformer_ctc_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_enes_conformer_ctc_large_codesw",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_enes_conformer_ctc_large_codesw",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_enes_conformer_ctc_large_codesw/versions/1.0.0/files/stt_enes_conformer_ctc_large_codesw.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_be_conformer_ctc_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_be_conformer_ctc_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_be_conformer_ctc_large/versions/1.12.0/files/stt_be_conformer_ctc_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_hr_conformer_ctc_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_hr_conformer_ctc_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_hr_conformer_ctc_large/versions/1.11.0/files/stt_hr_conformer_ctc_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_it_conformer_ctc_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_it_conformer_ctc_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_it_conformer_ctc_large/versions/1.13.0/files/stt_it_conformer_ctc_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_ru_conformer_ctc_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_ru_conformer_ctc_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_ru_conformer_ctc_large/versions/1.13.0/files/stt_ru_conformer_ctc_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_eo_conformer_ctc_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_eo_conformer_ctc_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_eo_conformer_ctc_large/versions/1.14.0/files/stt_eo_conformer_ctc_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_fastconformer_ctc_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_ctc_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_ctc_large/versions/1.0.0/files/stt_en_fastconformer_ctc_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_fastconformer_ctc_large_ls",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_ctc_large_ls",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_ctc_large_ls/versions/1.0.0/files/stt_en_fastconformer_ctc_large_ls.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_fastconformer_ctc_xlarge",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_ctc_xlarge",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_ctc_xlarge/versions/1.20.0/files/stt_en_fastconformer_ctc_xlarge.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_fastconformer_ctc_xxlarge",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_ctc_xxlarge",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_ctc_xxlarge/versions/1.20.1/files/stt_en_fastconformer_ctc_xxlarge.nemo",
        )
        results.append(model)

        return results


class MSEncDecCTCModelBPE(EncDecCTCModelBPE):
    """
    A Multi-Speaker CTC BPE model that fuses Sortformer diarization output
    with ASR encoder output before feeding into the CTC decoder.

    The model extends EncDecCTCModelBPE by optionally loading a pretrained
    Sortformer diarization model and fusing its frame-level speaker predictions
    with the ASR encoder output. The fusion can be done via:
      - 'projection': concatenation + linear projection (default)
      - 'sinusoidal': sinusoidal position encoding weighted by speaker posteriors
      - 'metacat': outer-product based fusion + linear projection
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

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

            # Pre-encode level diarization fusion
            self.use_pre_encode_diar_fusion = cfg.get('use_pre_encode_diar_fusion', False)
            if self.use_pre_encode_diar_fusion:
                self._setup_pre_encode_diar_fusion()
                logging.info("[MSEncDecCTCModelBPE] use_pre_encode_diar_fusion=True")
            else:
                logging.info("[MSEncDecCTCModelBPE] use_pre_encode_diar_fusion=False")

        else:
            self.diar = False
            self.use_fc_layer_fusion = False
            self.use_pre_encode_diar_fusion = False

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        """
        Override parent's dataloader setup to use the SOT dataset with
        ground-truth RTTM speaker targets when lhotse is enabled.

        Falls back to the parent implementation for non-lhotse configs.
        """
        if config.get("use_lhotse"):
            from nemo.collections.asr.data.audio_to_sot_text_lhotse import (
                LhotseSpeechToTextBpeDataset as LhotseSotDataset,
            )

            if hasattr(self, 'encoder') and self.encoder is not None:
                subsampling_factor = getattr(self.encoder, 'subsampling_factor', 8)
            else:
                subsampling_factor = self.cfg.get('encoder', {}).get('subsampling_factor', 8)

            dataset_cfg = {
                'num_speakers': self.cfg.get('max_num_speakers', 4),
                'sample_rate': self.cfg.get('sample_rate', 16000),
                'window_stride': self.cfg.get('preprocessor', {}).get('window_stride', 0.01),
                'subsampling_factor': subsampling_factor,
            }

            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank if not config.get("do_transcribe", False) else config.get("global_rank"),
                world_size=self.world_size if not config.get("do_transcribe", False) else config.get("world_size"),
                dataset=LhotseSotDataset(
                    tokenizer=self.tokenizer,
                    cfg=DictConfig(dataset_cfg),
                    return_cuts=config.get("do_transcribe", False),
                ),
                tokenizer=self.tokenizer,
            )

        return super()._setup_dataloader_from_config(config)

    # ── Diarization model initialization ───────────────────────────────

    def _init_diar_model(self):
        """
        Initialize the Sortformer diarization model from a pretrained checkpoint.
        """
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
        them on ``diarization_model.sortformer_modules``.
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

        diar_total_history = sm.spkcache_len + sm.fifo_len
        asr_left_context = scfg.last_channel_cache_size
        if self.cfg.get('simulate_streaming', False):
            if diar_total_history != asr_left_context:
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

            if scfg is not None and hasattr(scfg, 'valid_out_len'):
                vol = scfg.valid_out_len
                if isinstance(vol, list):
                    vol = vol[-1]
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
            mapping = list(range(n_asr))
        self._fc_fusion_layer_map = mapping

        # Projection layer (learnable): diar d_model -> ASR d_model
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
                out = output[0] if isinstance(output, tuple) else output
                self._diar_layer_outputs[layer_idx] = out.detach()
        return hook

    def _fuse_fc_layer_embeddings(self, cache_last_channel):
        """
        Fuse diar encoder per-layer hidden states into ASR cache_last_channel.

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
                aligned = projected[:, -cache_size:, :]
            else:
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
        (1) Sinusoidal: speaker position encoding weighted by diar predictions
        (2) Merge: project diar encoder's pre-encode embeddings to ASR dimension
        """
        d_asr = self.encoder.d_model

        # (1) Sinusoidal kernel
        self.pre_encode_diar_kernel = self.get_sinusoid_position_encoding(
            self.max_num_speakers, d_asr
        )
        logging.info(
            f"[pre_encode_diar_fusion] Sinusoidal kernel setup: "
            f"max_speakers={self.max_num_speakers}, d_asr={d_asr}"
        )

        # (2) Merge projection
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

    def _apply_pre_encode_diar_fusion(self, audio_chunk, chunk_lengths, chunk_diar_preds, drop_extra, gt_diar_preds=None):
        """
        Fuse diarization information at the pre-encode embedding level.

        Args:
            audio_chunk: (B, D_mel, T) mel features with pre-encode cache prepended.
            chunk_lengths: (B,) per-sample lengths in mel frames.
            chunk_diar_preds: (B, T_chunk, num_speakers) diar predictions.
            drop_extra: int, number of extra pre-encoded frames to drop.
            gt_diar_preds: (B, T_chunk, num_speakers) ground-truth speaker
                targets from RTTM, or None.  When provided (training), used
                directly.  When None (inference), model predictions are
                binarized to match the 0/1 training condition.

        Returns:
            fused_pre_encode: (B, T_valid, d_asr) fused pre-encode embeddings.
            pre_encode_lengths: (B,) lengths after pre-encode and dropping.
        """
        audio_signal_t = audio_chunk.transpose(1, 2)  # (B, D_mel, T) -> (B, T, D_mel)

        # ASR pre-encode (gradient flows if ASR not frozen)
        asr_pre_encode, pre_encode_lengths = self.encoder.pre_encode(
            x=audio_signal_t, lengths=chunk_lengths
        )
        pre_encode_lengths = pre_encode_lengths.to(torch.int64)

        if drop_extra > 0:
            asr_pre_encode = asr_pre_encode[:, drop_extra:, :]
            pre_encode_lengths = (pre_encode_lengths - drop_extra).clamp(min=0)

        # Resolve diar preds: GT (training) or binarized model (inference)
        if gt_diar_preds is not None:
            dp = gt_diar_preds
        else:
            dp = (chunk_diar_preds > 0.5).float()
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

        # (2) Merge diar pre-encode projection
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

    # ── ASR model initialization ───────────────────────────────────────

    def _init_asr_model(self):
        """
        Initialize the ASR encoder from a pretrained EncDecCTCModelBPE.
        Only the encoder weights are loaded (decoder is kept as initialized).
        """
        model_path = self.cfg.asr_model_path

        if model_path is not None and model_path.endswith('.nemo'):
            pretrained_asr_model = EncDecCTCModelBPE.restore_from(model_path, map_location="cpu")
            logging.info("ASR Model restored locally from {}".format(model_path))
        elif model_path.endswith('.ckpt'):
            pretrained_asr_model = EncDecCTCModelBPE.load_from_checkpoint(model_path, map_location="cpu")
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
        Override to handle init_from_nemo_model for MSEncDecCTCModelBPE.

        The pretrained ASR model is an EncDecCTCModelBPE, not an MSEncDecCTCModelBPE,
        so we must use EncDecCTCModelBPE.restore_from() instead of self.restore_from()
        to avoid a class/config mismatch when restoring the checkpoint.
        """
        if 'init_from_nemo_model' in cfg and cfg.init_from_nemo_model is not None:
            with open_dict(cfg):
                if isinstance(cfg.init_from_nemo_model, str):
                    model_path = cfg.init_from_nemo_model
                    restored_model = EncDecCTCModelBPE.restore_from(
                        model_path, map_location=map_location, strict=cfg.get("init_strict", True)
                    )
                    self.load_state_dict(restored_model.state_dict(), strict=False)
                    logging.info(f'Model checkpoint restored from nemo file with path : `{model_path}`')
                    del restored_model

                elif isinstance(cfg.init_from_nemo_model, (DictConfig, dict)):
                    model_load_dict = cfg.init_from_nemo_model
                    for model_load_cfg in model_load_dict.values():
                        model_path = model_load_cfg.path
                        restored_model = EncDecCTCModelBPE.restore_from(
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
            super().maybe_init_from_pretrained_checkpoint(cfg, map_location)

    # ── Diarization helpers ────────────────────────────────────────────

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

    def compute_diar_f1(self, signal, signal_len, gt_diar_preds, gt_diar_len=None):
        """
        Compute frame-level diarization F1, precision, and recall by comparing
        the diar model's binarized sigmoid output against ground-truth RTTM
        targets.

        Args:
            signal: [B, T] raw audio signal.
            signal_len: [B] audio lengths.
            gt_diar_preds: [B, T_gt, num_speakers] binary GT speaker targets.
            gt_diar_len: [B] valid frame counts for GT targets, or None.

        Returns:
            dict with keys 'f1', 'precision', 'recall' (Python floats).
        """
        with torch.no_grad():
            diar_preds = self.forward_diar(signal, signal_len)
            diar_binary = (diar_preds > 0.5).float()

            T_pred = diar_binary.shape[1]
            T_gt = gt_diar_preds.shape[1]
            T_min = min(T_pred, T_gt)
            diar_binary = diar_binary[:, :T_min, :]
            gt_aligned = gt_diar_preds[:, :T_min, :].float()

            if gt_diar_len is not None:
                frame_idx = torch.arange(T_min, device=diar_binary.device).unsqueeze(0)
                valid_mask = (frame_idx < gt_diar_len.unsqueeze(1)).unsqueeze(-1)
                diar_binary = diar_binary * valid_mask
                gt_aligned = gt_aligned * valid_mask

            tp = (diar_binary * gt_aligned).sum()
            fp = (diar_binary * (1 - gt_aligned)).sum()
            fn = ((1 - diar_binary) * gt_aligned).sum()

        return {
            'f1': (2 * tp / (2 * tp + fp + fn + 1e-8)).item(),
            'precision': (tp / (tp + fp + 1e-8)).item(),
            'recall': (tp / (tp + fn + 1e-8)).item(),
        }

    def fuse_diar_preds(self, encoded, diar_preds, gt_diar_preds=None):
        """
        Fuse diarization predictions with ASR encoder states.

        Args:
            encoded: [B, D, T] encoder output
            diar_preds: [B, T_diar, num_speakers] frame-level speaker predictions
            gt_diar_preds: [B, T_gt, num_speakers] ground-truth speaker
                targets from RTTM, or None.  When provided (training), used
                directly.  When None (inference), model predictions are
                binarized to match the 0/1 training condition.

        Returns:
            encoded: [B, D, T] fused encoder output
        """
        # Convert encoder output from (B, D, T) to (B, T, D) for fusion
        asr_enc_states = encoded.permute(0, 2, 1)  # (B, T, D)

        # Resolve diar preds: GT (training) or binarized model (inference)
        if gt_diar_preds is not None:
            diar_preds = gt_diar_preds
        else:
            diar_preds = (diar_preds > 0.5).float()

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

        # Convert back to (B, D, T) for CTC decoder
        return enc_states.permute(0, 2, 1)

    # ── Forward ────────────────────────────────────────────────────────

    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        gt_diar_preds=None,
    ):
        """
        Forward pass of the model. Adds diarization fusion after the encoder
        so that the fused representation is used by the CTC decoder.

        Args:
            input_signal: Tensor [B, T] of raw audio signals.
            input_signal_length: Vector [B] of audio lengths.
            processed_signal: Tensor [B, D, T] of preprocessed audio (e.g. from DALI).
            processed_signal_length: Vector [B] of processed audio lengths.
            gt_diar_preds: Tensor [B, T_target, num_speakers] ground-truth
                speaker targets from RTTM, or None.  When provided (training),
                used in place of model diar predictions.

        Returns:
            A tuple of 3 elements:
            1) log_probs: Tensor [B, T, V] log probabilities from the CTC decoder.
            2) encoded_len: Tensor [B] lengths after encoder.
            3) greedy_predictions: Tensor [B, T] argmax predictions.
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

        # --- ASR encoder forward (with optional pre-encode diar fusion) ---
        if self.use_pre_encode_diar_fusion and diar_preds is not None:
            with torch.set_grad_enabled(not self.cfg.get('freeze_asr', False)):
                fused_pre_encode, pre_encode_lengths = self._apply_pre_encode_diar_fusion(
                    processed_signal, processed_signal_length, diar_preds, drop_extra=0,
                    gt_diar_preds=gt_diar_preds,
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

        # --- Post-encode diar fusion (always applied when diar is available) ---
        if diar_preds is not None:
            encoded = self.fuse_diar_preds(encoded, diar_preds, gt_diar_preds=gt_diar_preds)

        # --- CTC decoder ---
        log_probs = self.decoder(encoder_output=encoded)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)

        return log_probs, encoded_len, greedy_predictions

    # ── PTL-specific methods ───────────────────────────────────────────

    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        # Unpack batch: 4-element (legacy) or 6-element (with GT diar targets)
        if isinstance(batch, DALIOutputs):
            signal, signal_len, transcript, transcript_len = batch
            gt_diar_preds = None
        elif len(batch) == 6:
            signal, signal_len, transcript, transcript_len, gt_diar_preds, gt_diar_len = batch
        else:
            signal, signal_len, transcript, transcript_len = batch
            gt_diar_preds = None

        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, encoded_len, predictions = self.forward(
                processed_signal=signal, processed_signal_length=signal_len,
                gt_diar_preds=gt_diar_preds,
            )
        else:
            log_probs, encoded_len, predictions = self.forward(
                input_signal=signal, input_signal_length=signal_len,
                gt_diar_preds=gt_diar_preds,
            )

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
        else:
            log_every_n_steps = 1

        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )

        # Add auxiliary losses, if registered
        loss_value = self.add_auxiliary_losses(loss_value)
        # InterCTC losses
        loss_value, tensorboard_logs = self.add_interctc_losses(
            loss_value, transcript, transcript_len, compute_wer=((batch_nb + 1) % log_every_n_steps == 0)
        )

        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        tensorboard_logs.update(
            {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }
        )

        if (batch_nb + 1) % log_every_n_steps == 0:
            self.wer.update(
                predictions=log_probs,
                targets=transcript,
                targets_lengths=transcript_len,
                predictions_lengths=encoded_len,
            )
            wer, _, _ = self.wer.compute()
            self.wer.reset()
            tensorboard_logs.update({'training_batch_wer': wer})

        return {'loss': loss_value, 'log': tensorboard_logs}

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        # Unpack batch: 4-element (legacy) or 6-element (with GT diar targets)
        if isinstance(batch, DALIOutputs):
            signal, signal_len, transcript, transcript_len = batch
            gt_diar_preds = None
        elif len(batch) == 6:
            signal, signal_len, transcript, transcript_len, gt_diar_preds, gt_diar_len = batch
        else:
            signal, signal_len, transcript, transcript_len = batch
            gt_diar_preds = None

        # Use GT diar preds during validation (matches training condition).
        use_gt_diar_for_val = self.cfg.get('use_gt_diar_for_validation', True)
        val_gt = gt_diar_preds if (use_gt_diar_for_val and gt_diar_preds is not None) else None

        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, encoded_len, predictions = self.forward(
                processed_signal=signal, processed_signal_length=signal_len,
                gt_diar_preds=val_gt,
            )
        else:
            log_probs, encoded_len, predictions = self.forward(
                input_signal=signal, input_signal_length=signal_len,
                gt_diar_preds=val_gt,
            )

        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )

        loss_value, metrics = self.add_interctc_losses(
            loss_value,
            transcript,
            transcript_len,
            compute_wer=True,
            log_wer_num_denom=True,
            log_prefix="val_",
        )

        self.wer.update(
            predictions=log_probs,
            targets=transcript,
            targets_lengths=transcript_len,
            predictions_lengths=encoded_len,
        )
        wer, wer_num, wer_denom = self.wer.compute()
        self.wer.reset()
        metrics.update({'val_loss': loss_value, 'val_wer_num': wer_num, 'val_wer_denom': wer_denom, 'val_wer': wer})

        # Diarization F1: compare model diar preds against GT
        if self.diar and gt_diar_preds is not None and signal is not None:
            diar_metrics = self.compute_diar_f1(signal, signal_len, gt_diar_preds, gt_diar_len)
            metrics['val_diar_f1'] = diar_metrics['f1']
            metrics['val_diar_precision'] = diar_metrics['precision']
            metrics['val_diar_recall'] = diar_metrics['recall']
            if batch_idx == 0:
                logging.info(
                    f"[ VALIDATION STEP ]    Diar F1={diar_metrics['f1']:.4f} "
                    f"(P={diar_metrics['precision']:.4f}, R={diar_metrics['recall']:.4f})"
                )

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        return metrics

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

    def on_validation_epoch_start(self) -> None:
        """Disable CUDA graphs for validation in MSEncDecCTCModelBPE.
        The parent class force-enables CUDA graphs here, but CUDA graph
        capture is incompatible with the dynamic diarization fusion path.
        """
        self.disable_cuda_graphs()

    def on_test_epoch_start(self) -> None:
        """Disable CUDA graphs for testing in MSEncDecCTCModelBPE.
        Same reason as on_validation_epoch_start.
        """
        self.disable_cuda_graphs()
