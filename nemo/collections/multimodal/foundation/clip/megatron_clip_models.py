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

import itertools
from typing import Any, List, Optional, Union
from functools import partial

import numpy as np
import torch
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.vision.data.megatron.vit_dataset import build_train_valid_datasets
from nemo.collections.vision.data.megatron.data_samplers import MegatronVisionPretrainingRandomBatchSampler
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
    MegatronPretrainingRandomBatchSampler,
)

from nemo.collections.vision.models.vision_base_model import MegatronVisionModel
from nemo.collections.vision.modules.vit.vit_backbone import VitBackbone, VitMlpHead

from nemo.collections.nlp.modules.common.megatron.module import (
    MegatronModule,
    Float16Module,
)

from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    get_linear_layer,
    init_method_normal,
    parallel_lm_logits,
    scaled_init_method_normal,
    average_losses_across_data_parallel_group,
    get_all_params_for_weight_decay_optimization,
    get_params_for_weight_decay_optimization,
)

from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.utils import logging
from nemo.core.classes.common import PretrainedModelInfo

try:
    from apex.transformer import parallel_state
    from apex.transformer.pipeline_parallel.schedules.common import build_model
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import (
        forward_backward_pipelining_without_interleaving,
    )
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_with_interleaving import (
        _forward_backward_pipelining_with_interleaving,
    )
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_no_pipelining import forward_backward_no_pipelining

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


class VitCLIPModel(MegatronModule):
    """Vision Transformer Model."""

    def __init__(self, model_cfg, output_dim,
                 pre_process=True, post_process=True):
        super(VitCLIPModel, self).__init__()

        scaled_init_method = (
            scaled_init_method_normal(model_cfg.init_method_std, model_cfg.num_layers)
            if model_cfg.use_scaled_init_method
            else init_method_normal(model_cfg.init_method_std)
        )

        self.hidden_size = model_cfg.hidden_size
        self.output_dim = output_dim
        self.pre_process = pre_process
        self.post_process = post_process
        self.backbone = VitBackbone(
            model_cfg,
            init_method=init_method_normal(model_cfg.init_method_std),
            scaled_init_method=scaled_init_method,
            pre_process=self.pre_process,
            post_process=self.post_process,
            class_token=False,
            single_token_output=False,
        )

        if self.post_process:
            # TODO(yuya): CLIP's ViT Head doesn't have bias
            self.head = get_linear_layer(
                self.hidden_size,
                self.output_dim,
                torch.nn.init.zeros_
            )

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.backbone.set_input_tensor(input_tensor)

    def forward(self, input):
        hidden_states = self.backbone(input)

        if self.post_process:
            hidden_states = self.head(hidden_states)
        hidden_states = hidden_states.contiguous()
        return hidden_states
