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

from nemo.collections.nlp.modules.common.megatron.language_model import get_language_model
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

    def __init__(self, model_cfg,
                 pre_process=True, post_process=True):
        super(VitCLIPModel, self).__init__()

        scaled_init_method = (
            scaled_init_method_normal(model_cfg.init_method_std, model_cfg.num_layers)
            if model_cfg.use_scaled_init_method
            else init_method_normal(model_cfg.init_method_std)
        )

        self.hidden_size = model_cfg.hidden_size
        self.output_dim = model_cfg.output_dim
        self.global_average_pool = model_cfg.global_average_pool
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
            self.head = torch.nn.Linear(
                self.hidden_size,
                self.output_dim,
                bias=False,
            )

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.backbone.set_input_tensor(input_tensor)

    def forward(self, input):
        hidden_states = self.backbone(input)

        if self.post_process:
            hidden_states = self.head(hidden_states)
        # TODO (yuya): is this necessary?
        hidden_states = hidden_states.contiguous()
        return hidden_states


class TextTransformerCLIPModel(MegatronModule):
    """Text Transformer Model."""
    def __init__(self, model_cfg,
                 pre_process=True, post_process=True):
        super(TextTransformerCLIPModel, self).__init__()


        self.output_dim = model_cfg.output_dim
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = model_cfg.fp16_lm_cross_entropy
        self.sequence_parallel = model_cfg.sequence_parallel
        self.gradient_accumulation_fusion = model_cfg.gradient_accumulation_fusion

        kv_channels = model_cfg.get("kv_channels")
        if kv_channels is None:
            assert (
                hidden_size % num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = hidden_size // num_attention_heads

        scaled_init_method = (
            scaled_init_method_normal(model_cfg.init_method_std, model_cfg.num_layers)
            if model_cfg.use_scaled_init_method
            else init_method_normal(model_cfg.init_method_std)
        )
        self.language_model, self._language_model_key = get_language_model(
            vocab_size=model_cfg.vocab_size,
            hidden_size=model_cfg.hidden_size,
            hidden_dropout=model_cfg.hidden_dropout,
            num_tokentypes=0,
            max_position_embeddings=model_cfg.max_position_embeddings,
            num_layers=model_cfg.num_layers,
            num_attention_heads=model_cfg.num_attention_heads,
            apply_query_key_layer_scaling=model_cfg.apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            ffn_hidden_size=model_cfg.ffn_hidden_size,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.causal,
            init_method=init_method_normal(model_cfg.init_method_std),
            scaled_init_method=scaled_init_method,
            pre_process=self.pre_process,
            post_process=self.post_process,
            init_method_std=model_cfg.init_method_std,
            use_cpu_initialization=model_cfg.use_cpu_initialization,
            precision=model_cfg.precision,
            fp32_residual_connection=model_cfg.fp32_residual_connection,
            activations_checkpoint_granularity=model_cfg.activations_checkpoint_granularity,
            activations_checkpoint_method=model_cfg.activations_checkpoint_method,
            activations_checkpoint_num_layers=model_cfg.activations_checkpoint_num_layers,
            activations_checkpoint_layers_per_pipeline=model_cfg.activations_checkpoint_layers_per_pipeline,
            normalization=model_cfg.normalization,
            layernorm_epsilon=model_cfg.layernorm_epsilon,
            bias_activation_fusion=model_cfg.bias_activation_fusion,
            bias_dropout_add_fusion=model_cfg.bias_dropout_add_fusion,
            masked_softmax_fusion=model_cfg.masked_softmax_fusion,
            gradient_accumulation_fusion=model_cfg.gradient_accumulation_fusion,
            persist_layer_norm=model_cfg.persist_layer_norm,
            openai_gelu=model_cfg.openai_gelu,
            onnx_safe=model_cfg.onnx_safe,
            sequence_parallel=model_cfg.sequence_parallel,
            transformer_engine=model_cfg.transformer_engine,
            fp8=model_cfg.fp8,
            fp8_e4m3=model_cfg.fp8_e4m3,
            fp8_hybrid=model_cfg.fp8_hybrid,
            fp8_margin=model_cfg.fp8_margin,
            fp8_interval=model_cfg.fp8_interval,
            fp8_amax_history_len=model_cfg.fp8_amax_history_len,
            fp8_amax_compute_algo=model_cfg.fp8_amax_compute_algo,
            use_emha=model_cfg.use_emha,
        )

        self.initialize_word_embeddings(
            init_method=init_method_normal(init_method_std), vocab_size=vocab_size, hidden_size=hidden_size
        )

        if self.post_process:
            self.head = torch.nn.Linear(
                self.hidden_size,
                self.output_dim,
                bias=False,
            )

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        token_type_ids=None,
        layer_past=None,
        get_key_value=False,
        encoder_input=None,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
        checkpoint_activations_all_layers=None,
    ):
        # input_ids: [b, s]
        # position_ids: [b, s]
        # attention_mask: [1, 1, s, s]

        lm_output = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            token_type_ids=token_type_ids,
            layer_past=layer_past,
            get_key_value=get_key_value,
            encoder_input=encoder_input,
            set_inference_key_value_memory=set_inference_key_value_memory,
            inference_max_sequence_len=inference_max_sequence_len,
            checkpoint_activations_all_layers=checkpoint_activations_all_layers,
        )

        if self.post_process:
            return self.head(lm_output)

        return lm_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):

        state_dict_ = {}
        state_dict_[self._language_model_key] = self.language_model.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars
        )
        # Save word_embeddings.
        # TODO (yuya): check if this is necessary
        # if self.post_process and not self.pre_process:
        #     state_dict_[self._word_embeddings_for_head_key] = self.word_embeddings.state_dict(
        #         destination, prefix, keep_vars
        #     )
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Load word_embeddings.
        # if self.post_process and not self.pre_process:
        #     self.word_embeddings.load_state_dict(state_dict[self._word_embeddings_for_head_key], strict=strict)
        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        self.language_model.load_state_dict(state_dict, strict=strict)
