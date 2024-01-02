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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import gelu

from nemo.collections.asr.modules.transformer.transformer_modules import MultiHeadAttention, PositionWiseFF
from nemo.collections.common.parts import form_attention_mask
import math

import numpy as np
from nemo.utils import logging

__all__ = ["SortformerEncoder"]




class MultiHeadAttentionWithScores(nn.Module):
    """
    Multi-head scaled dot-product attention layer.

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        num_attention_heads: number of heads in multi-head attention
        attn_score_dropout: probability of dropout applied to attention scores
        attn_layer_dropout: probability of dropout applied to the output of the
            whole layer, but before layer normalization
    """

    def __init__(self, hidden_size, num_attention_heads, attn_score_dropout=0.0, attn_layer_dropout=0.0):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number "
                "of attention heads (%d)" % (hidden_size, num_attention_heads)
            )
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attn_head_size = int(hidden_size / num_attention_heads)
        self.attn_scale = math.sqrt(math.sqrt(self.attn_head_size))

        self.query_net = nn.Linear(hidden_size, hidden_size)
        self.key_net = nn.Linear(hidden_size, hidden_size)
        self.value_net = nn.Linear(hidden_size, hidden_size)
        self.out_projection = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(attn_score_dropout)
        self.layer_dropout = nn.Dropout(attn_layer_dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attn_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, queries, keys, values, attention_mask):

        # attention_mask is needed to hide the tokens which correspond to [PAD]
        # in the case of BERT, or to hide the future tokens in the case of
        # vanilla language modeling and translation
        query = self.query_net(queries)
        key = self.key_net(keys)
        value = self.value_net(values)
        query = self.transpose_for_scores(query) / self.attn_scale
        key = self.transpose_for_scores(key) / self.attn_scale
        value = self.transpose_for_scores(value)

        # for numerical stability we pre-divide query and key by sqrt(sqrt(d))
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask.to(attention_scores.dtype)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)

        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.hidden_size,)
        context = context.view(*new_context_shape)

        # output projection
        output_states = self.out_projection(context)
        output_states = self.layer_dropout(output_states)
        return output_states, attention_scores

class PositionWiseEmbeddingFF(nn.Module):
    """
    Position-wise feed-forward network of Transformer block.

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        inner_size: number of neurons in the intermediate part of feed-forward
            net, usually is (4-8 x hidden_size) in the papers
        output_size: size of the output embeddings, usually is equal to hidden_size
        ffn_dropout: probability of dropout applied to net output
        hidden_act: activation function used between two linear layers
    """

    def __init__(self, hidden_size, inner_size, output_size, ffn_dropout=0.0, hidden_act="relu"):
        super().__init__()
        self.dense_in = nn.Linear(hidden_size, inner_size)
        self.dense_out = nn.Linear(inner_size, output_size)
        self.layer_dropout = nn.Dropout(ffn_dropout)
        ACT2FN = {"gelu": gelu, "relu": torch.relu}
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_states):
        output_states = self.dense_in(hidden_states)
        output_states = self.act_fn(output_states)
        output_states = self.dense_out(output_states)
        output_states = self.layer_dropout(output_states)
        return output_states

class SortformerEncoderBlock(nn.Module):
    """
    Building block of Sortformer encoder.

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        inner_size: number of neurons in the intermediate part of feed-forward
            net, usually is (4-8 x hidden_size) in the papers
        num_attention_heads: number of heads in multi-head attention
        attn_score_dropout: probability of dropout applied to attention scores
        attn_layer_dropout: probability of dropout applied to the output of the
            attention layers, but before layer normalization
        ffn_dropout: probability of dropout applied to FFN output
        hidden_act: activation function used between two linear layers in FFN
    """

    def __init__(
        self,
        hidden_size: int,
        inner_size: int,
        num_attention_heads: int = 1,
        attn_score_dropout: float = 0.0,
        attn_layer_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        hidden_act: str = "relu",
        pre_ln: bool = False,
        unit_dim: int = 4,
        sort_layer_on: bool = True,
        seq_var_sort: bool = False,
        sort_bin_order: bool = False,
        layer_arrival_time_sort: bool = False,
        num_classes: int = 4,
        detach_preds: bool = False,
        sort_layer_type: str = 'parallel',
    ):
        super().__init__()
        self.pre_ln = pre_ln
        self.sort_bin_order = sort_bin_order
        self.sort_layer_type = sort_layer_type
        self.sort_layer_on = sort_layer_on
        self.seq_var_sort = seq_var_sort
        self.num_classes = num_classes
        self.detach_preds = detach_preds
        if self.sort_layer_on:
            self.unit_dim = unit_dim
        else:
            self.unit_dim = hidden_size
        self.layer_norm_1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.first_sub_layer = MultiHeadAttentionWithScores(
            hidden_size, num_attention_heads, attn_score_dropout, attn_layer_dropout
        )
        self.layer_norm_2 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.second_sub_layer = PositionWiseFF(hidden_size, inner_size, ffn_dropout, hidden_act)
        self.third_sub_layer = PositionWiseEmbeddingFF(self.num_classes, inner_size, hidden_size, ffn_dropout, hidden_act)
        self.layer_arrival_time_sort = layer_arrival_time_sort
        self.hidden_to_spks = nn.Linear(hidden_size, self.num_classes)
        self.unit_emb_to_hidden = nn.Linear(self.unit_dim, hidden_size)
        self.dropout = nn.Dropout(ffn_dropout)
        self.sort_layer_type = sort_layer_type

    def forward_speaker_sigmoids(self, unit_emb_out):
        unit_emb_out = self.dropout(F.sigmoid(unit_emb_out))
        hidden_out = self.unit_emb_to_hidden(unit_emb_out)
        hidden_out = self.dropout(F.relu(hidden_out))
        spk_preds = self.hidden_to_spks(hidden_out)
        preds = nn.Sigmoid()(spk_preds)
        return preds    
        
    def forward_preln(self, encoder_query, encoder_mask, encoder_keys):
        """
        Pre-LayerNorm block
        Order of operations: LN -> Self-Attn -> Residual -> LN -> Cross-Attn -> Residual -> LN -> FFN
        """
        residual = encoder_query
        encoder_query = self.layer_norm_1(encoder_query)
        encoder_keys = self.layer_norm_1(encoder_keys)
        self_attn_output, attn_scores = self.first_sub_layer(encoder_query, encoder_keys, encoder_keys, encoder_mask)
        self_attn_output += residual

        residual = self_attn_output
        self_attn_output = self.layer_norm_2(self_attn_output)
        output_states = self.second_sub_layer(self_attn_output)
        output_states += residual

        return output_states
    
    def var_sort_pooling(self, output_states):
        bs, seq_len, emb_dim = output_states.shape
        sorted_output_states_list = []
        per_sample_vars = torch.var(output_states, dim=1)
        sort_indices = torch.sort(per_sample_vars, descending=True)[1]
        for i in range(bs):
            sorted_output_states_list.append(output_states[i, :, sort_indices[i]].unsqueeze(0))
        # offset_vals = torch.linspace(start=0, end=((bs-1)*emb_dim), steps=bs).int()
        # offset_vals_tile = torch.tile(offset_vals, (emb_dim, 1)).t() # (bs, emb_dim)
        # offset_added_sort_indices = sort_indices + offset_vals_tile.to(sort_indices.device).detach()
        # flatten_sort_indices = offset_added_sort_indices.flatten(0,1) 
        # flatten_output_states = output_states.transpose(0, 1).reshape((seq_len, bs*emb_dim))
        # # sorted_output_states = flatten_output_states[:, flatten_sort_indices].reshape_as(output_states) 
        # reshaped_before_transpose = flatten_output_states[:, flatten_sort_indices].reshape((seq_len, bs, emb_dim))
        # _sorted_output_states = flatten_output_states[:, flatten_sort_indices]
        sorted_output_states = torch.concat(sorted_output_states_list)
        return sorted_output_states
    
    def p2_norm(self, sorted_output_states):
        # import ipdb; ipdb.set_trace()
        return torch.nn.functional.normalize(sorted_output_states, p=2, dim=-1)

    def arrival_time_sort(self, sorted_output_states):
        eps = 1e-5
        # max_normed_states = sorted_output_states/torch.abs(sorted_output_states).max()
        # softmax_states = torch.softmax(max_normed_states+eps, dim=-1)
        ats_sorted_softmax_states = self.sort_probs_and_labels(sorted_output_states, discrete=False)
        # binary_states = max_normed_states.sign()
        # bs, seq_dim, short_dim = sorted_output_states.shape
        # offsets = torch.linspace(1, seq_dim, steps=seq_dim).to(log_sm.device).unsqueeze(0).unsqueeze(2).repeat((bs,1,short_dim))
        # offset_log_sm = log_sm + offsets
        return ats_sorted_output_states
   
    def find_first_nonzero(self, mat, max_cap_val=-1):
        non_zero_mask = mat != 0
        mask_max_values, mask_max_indices = torch.max(non_zero_mask, dim=1)
        mask_max_indices[mask_max_values == 0] = max_cap_val
        return mask_max_indices 
    
    def __forward_postln(self, encoder_query, encoder_mask, encoder_keys):
        """
        Post-LayerNorm block
        Order of operations: Self-Attn -> Residual -> LN -> Cross-Attn -> Residual -> LN -> FFN -> Residual -> LN
        """
        self_attn_output = self.first_sub_layer(encoder_query, encoder_keys, encoder_keys, encoder_mask)
        self_attn_output += encoder_query
        self_attn_output = self.layer_norm_1(self_attn_output)

        output_states = self.second_sub_layer(self_attn_output)
        output_states += self_attn_output
        output_states = self.layer_norm_2(output_states)

        return output_states 
    
    # def __sort_probs_and_labels(self, labels, discrete=True, thres=0.5):
    #     """
    #     Sorts probs and labels in descending order of signal_lengths.
    #     """
    #     max_cap_val = labels.shape[1] + 1 
    #     if not discrete:
    #         labels_discrete = torch.zeros_like(labels).to(labels.device)
    #         # mean = torch.mean(labels, dim=(1,2)).detach()
    #         # median_repeat = mean.unsqueeze(1).unsqueeze(1).repeat(1, labels.shape[1], labels.shape[2])
    #         # thres = 0.5
    #         # thres = torch.mean(labels, dim=(1,2)).detach()
    #         thres = torch.mean(labels).detach()
    #         labels_discrete[labels > thres] = 1
    #         # labels = labels_discrete
    #     else:
    #         labels_discrete = labels
        
    #     label_fz = self.find_first_nonzero(labels_discrete, max_cap_val)
    #     label_fz[label_fz == -1] = max_cap_val 
    #     sorted_inds = torch.sort(label_fz)[1]
    #     sorted_labels = labels.transpose(0,1)[:, torch.arange(labels.shape[0]).unsqueeze(1), sorted_inds].transpose(0, 1)
    #     # verify_sorted_labels = self.find_first_nonzero(sorted_labels, max_cap_val)
    #     return sorted_labels
    
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
            ax1 = torch.arange(labels_discrete.size(0)).unsqueeze(1)
            ax2 = torch.arange(labels_discrete.size(1)).unsqueeze(1)
            labels_discrete[ax1, ax2, max_inds[ax1, ax2]] = 1
            labels_discrete[labels <= thres] = 0
        else:
            labels_discrete = labels
        
        label_fz = self.find_first_nonzero(labels_discrete, max_cap_val)
        label_fz[label_fz == -1] = max_cap_val 
        sorted_inds = torch.sort(label_fz)[1]
        sorted_labels = labels.transpose(0,1)[:, torch.arange(labels.shape[0]).unsqueeze(1), sorted_inds].transpose(0, 1)
        return sorted_labels 
    
    def forward_postln(self, encoder_query, encoder_mask, encoder_keys):
        """
        Post-LayerNorm block
        Order of operations: Self-Attn -> Residual -> LN -> Cross-Attn -> Residual -> LN -> FFN -> Residual -> LN
        """
        self_attn_output, attn_score_mat = self.first_sub_layer(encoder_query, encoder_keys, encoder_keys, encoder_mask)
        self_attn_output += encoder_query
        self_attn_output = self.layer_norm_1(self_attn_output)

        output_states = self.second_sub_layer(self_attn_output)
        output_states_2nd = output_states + self_attn_output
        if self.sort_layer_on:
            if self.sort_bin_order and self.seq_var_sort:
                sorted_output_states = self.var_sort_pooling(output_states_2nd)
            else:
                sorted_output_states = output_states_2nd
            sorted_output_states_short = sorted_output_states[:, :, :self.unit_dim]
            if self.sort_bin_order:
                sorted_output_states_short = self.p2_norm(sorted_output_states_short)
                preds = self.forward_speaker_sigmoids(sorted_output_states_short)
                preds = self.sort_probs_and_labels(preds, discrete=False)
            else:
                preds = self.forward_speaker_sigmoids(sorted_output_states_short)
            if self.detach_preds: 
                preds = preds.detach()
            output_states = self.third_sub_layer(preds) + self.p2_norm(output_states_2nd)
        else:
            output_states = output_states_2nd
        # output_states = output_states_2nd + output_states_3rd
        # output_states = self.third_sub_layer(sorted_output_states_short) + output_states_2nd
        output_states = self.layer_norm_2(output_states)
        return output_states, attn_score_mat, preds
    
    def forward_and_sort_replace(self, encoder_query, encoder_mask, encoder_keys):
        """
        Post-LayerNorm block
        Order of operations: Self-Attn -> Residual -> LN -> Cross-Attn -> Residual -> LN -> FFN -> Residual -> LN
        """
        self_attn_output, attn_score_mat = self.first_sub_layer(encoder_query, encoder_keys, encoder_keys, encoder_mask)
        self_attn_output += encoder_query

        self_attn_output = self.layer_norm_1(self_attn_output)
        if self.sort_layer_on:
            if self.sort_bin_order and self.seq_var_sort:
                sorted_output_states = self.var_sort_pooling(self_attn_output)
            else:
                sorted_output_states = self_attn_output
            sorted_output_states_short = sorted_output_states[:, :, :self.unit_dim]
            if self.sort_bin_order:
                sorted_output_states_short = self.p2_norm(sorted_output_states_short)
                preds = self.forward_speaker_sigmoids(sorted_output_states_short)
                preds = self.sort_probs_and_labels(preds, discrete=False)
            else:
                preds = self.forward_speaker_sigmoids(sorted_output_states_short)
            if self.detach_preds: 
                preds = preds.detach()
            output_states = self.third_sub_layer(preds) + self.p2_norm(self_attn_output) 
        else:
            output_states = self.second_sub_layer(self_attn_output)
        output_states += self_attn_output
        output_states = self.layer_norm_2(output_states)
        return output_states, attn_score_mat, preds
    
    def forward_sort(self, encoder_query, encoder_mask, encoder_keys):
        """
        Post-LayerNorm block
        Order of operations: Self-Attn -> Residual -> LN -> Cross-Attn -> Residual -> LN -> FFN -> Residual -> LN
        """
        self_attn_output, attn_score_mat = self.first_sub_layer(encoder_query, encoder_keys, encoder_keys, encoder_mask)
        self_attn_output += encoder_query

        self_attn_output = self.layer_norm_1(self_attn_output)
        if self.sort_layer_on:
            if self.sort_layer_type == 'serial':
                self_attn_output = self.second_sub_layer(self_attn_output)
            if self.sort_bin_order and self.seq_var_sort:
                sorted_output_states = self.var_sort_pooling(self_attn_output)
            else:
                sorted_output_states = self_attn_output
            sorted_output_states_short = sorted_output_states[:, :, :self.unit_dim]
            if self.sort_bin_order:
                sorted_output_states_short = self.p2_norm(sorted_output_states_short)
                preds = self.forward_speaker_sigmoids(sorted_output_states_short)
                preds = self.sort_probs_and_labels(preds, discrete=False)
            else:
                preds = self.forward_speaker_sigmoids(sorted_output_states_short)
            if self.detach_preds: 
                preds = preds.detach()
            if self.sort_layer_type == 'parallel':
                output_states = self.p2_norm(self.third_sub_layer(preds)) + self.p2_norm(self.second_sub_layer(self_attn_output))
            elif self.sort_layer_type == 'replace':
                output_states = self.p2_norm(self.third_sub_layer(preds))
            else:
                raise NotImplementedError
        else:
            output_states = self.second_sub_layer(self_attn_output)
        output_states += self_attn_output
        output_states = self.layer_norm_2(output_states)
        return output_states, attn_score_mat, preds
 
    def forward(self, encoder_query, encoder_mask, encoder_keys):
        # if self.pre_ln:
        #     return self.forward_preln(encoder_query, encoder_mask, encoder_keys)
        # else:
        #     return self.forward_postln(encoder_query, encoder_mask, encoder_keys)
        if self.sort_layer_type in ['parallel', 'serial','replace']:
            return self.forward_sort(encoder_query, encoder_mask, encoder_keys)
        else:
            return self.forward_and_sort_replace(encoder_query, encoder_mask, encoder_keys)


class SortformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        inner_size: int,
        mask_future: bool = False,
        num_attention_heads: int = 1,
        attn_score_dropout: float = 0.0,
        attn_layer_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        hidden_act: str = "relu",
        pre_ln: bool = False,
        pre_ln_final_layer_norm: bool = True,
        unit_dim: int = 98,
        sort_layer_on: bool = True,
        seq_var_sort: bool = False,
        sort_bin_order: bool = False,
        layer_arrival_time_sort: bool = False,
        num_classes: int = 4,
        detach_preds: bool = False,
        sort_layer_type: str = 'parallel',
    ):
        super().__init__()

        if pre_ln and pre_ln_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)
        else:
            self.final_layer_norm = None

        layer = SortformerEncoderBlock(
            hidden_size,
            inner_size,
            num_attention_heads,
            attn_score_dropout,
            attn_layer_dropout,
            ffn_dropout,
            hidden_act,
            pre_ln,
            unit_dim,
            sort_layer_on,
            seq_var_sort,
            sort_bin_order,
            layer_arrival_time_sort,
            num_classes,
            detach_preds,
            sort_layer_type,
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.diag = 0 if mask_future else None
        self.sort_bin_order = sort_bin_order

    def _get_memory_states(self, encoder_states, encoder_mems_list=None, i=0):
        if encoder_mems_list is not None:
            memory_states = torch.cat((encoder_mems_list[i], encoder_states), dim=1)
        else:
            memory_states = encoder_states
        return memory_states

    def forward(self, encoder_states, encoder_mask, encoder_mems_list=None, return_mems=False):
        """
        Args:
            encoder_states: output of the embedding_layer (B x L_enc x H)
            encoder_mask: encoder inputs mask (B x L_enc)
            encoder_mems_list: list of the cached encoder hidden states
                for fast autoregressive generation which will be used instead
                of encoder_states as keys and values if not None
            return_mems: bool, whether to return outputs of all encoder layers
                or the last layer only
        """

        encoder_attn_mask = form_attention_mask(encoder_mask, self.diag)

        memory_states = self._get_memory_states(encoder_states, encoder_mems_list, 0)
        cached_mems_list = [memory_states]
        attn_score_mat_list = []
        encoder_states_list = []
        preds_list = []
        
        for i, layer in enumerate(self.layers):
            encoder_states, attn_score_mat, preds = layer(encoder_states, encoder_attn_mask, memory_states)
            attn_score_mat_list.append(attn_score_mat)
            preds_list.append(preds)
            memory_states = self._get_memory_states(encoder_states, encoder_mems_list, i + 1)
            cached_mems_list.append(memory_states)
            encoder_states_list.append(encoder_states)
            
        preds_layers = torch.stack(preds_list, dim=0)
        preds_mean = torch.mean(preds_layers, dim=0)
        
        if self.final_layer_norm is not None:
            encoder_states = self.final_layer_norm(encoder_states)
            memory_states = self._get_memory_states(encoder_states, encoder_mems_list, i + 1)
            cached_mems_list.append(memory_states)

        if return_mems:
            return cached_mems_list, attn_score_mat_list, encoder_states_list, preds_mean
        else:
            return cached_mems_list[-1], attn_score_mat_list, encoder_states_list, preds_mean
