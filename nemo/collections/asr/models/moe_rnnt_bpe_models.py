# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
RNNT BPE model with Mixture-of-Experts (MoE) encoder auxiliary loss support.

Extends EncDecRNNTBPEModel to collect and add the MoE load-balancing auxiliary
loss from MoEConformerEncoder during training.
"""

import torch

from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.utils import logging

__all__ = ['EncDecMoERNNTBPEModel']


class EncDecMoERNNTBPEModel(EncDecRNNTBPEModel):
    """Encoder-Decoder RNNT BPE model with MoE encoder auxiliary loss support.

    Inherits all functionality from :class:`EncDecRNNTBPEModel` and overrides
    :meth:`add_auxiliary_losses` to collect the MoE load-balancing loss from the
    encoder during training.

    Use this model class with :class:`MoEConformerEncoder` to train RNNT models
    with MoE feed-forward layers and the auxiliary load-balancing loss.
    """

    def add_auxiliary_losses(self, loss: torch.Tensor, reset_registry: bool = False) -> torch.Tensor:
        """Add auxiliary losses including MoE load-balancing loss.

        Args:
            loss: The primary loss value.
            reset_registry: Whether to reset the AccessMixin registry.

        Returns:
            Loss tensor with auxiliary losses added.
        """
        # Add standard auxiliary losses (adapter losses, etc.)
        loss = super().add_auxiliary_losses(loss, reset_registry=reset_registry)

        # Add MoE auxiliary load-balancing loss
        if hasattr(self.encoder, 'get_moe_auxiliary_loss'):
            moe_loss = self.encoder.get_moe_auxiliary_loss()
            if moe_loss is not None and moe_loss.requires_grad:
                loss = loss + moe_loss
                self.log('moe_aux_loss', moe_loss.detach())

        return loss
