# ! /usr/bin/python
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

import torch

from nemo.core.classes import Loss, Typing, typecheck
from nemo.core.neural_types import LabelsType, LengthsType, LossType, NeuralType, ProbsType
from nemo.collections.asr.parts.utils.offline_clustering import (
    getCosAffinityMatrix,
    cos_similarity,
    cos_similarity_batch,
    ScalerMinMax,
    ScalerMinMaxBatch,
    estimate_num_of_speakers_batch,
    get_binarized_batch_affinity_mat,
    batch_speaker_count,
)

__all__ = ['BCELoss']


class BCELoss(Loss, Typing):
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

    def __init__(self, reduction='sum', alpha=1.0, weight=torch.tensor([0.5, 0.5])):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = weight
        self.loss_f = torch.nn.BCELoss(weight=self.loss_weight, reduction=self.reduction)

    @typecheck()
    def forward(self, probs, labels, signal_lengths):
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

        return self.loss_f(probs, labels)

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

    def __init__(self, reduction='sum', alpha=1.0, weight=torch.tensor([0.5, 0.5])):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = weight

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
        affinity_loss = torch.abs(gt_affinity - batch_affinity_mat).sum()
        return affinity_loss
