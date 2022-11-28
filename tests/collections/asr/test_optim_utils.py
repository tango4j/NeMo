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


import numpy as np
import pytest
import torch

from nemo.collections.asr.parts.utils.optimization_utils import (
    linear_sum_assignment,
)



class TestDiarizationUtilFunctions:
    """
    Tests diarization and speaker-task related utils.
    Test functions include:
        - Segment interval merging function
        - Embedding merging
    """

    @pytest.mark.unit
    def test_combine_float_overlaps(self):
        intervals = [[0.25, 1.7], [1.5, 3.0], [2.8, 5.0], [5.5, 10.0]]
        target = [[0.25, 5.0], [5.5, 10.0]]
        merged = combine_float_overlaps(intervals)
        assert check_range_values(target, merged)

    @pytest.mark.unit
    def test_combine_int_overlaps(self):
        intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
        target = [[1, 6], [8, 10], [15, 18]]
        merged = combine_int_overlaps(intervals)
        assert check_range_values(target, merged)

    @pytest.mark.unit
    def test_combine_int_overlaps_edge(self):
        intervals = [[1, 4], [4, 5]]
        target = [[1, 5]]
        merged = combine_int_overlaps(intervals)
        assert check_range_values(target, merged)

    def test_embedding_merge(self):
        # TODO
        pass


class TestSpeakerClustering:
    """
    Test speaker clustering module
    Test functions include:
        - script module export
        - speaker counting feature
    """

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_lap_basics(self):

