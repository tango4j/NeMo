import os
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


import torch


class FeatureProcessor(ABC):

    @abstractmethod
    def process(self, training_example: dict) -> None:
        """
        Process the input training example dictionary.

        Args:
            training_example: training example dictionary.
        """
        raise NotImplementedError


class FeatureScaler(FeatureProcessor):

    def __init__(self, field: str, add_value: float = 0.0, div_value: float = 1.0):
        self.field = field
        self.add_value = add_value
        self.div_value = div_value

    def process(self, training_example: dict) -> None:
        feature = training_example[self.field]
        feature = (feature + self.add_value) / self.div_value
        training_example[self.field] = feature


class LogCompression(FeatureProcessor):

    def __init__(self, field: str, log_zero_guard_type: str = "add", log_zero_guard_value: float = 1.0):
        self.field = field

        if log_zero_guard_type == "add":
            self.guard_fn = self._add_guard
        elif log_zero_guard_type == "clamp":
            self.guard_fn = self._clamp_guard
        else:
            raise ValueError(f"Unsupported log zero guard type: '{log_zero_guard_type}'")

        self.guard_type = log_zero_guard_type
        self.guard_value = log_zero_guard_value

    def _add_guard(self, feature: torch.Tensor):
        return feature + self.guard_value

    def _clamp_guard(self, feature: torch.Tensor):
        return torch.clamp(feature, min=self.guard_value)

    def process(self, training_example: dict) -> None:
        feature = training_example[self.field]

        feature = self.guard_fn(feature)
        feature = torch.log(feature)

        training_example[self.field] = feature


class MeanVarianceNormalization(FeatureProcessor):

    def __init__(self, field: str, stats_path: Path, mask_field: Optional[str] = "voiced_mask"):
        self.field = field
        self.mask_field = mask_field

        if not os.path.exists(stats_path):
            raise ValueError(f"Statistics file does not exist: {stats_path}")

        with open(stats_path, 'r', encoding="utf-8") as pitch_f:
            stats_dict = json.load(pitch_f)
            self.mean = stats_dict["default"][f"{self.field}_mean"]
            self.std = stats_dict["default"][f"{self.field}_std"]

    def process(self, training_example: dict) -> None:
        feature = training_example[self.field]

        feature = (feature - self.mean) / self.std
        if self.mask_field:
            voiced_mask = training_example[self.mask_field]
            feature[~voiced_mask] = 0.0

        training_example[self.field] = feature


class MeanVarianceSpeakerNormalization(FeatureProcessor):

    def __init__(
        self,
        field: str,
        stats_path: Path,
        speaker_field: str = "speaker",
        mask_field: Optional[str] = "voiced_mask",
        fallback_to_default: bool = False
    ):
        self.field = field
        self.key_mean = f"{self.field}_mean"
        self.key_std = f"{self.field}_std"
        self.speaker_field = speaker_field
        self.mask_field = mask_field
        self.fallback_to_default = fallback_to_default

        if not os.path.exists(stats_path):
            raise ValueError(f"Statistics file does not exist: {stats_path}")

        with open(stats_path, 'r', encoding="utf-8") as pitch_f:
            self.stats_dict = json.load(pitch_f)

    def process(self, training_example: dict) -> None:
        feature = training_example[self.field]

        speaker = training_example[self.speaker_field]
        if speaker in self.stats_dict:
            stats = self.stats_dict[speaker]
        elif self.fallback_to_default:
            stats = self.stats_dict["default"]
        else:
            raise ValueError(f"Statistics not found for speaker: {speaker}")

        feature_mean = stats[self.key_mean]
        feature_std = stats[self.key_std]

        feature = (feature - feature_mean) / feature_std

        if self.mask_field:
            mask = training_example[self.mask_field]
            feature[~mask] = 0.0

        training_example[self.field] = feature
