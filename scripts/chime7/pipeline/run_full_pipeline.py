# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import logging

from asr.run_asr import run_asr
from diar.run_diar import run_diarization
from eval.run_chime_eval import run_chime_evaluation
from gss_process.run_gss_process import run_gss_process
from omegaconf import DictConfig, OmegaConf

from nemo.core.config import hydra_runner
from nemo.utils import logging


@hydra_runner(config_path="../", config_name="chime_config")
def main(cfg):
    cfg = DictConfig(OmegaConf.to_container(cfg, resolve=True))

    logging.info("Running Diarization")
    run_diarization(cfg)

    # Run GSS
    logging.info("Running GSS")
    run_gss_process(cfg)

    # Run ASR
    logging.info("Running ASR")
    run_asr(cfg)

    # Run evaluation
    logging.info("Running evaluation")
    run_chime_evaluation(cfg)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
