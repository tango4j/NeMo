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
import os
import time

import torch
from lightning.pytorch import Trainer, seed_everything
from omegaconf import OmegaConf
from torch.distributed.elastic.multiprocessing.errors import record

from nemo.collections.speechlm2 import SALM, DataModule, SALMDataset
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

if torch.cuda.is_available():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def _stage(message):
    rank = os.environ.get("RANK", "?")
    local_rank = os.environ.get("LOCAL_RANK", "?")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[SALM_STAGE {timestamp} rank={rank} local_rank={local_rank}] {message}", flush=True)


@hydra_runner(config_path="conf", config_name="salm")
def train(cfg):
    _stage("entered train()")
    OmegaConf.resolve(cfg)
    _stage("resolved config")
    if torch.cuda.is_available():
        _stage("initializing nccl process group")
        torch.distributed.init_process_group(backend="nccl")
        world_size = torch.distributed.get_world_size()
        _stage(f"initialized nccl process group world_size={world_size}")
    seed_everything(cfg.data.train_ds.seed)
    torch.set_float32_matmul_precision("medium")
    _stage("constructing Trainer")
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    _stage(f"constructed Trainer strategy={type(trainer.strategy).__module__}.{type(trainer.strategy).__name__}")
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")
    _stage(f"exp_manager complete log_dir={log_dir}")

    model_cls = SALM
    if cfg.model.get("use_nemo_automodel", False):
        from nemo.collections.speechlm2 import SALMAutomodel

        model_cls = SALMAutomodel

    _stage(
        "constructing model "
        f"class={model_cls.__name__} "
        f"pretrained_llm_weights={cfg.model.get('pretrained_llm_weights', None)} "
        f"pretrained_asr_weights={cfg.model.get('pretrained_asr_weights', None)}"
    )
    with trainer.init_module():
        model = model_cls(OmegaConf.to_container(cfg.model, resolve=True))
    _stage("constructed model")

    if hasattr(model, "build_dataset"):
        _stage("building dataset via model.build_dataset")
        dataset = model.build_dataset(tokenizer=model.tokenizer, data_cfg=cfg.data)
    else:
        _stage("building fallback SALMDataset")
        dataset = SALMDataset(tokenizer=model.tokenizer)
    _stage(f"constructed dataset class={type(dataset).__module__}.{type(dataset).__name__}")
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)
    _stage(f"constructed datamodule class={type(datamodule).__module__}.{type(datamodule).__name__}")

    _stage("starting trainer.fit")
    trainer.fit(model, datamodule)
    _stage("trainer.fit complete")


if __name__ == "__main__":
    record(train)()
