from asr.run_asr import run_asr
from diar.run_diar import run_diarization
from eval.run_chime_eval import run_chime_evaluation
from gss_process.run_gss_process import run_gss_process
from omegaconf import DictConfig, OmegaConf
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf import OmegaConf
import logging
import os

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
