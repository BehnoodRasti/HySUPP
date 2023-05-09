"""
Toolbox main file managed by hydra to handle experiments configurations
"""

import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@hydra.main(version_base=None, config_path="src/config", config_name="config")
def main(cfg: DictConfig) -> None:

    logger.info(f"Current working directory: {os.getcwd()}")
    logger.debug(OmegaConf.to_yaml(cfg))

    mode = cfg.mode

    if mode == "blind":
        from src.blind import main as _main
    elif mode == "supervised":
        from src.supervised import main as _main
    elif mode == "sparse":
        from src.sparse import main as _main
    elif mode == "extract":
        from src.extract import main as _main
    else:
        raise ValueError(f"Mode {mode} is invalid")

    try:
        _main(cfg)
    except Exception as e:
        logger.critical(e, exc_info=True)


if __name__ == "__main__":
    main()
