"""
Blind unmixing methods main source file
"""
import logging

from hydra.utils import instantiate

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(cfg):
    logger.debug("Blind Unmixing --- start...")

    # Instantiate objects
    noise = instantiate(cfg.noise)
    hsi = instantiate(
        cfg.data,
        noise=noise,
    )

    logger.info(hsi)

    logger.debug("Blind Unmixing --- end...")
    return None
