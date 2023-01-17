"""
Supervised unmixing methods main source file
"""
import logging

from hydra.utils import instantiate
import matplotlib.pyplot as plt

from src.utils.aligners import AbundancesAligner
from src.utils.metrics import SADAggregator, RMSEAggregator, SREAggregator

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(cfg):
    logger.debug("Supervised Unmixing --- start...")

    # Instantiate objects
    extractor = instantiate(cfg.extractor)
    noise = instantiate(cfg.noise)
    hsi = instantiate(
        cfg.data,
        noise=noise,
    )

    # Metrics
    RMSE = RMSEAggregator()
    SAD = SADAggregator()
    SRE = SREAggregator()

    # Log some info
    logger.info(hsi)

    for run in range(cfg.runs):

        # Reload new model for each run
        model = instantiate(cfg.model, hsi=hsi)
        # Apply noise to HSI
        hsi.apply_noise(seed=cfg.seed + run)

        # Project the data using SVD
        if cfg.projection:
            hsi.apply_projection()

        # Endmembers extraction
        E_hat = extractor.extract_endmembers(
            hsi=hsi, seed=cfg.seed + run, snr_input=noise.SNR
        )

        # Sample HSI
        Y, E_gt, A_gt, _ = hsi.sample()

        # Compute abundances
        A_hat = model.compute_abundances(Y, E_hat)

        # Align endmembers based on abundances MSE
        aligner = AbundancesAligner(Aref=A_gt)
        A1 = aligner.fit_transform(A_hat)
        E1 = aligner.transform_endmembers(E_hat)

        # Compute metrics
        RMSE.add_run(run, A_gt, A1, hsi.labels)
        SRE.add_run(run, A_gt, A1, hsi.labels)
        SAD.add_run(run, E_gt, E1, hsi.labels)

    # Aggregate metrics
    RMSE.aggregate()
    SAD.aggregate()
    SRE.aggregate()

    logger.debug("Supervised Unmixing --- end...")
