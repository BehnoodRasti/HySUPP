"""
Supervised unmixing methods main source file
"""
import logging

from hydra.utils import instantiate

from src.utils.aligners import AbundancesAligner
from src.utils.metrics import (
    SADAggregator,
    RMSEAggregator,
    SREAggregator,
    ERMSEAggregator,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(cfg):
    logger.debug("Supervised Unmixing --- start...")

    # Instantiate objects
    extractor = instantiate(cfg.extractor)
    noise = instantiate(cfg.noise)
    # Metrics
    RMSE = RMSEAggregator()
    SAD = SADAggregator()
    SRE = SREAggregator()
    ERMSE = ERMSEAggregator()

    for run in range(cfg.runs):

        hsi = instantiate(
            cfg.data,
            noise=noise,
        )

        if run == 0:
            # Log some info
            logger.info(hsi)
            hsi.plot_endmembers()
            hsi.plot_abundances()

        # Reload new model for each run
        model = instantiate(cfg.model, hsi=hsi)
        # # Apply noise to HSI
        # hsi.apply_noise(seed=cfg.seed + run)

        # # Project the data using SVD
        # if cfg.projection:
        #     hsi.apply_projection()

        # Sample HSI
        Y, E_gt, A_gt, _ = hsi.sample(seed=cfg.seed + run, projection=cfg.projection)

        # Endmembers extraction
        E_hat = extractor.extract_endmembers(
            Y=Y,
            p=hsi.p,
            seed=cfg.seed + run,
            snr_input=noise.SNR,
        )

        # Compute abundances
        A_hat = model.compute_abundances(Y, E_hat)

        # Align endmembers based on abundances MSE
        aligner = AbundancesAligner(Aref=A_gt)
        A1 = aligner.fit_transform(A_hat)
        E1 = aligner.transform_endmembers(E_hat)

        hsi.plot_endmembers(E0=E1, run=run)
        hsi.plot_abundances(A0=A1, run=run)

        # Compute metrics
        RMSE.add_run(run, A_gt, A1, hsi.labels)
        SRE.add_run(run, A1, A_gt, hsi.labels)  # NOTE Order is important
        SAD.add_run(run, E_gt, E1, hsi.labels)
        ERMSE.add_run(run, E1, E_gt, hsi.labels)

    # Aggregate metrics
    RMSE.aggregate()
    SAD.aggregate()
    ERMSE.aggregate()
    SRE.aggregate()

    logger.debug("Supervised Unmixing --- end...")
