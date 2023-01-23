"""
Blind unmixing methods main source file
"""
import logging

from hydra.utils import instantiate

from src.utils.aligners import AbundancesAligner
from src.utils.metrics import SADAggregator, RMSEAggregator, SREAggregator

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(cfg):
    logger.debug("Blind Unmixing --- start...")

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
        model = instantiate(cfg.model, hsi=hsi, root_matlab=cfg.MATLAB_root)
        # Apply noise to HSI
        hsi.apply_noise(seed=cfg.seed + run)

        # Project the data using SVD
        if cfg.projection:
            hsi.apply_projection()

        # Sample HSI
        Y, E_gt, A_gt, _ = hsi.sample()

        # Compute endmembers and abundances
        E_hat, A_hat = model.compute_endmembers_and_abundances(
            Y,
            hsi.p,
            seed=cfg.seed + run,
        )

        # Align endmembers based on abundances MSE
        aligner = AbundancesAligner(Aref=A_gt)
        A1 = aligner.fit_transform(A_hat)
        E1 = aligner.transform_endmembers(E_hat)

        # Compute metrics
        RMSE.add_run(run, A_gt, A1, hsi.labels)
        SRE.add_run(run, A1, A_gt, hsi.labels)  # NOTE Order is important
        SAD.add_run(run, E_gt, E1, hsi.labels)

    # Aggregate metrics
    RMSE.aggregate()
    SAD.aggregate()
    SRE.aggregate()

    logger.debug("Blind Unmixing --- end...")
