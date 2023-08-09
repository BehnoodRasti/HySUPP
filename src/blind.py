"""
Blind unmixing methods main source file
"""
import mlxp
from mlxp.launcher import _instance_from_config
import logging
import numpy as np

from src.data.utils import SVD_projection
from src.utils.aligners import AbundancesAligner
from src.utils.metrics import SRE, SADDegrees, aRMSE, eRMSE, compute_metric
from src.data.base import Estimate

log = logging.getLogger(__name__)


def main(ctx: mlxp.Context) -> None:

    cfg = ctx.config
    logger = ctx.logger
    log.info("Blind Unmixing - [START]")

    # Get noise
    noise = _instance_from_config(cfg.noise)
    # Get HSI
    hsi = _instance_from_config(cfg.data)
    # Print HSI information
    log.info(hsi)
    # Get data
    Y, p, _ = hsi.get_data()
    # Get image dimensions
    H, W = hsi.get_img_shape()
    # Normalize HSI
    # Y = (Y - Y.min()) / (Y.max() - Y.min())
    # Apply noise
    Y = noise.apply(Y)
    # L2 normalization
    if cfg.l2_normalization:
        normY = np.linalg.norm(Y, axis=0, ord=2, keepdims=True)
        Y = Y / normY
    # Apply SVD projection
    if cfg.projection:
        Y = SVD_projection(Y, p)
    # Build model
    model = _instance_from_config(cfg.model)
    # Solve unmixing
    E_hat, A_hat = model.compute_endmembers_and_abundances(
        Y,
        p,
        H=H,
        W=W,
    )

    logger.log_artifact(Estimate(E_hat, A_hat, H, W), "estimates")

    if hsi.has_GT():
        # Get ground truth
        E_gt, A_gt = hsi.get_GT()
        # Align based on abundances
        aligner = AbundancesAligner(Aref=A_gt)
        A1 = aligner.fit_transform(A_hat)
        E1 = aligner.transform_endmembers(E_hat)
        # Get labels
        labels = hsi.get_labels()
        # Compute and log metrics
        logger.log_metrics(
            compute_metric(
                SRE(),
                A_gt,
                A1,
                labels,
                detail=False,
                on_endmembers=False,
            ),
            log_name="SRE",
        )
        logger.log_metrics(
            compute_metric(
                aRMSE(),
                A_gt,
                A1,
                labels,
                detail=True,
                on_endmembers=False,
            ),
            log_name="aRMSE",
        )
        logger.log_metrics(
            compute_metric(
                SADDegrees(),
                E_gt,
                E1,
                labels,
                detail=True,
                on_endmembers=True,
            ),
            log_name="SAD",
        )
        logger.log_metrics(
            compute_metric(
                eRMSE(),
                E_gt,
                E1,
                labels,
                detail=True,
                on_endmembers=True,
            ),
            log_name="eRMSE",
        )

    log.info("Blind Unmixing - [END]")
