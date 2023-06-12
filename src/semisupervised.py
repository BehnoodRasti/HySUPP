"""
Semi-supervised unmixing methods main source file
"""
import mlxp
from mlxp.launcher import _instance_from_config
import logging
import numpy as np

from src.data.utils import SVD_projection
from src.utils.metrics import SRE, aRMSE, compute_metric
from src.data.base import Estimate

log = logging.getLogger(__name__)


def main(ctx: mlxp.Context) -> None:
    log.info("Semi-Supervised Unmixing - [START]...")
    cfg = ctx.config
    logger = ctx.logger

    # Get noise
    noise = _instance_from_config(cfg.noise)
    # Get HSI
    hsi = _instance_from_config(cfg.data)
    # Print HSI information
    log.info(hsi)
    # Get data
    Y, p, D = hsi.get_data()
    # Get image dimensions
    H, W = hsi.get_img_shape()
    # Apply noise
    Y = noise.apply(Y)
    # L2 normalization
    if cfg.l2_normalization:
        Y = Y / np.linalg.norm(Y, axis=0, ord=2, keepdims=True)
    # Apply SVD projection
    if cfg.projection:
        Y = SVD_projection(Y, p)
    # Build model
    model = _instance_from_config(cfg.model)
    # Solve unmixing
    A_hat = model.compute_abundances(Y, D, p=p, H=H, W=W)

    E_hat = np.zeros((Y.shape[0], p))

    logger.log_artifact(Estimate(E_hat, A_hat, H, W), "estimates")

    if hsi.has_GT():
        # Get ground truth
        _, A_gt = hsi.get_GT()
        # NOTE: Alignment not needed
        # Select only the first relevant components
        # NOTE Fix this code by using a custom index tied to the dataset
        index = hsi.get_index()
        A1 = A_hat[index]
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
    log.info(f"Semi-Supervised Unmixing - [END]")
