"""
Utility functions for data manipulation
"""

import logging
import numpy as np

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def SVD_projection(Y, p):
    log.debug(f"Y shape => {Y.shape}")
    V, SS, U = np.linalg.svd(Y, full_matrices=False)
    PC = np.diag(SS) @ U
    denoised_image_reshape = V[:, :p] @ PC[:p]
    log.debug(f"projected Y shape => {denoised_image_reshape.shape}")
    return np.clip(denoised_image_reshape, 0, 1)
