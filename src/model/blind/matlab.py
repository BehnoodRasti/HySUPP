"""
MATLAB blind unmixing methods
"""
import logging
import time
import os
import warnings

import numpy as np

from .base import BlindUnmixingModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

try:
    import matlab.engine
except Exception:
    warnings.warn("matlab.engine was not imported. MATLAB code will not work")


class NMFQMV(BlindUnmixingModel):
    """
    Python wrapper on the NMF-QMV matlab code found at
    https://github.com/LinaZhuang/NMF-QMV_demo
    """

    def __init__(
        self,
        root_matlab,
        term: str = "boundary",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.root_matlab = root_matlab
        path_to_NMFQMV = os.path.join(self.root_matlab, "NMF-QMV_demo")
        assert os.path.exists(path_to_NMFQMV), "Change path to your location of NMF-QMV"

        # Record hyperparameters
        self.term = term
        self.betas = np.logspace(2, -2, 5)
        self.drawfigs = "no"

        # Start matlab engine
        self.engine = matlab.engine.start_matlab()
        logger.debug("MATLAB engine started")
        # Go to SUnSAL code
        self.engine.cd(path_to_NMFQMV)

    def compute_endmembers_and_abundances(self, Y, p, H, W, *args, **kwargs):
        tic = time.time()

        L, N = Y.shape
        Y = Y.T.reshape(H, W, L)
        Y = Y.transpose(1, 0, 2)
        _, Ehat, Ahat = self.engine.NMF_QMV(
            matlab.double(Y.tolist()),
            matlab.double([p]),
            matlab.double(self.betas.tolist()),
            self.term,
            "DRAWFIGS",
            self.drawfigs,
            nargout=3,
        )

        Ehat = np.array(Ehat).astype(np.float32)
        Ahat = np.array(Ahat).astype(np.float32)

        tac = time.time()
        self.time = round(tac - tic, 2)
        logger.info(self.print_time())

        return Ehat, Ahat
