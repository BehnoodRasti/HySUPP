"""
MATLAB methods
"""
import logging
import time
import os
import warnings

import numpy as np

from .base import SparseUnmixingModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

try:
    import matlab.engine
except Exception:
    warnings.warn("matlab.engine was not imported. MATLAB code will not work")


class SUnSAL(SparseUnmixingModel):
    """
    Python wrapper on the SUnSAL matlab code found at
    https://github.com/ricardoborsoi/MUA_SparseUnmixing
    """

    def __init__(
        self,
        root_matlab,
        lambd=1e-3,
        ASC=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.root_matlab = root_matlab
        path_to_SUnSAL = os.path.join(self.root_matlab, "MUA_SparseUnmixing")
        assert os.path.exists(path_to_SUnSAL), "Change path to your location of SUnSAL"

        # Record hyperparameters
        self.lambd = lambd
        self.ASC = "yes" if ASC else "no"

        # Start matlab engine
        self.engine = matlab.engine.start_matlab()
        logger.debug("MATLAB engine started")
        # Go to SUnSAL code
        self.engine.cd(path_to_SUnSAL)

    def compute_abundances(self, Y, D, *args, **kwargs):
        tic = time.time()

        Ahat = self.engine.sunsal(
            matlab.double(D.tolist()),
            matlab.double(Y.tolist()),
            "lambda",
            matlab.double([self.lambd]),
            "ADDONE",
            self.ASC,
            "POSITIVITY",
            "yes",
            "TOL",
            matlab.double([1e-4]),
            "AL_ITERS",
            matlab.double([2000]),
            "verbose",
            "yes",
            nargout=1,
        )

        Ahat = np.array(Ahat).astype(np.float32)

        tac = time.time()
        self.time = round(tac - tic, 2)
        logger.info(self.print_time())

        return Ahat


class SUnSALTV(SparseUnmixingModel):
    """
    Python wrapper on the SUnSAL matlab code found at
    https://github.com/ricardoborsoi/MUA_SparseUnmixing
    """

    def __init__(
        self,
        hsi,
        root_matlab,
        lambd=1e-3,
        lambd_tv=1e-3,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.root_matlab = root_matlab
        path_to_SUnSALTV = os.path.join(self.root_matlab, "MUA_SparseUnmixing")
        assert os.path.exists(
            path_to_SUnSALTV
        ), "Change path to your location of SUnSAL"

        # Record hyperparameters
        self.lambd = lambd
        self.lambd_tv = lambd_tv
        self.H, self.W = hsi.H, hsi.W

        # Start matlab engine
        self.engine = matlab.engine.start_matlab()
        logger.debug("MATLAB engine started")
        # Go to SUnSAL code
        self.engine.cd(path_to_SUnSALTV)

    def compute_abundances(self, Y, D, *args, **kwargs):
        H, W = self.H, self.W
        tic = time.time()
        Ahat = self.engine.sunsal_tv(
            matlab.double(D.tolist()),
            matlab.double(Y.tolist()),
            "MU",
            matlab.double([0.05]),
            "ADDONE",
            "no",
            "POSITIVITY",
            "yes",
            "LAMBDA_1",
            matlab.double([self.lambd]),
            "LAMBDA_TV",
            matlab.double([self.lambd_tv]),
            "TV_TYPE",
            "niso",
            "IM_SIZE",
            matlab.double([H, W]),
            "AL_ITERS",
            matlab.double([200]),
            "verbose",
            "yes",
            nargout=1,
        )

        Ahat = np.array(Ahat).astype(np.float32)

        tac = time.time()
        self.time = round(tac - tic, 2)
        logger.info(self.print_time())

        return {"A": Ahat}


class S2WSU(SparseUnmixingModel):
    """
    Python wrapper on the SUnSAL matlab code found at
    https://github.com/ricardoborsoi/MUA_SparseUnmixing
    """

    def __init__(
        self,
        hsi,
        root_matlab,
        lambd=1e-3,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.root_matlab = root_matlab
        path_to_S2WSU = os.path.join(self.root_matlab, "MUA_SparseUnmixing")
        assert os.path.exists(path_to_S2WSU), "Change path to your location of S2WSU"

        # Record hyperparameters
        self.lambd = lambd
        self.H, self.W = hsi.H, hsi.W

        # Start matlab engine
        self.engine = matlab.engine.start_matlab()
        logger.debug("MATLAB engine started")
        # Go to SUnSAL code
        self.engine.cd(path_to_S2WSU)

    def compute_abundances(self, Y, D, *args, **kwargs):

        tic = time.time()
        H, W = self.H, self.W
        Ahat = self.engine.sunsal_tv_lw_sp(
            matlab.double(D.tolist()),
            matlab.double(Y.tolist()),
            "MU",
            matlab.double([0.5]),
            "ADDONE",
            "no",
            "POSITIVITY",
            "yes",
            "LAMBDA_1",
            matlab.double([self.lambd]),
            "IM_SIZE",
            matlab.double([H, W]),
            "AL_ITERS",
            matlab.double([5]),
            "verbose",
            "yes",
            nargout=1,
        )

        Ahat = np.array(Ahat).astype(np.float32)

        tac = time.time()
        self.time = round(tac - tic, 2)
        logger.info(self.print_time())

        return Ahat


class MUA_BPT(SparseUnmixingModel):
    """
    Python wrapper on the SUnSAL matlab code found at
    https://github.com/ricardoborsoi/MUA_SparseUnmixing
    """

    def __init__(
        self,
        hsi,
        root_matlab,
        lambda1_sp=0.005,
        lambda2_sp=0.1,
        beta=30,
        sideBPT=10,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.root_matlab = root_matlab
        path_to_MUA = os.path.join(self.root_matlab, "MUA_SparseUnmixing")
        assert os.path.exists(path_to_MUA), "Change path to your location of SUnSAL"

        # Record hyperparameters
        self.lambda1_sp = lambda1_sp
        self.lambda2_sp = lambda2_sp
        self.beta = beta
        self.sideBPT = sideBPT
        self.nl = hsi.H
        self.nc = hsi.W
        self.N = hsi.N
        self.L = hsi.L
        self.M = hsi.M

        # Start matlab engine
        self.engine = matlab.engine.start_matlab()
        logger.debug("MATLAB engine started")
        # Go to SUnSAL code
        self.engine.cd(path_to_MUA)

    def compute_abundances(self, Y, D, *args, **kwargs):
        tic = time.time()

        Ahat = self.engine.MUA_BPT(
            matlab.double(D.tolist()),
            matlab.double(Y.tolist()),
            matlab.double([self.nl]),
            matlab.double([self.nc]),
            matlab.double([self.N]),
            matlab.double([self.L]),
            matlab.double([self.M]),
            "lambda1_sp",
            matlab.double([self.lambda1_sp]),
            "lambda2_sp",
            matlab.double([self.lambda2_sp]),
            "beta",
            matlab.double([self.beta]),
            "sideBPT",
            matlab.double([self.sideBPT]),
            nargout=1,
        )

        Ahat = np.array(Ahat)

        tac = time.time()
        self.time = round(tac - tic, 2)
        logger.info(self.print_time())

        return Ahat
