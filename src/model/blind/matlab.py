"""
MATLAB blind unmixing methods
"""
import logging
import time
import os
import warnings

import numpy as np

from .base import BlindUnmixingModel

from src.model.extractors import VCA

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


class BayesianSMA(BlindUnmixingModel):
    """
    Python wrapper on the Bayesian matlab code from N. Dobigeon
    """

    def __init__(
        self,
        root_matlab,
        Nmc=100,
        bool_plot=0,
        Tgeo_method="vca",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.root_matlab = root_matlab
        path_to_BSMA = os.path.join(self.root_matlab, "Bayesian_SMA")
        assert os.path.exists(
            path_to_BSMA
        ), "Change path to your location of Bayesian SMA"

        # Record hyperparameters
        self.Nmc = Nmc
        self.bool_plot = bool_plot
        self.Tgeo_method = Tgeo_method

        # Start matlab engine
        self.engine = matlab.engine.start_matlab()
        logger.debug("MATLAB engine started")
        # Go to Bayesian SMA code
        self.engine.cd(path_to_BSMA)

    def compute_endmembers_and_abundances(self, Y, p, *args, **kwargs):

        tic = time.time()
        L, N = Y.shape
        Y = Y.T

        Tsigma2r = np.ones((p, 1)) * 10

        Tab_A, Tab_T, Tab_sigma2, matU, Y_bar, Nmc = self.engine.blind_unmix(
            matlab.double(Y.tolist()),
            matlab.double([p]),
            matlab.double([N]),
            matlab.double([self.Nmc]),
            self.Tgeo_method,
            matlab.double(Tsigma2r.tolist()),
            matlab.double([self.bool_plot]),
            nargout=6,
        )

        # Process output
        Nmc = int(Nmc)
        Tab_A = np.array(Tab_A).astype(np.float32)
        Tab_T = np.array(Tab_T).astype(np.float32)
        Tab_sigma2 = np.array(Tab_sigma2).astype(np.float32)
        matU = np.array(matU).astype(np.float32)
        Y_bar = np.array(Y_bar).astype(np.float32)

        Nbi = Nmc // 3
        Ahat = np.mean(Tab_A[Nbi:], 0)
        T_MMSE = np.mean(Tab_T[Nbi:], 0)
        Ehat = matU @ T_MMSE + Y_bar @ np.ones((1, p))

        tac = time.time()
        self.time = round(tac - tic, 2)
        logger.info(self.print_time())

        return Ehat, Ahat


class SCLSU(BlindUnmixingModel):
    def __init__(
        self,
        root_matlab,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.root_matlab = root_matlab
        path_to_SCLSU = os.path.join(self.root_matlab, "ELMM")
        assert os.path.exists(
            path_to_SCLSU
        ), "Change path to your location of ELMM/SCLSU"

        # Start matlab engine
        self.engine = matlab.engine.start_matlab()
        logger.debug("MATLAB engine started")
        # Go to SCLSU code
        self.engine.cd(path_to_SCLSU)

    def compute_endmembers_and_abundances(self, Y, p, H, W, *args, **kwargs):

        # Endmembers Extraction Algorithm
        E0 = VCA().extract_endmembers(Y, p)

        L, N = Y.shape
        Y_img = Y.T.reshape(H, W, L)

        Ahat, psi = self.engine.SCLSU(
            matlab.double(Y_img.tolist()),
            matlab.double(E0.tolist()),
            nargout=2,
        )

        Ahat = np.array(Ahat).astype(np.float32)
        psi = np.array(psi).astype(np.float32)

        # Process output
        E_full = np.zeros((L, p, N))
        for ii in range(N):
            E_full[:, :, ii] = E0 @ np.diag(psi[:, ii])

        Ehat = np.mean(E_full, axis=2)

        return Ehat, Ahat


class ELMM(BlindUnmixingModel):
    def __init__(
        self,
        root_matlab,
        lambda_s=0.5,
        lambda_a=0.01,
        lambda_psi=0.05,
        norm="1,1",
        verbose=False,
        maxiter_anls=100,
        maxiter_admm=100,
        eps_s=0.001,
        eps_a=0.001,
        eps_psi=0.001,
        eps_admm_abs=0.01,
        eps_admm_rel=0.01,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.root_matlab = root_matlab
        path_to_ELMM = os.path.join(self.root_matlab, "ELMM")
        assert os.path.exists(path_to_ELMM), "Change path to your location of ELMM"

        # Start matlab engine
        self.engine = matlab.engine.start_matlab()
        logger.debug("MATLAB engine started")
        # Go to ELMM code
        self.engine.cd(path_to_ELMM)

        # Hyperparameters setting
        self.lambda_s = lambda_s
        self.lambda_a = lambda_a
        self.lambda_psi = lambda_psi
        self.norm = norm
        self.verbose = verbose
        self.maxiter_anls = maxiter_anls
        self.maxiter_admm = maxiter_admm
        self.eps_s = eps_s
        self.eps_a = eps_a
        self.eps_psi = eps_psi
        self.eps_admm_abs = eps_admm_abs
        self.eps_admm_rel = eps_admm_rel

    def compute_endmembers_and_abundances(self, Y, p, H, W, *args, **kwargs):

        # Initialization
        # Endmembers Extraction Algorithm
        E0 = VCA().extract_endmembers(Y, p)
        _, A_init = SCLSU(self.root_matlab).compute_endmembers_and_abundances(
            Y, p, H, W
        )
        psis_init = np.ones_like(A_init)
        L, N = Y.shape
        Y_img = Y.T.reshape(H, W, L)

        lambda_psi = self.lambda_psi * np.ones(p)

        # Main algorithm
        A_ELMM, psis_ELMM, E_ELMM = self.engine.ELMM_ADMM(
            matlab.double(Y_img.tolist()),
            matlab.double(A_init.tolist()),
            matlab.double(psis_init.tolist()),
            matlab.double(E0.tolist()),
            matlab.double([self.lambda_s]),
            matlab.double([self.lambda_a]),
            matlab.double(lambda_psi.tolist()),
            self.norm,
            self.verbose,
            matlab.double([self.maxiter_anls]),
            matlab.double([self.maxiter_admm]),
            matlab.double([self.eps_s]),
            matlab.double([self.eps_a]),
            matlab.double([self.eps_psi]),
            matlab.double([self.eps_admm_abs]),
            matlab.double([self.eps_admm_rel]),
            nargout=3,
        )

        # Process output
        Ahat = np.array(A_ELMM).astype(np.float32)
        psis_ELMM = np.array(psis_ELMM).astype(np.float32)
        E_ELMM = np.array(E_ELMM).astype(np.float32)

        Ehat = E_ELMM.mean(2)

        return Ehat, Ahat
