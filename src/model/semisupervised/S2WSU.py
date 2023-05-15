"""
S2WSU simple Python implementation

Source: https://github.com/ricardoborsoi/MUA_SparseUnmixing
"""

import logging
import time

import numpy as np
import numpy.linalg as LA
import torch.nn.functional as F
import torch
from scipy.signal import convolve2d

from .base import SemiSupervisedUnmixingModel as SparseUnmixingModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class S2WSU(SparseUnmixingModel):
    def __init__(
        self,
        hsi,
        AL_iters=5,
        lambd=0.0,
        verbose=True,
        tol=1e-5,
        x0=0,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.AL_iters = AL_iters
        self.lambd = lambd
        self.verbose = verbose
        self.tol = tol
        self.x0 = x0

        self.H = hsi.H
        self.W = hsi.W

    @staticmethod
    def soft(b, t):
        max_b = np.maximum(np.abs(b) - t, 0)
        return b * (max_b / (max_b + t))

    def compute_abundances(self, Y, D, *args, **kwargs):
        tic = time.time()
        LD, M = D.shape
        L, N = Y.shape
        H, W = self.H, self.W

        assert L == LD, "Inconsistent number of channels for D and Y"

        lambd = self.lambd

        # # Compute mean norm
        # NOTE Legacy code
        # norm_d = np.sqrt(np.mean(D**2))
        # logger.debug(f"Norm D => {norm_d:.3e}")
        # # Rescale D, Y and lambda
        # D = D / norm_d
        # Y = Y / norm_d
        # lambd = lambd / norm_d**2
        logger.debug(f"Lambda initial value => {lambd:.3e}")

        # Constants and initialization
        mu = 0.5
        logger.debug(f"Mu initial value => {mu:.3e}")

        UF, sF, VF = LA.svd(D.T @ D)
        IF = UF @ (np.diag(1 / (sF + mu))) @ UF.T

        AA = LA.inv(D.T @ D + 2 * np.eye(M))

        # Initializations
        if self.x0 == 0:
            x = IF @ D.T @ Y
        else:
            x = self.x0

        u = x
        v1 = D @ x
        v2 = x
        v3 = x
        # # Scaled Lagrange Multipliers
        d1 = 0 * v1
        d2 = 0 * v2
        d3 = 0 * v3

        # # AL iterations
        tol = np.sqrt(N * M) * self.tol
        logger.debug(f"Tolerance => {tol:.3e}")
        k = 1
        i = 1
        res_p = float("inf")
        # res_d = float("inf")
        AL_iters2 = 60

        kernel = np.ones((3, 3))
        kernel[1, 1] = 0
        kernel[0, 0] = 1 / np.sqrt(2)
        kernel[2, 0] = 1 / np.sqrt(2)
        kernel[0, 2] = 1 / np.sqrt(2)
        kernel[2, 2] = 1 / np.sqrt(2)
        kernel = kernel / (4 + 4 / np.sqrt(2))

        while k <= AL_iters2:

            NU = np.zeros((M, N))
            X2 = np.reshape(v3 - d3, (M, H, W))
            for ii in range(M):
                NU[ii] = convolve2d(X2[ii], kernel, mode="same").flatten()

            w = 1 / (0.01 + np.abs(NU))

            NU2 = LA.norm(v3 - d3, axis=1, keepdims=True)
            w1 = w / NU2

            while (i <= self.AL_iters) and np.abs(res_p) > tol:
                # Save u to be used later
                if i % 10 == 1:
                    u0 = u

                # Minimize w.r.t. u
                u = AA @ (D.T @ (v1 + d1) + v2 + d2 + v3 + d3)

                # Minimize w.r.t. v1
                v1 = (Y + mu * (D @ u - d1)) / (1 + mu)

                # Minimize w.r.t. v2
                v2 = np.maximum(u - d2, 0)

                # Minimize w.r.t. v3
                v3 = self.soft(u - d3, (lambd / mu) * w1)

                # Lagrange multipliers update
                d1 = d1 - D @ u + v1
                d2 = d2 - u + v2
                d3 = d3 - u + v3

                if i % 10 == 1:
                    # primal residual
                    res_p = LA.norm(D @ u - v1) + LA.norm(u - v2) + LA.norm(u - v3)
                    if self.verbose:
                        logger.info(f"k = {k}, i = {i}, res_p = {res_p:.3e}")

                i += 1

            i = 1
            k += 1

        Ahat = u

        self.time = time.time() - tic
        logger.info(self.print_time())

        return Ahat
