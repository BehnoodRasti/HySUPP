"""
SUnSAL simple Python implementation

Source: https://github.com/etienne-monier/lib-unmixing/blob/master/unmixing.py
"""

import logging
import time

import numpy as np
import numpy.linalg as LA

from .base import SparseUnmixingModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CLSUnSAL(SparseUnmixingModel):
    def __init__(
        self,
        AL_iters=1000,
        lambd=0.0,
        verbose=True,
        tol=1e-4,
        mu=0.1,
        x0=0,
        # axis=1,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.AL_iters = AL_iters
        self.lambd = lambd
        # self.mu = mu
        self.verbose = verbose
        # self.positivity = positivity
        # self.addone = addone
        self.tol = tol
        self.x0 = x0
        self.mu = mu
        # self.axis = axis

    def compute_abundances(self, Y, D, *args, **kwargs):
        tic = time.time()
        LD, M = D.shape
        L, N = Y.shape

        assert L == LD, "Inconsistent number of channels for D and Y"

        lambd = self.lambd

        # # Compute mean norm
        # NOTE Legacy code
        norm_d = np.sqrt(np.mean(D**2))
        logger.debug(f"Norm D => {norm_d:.3e}")
        # Rescale D, Y and lambda
        D = D / norm_d
        Y = Y / norm_d
        lambd = lambd / norm_d**2
        logger.debug(f"Lambda initial value => {lambd:.3e}")

        # Constants and initialization
        # mu = 0.1
        mu = self.mu
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
        d1 = v1
        d2 = v2
        d3 = v3

        # # AL iterations
        tol = np.sqrt(N * M) * self.tol
        logger.debug(f"Tolerance => {tol:.3e}")
        k = 1
        res_p = float("inf")
        res_d = float("inf")

        while (k <= self.AL_iters) and ((np.abs(res_p) > tol) or (np.abs(res_d) > tol)):
            # Save u to be used later
            if k % 10 == 1:
                u0 = u

            # breakpoint()
            # Minimize w.r.t. u
            # NOTE Legacy (might be faster than solving linear system)
            u = AA @ (D.T @ (v1 + d1) + v2 + d2 + v3 + d3)
            # u = LA.solve(DD, D.T @ (v1 + d1) + v2 + d2 + v3 + d3)

            # Minimize w.r.t. v1
            v1 = (Y + mu * (D @ u - d1)) / (1 + mu)

            # Minimize w.r.t. v2
            current_fn = lambda b: self.vect_soft_thresh(b, lambd / mu)
            v2 = np.apply_along_axis(current_fn, axis=1, arr=u - d2)

            # Minimize w.r.t. v3
            v3 = np.maximum(u - d3, 0)

            # Lagrange multipliers update
            d1 = d1 - D @ u + v1
            d2 = d2 - u + v2
            d3 = d3 - u + v3

            # Update mu to keep primal and dual residuals within a factor of 10
            if k % 10 == 1:
                # primal residual
                res_p = LA.norm(D @ u - v1) + LA.norm(u - v2) + LA.norm(u - v3)
                # dual residual
                res_d = mu * LA.norm(u - u0)
                if self.verbose:
                    logger.info(
                        f"k = {k}, res_p = {res_p:.3e}, res_d = {res_d:.3e}, mu = {mu:.3e}"
                    )

                # Update mu
                if res_p > 10 * res_d:
                    mu = mu * 2

                if res_d > 10 * res_p:
                    mu = mu / 2

            k += 1

        Ahat = v3
        self.time = time.time() - tic
        logger.info(self.print_time())
        return Ahat

    @staticmethod
    def vect_soft_thresh(b, t):
        max_b = np.maximum(LA.norm(b) - t, 0)
        ret = b * (max_b) / (max_b + t)
        return ret
