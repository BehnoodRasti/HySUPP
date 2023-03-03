"""
SUnSAL simple Python implementation

Source: https://github.com/etienne-monier/lib-unmixing/blob/master/unmixing.py
"""

import logging
import time
import os

import numpy as np
import numpy.linalg as LA

from .base import SparseUnmixingModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SUnSAL(SparseUnmixingModel):
    def __init__(
        self,
        AL_iters=1000,
        lambd=0.0,
        verbose=True,
        positivity=False,
        addone=False,
        tol=1e-4,
        x0=0,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.AL_iters = AL_iters
        self.lambd = lambd
        self.verbose = verbose
        self.positivity = positivity
        self.addone = addone
        self.tol = tol
        self.x0 = x0

    def compute_abundances(self, Y, D, *args, **kwargs):
        tic = time.time()
        LD, M = D.shape
        L, N = Y.shape

        assert L == LD, "Inconsistent number of channels for D and Y"

        # Lambda for all pixels
        lambd = self.lambd * np.ones((M, N))

        # Compute mean norm
        # NOTE This typo led to better results
        # norm_d = np.sqrt(np.mean(D**2)) * (25 * M) / M
        # norm_d = np.sqrt(np.mean(D**2))
        # NOTE Align with original MATLAB code
        norm_d = np.sqrt(np.mean(D**2))
        logger.debug(f"norm D => {norm_d:.3e}")
        # # Rescale D, Y and lambda
        D = D / norm_d
        Y = Y / norm_d
        lambd = lambd / norm_d**2
        logger.debug(f"lambda value: {np.mean(lambd):.3e}")

        # Least squares
        if np.sum(lambd == 0) and not self.addone and not self.positivity:
            logger.debug("Least Squares")
            Ahat = LA.pinv(D) @ Y
            return Ahat

        # Constrained Least Squares (sum(x) = 1)
        SMALL = 1e-12
        B = np.ones((1, M))
        a = np.ones((1, N))

        if np.sum(lambd == 0) and self.addone and not self.positivity:
            logger.debug("Constrained Least Squares (sum(x) = 1)")
            F = D.T @ M
            # Test if F is invertible
            if LA.cond(F) > SMALL:
                # Compute the solution explicitely
                IF = LA.inv(F)
                Ahat = IF @ D.T @ Y - IF @ B.T @ (LA.inv(B @ IF @ B.T)) @ (
                    B @ IF @ D.T @ Y - a
                )
                return Ahat

        # Constants and initialization
        mu_AL = 0.01
        mu = 10 * np.mean(lambd) + mu_AL
        # mu = mu_AL
        logger.debug(f"mu initial value: {mu:.3e}")

        UF, sF, VF = LA.svd(D.T @ D)
        SF = np.diag(sF)
        IF = UF @ (np.diag(1 / (sF + mu))) @ UF.T

        Aux = IF @ B.T @ (LA.inv(B @ IF @ B.T))
        x_aux = Aux @ a
        IF1 = IF - Aux @ B @ IF
        yy = D.T @ Y

        # Initializations
        if self.x0 == 0:
            x = IF @ D.T @ Y
        else:
            x = self.x0

        z = x
        # Scaled Lagrange Multipliers
        d = 0 * z

        # AL iterations
        tol1 = np.sqrt(N * M) * self.tol
        tol2 = np.sqrt(N * M) * self.tol
        logger.debug(f"tolerance => {tol1:.3e}")
        i = 1
        res_p = float("inf")
        res_d = float("inf")
        mu_changed = 0

        # Constrained Least Squares (CLS) X >= 0

        if np.sum(lambd == 0) and not self.addone:
            logger.debug("Constrained Least Squares (x >= 0)")
            while (i <= self.AL_iters) and (
                (np.abs(res_p) > tol1) or (np.abs(res_d) > tol2)
            ):
                # Save z to be used later
                if i % 10 == 1:
                    z0 = z

                # Minimize w.r.t. z
                z = np.maximum(x - d, 0)
                # Minimize w.r.t. x
                x = IF @ (yy + mu * (z + d))
                # Lagrange multipliers update
                d = d - (x - z)

                # Update mu to keep primal and dual residuals within a factor of 10
                if i % 10 == 1:
                    # primal residual
                    res_p = LA.norm(x - z)
                    # dual residual
                    res_d = mu * LA.norm(z - z0)
                    if self.verbose:
                        logger.info(
                            f"i = {i}, res_p = {res_p:.3e}, res_d = {res_d:.3e}"
                        )

                    # update mu
                    if res_p > 10 * res_d:
                        mu = mu * 2
                        d = d / 2
                        mu_changed = 1

                    elif res_d > 10 * res_p:
                        mu = mu / 2
                        d = d * 2
                        mu_changed = 1

                    if mu_changed:
                        logger.debug(f"mu changed ({i}) => {mu}")
                        # Update IF and IF1
                        IF = UF @ np.diag(1 / (sF + mu)) @ UF.T
                        Aux = IF @ B.T @ (LA.inv(B @ IF @ B.T))
                        x_aux = Aux @ a
                        IF1 = IF - Aux @ B @ IF
                        mu_changed = 0

                i += 1

        # Fully Constraint Least Squares
        elif np.sum(lambd == 0) and self.addone:
            logger.debug("Fully Constrained Least Squares")
            while (i <= self.AL_iters) and (
                (np.abs(res_p) > tol1) or (np.abs(res_d) > tol2)
            ):
                # Save z to be used later
                if i % 10 == 1:
                    z0 = z

                # Minimize w.r.t. z
                z = np.maximum(x - d, 0)
                # Minimize w.r.t. x
                x = IF1 @ (yy + mu * (z + d)) + x_aux
                # Lagrange multipliers update
                d = d - (x - z)

                # Update mu to keep primal and dual residuals within a factor of 10
                if i % 10 == 1:
                    # primal residual
                    res_p = LA.norm(x - z)
                    # dual residual
                    res_d = mu * LA.norm(z - z0)
                    if self.verbose:
                        logger.info(
                            f"i = {i}, res_p = {res_p:.3e}, res_d = {res_d:.3e}"
                        )

                    # update mu
                    if res_p > 10 * res_d:
                        mu = mu * 2
                        d = d / 2
                        mu_changed = 1

                    elif res_d > 10 * res_p:
                        mu = mu / 2
                        d = d * 2
                        mu_changed = 1

                    if mu_changed:
                        # Update IF and IF1
                        IF = UF @ np.diag(1 / (sF + mu)) @ UF.T
                        Aux = IF @ B.T @ (LA.inv(B @ IF @ B.T))
                        x_aux = Aux @ a
                        IF1 = IF - Aux @ B @ IF
                        mu_changed = 0

                i += 1

        # Generic SUnSAL
        else:
            logger.debug("Generic SUnSAL")
            softthresh = lambda x, th: np.sign(x) * np.maximum(np.abs(x) - th, 0)

            while (i <= self.AL_iters) and (
                (np.abs(res_p) > tol1) or (np.abs(res_d) > tol2)
            ):
                if i % 10 == 1:
                    z0 = z

                # Minimize w.r.t. z
                z = softthresh(x - d, lambd / mu)
                # Test for positivity
                if self.positivity:
                    z = np.maximum(z, 0)

                # Test of Sum-to-one
                if self.addone:
                    x = IF1 @ (yy + mu * (z + d)) + x_aux
                else:
                    x = IF @ (yy + mu * (z + d))

                # Lagrange multipliers update
                d = d - (x - z)

                # Update mu to keep primal and dual residuals within a factor of 10
                if i % 10 == 1:
                    # primal residual
                    res_p = LA.norm(x - z)
                    # dual residual
                    res_d = mu * LA.norm(z - z0)
                    if self.verbose:
                        logger.info(
                            f"i = {i}, res_p = {res_p:.3e}, res_d = {res_d:.3e}"
                        )

                    # update mu
                    if res_p > 10 * res_d:
                        mu = mu * 2
                        d = d / 2
                        mu_changed = 1

                    elif res_d > 10 * res_p:
                        mu = mu / 2
                        d = d * 2
                        mu_changed = 1

                    if mu_changed:
                        # Update IF and IF1
                        logger.debug(f"mu changed ({i}) => {mu}")
                        IF = UF @ np.diag(1 / (sF + mu)) @ UF.T
                        Aux = IF @ B.T @ (LA.inv(B @ IF @ B.T))
                        x_aux = Aux @ a
                        IF1 = IF - Aux @ B @ IF
                        mu_changed = 0

                i += 1

        Ahat = z
        tac = time.time()
        self.time = tac - tic
        logger.info(self.print_time())
        return Ahat
