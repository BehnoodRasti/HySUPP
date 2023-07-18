import logging
import time
import os
import numpy as np
import numpy.linalg as LA

from .base import SemiSupervisedUnmixingModel as SparseUnmixingModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MUSIC_CSR(SparseUnmixingModel):
    def __init__(
        self,
        AL_iters=1000,
        lambd=0.1,
        verbose=True,
        tol=1e-4,
        mu=0.1,
        x0=0,
        projection_errors_threshold=0.045,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.AL_iters = AL_iters
        self.lambd = lambd
        self.verbose = verbose
        self.tol = tol
        self.x0 = x0
        self.mu = mu
        self.error_th = projection_errors_threshold

    def compute_abundances(
        self,
        Y,
        D,
        H,
        W,
        *args,
        **kwargs,
    ):
        tic = time.time()
        LD, M = D.shape
        L, N = Y.shape

        assert L == LD, "Inconsistent number of channels for D and Y"

        assert N == H * W, "Inconsistent number of pixels for Y"
        #################
        # Preprocessing #
        #################
        # Orthonormal basis estimation using HySime
        v_dim, E = HySime().count(Y.reshape(L, H, W).transpose(1, 2, 0))

        logger.info(f"Virtual dimension (Hysime) => {v_dim}")
        logger.info(f"Orthonormal basis shape => {E.shape}")

        # Projector construction
        P = np.eye(L) - E @ E.T

        # Compute projection residuals
        residuals = []
        for ii in range(M):
            res = LA.norm(P @ D[:, ii]) / LA.norm(D[:, ii])
            residuals.append(res)
        # Convert residuals to array
        residuals = np.array(residuals)

        # Reorder residuals based on error
        ordered_residuals = np.sort(residuals)
        ordered_residuals_indices = np.argsort(residuals)

        # Keep only indices for which residuals are lower than error threshold
        subset_count = sum(ordered_residuals < self.error_th)
        logger.info(f"Number of atoms to be kept: {subset_count}")
        keep_indices = ordered_residuals_indices[:subset_count]
        logger.debug(f"Indices kept: {keep_indices}")

        # Prune original dictionary
        D = D[:, keep_indices]

        # Recompute dictionary shape for subsequent operations
        LD, M = D.shape
        logger.info(f"Pruned dictionary dimensions: {D.shape}")

        # breakpoint()
        # CLSUnSAL

        lambd = self.lambd

        # # Compute mean norm
        # NOTE Legacy code
        norm_d = np.sqrt(np.mean(D**2))
        logger.info(f"Norm D => {norm_d:.3e}")
        # Rescale D, Y and lambda
        D = D / norm_d
        Y = Y / norm_d
        lambd = lambd / norm_d**2
        logger.info(f"Lambda initial value => {lambd:.3e}")

        # Constants and initialization
        # mu = 0.1
        mu = self.mu
        logger.info(f"Mu initial value => {mu:.3e}")

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
        logger.info(f"Tolerance => {tol:.3e}")
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


class HySime(object):
    """Hyperspectral signal subspace identification by minimum error."""

    def __init__(self):
        self.kf = None
        self.Ek = None

    def count(self, M):
        """
        Hyperspectral signal subspace estimation.

        Parameters:
            M: `numpy array`
                Hyperspectral data set (each row is a pixel)
                with ((m*n) x p), where p is the number of bands
                and (m*n) the number of pixels.

        Returns: `tuple integer, numpy array`
            * kf signal subspace dimension
            * Ek matrix which columns are the eigenvectors that span
              the signal subspace.

        Reference:
            Bioucas-Dias, Jose M., Nascimento, Jose M. P., 'Hyperspectral Subspace Identification',
            IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 46, NO. 8, AUGUST 2008.

        Copyright:
            Jose Nascimento (zen@isel.pt) & Jose Bioucas-Dias (bioucas@lx.it.pt)
            For any comments contact the authors.
        """
        h, w, numBands = M.shape
        Mr = np.reshape(M, (w * h, numBands))
        w, Rw = est_noise(Mr)
        self.kf, self.Ek = hysime(Mr, w, Rw)
        return self.kf, self.Ek


def est_noise(y, noise_type="additive"):
    """
    This function infers the noise in a
    hyperspectral data set, by assuming that the
    reflectance at a given band is well modelled
    by a linear regression on the remaining bands.

    Parameters:
        y: `numpy array`
            a HSI cube ((m*n) x p)

       noise_type: `string [optional 'additive'|'poisson']`

    Returns: `tuple numpy array, numpy array`
        * the noise estimates for every pixel (N x p)
        * the noise correlation matrix estimates (p x p)

    Copyright:
        Jose Nascimento (zen@isel.pt) and Jose Bioucas-Dias (bioucas@lx.it.pt)
        For any comments contact the authors
    """

    def est_additive_noise(r):
        small = 1e-6
        L, N = r.shape
        w = np.zeros((L, N), dtype=np.float64)
        RR = np.dot(r, r.T)
        RRi = np.linalg.pinv(RR + small * np.eye(L))
        RRi = np.matrix(RRi)
        for i in range(L):
            XX = RRi - (RRi[:, i] * RRi[i, :]) / RRi[i, i]
            RRa = RR[:, i]
            RRa[i] = 0
            beta = np.dot(XX, RRa)
            beta[0, i] = 0
            w[i, :] = r[i, :] - np.dot(beta, r)
        Rw = np.diag(np.diag(np.dot(w, w.T) / N))
        return w, Rw

    y = y.T
    L, N = y.shape
    # verb = 'poisson'
    if noise_type == "poisson":
        sqy = np.sqrt(y * (y > 0))
        u, Ru = est_additive_noise(sqy)
        x = (sqy - u) ** 2
        w = np.sqrt(x) * u * 2
        Rw = np.dot(w, w.T) / N
    # additive
    else:
        w, Rw = est_additive_noise(y)
    return w.T, Rw.T


def hysime(y, n, Rn):
    """
    Hyperspectral signal subspace estimation

    Parameters:
        y: `numpy array`
            hyperspectral data set (each row is a pixel)
            with ((m*n) x p), where p is the number of bands
            and (m*n) the number of pixels.

        n: `numpy array`
            ((m*n) x p) matrix with the noise in each pixel.

        Rn: `numpy array`
            noise correlation matrix (p x p)

    Returns: `tuple integer, numpy array`
        * kf signal subspace dimension
        * Ek matrix which columns are the eigenvectors that span
          the signal subspace.

    Copyright:
        Jose Nascimento (zen@isel.pt) & Jose Bioucas-Dias (bioucas@lx.it.pt)
        For any comments contact the authors
    """
    y = y.T
    n = n.T
    Rn = Rn.T
    L, N = y.shape
    Ln, Nn = n.shape
    d1, d2 = Rn.shape

    x = y - n

    Ry = np.dot(y, y.T) / N
    Rx = np.dot(x, x.T) / N
    E, dx, V = np.linalg.svd(Rx)

    Rn = Rn + np.sum(np.diag(Rx)) / L / 10**5 * np.eye(L)
    Py = np.diag(np.dot(E.T, np.dot(Ry, E)))
    Pn = np.diag(np.dot(E.T, np.dot(Rn, E)))
    cost_F = -Py + 2 * Pn
    kf = np.sum(cost_F < 0)
    ind_asc = np.argsort(cost_F)
    Ek = E[:, ind_asc[0:kf]]
    return kf, Ek  # Ek.T ?
