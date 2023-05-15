"""
SUnAA implementation
"""
import logging
import time

import numpy as np
import spams
from tqdm import tqdm

from .base import SemiSupervisedUnmixingModel as SparseUnmixingModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SUnAA(SparseUnmixingModel):
    def __init__(
        self,
        hsi,
        T,
        low_rank=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.p = hsi.p
        self.T = T  # number of iterations
        self.low_rank = low_rank

    def compute_abundances(self, Y, D, *args, **kwargs):
        def loss(a, b):
            return 0.5 * ((Y - (D @ b) @ a) ** 2).sum()

        def update_B(a, b):
            R = Y - (D @ b) @ a
            for jj in range(self.p):
                z_j = D @ b[:, jj]
                norm_aj = np.linalg.norm(a[jj])
                if norm_aj < 1e-10:
                    ZZ = z_j
                else:
                    ZZ = (R @ a[jj]) / (norm_aj**2) + z_j
                bb = spams.decompSimplex(np.asfortranarray(ZZ[:, np.newaxis]), DD)
                b[:, jj] = np.squeeze(bb.todense())
                R = R + (z_j - D @ b[:, jj])[:, np.newaxis] @ a[jj][np.newaxis, :]
            return b

        tic = time.time()

        _, N = Y.shape
        _, N_atoms = D.shape

        YY = np.asfortranarray(Y)
        DD = np.asfortranarray(D)
        B = (1 / N_atoms) * np.ones((N_atoms, self.p))
        A = (1 / self.p) * np.ones((self.p, N))

        logger.info(f"Initial loss => {loss(A, B):.2f}")

        progress = tqdm(range(self.T))
        for pp in progress:
            # B = update_B(A, B)
            B = update_B(A, B)
            # logger.debug(f"B update => {loss(A, B):.2f}")
            A = np.array(spams.decompSimplex(YY, np.asfortranarray(D @ B)).todense())
            # logger.debug(f"A update => {loss(A, B):.2f}")
            progress.set_postfix_str(f"loss={loss(A, B):.2f}")
            if np.isnan(loss(A, B)):
                # Restart
                pp = 0
                B = (1 / N_atoms) * np.ones((N_atoms, self.p))
                A = (1 / self.p) * np.ones((self.p, N))

        tac = time.time()

        self.time = round(tac - tic, 2)

        logger.info(f"{self} took {self.time}s")
        logger.info(f"Final loss => {loss(A, B):.2f}")

        self.E_hat = D @ B
        self.B = B
        self.A_lowrank = A  # low-rank abundances (i.e. $p$ endmembers)

        if self.low_rank:
            return A
        return B @ A  # redundant/full abundances
