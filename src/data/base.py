"""
HSI data base class
"""
import logging
from typing import Any
import os

from hydra.utils import to_absolute_path
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

from src import EPS

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

INTEGER_VALUES = ("H", "W", "M", "L", "p", "N")


class HSI:
    def __init__(
        self,
        name: str,
        data_dir: str = "./data",
        figs_dir: str = "./figs",
        noise=None,
    ) -> None:
        # Locate and check data file
        self.name = name
        filename = f"{self.name}.mat"
        path = to_absolute_path(os.path.join(data_dir, filename))
        logger.debug(f"Path to be opened: {path}")
        assert os.path.isfile(path)

        # Open data file
        data = sio.loadmat(path)
        logger.debug(f"Data keys: {data.keys()}")

        # Populate attributes based on data file values
        for key in filter(
            lambda k: not k.startswith("__"),
            data.keys(),
        ):
            self.__setattr__(
                key, data[key].item() if key in INTEGER_VALUES else data[key]
            )

        # Noise
        self.noise = noise
        # self.noise_applied = False
        self.projection_applied = False

        if "N" not in data.keys():
            self.N = self.H * self.W

        # Check data
        assert self.N == self.H * self.W
        assert self.E.shape == (self.L, self.p)
        assert self.Y.shape == (self.L, self.N)
        assert self.A.shape == (self.p, self.N)

        self.has_dict = False
        if "D" in data.keys():
            self.has_dict = True
            assert self.D.shape == (self.L, self.M)

        try:
            assert len(self.labels) == self.p
            tmp_labels = list(self.labels)
            self.labels = [s.strip(" ") for s in tmp_labels]

        except Exception:
            # Create numeroted labels
            self.labels = [f"#{ii}" for ii in range(self.p)]

        # Check physical constraints
        # Abundance Sum-to-One Constraint (ASC)
        assert np.allclose(self.A.sum(0), np.ones(self.N))
        # Abundance Non-negative Constraint (ANC)
        assert np.all(self.A >= -EPS)
        # Endmembers Non-negative Constraint (ENC)
        assert np.all(self.E >= -EPS)

        # Create output figures folder
        self.figs_dir = os.path.join(os.getcwd(), figs_dir)
        if self.figs_dir is not None:
            os.makedirs(self.figs_dir, exist_ok=True)

    def apply_noise(self, seed=0):
        # Apply noise but only once!
        # if not self.noise_applied:
        logger.info("Applying noise to input HSI")
        self.Y_noisy = self.noise.fit_transform(self.Y, seed=seed)
        # self.noise_applied = True
        # else:
        #     logger.debug("No noise were applied...")

    def apply_projection(self):
        if not self.projection_applied:
            logger.info("Applying SVD projection to input HSI")
            self.Y_noisy = self.svd_projection(self.Y_noisy, self.p)
            self.projection_applied = True
        else:
            logger.debug("No projection were applied...")

    @staticmethod
    def svd_projection(Y, p):
        V, SS, U = np.linalg.svd(Y, full_matrices=False)
        PC = np.diag(SS) @ U
        denoised_image_reshape = V[:, :p] @ PC[:p]
        return np.clip(denoised_image_reshape, 0, 1)

    def sample(self, expand_abundances=False):
        Y_noisy = np.copy(self.Y_noisy)
        E = np.copy(self.E)
        A = np.copy(self.A)
        if expand_abundances:
            A = np.vstack((A, np.zeros((self.M - self.p, self.N))))
            assert A.shape[0] == self.M
        D = np.copy(self.D) if self.has_dict else None
        return Y_noisy, E, A, D

    def __repr__(self) -> str:
        msg = f"HSI => {self.name}\n"
        msg += "------------------------------\n"
        msg += f"{self.L} bands,\n"
        msg += f"{self.H} lines, {self.W} samples ({self.N} pixels),\n"
        msg += f"{self.p} endmembers ({self.labels})\n"
        msg += f"GlobalMinValue: {self.Y.min()}, GlobalMaxValue: {self.Y.max()}\n"
        return msg

    def plot_endmembers(self):
        """
        Display endmembers
        """
        title = f"{self.name} - GT endmembers"
        ylabel = "Reflectance"
        xlabel = "# Bands"
        E = np.copy(self.E)

        plt.figure(figsize=(6, 6))
        for pp in range(self.p):
            plt.plot(E[:, pp], label=self.labels[pp])
        plt.title(title)
        plt.legend(frameon=True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        figname = f"{self.name}-endmembers-GT.png"
        plt.savefig(os.path.join(self.figs_dir, figname))
        plt.close()

    def plot_abundances(self):
        nrows, ncols = (1, self.p)
        title = f"{self.name} - GT abundances"
        A = np.copy(self.A)
        A = A.reshape(self.p, self.H, self.W)

        fig, ax = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(12, 4 * nrows),
        )
        kk = 0
        for ii in range(nrows):
            for jj in range(ncols):
                if nrows == 1:
                    curr_ax = ax[jj]
                else:
                    curr_ax = ax[ii, jj]
                mappable = curr_ax.imshow(A[kk], vmin=0.0, vmax=1.0)
                curr_ax.set_title(f"{self.labels[kk]}")
                curr_ax.axis("off")
                fig.colorbar(mappable, ax=curr_ax, location="right", shrink=0.5)

                kk += 1
                if kk == self.p:
                    break

        plt.suptitle(title)
        figname = f"{self.name}-abundances-GT.png"
        plt.savefig(os.path.join(self.figs_dir, figname))
        plt.close()
