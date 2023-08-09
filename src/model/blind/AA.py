"""
Archetypal Anaysis based blind unmixing methods
"""
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
import spams
from tqdm import tqdm

from .base import BlindUnmixingModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ArchetypalAnalysis(BlindUnmixingModel):
    def __init__(
        self,
        epsilon=1e-3,
        robust=False,
        computeXtX=True,
        stepsFISTA=3,
        stepsAS=50,
        randominit=True,
        numThreads=-1,
        *args,
        **kwargs,
    ):

        super().__init__()
        self.params = {
            "epsilon": epsilon,
            "robust": robust,
            "computeXtX": computeXtX,
            "stepsFISTA": stepsFISTA,
            "stepsAS": stepsAS,
            "randominit": randominit,
            "numThreads": numThreads,
        }

    def compute_endmembers_and_abundances(
        self,
        Y,
        p,
        *args,
        **kwargs,
    ):
        """
        Archetypal Analysis optimizer from SPAMS

        Parameters:
            Y: `numpy array`
                2D data matrix (L x N)

            p: `int`
                Number of endmembers

            E0: `numpy array`
                2D initial endmember matrix (L x p)
                Default: None

        Source: http://thoth.inrialpes.fr/people/mairal/spams/doc-python/html/doc_spams004.html#sec8
        """
        tic = time.time()
        Yf = np.asfortranarray(Y, dtype=np.float64)

        Ehat, Asparse, Bsparse = spams.archetypalAnalysis(
            Yf,
            p=p,
            returnAB=True,
            **self.params,
        )

        Ahat = np.array(Asparse.todense())
        self.B = np.array(Bsparse.todense())
        tac = time.time()
        self.time = round(tac - tic, 2)
        logger.info(self.print_time())

        return Ehat, Ahat


class EDAA(BlindUnmixingModel):
    def __init__(
        self,
        T=100,
        K1=5,
        K2=5,
        M=50,
        normalize=True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.T = T
        self.K1 = K1
        self.K2 = K2
        self.M = M
        self.normalize = normalize
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def compute_endmembers_and_abundances(
        self,
        Y,
        p,
        seed=0,
        *args,
        **kwargs,
    ):
        best_E = None
        best_A = None

        _, N = Y.shape

        def loss(a, b):
            return 0.5 * ((Y - (Y @ b) @ a) ** 2).sum()

        def residual_l1(a, b):
            return (Y - (Y @ b) @ a).abs().sum()

        def grad_A(a, b):
            YB = Y @ b
            ret = -YB.t() @ (Y - YB @ a)
            return ret

        def grad_B(a, b):
            return -Y.t() @ ((Y - Y @ b @ a) @ a.t())

        def update(a, b):
            return F.softmax(torch.log(a) + b, dim=0)

        def computeLA(b):
            YB = Y @ b
            S = torch.linalg.svdvals(YB)
            return S[0] * S[0]

        def max_correl(e):
            return np.max(np.corrcoef(e.T) - np.eye(p))

        results = {}

        tic = time.time()

        # # L2 Normalization here?
        # if self.normalize:
        #     logger.debug("Applying L2 Normalization on input data...")
        #     Y = Y / np.linalg.norm(Y, axis=0, ord=2, keepdims=True)

        # Convert data to tensor
        Y = torch.Tensor(Y)

        for m in tqdm(range(self.M)):
            torch.manual_seed(m + seed)
            generator = np.random.RandomState(m + seed)

            with torch.no_grad():

                # Matrix initialization
                B = F.softmax(0.1 * torch.rand((N, p)), dim=0)
                A = (1 / p) * torch.ones((p, N))

                # Send matrices on GPU
                Y = Y.to(self.device)
                A = A.to(self.device)
                B = B.to(self.device)

                # Random Step size factor
                factA = 2 ** generator.randint(-3, 4)

                # Compute step sizes
                self.etaA = factA / computeLA(B)
                self.etaB = self.etaA * ((p / N) ** 0.5)

                for _ in range(self.T):
                    for _ in range(self.K1):
                        A = update(A, -self.etaA * grad_A(A, B))

                    for _ in range(self.K2):
                        B = update(B, -self.etaB * grad_B(A, B))

                fit_m = residual_l1(A, B).item()
                # fit_m = loss(A, B).item()
                E = (Y @ B).cpu().numpy()
                A = A.cpu().numpy()
                B = B.cpu().numpy()
                Rm = max_correl(E)
                # Store results
                results[m] = {
                    "Rm": Rm,
                    "Em": E,
                    "Am": A,
                    "Bm": B,
                    "fit_m": fit_m,
                    "factA": factA,
                }

        min_fit_l1 = np.min([v["fit_m"] for v in results.values()])

        def fit_l1_cutoff(idx, tol=0.05):
            val = results[idx]["fit_m"]
            return (abs(val - min_fit_l1) / abs(val)) < tol

        sorted_indices = sorted(
            filter(fit_l1_cutoff, results),
            key=lambda x: results[x]["Rm"],
        )

        best_result_idx = sorted_indices[0]
        best_result = results[best_result_idx]

        best_E = best_result["Em"]
        best_A = best_result["Am"]
        self.B = best_result["Bm"]

        toc = time.time()
        self.time = toc - tic
        logger.info(self.print_time())

        return best_E, best_A
