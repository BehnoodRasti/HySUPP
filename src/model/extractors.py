"""
Endmembers extractor methods
"""

import logging

import numpy as np
import numpy.linalg as LA

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BaseExtractor:
    def __init__(self):
        self.seed = None

    def extract_endmembers(self, hsi, seed=0, *args, **kwargs):
        return NotImplementedError

    def __repr__(self):
        msg = f"{self.__class__.__name__}_seed{self.seed}"
        return msg


class GroundTruthEndmembers(BaseExtractor):
    def __init__(self):
        super().__init__()

    def extract_endmembers(self, hsi, *args, **kwargs):
        return hsi.E


class RandomPositiveMatrix(BaseExtractor):
    def __init__(self):
        super().__init__()

    def extract_endmembers(self, hsi, seed=0, *args, **kwargs):
        self.seed = seed
        generator = np.random.default_rng(seed=self.seed)
        return generator.random(size=(hsi.L, hsi.p))


class RandomPixels(BaseExtractor):
    def __init__(self):
        super().__init__()

    def extract_endmembers(self, hsi, seed=0, *args, **kwargs):
        self.seed = seed
        generator = np.random.default_rng(seed=self.seed)
        indices = generator.integers(low=0, high=hsi.N, size=hsi.p)
        pixels = hsi.Y[:, indices]
        assert pixels.shape == hsi.E.shape
        return pixels


class VCA(BaseExtractor):
    def __init__(self):
        super().__init__()

    def extract_endmembers(self, hsi, seed=0, snr_input=0, *args, **kwargs):
        """
        Vertex Component Analysis

        ------- Input variables -------------
        HSI containing the following variables =>
         Y - matrix with dimensions L(channels) x N(pixels)
             each pixel is a linear mixture of R endmembers
             signatures Y = M x s, where s = gamma x alfa
             gamma is a illumination perturbation factor and
             alfa are the abundance fractions of each endmember.
         p - positive integer number of endmembers in the scene

        ------- Output variables -----------
        E     - estimated mixing matrix (endmembers signatures)

        ------- Optional parameters---------
        snr_input - (float) signal to noise ratio (dB)
        ------------------------------------

        Author: Adrien Lagrange (adrien.lagrange@enseeiht.fr)
        This code is a translation of a matlab code provided by
        Jose Nascimento (zen@isel.pt) and Jose Bioucas Dias (bioucas@lx.it.pt)
        available at http://www.lx.it.pt/~bioucas/code.htm under a non-specified Copyright (c)
        Translation of last version at 22-February-2018 (Matlab version 2.1 (7-May-2004))

        more details on:
        Jose M. P. Nascimento and Jose M. B. Dias
        "Vertex Component Analysis: A Fast Algorithm to Unmix Hyperspectral Data"
        submited to IEEE Trans. Geosci. Remote Sensing, vol. .., no. .., pp. .-., 2004
        """
        Y = hsi.Y
        N, p = hsi.N, hsi.p
        self.seed = seed
        generator = np.random.default_rng(seed=self.seed)

        #############################################
        # SNR Estimates
        #############################################

        if snr_input == 0:
            y_m = np.mean(Y, axis=1, keepdims=True)
            Y_o = Y - y_m  # data with zero-mean
            Ud = LA.svd(np.dot(Y_o, Y_o.T) / float(N))[0][
                :, :p
            ]  # computes the R-projection matrix
            x_p = np.dot(Ud.T, Y_o)  # project the zero-mean data onto p-subspace

            SNR = self.estimate_snr(Y, y_m, x_p)

            logger.info(f"SNR estimated = {SNR}[dB]")
        else:
            SNR = snr_input
            logger.info(f"input SNR = {SNR}[dB]\n")

        SNR_th = 15 + 10 * np.log10(p)
        #############################################
        # Choosing Projective Projection or
        #          projection to p-1 subspace
        #############################################

        if SNR < SNR_th:
            logger.info("... Select proj. to R-1")

            d = p - 1
            if snr_input == 0:  # it means that the projection is already computed
                Ud = Ud[:, :d]
            else:
                y_m = np.mean(Y, axis=1, keepdims=True)
                Y_o = Y - y_m  # data with zero-mean

                Ud = LA.svd(np.dot(Y_o, Y_o.T) / float(N))[0][
                    :, :d
                ]  # computes the p-projection matrix
                x_p = np.dot(Ud.T, Y_o)  # project thezeros mean data onto p-subspace

            Yp = np.dot(Ud, x_p[:d, :]) + y_m  # again in dimension L

            x = x_p[:d, :]  #  x_p =  Ud.T * Y_o is on a R-dim subspace
            c = np.amax(np.sum(x**2, axis=0)) ** 0.5
            y = np.vstack((x, c * np.ones((1, N))))
        else:
            logger.info("... Select the projective proj.")

            d = p
            Ud = LA.svd(np.dot(Y, Y.T) / float(N))[0][
                :, :d
            ]  # computes the p-projection matrix

            x_p = np.dot(Ud.T, Y)
            Yp = np.dot(
                Ud, x_p[:d, :]
            )  # again in dimension L (note that x_p has no null mean)

            x = np.dot(Ud.T, Y)
            u = np.mean(x, axis=1, keepdims=True)  # equivalent to  u = Ud.T * r_m
            y = x / np.dot(u.T, x)

        #############################################
        # VCA algorithm
        #############################################

        indices = np.zeros((p), dtype=int)
        A = np.zeros((p, p))
        A[-1, 0] = 1

        for i in range(p):
            w = generator.random(size=(p, 1))
            f = w - np.dot(A, np.dot(LA.pinv(A), w))
            f = f / np.linalg.norm(f)

            v = np.dot(f.T, y)

            indices[i] = np.argmax(np.absolute(v))
            A[:, i] = y[:, indices[i]]  # same as x(:,indice(i))

        E = Yp[:, indices]

        logger.debug(f"Indices chosen to be the most pure: {indices}")
        self.indices = indices

        return E

    @staticmethod
    def estimate_snr(Y, r_m, x):
        L, N = Y.shape  # L number of bands (channels), N number of pixels
        p, N = x.shape  # p number of endmembers (reduced dimension)

        P_y = np.sum(Y**2) / float(N)
        P_x = np.sum(x**2) / float(N) + np.sum(r_m**2)
        snr_est = 10 * np.log10((P_x - p / L * P_y) / (P_y - P_x))

        return snr_est


class SiVM(BaseExtractor):
    def __init__(self):
        super().__init__()

    @staticmethod
    def Eucli_dist(x, y):
        a = np.subtract(x, y)
        return np.dot(a.T, a)

    def extract_endmembers(self, hsi, seed=0, *args, **kwargs):

        x, p = hsi.Y, hsi.p

        [D, N] = x.shape
        # If no distf given, use Euclidean distance function
        Z1 = np.zeros((1, 1))
        O1 = np.ones((1, 1))
        # Find farthest point
        d = np.zeros((p, N))
        I = np.zeros((p, 1))
        V = np.zeros((1, N))
        ZD = np.zeros((D, 1))
        for i in range(N):
            d[0, i] = self.Eucli_dist(x[:, i].reshape(D, 1), ZD)

        I = np.argmax(d[0, :])

        for i in range(N):
            d[0, i] = self.Eucli_dist(x[:, i].reshape(D, 1), x[:, I].reshape(D, 1))

        for v in range(1, p):
            D1 = np.concatenate(
                (d[0:v, I].reshape((v, I.size)), np.ones((v, 1))), axis=1
            )
            D2 = np.concatenate((np.ones((1, v)), Z1), axis=1)
            D4 = np.concatenate((D1, D2), axis=0)
            D4 = np.linalg.inv(D4)
            for i in range(N):
                D3 = np.concatenate((d[0:v, i].reshape((v, 1)), O1), axis=0)
                V[0, i] = np.dot(np.dot(D3.T, D4), D3)

            I = np.append(I, np.argmax(V))
            for i in range(N):
                d[v, i] = self.Eucli_dist(
                    x[:, i].reshape(D, 1), x[:, I[v]].reshape(D, 1)
                )

        per = np.argsort(I)
        I = np.sort(I)
        d = d[per, :]
        E = x[:, I]
        logger.debug(f"Indices chosen: {I}")
        return E
