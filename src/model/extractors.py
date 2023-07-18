"""
Endmembers extractor methods
"""

import logging
import warnings
import sys
import os
import time

import numpy as np
import numpy.linalg as LA
import numpy.linalg as lin

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from src import EPS

try:
    import matlab.engine
except Exception:
    warnings.warn("matlab.engine was not imported. MATLAB code will not work")


class BaseExtractor:
    def __init__(self):
        self.seed = None

    def extract_endmembers(self, Y, p, seed=0, *args, **kwargs):
        return NotImplementedError

    def __repr__(self):
        msg = f"{self.__class__.__name__}_seed{self.seed}"
        return msg

    def print_time(self, timer):
        msg = f"{self} took {timer:.2f} seconds..."
        return msg


class RandomPositiveMatrix(BaseExtractor):
    def __init__(self):
        super().__init__()

    def extract_endmembers(self, Y, p, seed=0, *args, **kwargs):
        L, N = Y.shape
        self.seed = seed
        generator = np.random.default_rng(seed=self.seed)
        return generator.random(size=(L, p))


class RandomPixels(BaseExtractor):
    def __init__(self):
        super().__init__()

    def extract_endmembers(self, Y, p, seed=0, *args, **kwargs):
        L, N = Y.shape
        self.seed = seed
        generator = np.random.default_rng(seed=self.seed)
        indices = generator.integers(low=0, high=N, size=p)
        pixels = Y[:, indices]
        assert pixels.shape == (L, p)
        return pixels


class VCA(BaseExtractor):
    def __init__(self):
        super().__init__()

    def extract_endmembers(self, Y, p, seed=0, snr_input=0, *args, **kwargs):
        """
        Vertex Component Analysis

        This code is a translation of a matlab code provided by
        Jose Nascimento (zen@isel.pt) and Jose Bioucas Dias (bioucas@lx.it.pt)
        available at http://www.lx.it.pt/~bioucas/code.htm under a non-specified Copyright (c)
        Translation of last version at 22-February-2018 (Matlab version 2.1 (7-May-2004))

        more details on:
        Jose M. P. Nascimento and Jose M. B. Dias
        "Vertex Component Analysis: A Fast Algorithm to Unmix Hyperspectral Data"
        submited to IEEE Trans. Geosci. Remote Sensing, vol. .., no. .., pp. .-., 2004
        """
        L, N = Y.shape
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

    def extract_endmembers(self, Y, p, seed=0, *args, **kwargs):

        x, p = Y, p

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


class SISAL(BaseExtractor):
    def __init__(
        self,
    ):
        super().__init__()

    @staticmethod
    def soft_neg(y, tau):
        """
        z = soft_neg(y,tau);

        negative soft (proximal operator of the hinge function)
        """

        z = np.maximum(np.abs(y + tau / 2) - tau / 2, 0)
        z = z / (z + tau / 2) * (y + tau / 2)
        return z

    def extract_endmembers(self, Y, p, seed=0, snr_input=0, *args, **kwargs):
        """
        M,Up,my,sing_values = sisal(Y,p,**kwargs)

        ----- Description ---------------

        Simplex identification via split augmented Lagrangian (SISAL) estimates
        the vertices  M={m_1,...m_p} of the (p-1)-dimensional simplex of minimum
        volume containing the vectors [y_1,...y_N], under the assumption that y_i
        belongs to a (p-1)  dimensional affine set.

        For details see

        [1] José M. Bioucas-Dias, "A variable splitting augmented lagrangian
        approach to linear spectral unmixing", First IEEE GRSS Workshop on
        Hyperspectral Image and Signal Processing - WHISPERS, 2009.
        http://arxiv.org/abs/0904.4635v1


        ----- Input ---------------------

        Y - matrix with dimension  L(channels) x N(pixels). Each pixel is a linear
            mixture of p endmembers signatures Y = M*x + noise.

        p - number of independent columns of M. Therefore, M spans a (p-1)-dimensional
            affine set. p is the number of endmembers.

        ----- Optional input ------------


        mm_iters - Maximum number of constrained quadratic programs
                Default: 80

        tau - Regularization parameter in the problem
                Q^* = arg min_Q  -\log abs(det(Q)) + tau*|| Q*yp ||_h
                    subject to np.ones((1,p))*Q=mq
                where mq = ones(1,N)*yp'inv(yp*yp) and ||x||_h is the "hinge"
                induced norm (see [1]).
            Default: 1

        mu - Augmented Lagrange regularization parameter
            Default: 1

        spherize - {True, False} Applies a spherization step to data such that the spherized
                data spans over the same range along any axis.
                Default: True

        tolf - Tolerance for the termination test (relative variation of f(Q))
            Default: 1e-2

        M0 - Initial M, dimension L x p.
            Defaults is given by the VCA algorithm.

        verbose - {0,1,2,3}
                        0 - work silently
                        1 - display simplex volume
                        2 - display figures
                        3 - display SISAL information
                        4 - display SISAL information and figures
                Default: 1

        ----- Output --------------------

        M - estimated endmember signature matrix L x p

        Up - isometric matrix spanning the same subspace as M, imension is L x p

        my - mean value of Y

        sing_values - (p-1) eigenvalues of Cy = (y-my)*(y-my)/N. The dynamic range
                    of these eigenvalues gives an idea of the  difficulty of the
                    underlying problem

        ----- Note ----------------------

        The identified affine set is given by
            {z\in R^p : z=Up(:,1:p-1)*a+my, a\in R^(p-1)}


        ----- License -------------------
        Author: Etienne Monier (etienne.monier@enseeiht.fr)

        This code is an improvement over a translation of a matlab code provided by
        Jose Nascimento (zen@isel.pt) and Jose Bioucas Dias (bioucas@lx.it.pt)
        available at http://www.lx.it.pt/~bioucas/code.htm under a non-specified Copyright (c)
        Translation of last version at 20-April-2018 (Matlab version 2.1 (7-May-2004))
        Improvements made on 07-Feb-2023.

        ----- Bug fixes ----------------

        - Matrix conditioning was not ideal prior to inverting.
          Fixed it by introducing (small) diagonal coefficients
        - Flattening was used in row-major mode ('C') but column-major
          is required to match MATLAB ('F')
        - Spherization is not used by default
          (showed worse results in terms of SAD)

        """

        #
        # -------------------------------------------------------------------------
        #
        #
        # --------------------------------------------------------------
        # test for number of required parametres
        # --------------------------------------------------------------

        # data set size
        L, N = Y.shape
        if L < p:
            raise ValueError("Insufficient number of columns in y")

        ##
        # --------------------------------------------------------------
        # Set the defaults for the optional parameters
        # --------------------------------------------------------------
        # maximum number of quadratic QPs

        MMiters = 40
        # spherize = True
        spherize = False
        # display only volume evolution
        # verbose = 3
        verbose = 3
        # soft constraint regularization parameter
        tau = 10
        # tau = 1
        # tau = 1000
        # Augmented Lagrangian regularization parameter
        mu = p * 1000 / N
        # no initial simplex
        M = 0
        # tolerance for the termination test
        tol_f = 1e-2

        ##
        # --------------------------------------------------------------
        # Local variables
        # --------------------------------------------------------------
        # maximum violation of inequalities
        slack = 1e-3
        # flag energy decreasing
        energy_decreasing = 0
        # used in the termination test
        f_val_back = float("inf")
        #
        # spherization regularization parameter
        lam_sphe = 1e-8
        # quadractic regularization parameter for the Hesssian
        # Hreg = = mu*I
        lam_quad = 1e-6
        # minimum number of AL iterations per quadratic problem
        AL_iters = 4
        # flag
        flaged = 0

        # --------------------------------------------------------------
        # Read the optional parameters
        # --------------------------------------------------------------

        for key in kwargs:
            Ukey = key.upper()

            if Ukey == "MM_ITERS":
                MMiters = kwargs[key]
            elif Ukey == "SPHERIZE":
                spherize = kwargs[key]
            elif Ukey == "MU":
                mu = kwargs[key]
            elif Ukey == "TAU":
                tau = kwargs[key]
            elif Ukey == "TOLF":
                tol_f = kwargs[key]
            elif Ukey == "M0":
                M = kwargs[key]
            elif Ukey == "VERBOSE":
                verbose = kwargs[key]
            elif Ukey == "SNR_INPUT":
                snr_input = kwargs[key]
            else:
                # Hmmm, something wrong with the parameter string
                # raise ValueError("Unrecognized option: {}".format(key))
                pass

        ##
        # --------------------------------------------------------------
        # set display mode
        # --------------------------------------------------------------
        if (verbose == 3) or (verbose == 4):
            warnings.filterwarnings("ignore")
        else:
            warnings.filterwarnings("always")

        ##
        # --------------------------------------------------------------
        # identify the affine space that best represent the data set y
        # --------------------------------------------------------------
        meanY = Y.mean(axis=1, keepdims=True)
        Y = Y - meanY
        Up, D, _ = lin.svd((Y @ Y.T) / N)
        Up = Up[:, : p - 1]
        d = D[: p - 1]

        # represent y in the subspace R^(p-1)
        Y = Up @ Up.T @ Y
        # lift y
        Y = Y + meanY
        # compute the orthogonal component of my
        meanY_ortho = meanY - Up @ Up.T.dot(meanY)
        # define another orthonormal direction
        Up = np.concatenate(
            (Up, meanY_ortho / np.sqrt(np.sum(meanY_ortho**2))),
            axis=1,
        )
        # get coordinates in R^p
        Y = Up.T @ Y

        ##
        # ------------------------------------------
        # spherize if requested
        # ------------------------------------------
        if spherize:
            logger.debug("Spherize requested...")
            Y = Up @ Y
            Y = Y - meanY
            C = np.diag(1 / np.sqrt(d + lam_sphe))
            IC = lin.inv(C)
            # Y = C.dot(np.transpose(Up[:, : p - 1])).dot(Y)
            Y = C @ Up[:, : p - 1].T @ Y
            # lift
            Y = np.concatenate((Y, np.ones((1, N))), axis=0)
            # Y[p-1,:] = 1
            # normalize to unit norm
            Y = Y / np.sqrt(p)

        ##
        # ---------------------------------------------
        #            Initialization
        # ---------------------------------------------
        if M == 0:
            # Initialize with VCA
            M = VCA().extract_endmembers(Y, p, seed=seed, snr_input=snr_input)
            # NOTE We do not expand Q and start directly from VCA estimate
        else:
            # Ensure that M is in the affine set defined by the data
            # M = M - Myp
            M = M - meanY
            M = Up[:, : p - 1] @ Up[:, : p - 1].T @ M
            # M = M + Myp
            M = M + meanY
            M = Up.T @ M  # represent in the data subspace
            # is sherization is set
            if spherize:
                # M = Up @ M - Myp
                M = Up @ M - meanY
                M = C @ Up[:, : p - 1].T @ M
                # lift
                M[p - 1, :] = 1
                # normalize to unit norm
                M = M / np.sqrt(p)

        Q0 = lin.inv(M)
        Q = Q0

        #
        # ---------------------------------------------
        #            Build constant matrices
        # ---------------------------------------------
        YYT = Y @ Y.T
        YYT_cond = YYT + 1e-3 * np.eye(
            p
        )  # NOTE better conditioning when solving linear system

        AAT = np.kron(YYT, np.eye(p))  # size p^2xp^2
        B = np.kron(np.eye(p), np.ones((1, p)))  # size pxp^2
        # qm = np.sum(lin.inv(Y @ Y.T) @ Y, axis=1)
        qm = np.sum(lin.solve(YYT_cond, Y), axis=1)  # NOTE Faster than solving inverse

        H = lam_quad * np.eye(p**2)
        F = H + mu * AAT  # equation (11) of [1]
        IF = lin.inv(F)

        # auxiliar constant matrices
        G = (
            IF @ B.T @ lin.inv(B @ IF @ B.T + lam_quad * np.eye(p))
        )  # NOTE better conditioning when inverting
        qm_aux = G.dot(qm)
        G = IF - G @ B @ IF

        ##
        # ---------------------------------------------------------------
        #          Main body- sequence of quadratic-hinge subproblems
        # ----------------------------------------------------------------

        # initializations
        Z = Q @ Y
        Bk = 0 * Z

        hinge = lambda x: np.maximum(-x, 0)

        # NOTE Matlab uses column-major mode ('F' for Fortran) for flattening
        for k in range(MMiters):

            IQ = lin.inv(Q)
            g = -IQ.T
            g = g.flatten(order="F")

            baux = H @ Q.flatten(order="F") - g

            q0 = Q.flatten(order="F")
            Q0 = Q

            # display the simplex volume
            if verbose == 1:
                if spherize:
                    # unscale
                    M = IQ * np.sqrt(p)
                    # remove offset
                    M = M[: p - 1, :]
                    # unspherize
                    M = Up[:, : p - 1].dot(IC).dot(M)
                    # sum ym
                    # M = M + Myp
                    M = M + meanY
                    M = Up.T.dot(M)
                else:
                    M = IQ

                logger.debug(
                    "iter = {0}, simplex volume = {1:.4e}  \n".format(
                        k, 1 / np.abs(lin.det(M))
                    )
                )

            if k == MMiters - 1:  # NOTE Fixed where this condition was never met
                AL_iters = 100

            # jj = 0
            while 1 > 0:
                q = Q.flatten(order="F")
                # initial function values (true and quadratic)
                f0_val = -np.log(np.abs(lin.det(Q))) + tau * np.sum(hinge(Q @ Y))
                f0_quad = (
                    (q - q0).T.dot(g)
                    + 0.5 * (q - q0).T.dot(H).dot(q - q0)
                    + tau * np.sum(hinge(Q.dot(Y)))
                )
                for i in range(AL_iters - 1):
                    # -------------------------------------------
                    # solve quadratic problem with constraints
                    # -------------------------------------------
                    dq_aux = Z + Bk  # matrix form
                    dtz_b = dq_aux @ Y.T
                    dtz_b = dtz_b.flatten(order="F")
                    b = baux + mu * dtz_b  # (11) of [1]
                    q = G.dot(b) + qm_aux  # (10) of [1]
                    # NOTE q was exploding => flatten order was wrong!
                    Q = np.reshape(q, (p, p), order="F")

                    # -------------------------------------------
                    # solve hinge
                    # -------------------------------------------
                    Z = self.soft_neg(Q @ Y - Bk, tau / mu)

                    # -------------------------------------------
                    # update Bk
                    # -------------------------------------------

                    Bk = Bk - (Q @ Y - Z)
                    if verbose == 3 or verbose == 4:
                        logger.debug(
                            "||Q*Y-Z|| = {0:.4e}".format(lin.norm(Q.dot(Y) - Z))
                        )

                f_quad = (
                    (q - q0).T.dot(g)
                    + 0.5 * (q - q0).T.dot(H).dot(q - q0)
                    + tau * np.sum(hinge(Q @ Y))
                )
                if verbose == 3 or verbose == 4:
                    logger.debug(
                        "MMiter = {0}, AL_iter, = {1},  f0_quad = {2:.4e}, f_quad = {3:.4e},".format(
                            k, i, f0_quad, f_quad
                        )
                    )

                f_val = -np.log(np.abs(lin.det(Q))) + tau * np.sum(hinge(Q.dot(Y)))

                if f0_quad >= f_quad:  # quadratic energy decreased
                    while f0_val < f_val:
                        if verbose == 3 or verbose == 4:
                            logger.debug(
                                "line search, MMiter = {0}, AL_iter, = {1},  f0_val = {2:.4e}, f_val = {3:.4e},".format(
                                    k, i, f0_val, f_val
                                )
                            )

                        # do line search
                        Q = (Q + Q0) / 2
                        f_val = (
                            -np.log(np.abs(lin.det(Q))) + tau * hinge(Q @ Y).sum()
                        )  # NOTE Fixed sum computation

                    break

        if spherize:
            M = lin.inv(Q)
            # refer to the initial affine set
            # unscale

            # remove offset
            M = M[: p - 1, :]
            # unspherize
            M = Up[:, : p - 1].dot(IC).dot(M)
            # sum ym
            # M = M + Myp
            M = M + meanY
        else:
            M = Up.dot(lin.inv(Q))

        return M


class VCA_MATLAB(BaseExtractor):
    def __init__(
        self,
        root_matlab,
    ):
        super().__init__()

        self.root_matlab = "/home/azouaoui/matlab"
        path_to_VCA = os.path.join(self.root_matlab, "sisal_demo")
        assert os.path.exists(path_to_VCA), "Change path to your location of VCA"

        # Start matlab engine
        self.engine = matlab.engine.start_matlab()
        logger.debug("MATLAB engine started...")
        # Go to VCA code
        self.engine.cd(path_to_VCA)

    def extract_endmembers(self, Y, p, snr_input=0, *args, **kwargs):
        tic = time.time()

        Ehat = self.engine.VCA(
            matlab.double(Y.tolist()),
            "Endmembers",
            matlab.double([p]),
            "snr",
            matlab.double([snr_input]),
        )
        Ehat = np.array(Ehat).astype(np.float32)
        tac = time.time()
        self.time = round(tac - tic, 2)
        logger.info(f"VCA MATLAB took {self.time} seconds")
        return Ehat


class SISAL_MATLAB(BaseExtractor):
    def __init__(
        self,
        root_matlab,
    ):
        super().__init__()

        self.root_matlab = root_matlab
        path_to_SISAL = os.path.join(self.root_matlab, "sisal_demo")
        assert os.path.exists(path_to_SISAL), "Change path to your location of SISAL"

        # Start matlab engine
        self.engine = matlab.engine.start_matlab()
        logger.debug("MATLAB engine started...")
        # Go to SISAL code
        self.engine.cd(path_to_SISAL)

    def extract_endmembers(self, Y, p, seed=0, *args, **kwargs):
        tic = time.time()

        Ehat = self.engine.sisal(
            matlab.double(Y.tolist()),
            matlab.double([p]),
            "spherize",
            "no",  # NOTE do not use spherization by default
            "MM_ITERS",
            matlab.double([40]),
            "TAU",
            matlab.double([10]),
            "verbose",
            matlab.int8([3]),
        )

        Ehat = np.array(Ehat).astype(np.float32)
        tac = time.time()
        self.time = round(tac - tic, 2)
        logger.info(f"SISAL MATLAB took {self.time} seconds")
        return Ehat
