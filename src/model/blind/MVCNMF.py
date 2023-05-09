"""
MVCNMF python implementation
"""

import logging
from math import factorial

import numpy as np
from tqdm import tqdm

from sklearn.decomposition import PCA

from .base import BlindUnmixingModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MVCNMF(BlindUnmixingModel):
    """Minimum-volume constrained non-negative matrix factorization (MVCNMF).

    Given a l * N matrix of N observations of l variables, identify latent
    variables subject to two criteria: the data is non-negative and the
    volume circumscribed by the simplex formed by the end members is
    the minimum possible. For details see references.

    Parameters
    ----------
    n_components : int
        Number of components to seek.
    regularization : float
        Importance of the simplex volume minimization relative to the model fit. Higher values weight the volume constraint more heavily.
    constraint : float
        The extent to which the sum-to-one constraint is required. Larger values more strongly enforce this constraint.

    Attributes
    ----------
    c : int
        Number of components.

    References
    ----------
    L. Miao and H. Qi, "Endmember Extraction From Highly Mixed Data Using Minimum Volume Constrained Nonnegative Matrix Factorization," in IEEE Transactions on Geoscience and Remote Sensing, vol. 45, no. 3, pp. 765-777, March 2007

    """

    def __init__(
        self,
        # n_components,
        regularization=0.5,
        constraint=1,
        learning_rate=100,
        max_iter=25,
        scaling=0.5,
        *args,
        **kwargs,
    ):

        # self.c = n_components
        self.constraint = constraint
        # self.tau = regularization / factorial(self.c - 1)
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.scaling = scaling

    @property
    def C(self):
        """c * c matrix of zeros. The first row is ones."""
        basis = np.zeros((self.c, self.c))
        basis[0, :] = 1
        return basis

    @property
    def B(self):
        """(c-1) * (c-1) identity matrix appended to a row of zeros."""
        basis = np.identity(self.c - 1)
        zeros = np.zeros((self.c - 1,))
        return np.vstack((zeros, basis))

    def Z(self, A, mean):
        """Augmented low-dimensional transformation of the factors."""
        if self.U.shape[0] == A.shape[0]:
            u = self.U
        elif self.U.shape[0] == A.shape[0] - 1:
            u = self.U_bar
        return self.C + np.dot(np.dot(self.B, u.T), A - np.array([mean] * self.c).T)

    @staticmethod
    def frobenius(Z):
        """Frobenius norm of a matrix.

        Parameters
        ----------
        Z : array-like
            A matrix.

        Returns
        -------
        float
            The Frobenius norm of `Z`.

        """
        return np.linalg.norm(Z)

    def simplex_volume(self, X, A, S):
        """The approximate volume of the simplex formed by the end members.

        Parameters
        ----------
        X : array-like
            l * N data matrix
        A : array-like
            l * c factor matrix
        S : array-like
            c * N loading matrix

        Returns
        -------
        float
            The approximate volume of the simplex formed by the end members.

        """
        return self.tau / 2 * np.linalg.det(self.Z(A, np.mean(X, axis=1))) ** 2

    def objective(self, X, A, S):
        """The minimisation criterion.

        Minimises both the model fit through the Frobenius norm and the minimum volume criterion.

        Parameters
        ----------
        X : array-like
            l * N data matrix
        A : array-like
            l * c factor matrix
        S : array-like
            c * N loading matrix

        Returns
        -------
        float
            The error in the overall fit.

        """
        return self.frobenius(X - np.dot(A, S)) + self.simplex_volume(X, A, S)

    def grad_a(self, X, A, S):
        """The gradient of the objective function with fixed S.

        Parameters
        ----------
        X : array-like
            l * N data matrix
        A : array-like
            l * c factor matrix
        S : array-like
            c * N loading matrix

        Returns
        -------
        gradient : array-like
            l * c gradient.

        """
        mean = np.mean(X, axis=1)
        frobenius_part = np.dot(np.dot(A, S) - X, S.T)
        if not np.isclose(np.linalg.det(self.Z(A, mean)), 0):
            geometric_part = (
                self.tau
                * np.square(np.linalg.det(self.Z(A, mean)))
                * self.U.dot(self.B.T).dot(np.linalg.inv(self.Z(A, mean)).T)
            )
        else:
            geometric_part = 0.0
        gradient = frobenius_part + geometric_part
        return gradient

    def grad_s(self, X, A, S):
        """The gradient of the objective function with fixed A.

        Parameters
        ----------
        X : array-like
            l * N data matrix
        A : array-like
            l * c factor matrix
        S : array-like
            c * N loading matrix

        Returns
        -------
        gradient : array-like
            c * N gradient.

        """
        return np.dot(A.T, np.dot(A, S) - X)

    def compute_endmembers_and_abundances(
        self,
        Y,
        p,
        fit_tolerance=1e-2,
        convergence_tolerance=1e-6,
        # learning_rate=100,
        # learning_rate=1,
        # learning_rate=1,
        # scaling=0.5,
        learning_tolerance=1e-4,
        # max_iter=1000,
        # max_iter=25,
        *args,
        **kwargs,
    ):
        """Fits the model by minimising the objective function.

        Parameters
        ----------
        Y : array-like
            L * N data matrix (note inverse of dimensions)
        fit_tolerance : float
            The accepted closeness-of-fit of the model.
        convergence_tolerance : float
            The lowest acceptable rate of change. Below this, the algorithm is
            assumed to have converged.
        learning_rate : float
            Initial learning rate. Higher values can lead to swifter convergence
            but can overshoot minima.
        scaling : float
            Rate of decrease of learning rate. Should be between zero and one.
        learning_tolerance : float
            Value weighting the gradient search. Larger values cause larger
            possible step sizes.
        max_iter : int
            Number of iterations allowed for convergence.

        Returns
        -------
        A : array-like
            l * c factor matrix, containing the end members.
        S : array-like
            c * N loading matrix, containing the relative abundance.

        """
        learning_rate = self.learning_rate
        max_iter = self.max_iter
        scaling = self.scaling
        X = Y
        self.c = p
        self.tau = self.regularization / factorial(self.c - 1)
        X_bar = self.augment(X)
        self.U = PCA(n_components=self.c - 1).fit(X.T).components_.T
        self.U_bar = PCA(n_components=self.c - 1).fit(self.augment(X).T).components_.T
        S = np.zeros((self.c, X.shape[1]))
        A = X[:, np.random.randint(0, X.shape[1], size=self.c)]
        o = 0
        progress = tqdm(range(max_iter))
        for _ in progress:
            alpha = self.get_alpha(
                X,
                A,
                S,
                learning_rate,
                1,
                scaling,
                learning_tolerance,
                max_iter=50,
            )
            A = self.A_new(X, A, S, alpha)
            A_bar = self.augment(A)
            beta = self.get_beta(
                X_bar,
                A_bar,
                S,
                learning_rate,
                1,
                scaling,
                learning_tolerance,
                max_iter=50,
            )
            S = self.S_new(X_bar, A_bar, S, beta)
            error_difference = np.abs(self.objective(X, A, S) - o)
            if error_difference < convergence_tolerance:
                print("Converged with error difference", error_difference)
                break
            o = self.objective(X, A, S)
            progress.set_postfix_str(f"obj={o:.3e}, err_diff={error_difference:3e}")
            if o < fit_tolerance:
                break

        return A, S

    def A_new(self, X, A, S, alpha):
        """Calculates updated factors.

        Parameters
        ----------
        X : array-like
            l * N data matrix
        A : array-like
            l * c factor matrix
        S : array-like
            c * N loading matrix
        alpha : float
            Step size. Calculate appropriate step size using `get_alpha`.

        Returns
        -------
        a_new : array_like
            l * c factor matrix

        """
        a_new = A - alpha * self.grad_a(X, A, S)
        a_new[a_new < 0] = 0
        return a_new

    def S_new(self, X, A, S, beta):
        """Calculates updated loadings.

        Parameters
        ----------
        X : array-like
            l * N data matrix
        A : array-like
            l * c factor matrix
        S : array-like
            c * N loading matrix
        beta : float
            Step size. Calculate appropriate step size using `get_alpha`.

        Returns
        -------
        s_new : array_like
            c * N loading matrix

        """
        s_new = S - beta * self.grad_s(X, A, S)
        s_new[s_new < 0] = 0
        return s_new

    def augment(self, Z):
        """Returns a copy of the matrix `Z` with a constant row appended."""
        return np.vstack((Z, self.constraint * np.ones((Z.shape[1],))))

    def get_alpha(self, X, A, S, alpha, m, scaling, learning_tolerance, max_iter=15):
        """Calculates an appropriate step size based on the Armijo rule.

        Parameters
        ----------
        X : array-like
            l * N data matrix
        A : array-like
            l * c factor matrix
        S : array-like
            c * N loading matrix
        alpha : float
            Initial guess for the step size.
        m : int
            Scaling exponent.
        scaling : float
            Factor to reduce the initial step size.
        learning_tolerance : float
            Controls rate of descent.
        max_iter : int
            Number of iterations to try.

        Returns
        -------
        alpha_new : float
            The optimised step size.

        """
        A_new = self.A_new(X, A, S, alpha)
        f_new = self.objective(X, A_new, S)
        f_old = self.objective(X, A, S)
        condition = (
            learning_tolerance
            * scaling
            * alpha
            * np.sum(np.dot(self.grad_a(X, A, S).T, A_new - A))
        )
        alpha_new = alpha * scaling
        if f_new - f_old <= condition or m > max_iter:
            return alpha_new
        else:
            return self.get_alpha(
                X,
                A,
                S,
                alpha_new,
                m + 1,
                scaling,
                learning_tolerance,
                max_iter=max_iter,
            )

    def get_beta(self, X, A, S, beta, m, scaling, learning_tolerance, max_iter=15):
        """Calculates an appropriate step size based on the Armijo rule.

        Parameters
        ----------
        X : array-like
            l * N data matrix
        A : array-like
            l * c factor matrix
        S : array-like
            c * N loading matrix
        beta : float
            Initial guess for the step size.
        m : int
            Scaling exponent.
        scaling : float
            Factor to reduce the initial step size.
        learning_tolerance : float
            Controls rate of descent.
        max_iter : int
            Number of iterations to try.

        Returns
        -------
        beta_new : float
            The optimised step size.

        """
        S_new = self.S_new(X, A, S, beta)
        f_new = self.objective(X, A, S_new)
        f_old = self.objective(X, A, S)
        condition = (
            learning_tolerance
            * scaling
            * beta
            * np.sum(np.dot(self.grad_s(X, A, S).T, S_new - S))
        )
        beta_new = beta * scaling
        if f_new - f_old < condition or m > max_iter:
            return beta_new
        else:
            return self.get_beta(
                X,
                A,
                S,
                beta_new,
                m + 1,
                scaling,
                learning_tolerance,
                max_iter=max_iter,
            )
