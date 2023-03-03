"""
Implement metrics used in Unmixing scenarios
"""
import logging

import numpy as np
import numpy.linalg as LA
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BaseMetric:
    def __init__(self):
        self.name = self.__class__.__name__

    @staticmethod
    def _check_input(X, Xref):
        assert X.shape == Xref.shape
        assert type(X) == type(Xref)
        return X, Xref

    def __call__(self, X, Xref):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.name}"


class MeanAbsoluteError(BaseMetric):
    def __init__(self):
        super().__init__()

    def __call__(self, E, Eref):
        E, Eref = self._check_input(E, Eref)

        normE = LA.norm(E, axis=0, keepdims=True)
        normEref = LA.norm(Eref, axis=0, keepdims=True)

        return 100 * (1 - np.abs((E / normE).T @ (Eref / normEref)))


class SpectralAngleDistance(BaseMetric):
    def __init__(self):
        super().__init__()

    def __call__(self, E, Eref):
        E, Eref = self._check_input(E, Eref)

        normE = LA.norm(E, axis=0, keepdims=True)
        normEref = LA.norm(Eref, axis=0, keepdims=True)

        tmp = (E / normE).T @ (Eref / normEref)
        ret = np.minimum(tmp, 1.0)  # NOTE Handle floating errors
        # return np.arccos((E / normE).T @ (Eref / normEref))
        return np.arccos(ret)


class SADDegrees(SpectralAngleDistance):
    def __init__(self):
        super().__init__()

    def __call__(self, E, Eref):
        tmp = super().__call__(E, Eref)
        return (np.diag(tmp) * (180 / np.pi)).mean()


class MSE(BaseMetric):
    def __init__(self):
        super().__init__()

    def __call__(self, E, Eref):
        E, Eref = self._check_input(E, Eref)

        normE = LA.norm(E, axis=0, keepdims=True)
        normEref = LA.norm(Eref, axis=0, keepdims=True)

        return np.sqrt(normE.T**2 + normEref**2 - 2 * (E.T @ Eref))


class aRMSE(BaseMetric):
    def __init__(self):
        super().__init__()

    def __call__(self, A, Aref):
        A, Aref = self._check_input(A, Aref)
        return 100 * np.sqrt(((A - Aref) ** 2).mean())


class SRE(BaseMetric):
    def __init__(self):
        super().__init__()

    def __call__(self, X, Xref):
        X, Xref = self._check_input(X, Xref)
        return 20 * np.log10(LA.norm(Xref, "fro") / LA.norm(Xref - X, "fro"))


class RunAggregator:
    def __init__(
        self,
        metric,
        use_endmembers=False,
        detail=True,
    ):
        """
        Aggregate runs by tracking a metric
        """
        self.metric = metric
        self.use_endmembers = use_endmembers
        self.filename = f"{metric}.json"
        self.data = {}
        self.df = None
        self.summary = None
        self.detail = detail

    def add_run(self, run, X, Xhat, labels):

        d = {}
        d["Overall"] = self.metric(X, Xhat)
        if self.detail:
            for ii, label in enumerate(labels):
                if self.use_endmembers:
                    x, xhat = X[:, ii][:, None], Xhat[:, ii][:, None]
                    d[label] = self.metric(x, xhat)
                else:
                    d[label] = self.metric(X[ii], Xhat[ii])

        logger.debug(f"Run {run}: {self.metric} => {d}")

        self.data[run] = d

    def aggregate(self, prefix=None):
        self.df = pd.DataFrame(self.data).T
        self.summary = self.df.describe().round(2)
        logger.info(f"{self.metric} summary:\n{self.summary}")
        self.save(prefix)

    def save(self, prefix=None):
        prefix = "" if prefix is None else f"{prefix}-"

        df_fname = f"{prefix}runs-{self.filename}"
        summary_fname = f"{prefix}summary-{self.filename}"

        self.df.to_json(df_fname)
        self.summary.to_json(summary_fname)


class SADAggregator(RunAggregator):
    def __init__(self):
        super().__init__(
            SADDegrees(),
            use_endmembers=True,
            detail=True,
        )


class RMSEAggregator(RunAggregator):
    def __init__(self):
        super().__init__(
            aRMSE(),
            use_endmembers=False,
            detail=True,
        )


class SREAggregator(RunAggregator):
    def __init__(self):
        super().__init__(
            SRE(),
            use_endmembers=False,
            detail=False,
        )
