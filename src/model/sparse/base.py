"""
Sparse unmixing base model
"""

from src.model.base import UnmixingModel


class SparseUnmixingModel(UnmixingModel):
    def __init__(self):
        super().__init__()

    def compute_abundances(
        self,
        Y,
        D,
        *args,
        **kwargs,
    ):
        raise NotImplementedError(f"Solver is not implemented for {self}")
