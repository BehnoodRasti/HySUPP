"""
Base model declaration for blind unmixing methods
"""

from src.model.base import UnmixingModel


class BlindUnmixingModel(UnmixingModel):
    def __init__(
        self,
    ):
        super().__init__()

    def compute_endmembers_and_abundances(
        self,
        Y,
        p,
        *args,
        **kwargs,
    ):
        raise NotImplementedError(f"Solver is not implemented for {self}")
