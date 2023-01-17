"""
Base model declaration for supervised unmixing methods
"""

import logging

from src.model.base import UnmixingModel


class SupervisedUnmixingModel(UnmixingModel):
    def __init__(
        self,
    ):
        super().__init__()

    def compute_abundances(
        self,
        Y,
        E,
        *args,
        **kwargs,
    ):
        raise NotImplementedError("Solver is not implemented")
