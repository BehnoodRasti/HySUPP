"""
Model related globals
"""
import warnings

try:
    import matlab.engine
except Exception:
    warnings.warn(
        "`matlab.engine` was not imported. MATLAB related code will not work."
    )


class UnmixingModel:
    def __init__(self):
        self.time = 0

    def __repr__(self):
        msg = f"{self.__class__.__name__}"
        return msg

    def print_time(self):
        return f"{self} took {self.time:.2f}s"
