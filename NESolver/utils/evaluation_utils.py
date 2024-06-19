"""An utility module that handles evaluation in the project.

Author: Weixun Luo
Date: 08/04/2024
"""
import numpy as np


PRECISION = 1e-5  # The maximum percentage error allowed


"""----- Element-wise Error -----"""
# {{{ compute_absolute_error
def compute_absolute_error(
    candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.abs(candidate - reference)
# }}}

# {{{ compute_absolute_percentage_error
def compute_absolute_percentage_error(
    candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.abs((reference-candidate) / (reference+1e-30)) * 100.0
# }}}

# {{{ compute_squared_error
def compute_squared_error(
    candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.power(candidate-reference, 2.0)
# }}}


"""----- Mean Element-wise Error -----"""
# {{{ compute_mean_absolute_error
def compute_mean_absolute_error(
    candidate: np.ndarray,
    reference: np.ndarray,
    axis: int | None = None,
) -> float | np.ndarray:
    return np.nanmean(compute_absolute_error(candidate, reference), axis)
# }}}

# {{{ compute_mean_absolute_percentage_error
def compute_mean_absolute_percentage_error(
    candidate: np.ndarray,
    reference: np.ndarray,
    axis: int | None = None,
) -> float | np.ndarray:
    return np.nanmean(
        compute_absolute_percentage_error(candidate, reference), axis)
# }}}

# {{{ compute_mean_squared_error
def compute_mean_squared_error(
    candidate: np.ndarray,
    reference: np.ndarray,
    axis: int | None = None,
) -> float | None:
    return np.nanmean(compute_squared_error(candidate, reference), axis)
# }}}


"""----- Numerically Close -----"""
# {{{ is_numerically_close
def is_numerically_close(
    candidate: np.ndarray,
    reference: np.ndarray,
    precision: float = PRECISION,
) -> np.ndarray:
    return np.isclose(candidate, reference, atol=0.0, rtol=PRECISION)
# }}}
