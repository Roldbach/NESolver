"""A module that handles result evaluation in the project.

Author: Weixun Luo
Date: 08/04/2024
"""
import warnings

import numpy as np
from scipy import stats


PRECISION = 1e-5  # The maximum relative difference between the derived and true.
SIGNIFICANCE_LEVEL = 0.05


"""----- Element-wise Error -----"""
# {{{ compute_absolute_error
def compute_absolute_error(
    candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.abs(candidate - reference)
# }}}

# {{{ compute_absolute_percentage_error
def compute_absolute_percentage_error(
    candidate: np.ndarray,
    reference: np.ndarray,
    require_scaling: bool = True,
) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        error = np.abs((reference-candidate) / (reference+1E-30))
        if require_scaling:
            error *= 100.0
    return error
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
    require_scaling: bool = True,
) -> float | np.ndarray:
    return np.nanmean(
        compute_absolute_percentage_error(candidate, reference, require_scaling),
        axis)
# }}}

# {{{ compute_mean_squared_error
def compute_mean_squared_error(
    candidate: np.ndarray,
    reference: np.ndarray,
    axis: int | None = None,
) -> float | None:
    return np.nanmean(compute_squared_error(candidate, reference), axis)
# }}}


"""----- Difference -----"""
# {{{ is_numerically_close
def is_numerically_close(
    candidate: np.ndarray,
    reference: np.ndarray,
    precision: float = PRECISION,
) -> np.ndarray:
    return np.isclose(candidate, reference, atol=0.0, rtol=PRECISION)
# }}}

# {{{ is_statistically_close
def is_statistically_close(
    candidate: np.ndarray,
    reference: np.ndarray,
    significance_level: float = SIGNIFICANCE_LEVEL,
) -> tuple[np.ndarray, np.ndarray]:
    is_statistically_close_all = []
    is_difference_normal_all = []
    for i in range(candidate.shape[1]):
        is_statistically_close, is_difference_normal = _is_statistically_close_single(
            candidate[:,i], reference[:,i], significance_level)
        is_statistically_close_all.append(is_statistically_close)
        is_difference_normal_all.append(is_difference_normal)
    return np.array(is_statistically_close_all), np.array(is_difference_normal_all)
# }}}

# {{{ _is_statistically_close_single
def _is_statistically_close_single(
    candidate_column: np.ndarray,
    reference_column: np.ndarray,
    significance_level: float,
) -> tuple[bool, bool]:
    is_difference_normal = _run_shapiro_test(
        candidate_column-reference_column, significance_level)
    if is_difference_normal:
        is_statistically_close = _run_two_sided_paired_t_test(
            candidate_column, reference_column, significance_level)
    else:
        is_statistically_close = _run_wilcoxon_signed_rank_test(
            candidate_column, reference_column, significance_level)
    return is_statistically_close, is_difference_normal
# }}}

# {{{ _run_shapiro_test
def _run_shapiro_test(candidate: np.ndarray, significance_level: float) -> bool:
    _, p_value = stats.shapiro(candidate)
    return p_value >= significance_level
# }}}

# {{{ _run_two_sided_paired_t_test
def _run_two_sided_paired_t_test(
    candidate: np.ndarray,
    reference: np.ndarray,
    significance_level: float,
) -> bool:
    _, p_value = stats.ttest_rel(candidate, reference)
    return p_value >= significance_level
# }}}

# {{{ _run_wilcoxon_signed_rank_test
def _run_wilcoxon_signed_rank_test(
    candidate: np.ndarray,
    reference: np.ndarray,
    significance_level: float,
) -> bool:
    _, p_value = stats.wilcoxon(candidate, reference)
    return p_value >= significance_level
# }}}
