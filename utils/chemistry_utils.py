"""A utility module that handles chemistry related computation in the project.

Author: Weixun Luo
Date: 13/04/2024
"""
import numpy as np


A = 0.51  # Temperature-dependent constant in the extended Debye-Huckel equation
B = 1/305  # Ion size coefficient in the extended Debye-Huckel equation (pm)
Ag_AgCl_INTERCEPT = 0.197  # Standard reduction potential of Ag/AgCl reference
                           # electrode with saturated KCl at 25 celsius
FARADAY_CONSTANT = 9.6485309e+4  # Faraday constant, F (C mol^-1)
GAS_CONSTANT = 8.314510  # Gas constant, R (J K^-1 mol^-1)
TEMPERATURE = 298.15  # Standard temperature (Kelvin)


"""----- Activity / Activity Coefficient -----"""
# {{{ convert_concentration_to_activity
def convert_concentration_to_activity(
    concentration: np.ndarray,
    charge: np.ndarray,
    ion_size: np.ndarray,
) -> np.ndarray:
    activity_coefficient = _compute_activity_coefficient(
        concentration, charge, ion_size)
    activity = activity_coefficient * concentration
    return activity
# }}}

# {{{ _compute_activity_coefficient
def _compute_activity_coefficient(
    concentration: np.ndarray,
    charge: np.ndarray,
    ion_size: np.ndarray,
) -> np.ndarray:
    return np.apply_along_axis(
        func1d = _compute_activity_coefficient_row,
        axis = 1,
        arr = concentration,
        charge = charge,
        ion_size = ion_size,
    ).reshape(concentration.shape)
# }}}

# {{{ _compute_activity_coefficient_row
def _compute_activity_coefficient_row(
    concentration_row: np.ndarray,
    charge: np.ndarray,
    ion_size: np.ndarray,
) -> float:
    root_ionic_stregth = np.sqrt(0.5* concentration_row @ np.power(charge,2).T)
    numerator = -A * np.power(charge,2) * root_ionic_strength
    denominator = 1 + B*root_ionic_strength*ion_size
    activity_coefficient = np.power(10.0, numerator/denominator)
    return activity_coefficient
# }}}


"""----- Activity Power -----"""
# {{{ compute_activity_power
def compute_activity_power(charge: np.ndarray) -> np.ndarray:
    return np.abs(charge.reshape((-1,1,1)) / charge)
# }}}


"""----- Response Slope -----"""
# {{{ compute_Nernst_slope
def compute_Nernst_slope(
    charge: np.ndarray,
    gas_constant: float = GAS_CONSTANT,
    temperature: float = TEMPERATURE,
    faraday_constant: float = FARADAY_CONSTANT,
) -> np.ndarray:
    numerator = 2.303 * gas_constant * temperature
    denominator = faraday_constant * charge
    slope = numerator / denominator
    return slope
# }}}
