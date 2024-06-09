"""A utility module that handles chemistry related computation in the project.

Author: Weixun Luo
Date: 13/04/2024
"""
import numpy as np


A = 0.51  # Temperature-dependent constant in the extended Debye-Huckel equation
B = 1/305  # Ion size coefficient in the extended Debye-Huckel equation (pm)
DRIFT_VALUE = 0.197  # Standard reduction potential of Ag/AgCl reference
                     # electrode with saturated KCl at 25 degrees
                     # -> Reference: Book, P345
FARADAY_CONSTANT = 9.6485309e+4  # Faraday constant, F (C mol^-1)
GAS_CONSTANT = 8.314510  # Gas constant, R (J K^-1 mol^-1)
TEMPERATURE = 298.15  # Standard temperature (Kelvin)


"""----- Activity -----"""
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


"""----- Activity Coefficient -----"""
# {{{ _compute_activity_coefficient
def _compute_activity_coefficient(
    concentration: np.ndarray,
    charge: np.ndarray,
    ion_size: np.ndarray,
) -> np.ndarray:
    """Compute the activity coefficient according to the extended Debye-Huckel
    equation.

    Argument
        - concentration: A numpy.ndarray that contains concentration of
                         ions (M) with shape (#sample, #ISE).
        - charge: A numpy.ndarray that contains signed charge number of ions
                  with shape (1, #ISE).
        - ion_size: A numpy.ndarray that contains the size of ions (pm) with
                    shape (1, #ISE).

    Return
        A numpy.ndarray that specifies the corresponding activity coefficients
    of each ion with shape (#sample, #ISE).
    """
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
    """Compute the activity coefficient according to the extended Debye-Huckel
    equation.

    This function only accepts row vectors as input.

    Argument
        - concentration_row: A numpy.ndarray that contains concentration of
                             ions with shape (1, #ISE).
        - charge: A numpy.ndarray that contains signed charge number of ions
                  with shape (1, #ISE).
        - ion_size: A numpy.ndarray that contains the size of ions (pm) with
                    shape (1, #ISE).
    Return
        A float that specifies the corresponding activity coefficients of
    each ion with shape (1, #ISE).
    """
    root_ionic_strength = _compute_root_ionic_strength(
        concentration_row, charge)
    numerator = -A * np.power(charge,2) * root_ionic_strength
    denominator = 1 + B*root_ionic_strength*ion_size
    activity_coefficient = np.power(10.0, numerator/denominator)
    return activity_coefficient
# }}}


"""----- Activity Power -----"""
# {{{ compute_activity_power
def compute_activity_power(charge: np.ndarray) -> np.ndarray:
    """Compute the power of activity according to the Nikolsky-Eisenman equation.

    Argument
        - charge: A numpy.ndarray that contains signed charge number of ions
                  with shape (1, #ISE).

    Return
        A numpy.ndarray that contains the power of activity with shape
    (#ISE, 1, #ISE).
    """
    return np.abs(charge.reshape((-1,1,1)) / charge)
# }}}


"""----- Ionic Strength -----"""
# {{{ _compute_root_ionic_strength
def _compute_root_ionic_strength(
    concentration: np.ndarray, charge: np.ndarray) -> np.ndarray:
    return np.sqrt(_compute_ionic_strength(concentration, charge))
# }}}

# {{{ _compute_ionic_strength
def _compute_ionic_strength(
    concentration: np.ndarray, charge: np.ndarray) -> np.ndarray:
    return 0.5 * concentration @ np.power(charge,2).T
# }}}


"""----- Slope -----"""
# {{{ compute_Nernst_slope
def compute_Nernst_slope(charge: np.ndarray) -> np.ndarray:
    """Compute the potential slope of ISEs according to the Nikolsky-Eisenman
    equation."""
    numerator = 2.303 * GAS_CONSTANT * TEMPERATURE
    denominator = FARADAY_CONSTANT * charge
    slope = numerator / denominator
    return slope
# }}}


"""----- Drift -----"""
# {{{ compute_Ag_AgCl_drift
def compute_Ag_AgCl_drift(sensor_number: int) -> np.ndarray:
    """Compute the potential drift of ISEs based on the reduction potential of
    Ag/AgCl reference electrode operated under 298.15K.
    """
    return np.array(
        [DRIFT_VALUE for _ in range(sensor_number)], dtype=np.float64).reshape((1,-1))
# }}}
