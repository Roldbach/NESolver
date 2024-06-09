"""A utility module that handles data simulation in the project.

Author: Weixun Luo
Date: 30/03/2024
"""
import numpy as np
import torch
from torch import nn

from utils import chemistry_utils
from utils import matrix_utils
from utils import typing_utils


# {{{ DataPackSimulator
class DataPackSimulator:
    """A class that can simulate the data pack based on the Nikolsky-Eisenman
    equation.

    Attribute
        - _sensor_number: An int that specifies the number of sensors. It is
                          assumed to be the same as the number of ions.
        - _charge: A numpy.ndarray that contains the signed charge numbers of
                   ions with shape (1, #ISE).
        - _ion_size: A numpy.ndarray that contains the size of ions with shape
                     (1, #ISE).
        - _selectivity_coefficient: A numpy.ndarray that contains the reshaped
                                    selectivity coefficients with shape
                                    (#ISE, #ISE, 1).
        - _slope: A numpy.ndarray that contains the potential slopes of ISEs
                  with shape (#ISE, 1, 1).
        - _drift: A numpy.ndarray that contains the potential drifts of ISEs
                  with shape (#ISE, 1, 1).

    Property
        - charge: A numpy.ndarray that contains the signed charge numbers of
                  ions with shape (1, #ISE).
        - ion_size: A numpy.ndarray that contains the size of ions with shape
                    (1, #ISE).
        - selectivity_coefficient: A numpy.ndarray that specifies the selectivity
                                   coefficients of ISEs with shape (#ISE, #ISE).
        - slope: A numpy.ndarray that contains the potential slopes of ISEs with
                 shape (1, #ISE).
        - drift: A numpy.ndarray that contains the potential drift of ISEs with
                 shape (1, #ISE).
    """

    # {{{ __init__
    def __init__(
        self,
        charge: np.ndarray,
        ion_size: np.ndarray,
        selectivity_coefficient: np.ndarray,
        noise_parameter: tuple[float, float],
    ) -> None:
        self._sensor_number = self._construct_sensor_number(charge)
        self._charge = self._construct_charge(charge)
        self._ion_size = self._construct_ion_size(ion_size)
        self._selectivity_coefficient = self._construct_selectivity_coefficient(
            selectivity_coefficient)
        self._slope = self._construct_slope(noise_parameter)
        self._drift = self._construct_drift(noise_parameter)
    # }}}

    # {{{ _construct_sensor_number
    def _construct_sensor_number(self, charge: np.ndarray) -> int:
        return charge.shape[1]
    # }}}

    # {{{ _construct_charge
    def _construct_charge(self, charge: np.ndarray) -> np.ndarray:
        return matrix_utils.build_row_array(charge)
    # }}}

    # {{{ _construct_ion_size
    def _construct_ion_size(self, ion_size: np.ndarray) -> np.ndarray:
        return matrix_utils.build_array(ion_size)
    # }}}

    # {{{ _construct_selectivity_coefficient
    def _construct_selectivity_coefficient(
        self, selectivity_coefficient: np.ndarray) -> np.ndarray:
        selectivity_coefficient =  matrix_utils.build_array(
            selectivity_coefficient)
        selectivity_coefficient = selectivity_coefficient.reshape(
            (self._sensor_number, self._sensor_number, 1))
        return selectivity_coefficient
    # }}}

    # {{{ _construct_slope
    def _construct_slope(
        self, noise_parameter: tuple[float, float]) -> np.ndarray:
        slope = chemistry_utils.compute_Nernst_slope(self.charge)
        slope = self._add_noise(slope, noise_parameter)
        slope = slope.reshape((self._sensor_number,1,1))
        return slope
    # }}}

    # {{{ _construct_drift
    def _construct_drift(
        self, noise_parameter: tuple[float, float]) -> np.ndarray:
        drift = chemistry_utils.compute_Ag_AgCl_drift(self._sensor_number)
        drift = self._add_noise(drift, noise_parameter)
        drift = drift.reshape((self._sensor_number,1,1))
        return drift
    # }}}

    # {{{ _add_noise
    def _add_noise(
        self, candidate: np.ndarray, parameter: tuple[float, float],
    ) -> np.ndarray:
        candidate += np.random.normal(*parameter, candidate.shape)
        return candidate
    # }}}

    # {{{ @property: charge
    @property
    def charge(self) -> np.ndarray:
        return matrix_utils.build_row_array(self._charge)
    # }}}

    # {{{ @property: ion_size
    @property
    def ion_size(self) -> np.ndarray:
        return matrix_utils.build_row_array(self._ion_size)
    # }}}

    # {{{ @property: selectivity_coefficient
    @property
    def selectivity_coefficient(self) -> np.ndarray:
        selectivity_coefficient = matrix_utils.build_array(
            self._selectivity_coefficient)
        selectivity_coefficient = selectivity_coefficient.reshape(
            (self._sensor_number, self._sensor_number))
        return selectivity_coefficient
    # }}}

    # {{{ @property: slope
    @property
    def slope(self) -> np.ndarray:
        return matrix_utils.build_row_array(self._slope)
    # }}}

    # {{{ @property: drift
    @property
    def drift(self) -> np.ndarray:
        return matrix_utils.build_row_array(self._drift)
    # }}}

    # {{{ simulate
    def simulate(self, scope_to_sample_number: dict) -> typing_utils.DataPack:
        concentration = self._simulate_concentration(scope_to_sample_number)
        activity = self._simulate_activity(concentration)
        potential = self._simulate_potential(activity)
        return {
            'charge': self.charge,
            'ion_size': self.ion_size,
            'selectivity_coefficient': self.selectivity_coefficient,
            'slope': self.slope,
            'drift': self.drift,
            'concentration': concentration,
            'activity': activity,
            'potential': potential,
        }
    # }}}

    # {{{ _simulate_concentration
    def _simulate_concentration(
        self, scope_to_sample_number: dict) -> np.ndarray:
        return np.vstack([
            self._simulate_concentration_helper(scope, sample_number)
            for scope, sample_number in scope_to_sample_number.items()
        ])
    # }}}

    # {{{ _simulate_concentration_helper
    def _simulate_concentration_helper(
        self, scope: tuple[float,float], sample_number: int,
    ) -> np.ndarray:
        concentration = matrix_utils.sample_uniform_distribution_array(
            (sample_number,self._sensor_number-1), scope)
        concentration = self._append_Cl_concentration(concentration)
        return concentration
    # }}}

    # {{{ _append_Cl_concentration
    def _append_Cl_concentration(self, concentration: np.ndarray) -> np.ndarray:
        """Append the concentration of Cl ions at the last column."""
        Cl_concentration = self._compute_Cl_concentration(concentration)
        concentration = np.hstack((concentration, Cl_concentration))
        return concentration
    # }}}

    # {{{ _compute_Cl_concentration
    def _compute_Cl_concentration(self, concentration: np.ndarray) -> np.ndarray:
        """Build the concentration of Cl ions as the only anions."""
        Cl_concentration = np.sum(self.charge[0,:-1]*concentration, 1)
        Cl_concentration = Cl_concentration.reshape((-1,1))
        return Cl_concentration
    # }}}

    # {{{ _simulate_activity
    def _simulate_activity(self, concentration: np.ndarray) -> np.ndarray:
        return chemistry_utils.convert_concentration_to_activity(
            concentration, self.charge, self.ion_size)
    # }}}

    # {{{ _simulate_potential
    def _simulate_potential(self, activity: np.ndarray) -> np.ndarray:
        activity = self._pre_process_activity(activity)
        potential = np.log10(activity @ self._selectivity_coefficient)
        potential = self._drift + self._slope*potential
        potential = np.transpose(potential, (1,0,2))
        potential = potential.reshape((-1, self._sensor_number))
        return potential
    # }}}

    # {{{ _pre_process_activity
    def _pre_process_activity(self, activity: np.ndarray) -> np.ndarray:
        """Expand and power the activity of ions according to the Nikolsky-
        Eisenman equation.

        Argument
            - activity: A numpy.ndarray that contains activity of ions with
                        shape (#sample, #ISE).

        Return:
            A numpy.ndarray that contains expanded and powered activity of ions
        with shape (#ISE, #sample, #ISE).
        """
        activity_power = chemistry_utils.compute_activity_power(
            self.charge)
        activity = np.tile(activity, (activity.shape[1],1,1))
        activity = np.power(activity, activity_power)
        return activity
    # }}}
# }}}
