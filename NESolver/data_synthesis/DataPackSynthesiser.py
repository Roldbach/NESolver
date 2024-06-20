"""A class that can synthesis data used for multivariate ion analysis.

Author: Weixun Luo
Date: 30/03/2024
"""
import numpy as np
import torch
from torch import nn

from NESolver.utils import chemistry_utils
from NESolver.utils import matrix_utils
from NESolver.utils import typing_utils


# {{{ DataPackSynthesiser
class DataPackSynthesiser:
    """A class that can synthesis data used for multivariate ion analysis.

    Attribute
        - _sensor_number: An int that specifies the number of sensors. It is
                          assumed to be the same as the number of ions.
        - _charge: A numpy.ndarray that contains the signed charge numbers of
                   ions with shape (1, #ISE).
        - _ion_size: A numpy.ndarray that contains the size of ions with shape
                     (1, #ISE).
        - _response_intercept: A numpy.ndarray that contains the response
                               intercepts of ISEs with shape (#ISE, 1, 1).
        - _response_slope: A numpy.ndarray that contains the response slopes of
                           ISEs with shape (#ISE, 1, 1).
        - _selectivity_coefficient: A numpy.ndarray that contains the reshaped
                                    selectivity coefficients with shape
                                    (#ISE, #ISE, 1).

    Property
        - charge: A numpy.ndarray that contains the signed charge numbers of
                  ions with shape (1, #ISE).
        - ion_size: A numpy.ndarray that contains the size of ions with shape
                    (1, #ISE).
        - response_intercept: A numpy.ndarray that contains the response
                              intercepts of ISEs with shape (1, #ISE).
        - response_slope: A numpy.ndarray that contains the response slopes of
                          ISEs with shape (1, #ISE).
        - selectivity_coefficient: A numpy.ndarray that specifies the selectivity
                                   coefficients of ISEs with shape (#ISE, #ISE).
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
        self._response_intercept = self._construct_response_intercept(
            noise_parameter)
        self._response_slope = self._construct_response_slope(noise_parameter)
        self._selectivity_coefficient = self._construct_selectivity_coefficient(
            selectivity_coefficient)
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

    # {{{ _construct_response_intercept
    def _construct_response_intercept(
        self, noise_parameter: tuple[float, float]) -> np.ndarray:
        response_intercept = matrix_utils.build_ones_array(
            (self._sensor_number, 1, 1))
        response_intercept *= chemistry_utils.Ag_AgCl_RESPONSE_INTERCEPT
        response_intercept = matrix_utils.add_gaussian_noise_array(
            response_intercept, noise_parameter)
        return response_intercept
    # }}}

    # {{{ _construct_response_slope
    def _construct_response_slope(
        self, noise_parameter: tuple[float, float]) -> np.ndarray:
        response_slope = chemistry_utils.compute_Nernst_response_slope(
            self._charge)
        response_slope = matrix_utils.add_gaussian_noise_array(
            response_slope, noise_parameter)
        response_slope = response_slope.reshape((self._sensor_number,1,1))
        return response_slope
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

    # {{{ @property: response_intercept
    @property
    def response_intercept(self) -> np.ndarray:
        return matrix_utils.build_row_array(self._response_intercept)
    # }}}

    # {{{ @property: response_slope
    @property
    def response_slope(self) -> np.ndarray:
        return matrix_utils.build_row_array(self._response_slope)
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

    # {{{ synthesis
    def synthesis(self, scope_to_sample_number: dict) -> typing_utils.DataPack:
        concentration = self._synthesis_concentration(scope_to_sample_number)
        activity = self._synthesis_activity(concentration)
        response = self._synthesis_response(activity)
        return {
            'charge': self.charge,
            'ion_size': self.ion_size,
            'response_intercept': self.response_intercept,
            'response_slope': self.response_slope,
            'selectivity_coefficient': self.selectivity_coefficient,
            'concentration': concentration,
            'activity': activity,
            'response': response,
        }
    # }}}

    # {{{ _synthesis_concentration
    def _synthesis_concentration(
        self, scope_to_sample_number: dict) -> np.ndarray:
        return np.vstack([
            self._synthesis_concentration_helper(scope, sample_number)
            for scope, sample_number in scope_to_sample_number.items()
        ])
    # }}}

    # {{{ _synthesis_concentration_helper
    def _synthesis_concentration_helper(
        self, scope: tuple[float,float], sample_number: int,
    ) -> np.ndarray:
        concentration = matrix_utils.sample_uniform_distribution_array(
            (sample_number, self._sensor_number-1), scope)
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
        Cl_concentration = np.sum(self._charge[0,:-1]*concentration, 1)
        Cl_concentration = Cl_concentration.reshape((-1,1))
        return Cl_concentration
    # }}}

    # {{{ _synthesis_activity
    def _synthesis_activity(self, concentration: np.ndarray) -> np.ndarray:
        return chemistry_utils.convert_concentration_to_activity(
            concentration, self._charge, self._ion_size)
    # }}}

    # {{{ _synthesis_response
    def _synthesis_response(self, activity: np.ndarray) -> np.ndarray:
        activity = self._pre_process_activity(activity)
        response = np.log10(activity @ self._selectivity_coefficient)
        response = self._response_intercept + self._response_slope*response
        response = np.transpose(response, (1,0,2))
        response = response.reshape((-1, self._sensor_number))
        return response
    # }}}

    # {{{ _pre_process_activity
    def _pre_process_activity(self, activity: np.ndarray) -> np.ndarray:
        activity_power = chemistry_utils.compute_Nikolsky_Eisenman_activity_power(
            self._charge)
        activity = np.tile(activity, (activity.shape[1],1,1))
        activity = np.power(activity, activity_power)
        return activity
    # }}}
# }}}
