"""A utility module that handles optimisation related operations in the project.

Author: Weixun Luo
Date: 10/04/2024
"""
import abc
import typing

import numpy as np
from scipy import optimize
from sklearn import cross_decomposition
from sklearn import linear_model
from sklearn import model_selection
import torch
from torch import nn

from NESolver.utils import chemistry_utils
from NESolver.utils import io_utils
from NESolver.utils import matrix_utils


"""----- Agent -----"""
# {{{ Agent
class Agent(abc.ABC):
    """An interface that defines an agent that can perform multivariate ion
    analysis.
    """

    # {{{ __init__
    @abc.abstractmethod
    def __init__(self) -> None:
        pass
    # }}}

    # {{{ @property: response_intercept
    @property
    @abc.abstractmethod
    def response_intercept(self) -> np.ndarray:
        pass
    # }}}

    # {{{ @property: response_slope
    @property
    @abc.abstractmethod
    def response_slope(self) -> np.ndarray:
        pass
    # }}}

    # {{{ @property: selectivity_coefficient
    @property
    @abc.abstractmethod
    def selectivity_coefficient(self) -> np.ndarray:
        pass
    # }}}

    # {{{ forward_solve
    @abc.abstractmethod
    def forward_solve(self, concentration: np.ndarray) -> np.ndarray:
        pass
    # }}}

    # {{{ backward_solve
    @abc.abstractmethod
    def backward_solve(self, response: np.ndarray) -> np.ndarray:
        pass
    # }}}
# }}}

# {{{ TrainableAgent
class TrainableAgent(nn.Module, Agent):
    """An interface that defines an agent that can be trained to perform
    multivariate ion analysis.
    """

    # {{{ forward
    @abc.abstractmethod
    def forward(self, candidate: torch.Tensor) -> torch.Tensor:
        pass
    # }}}

    # {{{ load_weight
    @abc.abstractmethod
    def load_weight(self, weight_file_path: str) -> None:
        pass
    # }}}

    # {{{ save_weight
    @abc.abstractmethod
    def save_weight(self, weight_file_path: str) -> None:
        pass
    # }}}
# }}}


"""----- Optimisation -----"""
# {{{ ActivitySolver
class ActivitySolver:
    """A class that inversely solves the ion activity giving the response of ISEs.
    """

    # {{{ __init__
    def __init__(self, charge: np.ndarray) -> None:
        self._activity_power = self._construct_activity_power(charge)
    # }}}

    # {{{ _construct_activity_power
    def _construct_activity_power(self, charge: np.ndarray) -> np.ndarray:
        activity_power = np.tile(charge.T, (1,charge.shape[1]))
        activity_power = activity_power / charge
        activity_power = np.abs(activity_power)
        return activity_power
    # }}}

    # {{{ solve
    def solve(
        self,
        response: np.ndarray,
        selectivity_coefficient: np.ndarray,
    ) -> np.ndarray:
        return np.apply_along_axis(
            func1d = self._solve_row,
            axis = 1,
            arr = response,
            selectivity_coefficient = selectivity_coefficient,
        ).reshape(response.shape)
    # }}}

    # {{{  _solve_row
    def _solve_row(
        self,
        response_row: np.ndarray,
        selectivity_coefficient: np.ndarray,
    ) -> np.ndarray:
        return optimize.root(
            fun = self._build_objective_function(
                response_row, selectivity_coefficient),
            x0 = response_row,
            jac = self._build_Jacobian(selectivity_coefficient),
        ).x
    # }}}

    # {{{ _build_objective_function
    def _build_objective_function(
        self,
        response_row: np.ndarray,
        selectivity_coefficient: np.ndarray,
    ) -> typing.Callable:

        # {{{ objective_function
        def objective_function(activity_row: np.ndarray) -> np.ndarray:
            output = np.tile(activity_row, (activity_row.size,1))
            output = np.power(output, self._activity_power)
            output *= selectivity_coefficient
            output = np.hstack((output, -1.0*response_row.reshape((-1,1))))
            output = np.sum(output, 1)
            return output
        # }}}

        return objective_function
    # }}}

    # {{{ _build_Jacobian
    def _build_Jacobian(
        self, selectivity_coefficient: np.ndarray) -> typing.Callable:

        # {{{ Jacobian
        def Jacobian(activity_row: np.ndarray) -> np.ndarray:
            output = np.tile(activity_row, (activity_row.size,1))
            output = np.power(output, self._activity_power-1)
            output *= self._activity_power
            output *= selectivity_coefficient
            return output
        # }}}

        return Jacobian
    # }}}
# }}}

# {{{ ActivityCoefficientSolver
class ActivityCoefficientSolver:
    """A class that solves ion activity coefficients based on the extended
    Debye-Huckel equation.
    """

    # {{{ __init__
    def __init__(
        self,
        charge: np.ndarray,
        ion_size: np.ndarray,
    ) -> None:
        self._sensor_number = self._construct_sensor_number(charge)
        self._charge_squared = self._construct_charge_squared(charge)
        self._ion_size = self._construct_ion_size(ion_size)
    # }}}

    # {{{ _construct_sensor_number
    def _construct_sensor_number(self, charge: np.ndarray) -> int:
        return charge.shape[1]
    # }}}

    # {{{ _construct_charge_squared
    def _construct_charge_squared(self, charge: np.ndarray) -> np.ndarray:
        return np.power(charge, 2).flatten()
    # }}}

    # {{{ _construct_ion_size
    def _construct_ion_size(self, ion_size: np.ndarray) -> np.ndarray:
        return matrix_utils.build_array(ion_size).flatten()
    # }}}

    # {{{ solve
    def solve(self, activity: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(
            func1d = self._solve_row,
            axis = 1,
            arr = activity,
        ).reshape(activity.shape)
    # }}}

    # {{{ _solve_row
    def _solve_row(self, activity_row: np.ndarray) -> np.ndarray:
        return optimize.root(
            fun = self._build_objective_function(activity_row),
            x0 = matrix_utils.build_ones_array(activity_row.shape),
            jac = self._build_Jacobian(activity_row),
        ).x
    # }}}

    # {{{ _build_objective_function
    def _build_objective_function(
        self, activity_row: np.ndarray) -> typing.Callable:

        # {{{ objective_function
        def objective_function(
            activity_coefficient_row: np.ndarray) -> np.ndarray:
            ionic_strength_root = self._compute_root_ionic_strength(
                activity_coefficient_row, activity_row)
            numerator = chemistry_utils.A * ionic_strength_root * self._charge_squared
            denominator = 1 + chemistry_utils.B*ionic_strength_root*self._ion_size
            output = numerator / denominator
            output += np.log10(activity_coefficient_row)
            return output
        # }}}

        return objective_function
    # }}}

    # {{{ _compute_root_ionic_strength
    def _compute_root_ionic_strength(
        self,
        activity_coefficient_row: np.ndarray,
        activity_row: np.ndarray,
    ) -> float:
        return np.sqrt(
            self._compute_ionic_strength(activity_coefficient_row, activity_row))
    # }}}

    # {{{ _compute_ionic_strength
    def _compute_ionic_strength(
        self,
        activity_coefficient_row: np.ndarray,
        activity_row: np.ndarray,
    ) -> float:
        concentration_row = activity_row / activity_coefficient_row
        ionic_strength = 0.5 * np.dot(concentration_row, self._charge_squared)
        return ionic_strength
    # }}}

    # {{{ _build_Jacobian
    def _build_Jacobian(self, activity_row: np.ndarray) -> typing.Callable:

        # {{{
        def Jacobian(activity_coefficient_row: np.ndarray) -> np.ndarray:
            return matrix_utils.build_array([
                self._compute_Jacobian_element(
                    i, j, activity_coefficient_row, activity_row)
                for i in range(self._sensor_number)
                for j in range(self._sensor_number)
            ]).reshape((self._sensor_number, self._sensor_number))
        # }}}

        return Jacobian
    # }}}

    # {{{ _compute_Jacobian_element
    def _compute_Jacobian_element(
        self,
        row_index: int,
        column_index: int,
        activity_coefficient_row: np.ndarray,
        activity_row: np.ndarray,
    ) -> float:
        fraction_term_prime = self._compute_fraction_term_prime(
            row_index, activity_coefficient_row, activity_row)
        root_ionic_strength_prime = self._compute_root_ionic_strength_prime(
            column_index, activity_coefficient_row, activity_row)
        output = fraction_term_prime * root_ionic_strength_prime
        if row_index == column_index:
            output += 1 / (np.log(10)*activity_coefficient_row[row_index])
        return output
    # }}}

    # {{{ _compute_fraction_term_prime
    def _compute_fraction_term_prime(
        self,
        row_index: int,
        activity_coefficient_row: np.ndarray,
        activity_row: np.ndarray,
    ) -> float:
        ionic_strength_root = self._compute_root_ionic_strength(
            activity_row, activity_coefficient_row)
        numerator = self._compute_fraction_term_prime_numerator(
            row_index, ionic_strength_root)
        denominator = self._compute_fraction_term_prime_denominator(
            row_index, ionic_strength_root)
        output = numerator / denominator
        return output
    # }}}

    # {{{ _compute_fraction_term_prime_numerator
    def _compute_fraction_term_prime_numerator(
        self, row_index: int, ionic_strength_root: float) -> float:
        output = 1 + 2*chemistry_utils.B*self._ion_size[row_index]*ionic_strength_root
        output *= (chemistry_utils.A * self._charge_squared[row_index])
        return output
    # }}}

    # {{{ _compute_fraction_term_prime_denominator
    def _compute_fraction_term_prime_denominator(
        self, row_index: int, ionic_strength_root: float) -> float:
        output = 1 + chemistry_utils.B*self._ion_size[row_index]*ionic_strength_root
        output = np.power(output, 2)
        return output
    # }}}

    # {{{ _compute_root_ionic_strength_prime
    def _compute_root_ionic_strength_prime(
        self,
        column_index: int,
        activity_coefficient_row: np.ndarray,
        activity_row: np.ndarray,
    ) -> float:
        ionic_strength = self._compute_ionic_strength(
            activity_coefficient_row, activity_row)
        output = np.power(0.5,1.5) * np.power(ionic_strength,-0.5)
        output *= np.power(activity_coefficient_row[column_index], -2)
        output *= (-1.0 * activity_row[column_index] * self._charge_squared[column_index])
        return output
    # }}}
#}}}

# {{{ ResponseSolver
class ResponseSolver(nn.Module):
    """A class that quantifies sensor characteristics based on the
    Nikolsky-Eisenman equation.
    """

    # {{{ __init__
    def __init__(
        self,
        charge: np.ndarray,
        response_intercept_initialisation: str = 'Ag/AgCl',
        response_slope_initialisation: str = 'nernst',
        selectivity_coefficient_initialisation: str = 'eye',
    ) -> None:
        super().__init__()
        self._sensor_number = self._construct_sensor_number(charge)
        self._response_intercept = self._construct_response_intercept(
            response_intercept_initialisation)
        self._response_slope = self._construct_response_slope(
            response_slope_initialisation, charge)
        self._selectivity_coefficient = self._construct_selectivity_coefficient(
            selectivity_coefficient_initialisation)
    # }}}

    # {{{ _construct_sensor_number
    def _construct_sensor_number(self, charge: np.ndarray) -> int:
        return charge.shape[1]
    # }}}

    # {{{ _construct_response_intercept
    def _construct_response_intercept(self, initialisation: str) -> nn.Parameter:
        weight = matrix_utils.initialise_weight_tensor(
            (self._sensor_number,1,1), initialisation)
        response_intercept = nn.Parameter(weight)
        return response_intercept
    # }}}

    # {{{ _construct_response_slope
    def _construct_response_slope(
        self, initialisation: str, charge: np.ndarray) -> nn.Parameter:
        weight = matrix_utils.initialise_weight_tensor(
            (self._sensor_number,1,1), initialisation, charge=charge)
        response_slope = nn.Parameter(weight)
        return response_slope
    # }}}

    # {{{ _construct_selectivity_coefficient
    def _construct_selectivity_coefficient(
        self, initialisation: str) -> nn.Parameter:
        weight = matrix_utils.initialise_weight_tensor(
            shape = (self._sensor_number, self._sensor_number),
            initialisation = initialisation,
        ).unsqueeze(2)
        selectivity_coefficient = nn.Parameter(weight)
        selectivity_coefficient = self._register_gradient_hook(
            selectivity_coefficient)
        return selectivity_coefficient
    # }}}

    # {{{ _register_gradient_hook
    def _register_gradient_hook(
        self, selectivity_coefficient: nn.Parameter) -> nn.Parameter:

        # {{{ _gradient_hook
        def _gradient_hook(gradient: torch.Tensor) -> torch.Tensor:
            gradient = gradient.squeeze(2)
            gradient.fill_diagonal_(0.0)
            gradient[-1,:].fill_(0.0)
            gradient[:,-1].fill_(0.0)
            gradient = gradient.unsqueeze(2)
            return gradient
        # }}}

        selectivity_coefficient.register_hook(_gradient_hook)
        return selectivity_coefficient
    # }}}

    # {{{ @property: response_intercept
    @property
    def response_intercept(self) -> np.ndarray:
        return self._response_intercept.clone().detach().numpy().reshape((1,-1))
    # }}}

    # {{{ @property: response_slope
    @property
    def response_slope(self) -> np.ndarray:
        return self._response_slope.clone().detach().numpy().reshape((1,-1))
    # }}}

    # {{{ @property: selectivity_coefficient
    @property
    def selectivity_coefficient(self) -> np.ndarray:
        selectivity_coefficient = self._selectivity_coefficient.clone().detach().numpy()
        selectivity_coefficient = selectivity_coefficient.reshape(
            (self._sensor_number, self._sensor_number))
        return selectivity_coefficient
    # }}}

    # {{{ forward
    def forward(self, candidate: torch.Tensor) -> torch.Tensor:
        output = torch.log10(candidate @ self._selectivity_coefficient)
        output = self._response_intercept + self._response_slope*output
        return output
    # }}}
# }}}

# {{{ OptimisationAgent
class OptimisationAgent(TrainableAgent):
    """An Agent that performs multivariate ion analysis based on optimisation and
    numerically solving equations.
    """

    # {{{ __init__
    def __init__(self, charge: np.ndarray, ion_size: np.ndarray) -> None:
        super().__init__()
        self._charge = self._construct_charge(charge)
        self._ion_size = self._construct_ion_size(ion_size)
        self._activity_power = self._construct_activity_power()
        self._activity_solver = self._construct_activity_solver(charge)
        self._activity_coefficient_solver = self._construct_activity_coefficient_solver(
            charge, ion_size)
        self._response_solver = self._construct_response_solver(charge)
    # }}}

    # {{{ _construct_charge
    def _construct_charge(self, charge: np.ndarray) -> np.ndarray:
        return matrix_utils.build_array(charge)
    # }}}

    # {{{ _construct_ion_size
    def _construct_ion_size(self, ion_size: np.ndarray) -> np.ndarray:
        return matrix_utils.build_array(ion_size)
    # }}}

    # {{{ _construct_activity_power
    def _construct_activity_power(self) -> np.ndarray:
        return chemistry_utils.compute_Nikolsky_Eisenman_activity_power(
            self._charge)
    # }}}

    # {{{ _construct_activity_solver
    def _construct_activity_solver(self, charge: np.ndarray) -> ActivitySolver:
        return ActivitySolver(charge)
    # }}}

    # {{{ _construct_activity_coefficient_solver
    def _construct_activity_coefficient_solver(
        self, charge: np.ndarray, ion_size: np.ndarray,
    ) -> ActivityCoefficientSolver:
        return ActivityCoefficientSolver(charge, ion_size)
    # }}}

    # {{{ _construct_response_solver
    def _construct_response_solver(self, charge: np.ndarray) -> ResponseSolver:
        return ResponseSolver(charge)
    # }}}

    # {{{ @property: response_intercept
    @property
    def response_intercept(self) -> np.ndarray:
        return self._response_solver.response_intercept
    # }}}

    # {{{ @property: response_slope
    @property
    def response_slope(self) -> np.ndarray:
        return self._response_solver.response_slope
    # }}}

    # {{{ @property: selectivity_coefficient
    @property
    def selectivity_coefficient(self) -> np.ndarray:
        return self._response_solver.selectivity_coefficient
    # }}}

    # {{{ forward_solve
    def forward_solve(self, concentration: np.ndarray) -> np.ndarray:
        activity = self._pre_process_concentration_forward(concentration)
        self._response_solver.eval()
        with torch.no_grad():
            response = self._response_solver(activity)
        response = response.numpy().squeeze()
        return response
    # }}}

    # {{{ _pre_process_concentration_forward
    def _pre_process_concentration_forward(
        self, concentration: np.ndarray) -> torch.Tensor:
        activity = chemistry_utils.convert_concentration_to_activity(
            concentration, self._charge, self._ion_size)
        for _ in range(2):
            activity = np.expand_dims(activity, 1)
        activity = np.tile(activity, (1,activity.shape[3],1,1))
        activity = np.power(activity, self._activity_power)
        activity = matrix_utils.build_tensor(activity)
        return activity
    # }}}

    # {{{ backward_solve
    def backward_solve(self, response: np.ndarray) -> np.ndarray:
        response = self._pre_process_response_backward(response)
        activity = self._activity_solver.solve(
            response, self._response_solver.selectivity_coefficient)
        activity_coefficient = self._activity_coefficient_solver.solve(activity)
        concentration = activity / activity_coefficient
        return concentration
    # }}}

    # {{{ _pre_process_response_backward
    def _pre_process_response_backward(self, response: np.ndarray) -> np.ndarray:
        response = response - self._response_solver.response_intercept
        response = response / self._response_solver.response_slope
        response = np.power(10, response)
        return response
    # }}}

    # {{{ forward
    def forward(self, candidate: torch.Tensor) -> torch.Tensor:
        return self._response_solver(candidate)
    # }}}

    # {{{ load_weight
    def load_weight(self, weight_file_path: str) -> None:
        self._response_solver = io_utils.load_state_dictionary(
            weight_file_path, self._response_solver)
    # }}}

    # {{{ save_weight
    def save_weight(self, weight_file_path: str) -> None:
        io_utils.save_state_dictionary(self._response_solver, weight_file_path)
    # }}}

    # {{{ clamp_selectivity_coefficient
    def clamp_selectivity_coefficient(self) -> None:
        self._response_solver._selectivity_coefficient.data.clamp_min_(0)
    # }}}
# }}}


"""----- Regression -----"""
# {{{ RegressionAgent
class RegressionAgent(Agent):
    """An Agent that performs multivariate ion analysis based on regression."""

    # {{{ __init__
    def __init__(self, sensor_number: int) -> None:
        self._response_intercept = self._construct_response_intercept(
            sensor_number)
        self._response_slope = self._construct_response_slope(sensor_number)
        self._selectivity_coefficient = self._construct_selectivity_coefficient(
            sensor_number)
    # }}}

    # {{{ _construct_response_intercept
    def _construct_response_intercept(self, sensor_number: int) -> np.ndarray:
        return matrix_utils.build_zeros_array((1, sensor_number))
    # }}}

    # {{{ _construct_response_slope
    def _construct_response_slope(self, sensor_number: int) -> np.ndarray:
        return matrix_utils.build_zeros_array((1, sensor_number))
    # }}}

    # {{{ _construct_selectivity_coefficient
    def _construct_selectivity_coefficient(
        self, sensor_number: int) -> np.ndarray:
        return matrix_utils.build_zeros_array((sensor_number, sensor_number))
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
        return matrix_utils.build_array(self._selectivity_coefficient)
    # }}}

    # {{{ forward_solve
    def forward_solve(self, concentration: np.ndarray) -> np.ndarray:
        response = np.log10(concentration @ self._selectivity_coefficient.T)
        response = self._response_intercept + self._response_slope*response
        return response
    # }}}

    # {{{ backward_solve
    def backward_solve(self, response: np.ndarray) -> np.ndarray:
        response = (response-self._response_intercept) / self._response_slope
        response = np.power(10, response)
        concentration = response @ np.linalg.pinv(self._selectivity_coefficient.T)
        return concentration
    # }}}

    # {{{ calibrate
    def calibrate(self, concentration: np.ndarray, response: np.ndarray) -> None:
        self._calibrate_response_intercept_and_slope(concentration, response)
        self._calibrate_selectivity_coefficient(concentration, response)
    # }}}

    # {{{ _calibrate_response_intercept_and_slope
    def _calibrate_response_intercept_and_slope(
        self, concentration: np.ndarray, response: np.ndarray) -> None:
        concentration = np.log10(concentration)
        for i in range(concentration.shape[1]):
            self._response_intercept[0,i], self._response_slope[0,i] = self._compute_response_intercept_and_slope_single(
                concentration[:,i], response[:,i])
    # }}}

    # {{{ _compute_response_intercept_and_slope_single
    def _compute_response_intercept_and_slope_single(
        self,
        concentration_column: np.ndarray,
        response_column: np.ndarray,
    ) -> tuple[float,float]:
        pass
    # }}}

    # {{{ _calibrate_selectivity_coefficient
    def _calibrate_selectivity_coefficient(
        self, concentration: np.ndarray, response: np.ndarray) -> None:
        response = np.power(
            10, (response-self._response_intercept)/self._response_slope)
        for i in range(concentration.shape[1]):
            self._selectivity_coefficient[i,:] = self._compute_selectivity_coefficient_single(
                concentration, response[:,i])
    # }}}

    # {{{ _compute_selectivity_coefficient_single
    def _compute_selectivity_coefficient_single(
        self,
        concentration: np.ndarray,
        response_column: np.ndarray,
    ) -> np.ndarray:
        pass
    # }}}
# }}}

# {{{ BayesianRegressionAgent
class BayesianRegressionAgent(RegressionAgent):
    """A class that can run forward/backward solving based on bayesian least
    squares regression.
    """

    # {{{ __init__
    def __init__(self, sensor_number: int) -> None:
        super().__init__(sensor_number)
    # }}}

    # {{{ _compute_response_intercept_and_slope_single
    def _compute_response_intercept_and_slope_single(
        self, concentration_column: np.ndarray, response_column: np.ndarray,
    ) -> tuple[float,float]:
        grid_searcher = model_selection.GridSearchCV(
            estimator = linear_model.BayesianRidge(),
            param_grid = {
                'max_iter': [1000],
                'tol': [1e-6],
                'alpha_1': [np.power(10.0,exponent) for exponent in range(-9,0)],
                'alpha_2': [np.power(10.0,exponent) for exponent in range(-9,0)],
                'lambda_1': [np.power(10.0,exponent) for exponent in range(-9,0)],
                'lambda_1': [np.power(10.0,exponent) for exponent in range(-9,0)],
                'fit_intercept': [True],
            },
            scoring = 'neg_mean_squared_error',
            n_jobs = -1,
        )
        grid_searcher.fit(concentration_column.reshape((-1,1)), response_column)
        return (
            grid_searcher.best_estimator_.intercept_,
            grid_searcher.best_estimator_.coef_,
        )
    # }}}

    # {{{ _compute_selectivity_coefficient_single
    def _compute_selectivity_coefficient_single(
        self,
        concentration: np.ndarray,
        response_column: np.ndarray,
    ) -> np.ndarray:
        grid_searcher = model_selection.GridSearchCV(
            estimator = linear_model.BayesianRidge(),
            param_grid = {
                'max_iter': [1000],
                'tol': [1e-6],
                'alpha_1': [np.power(10.0,exponent) for exponent in range(-9,0)],
                'alpha_2': [np.power(10.0,exponent) for exponent in range(-9,0)],
                'lambda_1': [np.power(10.0,exponent) for exponent in range(-9,0)],
                'lambda_1': [np.power(10.0,exponent) for exponent in range(-9,0)],
                'fit_intercept': [False],
            },
            scoring = 'neg_mean_squared_error',
            n_jobs = -1,
        )
        grid_searcher.fit(concentration, response_column)
        return grid_searcher.best_estimator_.coef_ 
    # }}}
# }}}

# {{{ OrdinaryRegressionAgent
class OrdinaryRegressionAgent(RegressionAgent):
    """A class that can run forward/backward solving based on ordinary least
    squares regression.
    """

    # {{{ __init__
    def __init__(self, sensor_number: int) -> None:
        super().__init__(sensor_number)
    # }}}

    # {{{ _compute_response_intercept_and_slope_single
    def _compute_response_intercept_and_slope_single(
        self,
        concentration_column: np.ndarray,
        response_column: np.ndarray,
    ) -> tuple[float, float]:
        regressor = linear_model.LinearRegression(fit_intercept=True)
        regressor.fit(concentration_column.reshape((-1,1)), response_column)
        return regressor.intercept_, regressor.coef_
    # }}}

    # {{{ _compute_selectivity_coefficient_single
    def _compute_selectivity_coefficient_single(
        self,
        concentration: np.ndarray,
        response_column: np.ndarray,
    ) -> np.ndarray:
        regressor = linear_model.LinearRegression(fit_intercept=False)
        regressor.fit(concentration, response_column)
        return regressor.coef_
    # }}}
# }}}

# {{{ PartialRegressionAgent
class PartialRegressionAgent(RegressionAgent):
    """A class that can run forward/backward solving based on partial least
    squares regression.
    """

    # {{{ __init__
    def __init__(self, sensor_number: int) -> None:
        super().__init__(sensor_number)
    # }}}

    # {{{ _compute_response_intercept_and_slope_single
    def _compute_response_intercept_and_slope_single(
        self,
        concentration_column: np.ndarray,
        response_column: np.ndarray,
    ) -> tuple[float,float]:
        regressor = cross_decomposition.PLSRegression(
            n_components=1, scale=True, max_iter=1000)
        regressor.fit(concentration_column.reshape((-1,1)), response_column)
        return regressor.intercept_, regressor.coef_
    # }}}

    # {{{ _compute_selectivity_coefficient_single
    def _compute_selectivity_coefficient_single(
        self,
        concentration: np.ndarray,
        response_column: np.ndarray,
    ) -> np.ndarray:
        grid_searcher = model_selection.GridSearchCV(
            estimator = cross_decomposition.PLSRegression(),
            param_grid = {
                'n_components': [i for i in range(1,concentration.shape[1]+1)],
                'scale': [True],
                'max_iter': [1000],
            },
            scoring = 'neg_mean_squared_error',
            n_jobs = -1,
        )
        grid_searcher.fit(concentration, response_column)
        return grid_searcher.best_estimator_.coef_
    # }}}
# }}}
