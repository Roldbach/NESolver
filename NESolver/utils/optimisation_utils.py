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
from NESolver.utils import data_processing_utils
from NESolver.utils import io_utils
from NESolver.utils import matrix_utils


VANILLA_POTENTIAL_SOLVER_PARAMETER = {
    'selectivity_coefficient': 'uniform',
    'slope': 'zeros',
    'drift': 'zeros',
}
NOVEL_POTENTIAL_SOLVER_PARAMETER = {
    'selectivity_coefficient': 'eye',
    'slope': 'nernst',
    'drift': 'Ag/AgCl',
}


"""----- Agent -----"""
# {{{ Agent
class Agent(abc.ABC):
    """An interface that defines an agent that can run forward/backward solving.
    """

    # {{{ __init__
    @abc.abstractmethod
    def __init__(self) -> None:
        pass
    # }}}

    # {{{ @property: selectivity_coefficient
    @property
    @abc.abstractmethod
    def selectivity_coefficient(self) -> np.ndarray:
        pass
    # }}}

    # {{{ @property: slope
    @property
    @abc.abstractmethod
    def slope(self) -> np.ndarray:
        pass
    # }}}

    # {{{ @property: drift
    @property
    @abc.abstractmethod
    def drift(self) -> np.ndarray:
        pass
    # }}}

    # {{{ forward_solve
    @abc.abstractmethod
    def forward_solve(self, concentration: np.ndarray) -> np.ndarray:
        pass
    # }}}

    # {{{ backward_solve
    @abc.abstractmethod
    def backward_solve(
        self, potential: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass
    # }}}
# }}}

# {{{ TrainableAgent
class TrainableAgent(nn.Module, Agent):
    """An interface that defines an agent that can be trained."""

    # {{{ forward
    @abc.abstractmethod
    def forward(self, candidate: torch.Tensor) -> torch.Tensor:
        pass
    # }}}

    # {{{ infer
    @abc.abstractmethod
    def infer(
        self,
        candidate: np.ndarray,
        reference: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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


"""----- Artificial Neural Network -----"""
# {{{ NeuralNetwork
class NeuralNetwork(nn.Module):
    """A class that can predict ion concentrations through deep learning."""

    # {{{ __init__
    def __init__(self, sensor_number: int) -> None:
        super().__init__()
        self._layer = nn.Sequential(
            nn.Linear(sensor_number, sensor_number).double(),
            nn.Sigmoid(),
            nn.Linear(sensor_number, sensor_number).double(),
            nn.Sigmoid(),
            nn.Linear(sensor_number, sensor_number).double(),
        )
    # }}}

    # {{{ forward
    def forward(self, candidate: torch.Tensor) -> torch.Tensor:
        return self._layer(candidate)
    # }}}
# }}}

# {{{ NeuralNetworkAgent
class NeuralNetworkAgent(TrainableAgent):
    """An agent that can run backward solving through deep learning."""

    # {{{ __init__
    def __init__(self, charge: np.ndarray, ion_size: np.ndarray) -> None:
        super().__init__()
        self._charge = self._construct_charge(charge)
        self._ion_size = self._construct_ion_size(ion_size)
        self._neural_network = self._construct_neural_network()
        self._data_processor = self._construct_data_processor()
    # }}}

    # {{{ _construct_charge
    def _construct_charge(self, charge: np.ndarray) -> np.ndarray:
        return matrix_utils.build_array(charge)
    # }}}

    # {{{ _construct_ion_size
    def _construct_ion_size(self, ion_size: np.ndarray) -> np.ndarray:
        return matrix_utils.build_array(ion_size)
    # }}}

    # {{{ _construct_neural_network
    def _construct_neural_network(self) -> NeuralNetwork:
        return NeuralNetwork(self._charge.shape[1])
    # }}}
    
    # {{{ _construct_data_processor
    def _construct_data_processor(
        self) -> data_processing_utils.NeuralNetworkAgentDataProcessor:
        return data_processing_utils.NeuralNetworkAgentDataProcessor()
    # }}}

    # {{{ selectivity_coefficient
    @property
    def selectivity_coefficient(self) -> None:
        pass
    # }}}

    # {{{ slope
    @property
    def slope(self) -> None:
        pass
    # }}}

    # {{{ drift
    @property
    def drift(self) -> None:
        pass
    # }}}

    # {{{ forward_solve
    def forward_solve(self, concentration: np.ndarray) -> np.ndarray:
        pass
    # }}}

    # {{{ backward_solve
    def backward_solve(
        self, potential: np.ndarray) -> np.ndarray:
        potential = self._data_processor.pre_process_candidate_backward(potential)
        with torch.no_grad():
            concentration = self._neural_network(potential)
        concentration = self._data_processor.post_process_prediction_backward(
            concentration)
        activity = chemistry_utils.convert_concentration_to_activity(
            concentration, self._charge, self._ion_size)
        return activity, concentration
    # }}}

    # {{{ forward
    def forward(self, candidate: torch.Tensor) -> torch.Tensor:
        return self._neural_network(candidate)
    # }}}

    # {{{ infer
    def infer(
        self,
        candidate: np.ndarray,
        reference: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidate = self._data_processor.pre_process_candidate_backward(
            candidate)
        reference = self._data_processor.pre_process_reference_backward(
            reference)
        prediction = self._neural_network(candidate)
        return prediction, reference
    # }}}

    # {{{ load_weight
    def load_weight(self, weight_file_path: str) -> None:
        self._neural_network = io_utils.load_state_dictionary(
            weight_file_path, self._neural_network)
    # }}}

    # {{{ save_weight
    def save_weight(self, weight_file_path: str) -> None:
        io_utils.save_state_dictionary(self._neural_network, weight_file_path)
    # }}}
# }}}


"""----- Numerical Solution -----"""
"""----- Activity -----"""
# {{{ ActivitySolver
class ActivitySolver:
    """A class that can inversely solve the activity of ions giving the
    transformed potential of ISEs (i.e., backward solving).
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
        self, potential: np.ndarray, selectivity_coefficient: np.ndarray,
    ) -> np.ndarray:
        return np.apply_along_axis(
            func1d = self._solve_row,
            axis = 1,
            arr = potential,
            selectivity_coefficient = selectivity_coefficient,
        ).reshape(potential.shape)
    # }}}

    # {{{  _solve_row
    def _solve_row(
        self, potential_row: np.ndarray, selectivity_coefficient: np.ndarray,
    ) -> np.ndarray:
        return optimize.root(
            fun = self._build_objective_function(
                potential_row, selectivity_coefficient),
            x0 = potential_row,
            jac = self._build_Jacobian(selectivity_coefficient),
        ).x
    # }}}

    # {{{ _build_objective_function
    def _build_objective_function(
        self, potential_row: np.ndarray, selectivity_coefficient: np.ndarray,
    ) -> typing.Callable:

        # {{{ objective_function
        def objective_function(activity_row: np.ndarray) -> np.ndarray:
            output = np.tile(activity_row, (activity_row.size,1))
            output = np.power(output, self._activity_power)
            output *= selectivity_coefficient
            output = np.hstack((
                output, -1.0*potential_row.reshape((-1,1)),
            ))
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

"""----- Activity Coefficient -----"""
# {{{ ActivityCoefficientSolver
class ActivityCoefficientSolver:
    """A class that can derive activity coefficients of ions through solving the
    extended Debye-Huckel equation.
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

"""----- Potential -----"""
# {{{ PotentialSolver
class PotentialSolver(nn.Module):
    """A class that can solve the underlying parameters of ISEs based on the
    Nikolsky-Eisenman equation.

    Attribute
        - _sensor_number: An int that specifies the number of ISEs.
        - _selectivity_coefficient: A torch.nn.Parameter that specifies the
                                    selectivity coefficients of ISEs with shape
                                    (#ISE, #ISE, 1).
        - _slope: A torch.nn.Parameter that specifies the slope of ISEs with
                  shape (#ISE, 1, 1).
        - _drift: A torch.nn.Parameter that specifies the potential drift of ISEs
                  with shape (#ISE, 1, 1).

    Property
        - selectivity_coefficient: A numpy.ndarray that specifies the selectivity
                                   coefficients of ISEs with shape (#ISE, #ISE).
        - slope: A numpy.ndarray that specifies the slope of ISEs with shape
                 (1, #ISE).
        - drift: A numpy.ndarray that specifies the potential drift of ISEs with 
                 shape (1, #ISE).
    """

    # {{{ __init__
    def __init__(
        self,
        charge: np.ndarray,
        selectivity_coefficient_initialisation: str,
        slope_initialisation: str,
        drift_initialisation: str,
    ) -> None:
        super().__init__()
        self._sensor_number = self._construct_sensor_number(charge)
        self._selectivity_coefficient = self._construct_selectivity_coefficient(
            selectivity_coefficient_initialisation)
        self._slope = self._construct_slope(slope_initialisation, charge)
        self._drift = self._construct_drift(drift_initialisation)
    # }}}

    # {{{ _construct_sensor_number
    def _construct_sensor_number(self, charge: np.ndarray) -> int:
        return charge.shape[1]
    # }}}

    # {{{ _construct_selectivity_coefficient
    def _construct_selectivity_coefficient(
        self, selectivity_coefficient_initialisation: str) -> nn.Parameter:
        weight = matrix_utils.initialise_weight_tensor(
            shape = (self._sensor_number, self._sensor_number),
            initialisation = selectivity_coefficient_initialisation,
        ).unsqueeze(2)
        selectivity_coefficient = nn.Parameter(weight)
        return selectivity_coefficient
    # }}}

    # {{{ _construct_slope
    def _construct_slope(
        self,
        slope_initialisation: str,
        charge: np.ndarray,
    ) -> nn.Parameter:
        weight = matrix_utils.initialise_weight_tensor(
            (self._sensor_number,1,1), slope_initialisation, charge=charge)
        slope = nn.Parameter(weight)
        return slope
    # }}}

    # {{{ _construct_drift
    def _construct_drift(self, drift_initialisation: str) -> nn.Parameter:
        weight = matrix_utils.initialise_weight_tensor(
            (self._sensor_number,1,1), drift_initialisation)
        drift = nn.Parameter(weight)
        return drift
    # }}}

    # {{{ @property: selectivity_coefficient
    @property
    def selectivity_coefficient(self) -> np.ndarray:
        selectivity_coefficient = self._selectivity_coefficient.clone().detach().numpy()
        selectivity_coefficient = selectivity_coefficient.reshape(
            (-1, self._sensor_number))
        return selectivity_coefficient
    # }}}

    # {{{ @property: slope
    @property
    def slope(self) -> np.ndarray:
        return self._slope.data.clone().detach().numpy().reshape((1,-1))
    # }}}

    # {{{ @property: drift
    @property
    def drift(self) -> np.ndarray:
        return self._drift.data.clone().detach().numpy().reshape((1,-1))
    # }}}

    # {{{ forward
    def forward(self, candidate: torch.Tensor) -> torch.Tensor:
        output = torch.log10(candidate @ self._selectivity_coefficient)
        output = self._drift + self._slope*output
        return output
    # }}}
# }}}

# {{{ VanillaPotentialSolver
class VanillaPotentialSolver(PotentialSolver):
    """A type of PotentialSolver that utilises vanilla settings."""

    # {{{ __init__
    def __init__(
        self,
        charge: np.ndarray,
        selectivity_coefficient_initialisation: str = VANILLA_POTENTIAL_SOLVER_PARAMETER['selectivity_coefficient'],
        slope_initialisation: str = VANILLA_POTENTIAL_SOLVER_PARAMETER['slope'],
        drift_initialisation: str = VANILLA_POTENTIAL_SOLVER_PARAMETER['drift'],
    ) -> None:
        super().__init__(
            charge,
            selectivity_coefficient_initialisation,
            slope_initialisation,
            drift_initialisation,
        )
    # }}}
# }}}

# {{{ NovelPotentialSolver
class NovelPotentialSolver(PotentialSolver):
    """A type of PotentialSolver that utilises novel settings."""

    # {{{ __init__
    def __init__(
        self,
        charge: np.ndarray,
        selectivity_coefficient_initialisation: str = NOVEL_POTENTIAL_SOLVER_PARAMETER['selectivity_coefficient'],
        slope_initialisation: str = NOVEL_POTENTIAL_SOLVER_PARAMETER['slope'],
        drift_initialisation: str = NOVEL_POTENTIAL_SOLVER_PARAMETER['drift'],
    ) -> None:
        super().__init__(
            charge,
            selectivity_coefficient_initialisation,
            slope_initialisation,
            drift_initialisation,
        )
    # }}}

    # {{{ _construct_selectivity_coefficient
    def _construct_selectivity_coefficient(
        self, selectivity_coefficient_initialisation: str) -> nn.Parameter:
        selectivity_coefficient = super()._construct_selectivity_coefficient(
            selectivity_coefficient_initialisation)
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
# }}}

"""----- Numerical Agent -----"""
# {{{ NumericalAgent
class NumericalAgent(TrainableAgent):
    """An agent that can run forward/backward solving through numerical solving."""

    # {{{ __init__
    def __init__(
        self,
        charge: np.ndarray,
        ion_size: np.ndarray,
        potential_solver_class: PotentialSolver,
    ) -> None:
        super().__init__()
        self._activity_solver = self._construct_activity_solver(charge)
        self._activity_coefficient_solver = self._construct_activity_coefficient_solver(
            charge, ion_size)
        self._potential_solver = self._construct_potential_solver(
            charge, potential_solver_class)
        self._data_processor = self._construct_data_processor(charge, ion_size)
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

    # {{{ _construct_potential_solver
    def _construct_potential_solver(
        self,
        charge: np.ndarray,
        potential_solver_class: PotentialSolver,
    ) -> PotentialSolver:
        return potential_solver_class(charge)
    # }}}

    # {{{ _construct_data_processor
    def _construct_data_processor(
        self,
        charge: np.ndarray,
        ion_size: np.ndarray,
    ) -> data_processing_utils.NumericalAgentDataProcessor:
        return data_processing_utils.NumericalAgentDataProcessor(charge, ion_size)
    # }}}

    # {{{ @property: selectivity_coefficient
    @property
    def selectivity_coefficient(self) -> np.ndarray:
        return self._potential_solver.selectivity_coefficient
    # }}}

    # {{{ @property: slope
    @property
    def slope(self) -> np.ndarray:
        return self._potential_solver.slope
    # }}}

    # {{{ @property: drift
    @property
    def drift(self) -> np.ndarray:
        return self._potential_solver.drift
    # }}}

    # {{{ forward_solve
    def forward_solve(self, concentration: np.ndarray) -> np.ndarray:
        activity = self._data_processor.pre_process_candidate_forward(
            concentration)
        self._potential_solver.eval()
        with torch.no_grad():
            potential = self._potential_solver(activity)
        potential = self._data_processor.post_process_prediction_forward(potential)
        return potential
    # }}}

    # {{{ backward_solve
    def backward_solve(
        self, potential: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        potential = self._data_processor.pre_process_candidate_backward(
            potential, self._potential_solver.slope, self._potential_solver.drift)
        activity = self._activity_solver.solve(
            potential, self._potential_solver.selectivity_coefficient)
        activity_coefficient = self._activity_coefficient_solver.solve(
            activity)
        concentration = activity / activity_coefficient
        return concentration
    # }}}

    # {{{ forward
    def forward(self, candidate: torch.Tensor) -> torch.Tensor:
        return self._potential_solver(candidate)
    # }}}

    # {{{ infer
    def infer(
        self,
        candidate: np.ndarray,
        reference: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidate = self._data_processor.pre_process_candidate_forward(
            candidate)
        reference = self._data_processor.pre_process_reference_forward(
            reference)
        prediction = self._potential_solver(candidate)
        return prediction, reference
    # }}}

    # {{{ load_weight
    def load_weight(self, weight_file_path: str) -> None:
        self._potential_solver = io_utils.load_state_dictionary(
            weight_file_path, self._potential_solver)
    # }}}

    # {{{ save_weight
    def save_weight(self, weight_file_path: str) -> None:
        io_utils.save_state_dictionary(self._potential_solver, weight_file_path)
    # }}}
# }}}

# {{{ VanillaNumericalAgent
class VanillaNumericalAgent(NumericalAgent):
    """An agent that can run forward/backward solving through numerical solving."""

    # {{{ __init__
    def __init__(self, charge: np.ndarray, ion_size: np.ndarray) -> None:
        super().__init__(charge, ion_size, VanillaPotentialSolver)
    # }}}
# }}}

# {{{ NovelNumericalAgent
class NovelNumericalAgent(NumericalAgent):
    """An agent that can run forward/backward solving through numerical solving."""

    # {{{ __init__
    def __init__(self, charge: np.ndarray, ion_size: np.ndarray) -> None:
        super().__init__(charge, ion_size, NovelPotentialSolver)
    # }}}

    # {{{ clamp_selectivity_coefficient
    def clamp_selectivity_coefficient(self) -> None:
        self._potential_solver._selectivity_coefficient.data.clamp_min_(0)
    # }}}
# }}}


"""----- Regression -----"""
# {{{ RegressionAgent
class RegressionAgent(Agent):
    """A class that can run forward/backward solving based on regression."""

    # {{{ __init__
    def __init__(self) -> None:
        self._selectivity_coefficient = None
        self._selectivity_coefficient_pseudo_inverse = None
        self._slope = None
        self._drift = None
    # }}}

    # {{{ @property: selectivity_coefficient
    @property
    def selectivity_coefficient(self) -> np.ndarray:
        return matrix_utils.build_array(self._selectivity_coefficient)
    # }}}

    # {{{ @property: slope
    @property
    def slope(self) -> np.ndarray:
        return matrix_utils.build_array(self._slope)
    # }}}

    # {{{ @property: drift
    @property
    def drift(self) -> np.ndarray:
        return matrix_utils.build_array(self._drift)
    # }}}

    # {{{ forward_solve
    def forward_solve(self, concentration: np.ndarray) -> np.ndarray:
        potential = np.log10(concentration @ self._selectivity_coefficient.T)
        potential = self._drift + self._slope*potential
        return potential
    # }}}

    # {{{ backward_solve
    def backward_solve(self, potential: np.ndarray) -> np.ndarray:
        potential = np.power(10, (potential-self._drift)/self._slope)
        concentration = potential @ self._selectivity_coefficient_pseudo_inverse
        return concentration
    # }}}

    # {{{ calibrate
    def calibrate(
        self, concentration: np.ndarray, potential: np.ndarray) -> None:
        self._slope, self._drift = self._calibrate_slope_and_drift(
            concentration, potential)
        self._selectivity_coefficient = self._calibrate_selectivity_coefficient(
            concentration, potential)
        self._selectivity_coefficient_pseudo_inverse = np.linalg.pinv(
            self._selectivity_coefficient.T)
    # }}}

    # {{{ _calibrate_slope_and_drift
    def _calibrate_slope_and_drift(
        self, concentration: np.ndarray, potential: np.ndarray) -> None:
        slope = matrix_utils.build_zeros_array((1, concentration.shape[1]))
        drift = matrix_utils.build_zeros_array((1, concentration.shape[1]))
        concentration = np.log10(concentration)
        for i in range(concentration.shape[1]):
            slope[0,i], drift[0,i] = self._calibrate_slope_and_drift_single(
                concentration[:,i], potential[:,i])
        return slope, drift
    # }}}

    # {{{ _calibrate_slope_and_drift_single
    def _calibrate_slope_and_drift_single(
        self, concentration_column: np.ndarray, potential_column: np.ndarray,
    ) -> tuple[float,float]:
        pass
    # }}}

    # {{{ _calibrate_selectivity_coefficient
    def _calibrate_selectivity_coefficient(
        self, concentration: np.ndarray, potential: np.ndarray) -> None:
        selectivity_coefficient = matrix_utils.build_zeros_array(
            (concentration.shape[1], concentration.shape[1]))
        potential = np.power(10, (potential-self._drift)/self._slope)
        for i in range(concentration.shape[1]):
            selectivity_coefficient[i,:] = self._calibrate_selectivity_coefficient_single(
                concentration, potential[:,i])
        return selectivity_coefficient
    # }}}

    # {{{ _calibrate_selectivity_coefficient_single
    def _calibrate_selectivity_coefficient_single(
        self,
        concentration: np.ndarray,
        potential_column: np.ndarray,
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
    def __init__(self) -> None:
        super().__init__()
    # }}}

    # {{{ _calibrate_slope_and_drift_single
    def _calibrate_slope_and_drift_single(
        self, concentration_column: np.ndarray, potential_column: np.ndarray,
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
        grid_searcher.fit(concentration_column.reshape((-1,1)), potential_column)
        return (
            grid_searcher.best_estimator_.coef_,
            grid_searcher.best_estimator_.intercept_,
        )
    # }}}

    # {{{ _calibrate_selectivity_coefficient_single
    def _calibrate_selectivity_coefficient_single(
        self,
        concentration: np.ndarray,
        potential_column: np.ndarray,
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
        grid_searcher.fit(concentration, potential_column)
        return grid_searcher.best_estimator_.coef_ 
    # }}}
# }}}

# {{{ OrdinaryRegressionAgent
class OrdinaryRegressionAgent(RegressionAgent):
    """A class that can run forward/backward solving based on ordinary least
    squares regression.
    """

    # {{{ __init__
    def __init__(self) -> None:
        super().__init__()
    # }}}

    # {{{ _calibrate_slope_and_drift_single
    def _calibrate_slope_and_drift_single(
        self, concentration_column: np.ndarray, potential_column: np.ndarray,
    ) -> tuple[float,float]:
        regressor = linear_model.LinearRegression(fit_intercept=True)
        regressor.fit(concentration_column.reshape((-1,1)), potential_column)
        return regressor.coef_, regressor.intercept_
    # }}}

    # {{{ _calibrate_selectivity_coefficient_single
    def _calibrate_selectivity_coefficient_single(
        self,
        concentration: np.ndarray,
        potential_column: np.ndarray,
    ) -> np.ndarray:
        regressor = linear_model.LinearRegression(fit_intercept=False)
        regressor.fit(concentration, potential_column)
        return regressor.coef_
    # }}}
# }}}

# {{{ PartialRegressionAgent
class PartialRegressionAgent(RegressionAgent):
    """A class that can run forward/backward solving based on partial least
    squares regression.
    """

    # {{{ __init__
    def __init__(self) -> None:
        super().__init__()
    # }}}

    # {{{ _calibrate_slope_and_drift_single
    def _calibrate_slope_and_drift_single(
        self, concentration_column: np.ndarray, potential_column: np.ndarray,
    ) -> tuple[float,float]:
        regressor = cross_decomposition.PLSRegression(
            n_components=1, scale=False, max_iter=1000)
        regressor.fit(concentration_column.reshape((-1,1)), potential_column)
        return regressor.coef_, regressor.intercept_
    # }}}

    # {{{ _calibrate_selectivity_coefficient_single
    def _calibrate_selectivity_coefficient_single(
        self,
        concentration: np.ndarray,
        potential_column: np.ndarray,
    ) -> np.ndarray:
        grid_searcher = model_selection.GridSearchCV(
            estimator = cross_decomposition.PLSRegression(),
            param_grid = {
                'n_components': [i for i in range(1,concentration.shape[1]+1)],
                'scale': [False],
                'max_iter': [1000],
            },
            scoring = 'neg_mean_squared_error',
            n_jobs = -1,
        )
        grid_searcher.fit(concentration, potential_column)
        return grid_searcher.best_estimator_.coef_
    # }}}
# }}}
