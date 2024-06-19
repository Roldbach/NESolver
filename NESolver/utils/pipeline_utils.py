"""A utility module that builds pipelines used in the project.

Author: Weixun Luo
Date: 03/04/2024
"""
import collections
import functools
import sys
import time
import typing

import numpy as np
from sklearn import model_selection
from skorch import net
import torch
from torch import nn
from torch import optim
from torchmetrics import regression

from NESolver.utils import chemistry_utils
from NESolver.utils import data_processing_utils
from NESolver.utils import data_simulation_utils
from NESolver.utils import evaluation_utils
from NESolver.utils import format_utils
from NESolver.utils import io_utils
from NESolver.utils import matrix_utils
from NESolver.utils import optimisation_utils
from NESolver.utils import typing_utils


EARLY_STOPPER_PARAMETER = {
    'patience': 1000,
    'delta': 0.0,
    'checkpoint_file_path': '.checkpoint.pth',
}
EPOCH_OUTCOME_LIMIT = 15  # The number of lines that can be seen in terminal
HYPERPARAMETER_SPACE = {
    'max_epochs': [1000],
    'lr': [1e-3, 1e-4, 1e-5],
    'optimizer': [optim.Adam],
    'criterion': [nn.MSELoss, regression.MeanSquaredLogError],
}


"""----- Timing -----"""
# {{{ timer
class timer:
    """A decorator that can record and display the run-time of functions."""

    # {{{ __init__
    def __init__(self) -> None:
        self._time_start = None
        self._time_end = None
    # }}}

    # {{{ __call__
    def __call__(self, function: typing.Callable) -> typing.Callable:
        @functools.wraps(function)
        def _decorator(*args, **kwargs) -> typing.Callable:
            self._start_timing()
            output = function(*args, **kwargs)
            self._end_timing()
            self._print_timing()
            return output

        return _decorator
    # }}}

    # {{{ _start_timing
    def _start_timing(self) -> None:
        self._time_start = time.time()
    # }}}

    # {{{ _end_timing
    def _end_timing(self) -> None:
        self._time_end = time.time()
    # }}}

    # {{{ _print_timing
    def _print_timing(self) -> None:
        time_run = format_utils.format_float_value(self._time(), 's')
        print('')
        print(f'Completed in: {time_run}')
        print('')
    # }}}

    # {{{ _time
    def _time(self) -> float:
        return self._time_end - self._time_start
    # }}}
# }}}


"""----- Early Stopping -----"""
# {{{ EarlyStopper
class EarlyStopper:
    """A class that can early stop the training of the agent if its validation
    loss doesn't improve lasting for a given number of epochs.
    """

    # {{{ __init__
    def __init__(
        self,
        patience = EARLY_STOPPER_PARAMETER['patience'],
        delta = EARLY_STOPPER_PARAMETER['delta'],
        checkpoint_file_path = EARLY_STOPPER_PARAMETER['checkpoint_file_path'],
    ) -> None:
        self._patience = patience
        self._delta = delta
        self._checkpoint_file_path = checkpoint_file_path
        self._validation_loss_optimal = np.inf
        self._counter = 0
    # }}}

    # {{{ __call__
    def __call__(
        self,
        validation_loss: float,
        agent: nn.Module,
    ) -> bool:
        if validation_loss == float('nan'):
            self._load_checkpoint(agent)
            return True
        if validation_loss+self._delta <= self._validation_loss_optimal:
            self._validation_loss_optimal = validation_loss
            self._counter = 0
            self._save_checkpoint(agent)
            return False
        else:
            self._counter += 1
            if self._counter == self._patience:
                self._load_checkpoint(agent)
                return True
            return False
    # }}}

    # {{{ _load_checkpoint
    def _load_checkpoint(self, agent: nn.Module) -> None:
        agent.load_weight(self._checkpoint_file_path)
    # }}}

    # {{{ _save_checkpoint
    def _save_checkpoint(self, agent: nn.Module) -> None:
        agent.save_weight(self._checkpoint_file_path)
    # }}}
# }}}


"""----- Hyperparameter Search -----"""
# {{{ HyperparameterSearchPipeline
class HyperparameterSearchPipeline:
    """A pipeline that can search optimal hyperparametes for agent training."""

    # {{{ __init__
    def __init__(
        self,
        agent_class: nn.Module,
        charge: np.ndarray,
        ion_size: np.ndarray,
        activity: np.ndarray,
        potential: np.ndarray,
        hyperparameter_space: dict = HYPERPARAMETER_SPACE,
    ) -> None:
        self._dataset = self._construct_dataset(charge, activity, potential)
        self._grid = self._construct_grid(
            agent_class, charge, ion_size, hyperparameter_space)
    # }}}

    # {{{ _construct_grid
    def _construct_grid(
        self,
        agent_class: nn.Module,
        charge: np.ndarray,
        ion_size: np.ndarray,
        hyperparameter_space: dict,
    ) -> model_selection.GridSearchCV:
        return model_selection.GridSearchCV(
            estimator = self._build_estimator(agent_class, charge, ion_size),
            param_grid = hyperparameter_space,
            scoring = self._build_score_function(),
            refit = False,
            n_jobs = -1,
            verbose = 0,
        )
    # }}}

    # {{{ _build_estimator
    def _build_estimator(
        self,
        agent_class: nn.Module,
        charge: np.ndarray,
        ion_size: np.ndarray,
    ) -> net.NeuralNet:
        return net.NeuralNet(
            module = agent_class,
            module__charge = charge,
            module__ion_size = ion_size,
            criterion = None,
            batch_size = len(self._dataset),
            verbose = 0,
        )
    # }}}

    # {{{ _build_score_function
    def _build_score_function(self) -> typing.Callable:

        # {{{ _score_function
        def _score_function(
            estimator: net.NeuralNet,
            candidate: np.ndarray,
            reference: np.ndarray,
        ) -> float:
            prediction = estimator.predict(candidate)
            mean_squared_error = evaluation_utils.compute_mean_squared_error(
                prediction, reference)
            score = -1.0 * mean_squared_error
            return score
        # }}}

        return _score_function
    # }}}

    # {{{ search
    @timer()
    def search(self) -> typing_utils.Hyperparameter:
        print('##### Hyperparameter Grid Search #####')
        hyperparameter_optimal, score_optimal = self._search()
        self._print_search_outcome(hyperparameter_optimal, score_optimal)
        return hyperparameter_optimal
    # }}}

    # {{{ _search
    def _search(self) -> tuple[dict, float]:
        search_outcome = self._grid.fit(
            self._dataset.activity, self._dataset.potential)
        return search_outcome.best_params_, search_outcome.best_score_
    # }}}

    # {{{ _print_search_outcome
    def _print_search_outcome(
        self, hyperparameter_optimal: dict, score_optimal: float) -> None:
        score_optimal = format_utils.format_scientific_value(score_optimal)
        print('- Optimal Hyperparameter')
        for name, parameter in hyperparameter_optimal.items():
            print(f'\t - {name}: {parameter}')
        print(f'- Optimal L2 Score: {score_optimal}')
    # }}}
# }}}

# {{{ SolverHyperparameterSearchPipeline
class SolverHyperparameterSearchPipeline(HyperparameterSearchPipeline):
    """A pipeline that can search optimal hyperparameters for Solver training."""

    # {{{ __init__
    def __init__(
        self,
        agent_class: optimisation_utils.NumericalAgent,
        charge: np.ndarray,
        ion_size: np.ndarray,
        activity: np.ndarray,
        potential: np.ndarray,
        hyperparameter_space: dict = HYPERPARAMETER_SPACE,
    ) -> None:
        super().__init__(
            agent_class,
            charge,
            ion_size,
            activity,
            potential,
            hyperparameter_space,
        )
    # }}}
# }}}


"""----- Training -----"""
# {{{ AgentTrainingPipline
class AgentTrainingPipeline:
    """A pipeline that trains an agent to fit the underlying parameters of ISEs.
    """

    # {{{ __init__
    def __init__(
        self,
        agent_class: typing.Callable,
        charge: np.ndarray,
        ion_size: np.ndarray,
        candidate: np.ndarray,
        reference: np.ndarray,
        slice_training: slice,
        slice_validation: slice,
        learning_rate: float,
        optimiser_class: optim.Optimizer,
        criterion_class: nn.Module,
        epoch_outcome_limit: int = EPOCH_OUTCOME_LIMIT,
    ) -> None:
        self._agent = self._construct_agent(agent_class, charge, ion_size)
        self._candidate = self._construct_candidate(candidate)
        self._reference = self._construct_reference(reference)
        self._slice_training = slice_training
        self._slice_validation = slice_validation
        self._optimiser = self._construct_optimiser(
            optimiser_class, learning_rate)
        self._criterion = self._construct_criterion(criterion_class)
        self._epoch = 0
        self._training_loss_all, self._validation_loss_all = [], []
        self._early_stopper = self._construct_early_stopper()
        self._epoch_outcome_deque = self._construct_epoch_outcome_deque(
            epoch_outcome_limit)
    # }}}

    # {{{ _construct_agent
    def _construct_agent(
        self,
        agent_class: typing.Callable,
        charge: np.ndarray,
        ion_size: np.ndarray,
    ) -> optimisation_utils.TrainableAgent:
        return agent_class(charge, ion_size)
    # }}}

    # {{{ _construct_candidate
    def _construct_candidate(self, candidate: np.ndarray) -> np.ndarray:
        return matrix_utils.build_array(candidate)
    # }}}

    # {{{ _construct_reference
    def _construct_reference(self, reference: np.ndarray) -> np.ndarray:
        return matrix_utils.build_array(reference)
    # }}}

    # {{{ _construct_optimiser
    def _construct_optimiser(
        self,
        optimiser_class: optim.Optimizer,
        learning_rate: float,
    ) -> optim.Optimizer:
        return optimiser_class(self._agent.parameters(), learning_rate)
    # }}}

    # {{{ _construct_criterion
    def _construct_criterion(self, criterion_class: nn.Module) -> nn.Module:
        return criterion_class()
    # }}}

    # {{{ _construct_early_stopper
    def _construct_early_stopper(self) -> EarlyStopper:
        return EarlyStopper()
    # }}}

    # {{{ _construct_epoch_outcome_deque
    def _construct_epoch_outcome_deque(
        self, epoch_outcome_limit) -> collections.deque:
        return collections.deque(maxlen=epoch_outcome_limit)
    # }}}

    # {{{ train
    @timer()
    def train(self) -> tuple:
        print('##### Forward Solving #####')
        while True:
            self._epoch += 1
            self._train()
            self._validate()
            self._append_epoch_outcome()
            self._print_epoch_outcome()
            if self._is_stopping():
                return (
                    self._agent,
                    self._training_loss_all,
                    self._validation_loss_all,
                )
    # }}}

    # {{{ _train
    def _train(self) -> None:
        self._agent.train()
        prediction, reference = self._agent.infer(
            self._candidate[self._slice_training,:],
            self._reference[self._slice_training,:],
        )
        training_loss = self._criterion(prediction, reference)
        training_loss.backward()
        self._optimiser.step()
        self._optimiser.zero_grad()
        self._training_loss_all.append(training_loss.item())
    # }}}

    # {{{ _validate
    def _validate(self) -> float:
        self._agent.eval()
        with torch.no_grad():
            prediction, reference = self._agent.infer(
                self._candidate[self._slice_validation,:],
                self._reference[self._slice_validation,:],
            )
            validation_loss = self._criterion(prediction, reference)
        self._validation_loss_all.append(validation_loss.item())
    # }}}

    # {{{ _append_epoch_outcome
    def _append_epoch_outcome(self) -> str:
        training_loss = format_utils.format_scientific_value(
            self._training_loss_all[-1])
        validation_loss = format_utils.format_scientific_value(
            self._validation_loss_all[-1])
        self._epoch_outcome_deque.append((
            f'Epoch {self._epoch} \t'
            f'Training Loss: {training_loss} \t'
            f'Validation Loss: {validation_loss} \t'
        ))
    # }}}

    # {{{ _print_epoch_outcome
    def _print_epoch_outcome(self) -> None:
        if self._epoch <= EPOCH_OUTCOME_LIMIT:
            print(self._epoch_outcome_deque[-1])
        else:
            sys.stdout.write(f'\033[{EPOCH_OUTCOME_LIMIT}F')
            sys.stdout.write('\033[K')
            print('\n'.join(self._epoch_outcome_deque))
    # }}}

    # {{{ _is_stopping
    def _is_stopping(self) -> bool:
        return self._early_stopper(self._validation_loss_all[-1], self._agent)
    # }}}
# }}}

# {{{ NeuralNetworkAgentTrainingPipeline
class NeuralNetworkAgentTrainingPipeline(AgentTrainingPipeline):
    """A pipeline that trains a NeuralNetworkAgent to run backward solving."""

    # {{{ __init__
    def __init__(
        self,
        charge: np.ndarray,
        ion_size: np.ndarray,
        concentration: np.ndarray,
        potential: np.ndarray,
        slice_training: slice,
        slice_validation: slice,
        learning_rate: float,
        optimiser_class: optim.Optimizer,
        criterion_class: nn.Module,
    ) -> None:
        super().__init__(
            optimisation_utils.NeuralNetworkAgent,
            charge,
            ion_size,
            potential,
            concentration,
            slice_training,
            slice_validation,
            learning_rate,
            optimiser_class,
            criterion_class,
        )
    # }}}
# }}}

# {{{ VanillaNumericalAgentTrainingPipeline
class VanillaNumericalAgentTrainingPipeline(AgentTrainingPipeline):
    """A pipeline that trains a VanillaNumericalAgent to run forward solving."""

    # {{{ __init__
    def __init__(
        self,
        charge: np.ndarray,
        ion_size: np.ndarray,
        concentration: np.ndarray,
        potential: np.ndarray,
        slice_training: slice,
        slice_validation: slice,
        learning_rate: float,
        optimiser_class: optim.Optimizer,
        criterion_class: nn.Module,
    ) -> None:
        super().__init__(
            optimisation_utils.VanillaNumericalAgent,
            charge,
            ion_size,
            concentration,
            potential,
            slice_training,
            slice_validation,
            learning_rate,
            optimiser_class,
            criterion_class,
        )
    # }}}
# }}}

# {{{ NovelNumericalAgentTrainingPipeline
class NovelNumericalAgentTrainingPipeline(AgentTrainingPipeline):
    """A pipeline that trains a VanillaNumericalAgent to run forward solving."""

    # {{{ __init__
    def __init__(
        self,
        charge: np.ndarray,
        ion_size: np.ndarray,
        concentration: np.ndarray,
        potential: np.ndarray,
        slice_training: slice,
        slice_validation: slice,
        learning_rate: float,
        optimiser_class: optim.Optimizer,
        criterion_class: nn.Module,
    ) -> None:
        super().__init__(
            optimisation_utils.NovelNumericalAgent,
            charge,
            ion_size,
            concentration,
            potential,
            slice_training,
            slice_validation,
            learning_rate,
            optimiser_class,
            criterion_class,
        )
    # }}}

    # {{{ _train
    def _train(self) -> None:
        self._agent.train()
        prediction, reference = self._agent.infer(
            self._candidate[self._slice_training,:],
            self._reference[self._slice_training,:],
        )
        training_loss = self._criterion(prediction, reference)
        training_loss.backward()
        self._optimiser.step()
        self._optimiser.zero_grad()
        self._agent.clamp_selectivity_coefficient()
        self._training_loss_all.append(training_loss.item())
    # }}}
# }}}


"""----- Evaluation -----"""
# {{{ AgentEvaluationPipeline
class AgentEvaluationPipeline:
    """A pipeline that evaluates the trained agent."""

    # {{{ __init__
    def __init__(
        self,
        agent: optimisation_utils.Agent,
        selectivity_coefficient: np.ndarray,
        slope: np.ndarray,
        drift: np.ndarray,
        concentration: np.ndarray,
        activity: np.ndarray,
        potential: np.ndarray,
    ) -> None:
        self._agent = agent
        self._selectivity_coefficient = self._construct_selectivity_coefficient(
            selectivity_coefficient)
        self._slope = self._construct_slope(slope)
        self._drift = self._construct_drift(drift)
        self._concentration = self._construct_concentration(concentration)
        self._activity = self._construct_activity(activity)
        self._potential = self._construct_potential(potential)
        self._evaluation_outcome = self._construct_evaluation_outcome()
    # }}}

    # {{{ _construct_selectivity_coefficient
    def _construct_selectivity_coefficient(
        self, selectivity_coefficient: np.ndarray) -> np.ndarray:
        return matrix_utils.build_array(selectivity_coefficient)
    # }}}

    # {{{ _construct_slope
    def _construct_slope(self, slope: np.ndarray) -> np.ndarray:
        return matrix_utils.build_array(slope)
    # }}}

    # {{{ _construct_drift
    def _construct_drift(self, drift: np.ndarray) -> np.ndarray:
        return matrix_utils.build_array(drift)
    # }}}

    # {{{ _construct_concentration
    def _construct_concentration(self, concentration: np.ndarray) -> np.ndarray:
        return matrix_utils.build_array(concentration)
    # }}}

    # {{{ _construct_activity
    def _construct_activity(self, activity: np.ndarray) -> np.ndarray:
        return matrix_utils.build_array(activity)
    # }}}

    # {{{ _construct_potential
    def _construct_potential(self, potential: np.ndarray) -> np.ndarray:
        return matrix_utils.build_array(potential)
    # }}}

    # {{{ _construct_evaluation_outcome
    def _construct_evaluation_outcome(self) -> collections.defaultdict:
        return collections.defaultdict(dict)
    # }}}

    # {{{ evaluate
    def evaluate(self) -> collections.defaultdict:
        print('##### Evaluation #####')
        self._evaluate_drift()
        self._evaluate_slope()
        self._evaluate_selectivity_coefficient()
        self._evaluate_potential()
        self._evaluate_concentration()
        return self._evaluation_outcome
    # }}}

    # {{{ _evaluate_drift
    def _evaluate_drift(self) -> None:
        self._evaluation_outcome['drift']['sensor_wise_error'] = evaluation_utils.compute_absolute_percentage_error(
            self._agent.drift, self._drift).flatten()
        self._evaluation_outcome['drift']['overall_error'] = evaluation_utils.compute_mean_absolute_percentage_error(
            self._agent.drift, self._drift)
        self._evaluation_outcome['drift']['is_numerically_close'] = evaluation_utils.is_numerically_close(
            self._agent.drift, self._drift).flatten()
        error_sensor_wise = format_utils.format_scientific_array(
            self._evaluation_outcome['drift']['sensor_wise_error'])
        error_overall = format_utils.format_scientific_value(
            self._evaluation_outcome['drift']['overall_error'])
        print('##### Response Intercept #####')
        print(f'- Derived: {self._agent.drift.flatten().tolist()}')
        print(f'- True: {self._drift.flatten().tolist()}')
        print(f'- Is numerically close? {self._evaluation_outcome["drift"]["is_numerically_close"].tolist()}')
        print(f'- Sensor-wise Percentage Error (%): {error_sensor_wise}')
        print(f'- Overall Percentage Error (%): {error_overall}')
        print('')
    # }}}

    # {{{ _evaluate_slope
    def _evaluate_slope(self) -> None:
        self._evaluation_outcome['slope']['sensor_wise_error'] = evaluation_utils.compute_absolute_error(
            self._agent.slope, self._slope).flatten()
        self._evaluation_outcome['slope']['overall_error'] = evaluation_utils.compute_mean_absolute_error(
            self._agent.slope, self._slope)
        self._evaluation_outcome['slope']['is_numerically_close'] = evaluation_utils.is_numerically_close(
            self._agent.slope, self._slope).flatten()
        error_sensor_wise = format_utils.format_scientific_array(
            self._evaluation_outcome['slope']['sensor_wise_error'])
        error_overall = format_utils.format_scientific_value(
            self._evaluation_outcome['slope']['overall_error'])
        print('##### Response Slope #####')
        print(f'- Derived: {self._agent.slope.flatten().tolist()}')
        print(f'- True: {self._slope.flatten().tolist()}')
        print(f'- Is numerically close? {self._evaluation_outcome["slope"]["is_numerically_close"].tolist()}')
        print(f'- Sensor-wise Percentage Error (%): {error_sensor_wise}')
        print(f'- Overall Percentage Error (%): {error_overall}')
        print('')
    # }}}

    # {{{ _evaluate_selectivity_coefficient
    def _evaluate_selectivity_coefficient(self) -> None:
        self._evaluation_outcome['selectivity_coefficient']['sensor_wise_error'] = evaluation_utils.compute_mean_absolute_error(
            self._agent.selectivity_coefficient,
            self._selectivity_coefficient,
            axis = 1,
        )
        self._evaluation_outcome['selectivity_coefficient']['overall_error'] = evaluation_utils.compute_mean_absolute_error(
            self._agent.selectivity_coefficient,
            self._selectivity_coefficient,
        )
        self._evaluation_outcome['selectivity_coefficient']['is_numerically_close'] = evaluation_utils.is_numerically_close(
            self._agent.selectivity_coefficient,
            self._selectivity_coefficient,
        )
        error_sensor_wise = format_utils.format_scientific_array(
            self._evaluation_outcome['selectivity_coefficient']['sensor_wise_error'])
        error_overall = format_utils.format_scientific_value(
            self._evaluation_outcome['selectivity_coefficient']['overall_error'])
        print('##### Selectivity Coefficient #####')
        print('- Derived:')
        print(self._agent.selectivity_coefficient)
        print('- True:')
        print(self._selectivity_coefficient)
        print('- Is numerically close?')
        print(self._evaluation_outcome['selectivity_coefficient']['is_numerically_close'])
        print(f'- Sensor-wise L1 Error (a.u.): {error_sensor_wise}')
        print(f'- Overall L1 Error (a.u.): {error_overall}')
        print('')
    # }}}

    # {{{ _evaluate_potential
    def _evaluate_potential(self) -> None:
        potential = self._agent.forward_solve(self._concentration)
        self._evaluation_outcome['potential']['sensor_wise_error'] = evaluation_utils.compute_mean_absolute_percentage_error(
            potential, self._potential, 0).flatten()
        self._evaluation_outcome['potential']['overall_error'] = evaluation_utils.compute_mean_absolute_percentage_error(
            potential, self._potential)
        error_sensor_wise = format_utils.format_scientific_array(
            self._evaluation_outcome['potential']['sensor_wise_error'])
        error_overall = format_utils.format_scientific_value(
            self._evaluation_outcome['potential']['overall_error'])
        #self._evaluation_outcome['potential']['is_statistically_close'], is_difference_normal = evaluation_utils.is_statistically_close(
            #potential, self._potential)
        print('##### Forward Accuracy (Sensor Response) #####')
        print(f'- Sensor-wise Percentage Error (%): {error_sensor_wise}')
        print(f'- Overall Percentage Error (%): {error_overall}')
        #print(f'- Is statistically close? {self._evaluation_outcome["potential"]["is_statistically_close"].flatten().tolist()}')
        #print(f'- Is checked by t-test? {is_difference_normal.flatten().tolist()}')
        print('')
    # }}}

    # {{{ _evaluate_activity
    def _evaluate_activity(self) -> None:
        activity, _ = self._agent.backward_solve(self._potential)
        self._evaluation_outcome['activity']['sensor_wise_error'] = evaluation_utils.compute_mean_absolute_percentage_error(
            activity, self._activity, 0).flatten()
        self._evaluation_outcome['activity']['overall_error'] = evaluation_utils.compute_mean_absolute_percentage_error(
            activity, self._activity)
        self._evaluation_outcome['activity']['is_statistically_close'], is_difference_normal = evaluation_utils.is_statistically_close(
            activity, self._activity)
        error_sensor_wise = format_utils.format_scientific_array(
            self._evaluation_outcome['activity']['sensor_wise_error'])
        error_overall = format_utils.format_scientific_value(
            self._evaluation_outcome['activity']['overall_error'])
        print('##### Backward Accuracy - Activity #####')
        print(f'- Sensor-wise Percentage Error (%): {error_sensor_wise}')
        print(f'- Overall Percentage Error (%): {error_overall}')
        print(f'- Is statistically close? {self._evaluation_outcome["activity"]["is_statistically_close"].flatten().tolist()}')
        print(f'- Is checked by t-test? {is_difference_normal.flatten().tolist()}')
        print('')
    # }}}

    # {{{ _evaluate_concentration
    def _evaluate_concentration(self) -> None:
        concentration = self._agent.backward_solve(self._potential)
        self._evaluation_outcome['concentration']['sensor_wise_error'] = evaluation_utils.compute_mean_absolute_percentage_error(
            concentration, self._concentration, 0).flatten()
        self._evaluation_outcome['concentration']['overall_error'] = evaluation_utils.compute_mean_absolute_percentage_error(
            concentration, self._concentration)
        #self._evaluation_outcome['concentration']['is_statistically_close'], is_difference_normal = evaluation_utils.is_statistically_close(
            #concentration, self._concentration)
        error_sensor_wise = format_utils.format_scientific_array(
            self._evaluation_outcome['concentration']['sensor_wise_error'])
        error_overall = format_utils.format_scientific_value(
            self._evaluation_outcome['concentration']['overall_error'])
        print('##### Backward Accuracy (Ion Concentration) #####')
        print(f'- Sensor-wise Percentage Error (%): {error_sensor_wise}')
        print(f'- Overall Percentage Error (%): {error_overall}')
        #print(f'- Is statistically close? {self._evaluation_outcome["concentration"]["is_statistically_close"].flatten().tolist()}')
        #print(f'- Is checked by t-test? {is_difference_normal.flatten().tolist()}')
        print('')
    # }}}
# }}}

# {{{ NeuralNetworkAgentEvaluationPipeline
class NeuralNetworkAgentEvaluationPipeline(AgentEvaluationPipeline):
    """A pipeline that evaluates the trained NeuralNetworkAgent."""

    # {{{ __init__
    def __init__(
        self,
        agent: optimisation_utils.NeuralNetwork,
        selectivity_coefficient: np.ndarray,
        slope: np.ndarray,
        drift: np.ndarray,
        concentration: np.ndarray,
        activity: np.ndarray,
        potential: np.ndarray,
    ) -> None:
        super().__init__(
            agent,
            selectivity_coefficient,
            slope,
            drift,
            concentration,
            activity,
            potential,
        )
    # }}}

    # {{{ _evaluate_selectivity_coefficient
    def _evaluate_selectivity_coefficient(self) -> None:
        pass
    # }}}

    # {{{ _evaluate_slope
    def _evaluate_slope(self) -> None:
        pass
    # }}}

    # {{{ _evaluate_drift
    def _evaluate_drift(self) -> None:
        pass
    # }}}

    # {{{ _evaluate_potential
    def _evaluate_potential(self) -> None:
        pass
    # }}}
# }}}

# {{{ RegressonAgentEvaluationPipeline
class RegressionAgentEvaluationPipeline(AgentEvaluationPipeline):
    """A pipeline that evaluates the calibrated RegressionAgent."""

    # {{{ __init__
    def __init__(
        self,
        agent: optimisation_utils.Agent,
        selectivity_coefficient: np.ndarray,
        slope: np.ndarray,
        drift: np.ndarray,
        concentration: np.ndarray,
        activity: np.ndarray,
        potential: np.ndarray,
    ) -> None:
        super().__init__(
            agent,
            selectivity_coefficient,
            slope,
            drift,
            concentration,
            activity,
            potential,
        )
    # }}}
# }}}

# {{{ NumericalAgentEvaluationPipeline
class NumericalAgentEvaluationPipeline(AgentEvaluationPipeline):
    """A pipeline that evaluates the trained NumericalAgent."""

    # {{{ __init__
    def __init__(
        self,
        agent: optimisation_utils.NumericalAgent,
        selectivity_coefficient: np.ndarray,
        slope: np.ndarray,
        drift: np.ndarray,
        concentration: np.ndarray,
        activity: np.ndarray,
        potential: np.ndarray,
    ) -> None:
        super().__init__(
            agent,
            selectivity_coefficient,
            slope,
            drift,
            concentration,
            activity,
            potential,
        )
    # }}}
# }}}
