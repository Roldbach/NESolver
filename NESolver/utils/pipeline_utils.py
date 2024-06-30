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
from NESolver.utils import data_utils
from NESolver.utils import evaluation_utils
from NESolver.utils import format_utils
from NESolver.utils import io_utils
from NESolver.utils import matrix_utils
from NESolver.utils import optimisation_utils
from NESolver.utils import typing_utils


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
        patience: int = 1000,
        delta: float = 0.0,
        checkpoint_file_path: str = '.checkpoint.pth',
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


"""----- Training -----"""
# {{{ TrainingPipline
class TrainingPipeline:
    """A pipeline that trains an agent to fit the underlying parameters of ISEs.
    """

    # {{{ __init__
    def __init__(
        self,
        charge: np.ndarray,
        ion_size: np.ndarray,
        concentration: np.ndarray,
        response: np.ndarray,
        slice_training: slice,
        slice_validation: slice,
        learning_rate: float,
        optimiser_class: optim.Optimizer,
        criterion_class: nn.Module,
        epoch_outcome_limit: int = 15,
    ) -> None:
        self._agent = self._construct_agent(charge, ion_size)
        self._data_loader_training = self._construct_data_loader(
            charge,
            ion_size,
            concentration[slice_training,...],
            response[slice_training,...],
        )
        self._data_loader_validation = self._construct_data_loader(
            charge,
            ion_size,
            concentration[slice_validation,...],
            response[slice_validation,...],
        )
        self._optimiser = self._construct_optimiser(
            optimiser_class, learning_rate)
        self._criterion = self._construct_criterion(criterion_class)
        self._epoch = 0
        self._training_loss_all, self._validation_loss_all = [], []
        self._early_stopper = self._construct_early_stopper()
        self._epoch_outcome_deque = self._construct_epoch_outcome_deque(
            epoch_outcome_limit)
        self._epoch_outcome_limit = epoch_outcome_limit
    # }}}

    # {{{ _construct_agent
    def _construct_agent(
        self,
        charge: np.ndarray,
        ion_size: np.ndarray,
    ) -> optimisation_utils.OptimisationAgent:
        return optimisation_utils.OptimisationAgent(charge, ion_size)
    # }}}

    # {{{ _construct_data_loader
    def _construct_data_loader(
        self,
        charge: np.ndarray,
        ion_size: np.ndarray,
        concentration: np.ndarray,
        response: np.ndarray,
    ) -> data_utils.OptimisationAgentForwardDataLoader:
        return data_utils.OptimisationAgentForwardDataLoader(
            charge, ion_size, concentration, response)
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
        training_loss = 0.0
        self._agent.train()
        for candidate_batch, reference_batch in self._data_loader_training:
            training_loss += self._train_batch(candidate_batch, reference_batch)
        self._training_loss_all.append(
            training_loss / len(self._data_loader_training))
    # }}}

    # {{{ _train_batch
    def _train_batch(
        self,
        candidate_batch: torch.Tensor,
        reference_batch: torch.Tensor,
    ) -> float:
        prediction_batch = self._agent(candidate_batch)
        training_loss_batch = self._criterion(prediction_batch, reference_batch)
        training_loss_batch.backward()
        self._optimiser.step()
        self._optimiser.zero_grad()
        self._agent.clamp_selectivity_coefficient()
        return training_loss_batch.item()
    # }}}

    # {{{ _validate
    def _validate(self) -> None:
        validation_loss = 0.0
        self._agent.eval()
        with torch.no_grad():
            for candidate_batch, reference_batch in self._data_loader_validation:
                validation_loss += self._validate_batch(
                    candidate_batch, reference_batch)
        self._validation_loss_all.append(
            validation_loss / len(self._data_loader_validation))
    # }}}

    # {{{ _validate_batch
    def _validate_batch(
        self,
        candidate_batch: torch.Tensor,
        reference_batch: torch.Tensor,
    ) -> float:
        prediction_batch = self._agent(candidate_batch)
        validation_loss_batch = self._criterion(prediction_batch, reference_batch)
        return validation_loss_batch.item()
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
        if self._epoch <= self._epoch_outcome_limit:
            print(self._epoch_outcome_deque[-1])
        else:
            sys.stdout.write(f'\033[{self._epoch_outcome_limit}F')
            sys.stdout.write('\033[K')
            print('\n'.join(self._epoch_outcome_deque))
    # }}}

    # {{{ _is_stopping
    def _is_stopping(self) -> bool:
        return self._early_stopper(self._validation_loss_all[-1], self._agent)
    # }}}
# }}}


"""----- Evaluation -----"""
# {{{ EvaluationPipeline
class EvaluationPipeline:
    """A pipeline that evaluates the agent performance in multivariate ion
    analysis.
    """

    # {{{ __init__
    def __init__(
        self,
        agent: optimisation_utils.Agent,
        response_intercept: np.ndarray,
        response_slope: np.ndarray,
        selectivity_coefficient: np.ndarray,
        concentration: np.ndarray,
        response: np.ndarray,
    ) -> None:
        self._agent = agent
        self._response_intercept = self._construct_response_intercept(
            response_intercept)
        self._response_slope = self._construct_response_slope(response_slope)
        self._selectivity_coefficient = self._construct_selectivity_coefficient(
            selectivity_coefficient)
        self._concentration = self._construct_concentration(concentration)
        self._response = self._construct_response(response)
    # }}}

    # {{{ _construct_response_intercept
    def _construct_response_intercept(
        self, response_intercept: np.ndarray) -> np.ndarray:
        return matrix_utils.build_row_array(response_intercept)
    # }}}

    # {{{ _construct_response_slope
    def _construct_response_slope(self, response_slope: np.ndarray) -> np.ndarray:
        return matrix_utils.build_row_array(response_slope)
    # }}}

    # {{{ _construct_selectivity_coefficient
    def _construct_selectivity_coefficient(
        self, selectivity_coefficient: np.ndarray) -> np.ndarray:
        return matrix_utils.build_array(selectivity_coefficient)
    # }}}

    # {{{ _construct_concentration
    def _construct_concentration(self, concentration: np.ndarray) -> np.ndarray:
        return matrix_utils.build_array(concentration)
    # }}}

    # {{{ _construct_response
    def _construct_response(self, response: np.ndarray) -> np.ndarray:
        return matrix_utils.build_array(response)
    # }}}

    # {{{ evaluate
    def evaluate(self) -> collections.defaultdict:
        print('##### Evaluation #####')
        self._evaluate_forward_accuracy()
        self._evaluate_backward_accuracy()
        self._evaluate_response_intercept()
        self._evaluate_response_slope()
        self._evaluate_selectivity_coefficient()
    # }}}

    # {{{ _evaluate_forward_accuracy
    def _evaluate_forward_accuracy(self) -> None:
        response = self._agent.forward_solve(self._concentration)
        error_sensor_wise = evaluation_utils.compute_mean_absolute_percentage_error(
            response, self._response, 0)
        error_sensor_wise = format_utils.format_scientific_array(error_sensor_wise)
        error_overall = evaluation_utils.compute_mean_absolute_percentage_error(
            response, self._response)
        error_overall = format_utils.format_scientific_value(error_overall)
        print('##### Forward Accuracy (Sensor Response) #####')
        print(f'- Sensor-wise Percentage Error (%): {error_sensor_wise}')
        print(f'- Overall Percentage Error (%): {error_overall}')
        print('')
    # }}}

    # {{{ _evaluate_backward_accuracy
    def _evaluate_backward_accuracy(self) -> None:
        concentration = self._agent.backward_solve(self._response)
        error_sensor_wise = evaluation_utils.compute_mean_absolute_percentage_error(
            concentration, self._concentration, 0)
        error_sensor_wise = format_utils.format_scientific_array(error_sensor_wise)
        error_overall = evaluation_utils.compute_mean_absolute_percentage_error(
            concentration, self._concentration)
        error_overall = format_utils.format_scientific_value(error_overall)
        print('##### Backward Accuracy (Ion Concentration) #####')
        print(f'- Sensor-wise Percentage Error (%): {error_sensor_wise}')
        print(f'- Overall Percentage Error (%): {error_overall}')
        print('')
    # }}}

    # {{{ _evaluate_response_intercept
    def _evaluate_response_intercept(self) -> None:
        is_numerically_close = evaluation_utils.is_numerically_close(
            self._agent.response_intercept, self._response_intercept)
        error_sensor_wise = evaluation_utils.compute_absolute_percentage_error(
            self._agent.response_intercept, self._response_intercept)
        error_sensor_wise = format_utils.format_scientific_array(error_sensor_wise)
        error_overall = evaluation_utils.compute_mean_absolute_percentage_error(
            self._agent.response_intercept, self._response_intercept)
        error_overall = format_utils.format_scientific_value(error_overall)
        print('##### Response Intercept #####')
        print(f'- Derived: {self._agent.response_intercept.flatten().tolist()}')
        print(f'- True: {self._response_intercept.flatten().tolist()}')
        print(f'- Is numerically close? {is_numerically_close.flatten().tolist()}')
        print(f'- Sensor-wise Percentage Error (%): {error_sensor_wise}')
        print(f'- Overall Percentage Error (%): {error_overall}')
        print('')
    # }}}

    # {{{ _evaluate_response_slope
    def _evaluate_response_slope(self) -> None:
        is_numerically_close = evaluation_utils.is_numerically_close(
            self._agent.response_slope, self._response_slope)
        error_sensor_wise = evaluation_utils.compute_absolute_percentage_error(
            self._agent.response_slope, self._response_slope)
        error_sensor_wise = format_utils.format_scientific_array(error_sensor_wise)
        error_overall = evaluation_utils.compute_mean_absolute_percentage_error(
            self._agent.response_slope, self._response_slope)
        error_overall = format_utils.format_scientific_value(error_overall)
        print('##### Response Slope #####')
        print(f'- Derived: {self._agent.response_slope.flatten().tolist()}')
        print(f'- True: {self._response_slope.flatten().tolist()}')
        print(f'- Is numerically close? {is_numerically_close.flatten().tolist()}')
        print(f'- Sensor-wise Percentage Error (%): {error_sensor_wise}')
        print(f'- Overall Percentage Error (%): {error_overall}')
        print('')
    # }}}

    # {{{ _evaluate_selectivity_coefficient
    def _evaluate_selectivity_coefficient(self) -> None:
        is_numerically_close = evaluation_utils.is_numerically_close(
            self._agent.selectivity_coefficient, self._selectivity_coefficient)
        error_sensor_wise = evaluation_utils.compute_mean_absolute_error(
            self._agent.selectivity_coefficient,
            self._selectivity_coefficient,
            axis = 1,
        )
        error_sensor_wise = format_utils.format_scientific_array(error_sensor_wise)
        error_overall = evaluation_utils.compute_mean_absolute_error(
            self._agent.selectivity_coefficient, self._selectivity_coefficient)
        error_overall = format_utils.format_scientific_value(error_overall)
        print('##### Selectivity Coefficient #####')
        print('- Derived:')
        print(self._agent.selectivity_coefficient)
        print('- True:')
        print(self._selectivity_coefficient)
        print('- Is numerically close?')
        print(is_numerically_close)
        print(f'- Sensor-wise L1 Error (a.u.): {error_sensor_wise}')
        print(f'- Overall L1 Error (a.u.): {error_overall}')
        print('')
    # }}}
# }}}
