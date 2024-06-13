#! /usr/bin/env python
"""Script used to run the full pipeline including hyperparameter searching,
solver training and evaluation.

Author: Weixun Luo
Date: 28/04/2024
"""
import collections
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from configuration import path_configuration
from utils import io_utils
from utils import optimisation_utils
from utils import pipeline_utils
from utils import typing_utils

from torchmetrics import regression


DATA_PACK_FILE_PATH = f'{path_configuration.DATA_DIRECTORY_PATH}/Na-K-Cl_simulated_clean.npz'
SEED = 0  # Seed used for the control of randomness
SOLVER_CLASS = optimisation_utils.NovelSolver
SLICE_TRAINING = slice(10)
SLICE_VALIDATION = slice(1000,1100)
SLICE_TESTING = slice(1100,1200)


# {{{ main
def main(
    data_pack_file_path: str = DATA_PACK_FILE_PATH,
    solver_class: optimisation_utils.Solver = SOLVER_CLASS,
    slice_training: slice = SLICE_TRAINING,
    slice_validation: slice = SLICE_VALIDATION,
    slice_testing: slice = SLICE_TESTING,
    seed: int = SEED,
) -> int:
    set_seed(seed)
    data_pack = load_data_pack(data_pack_file_path)
    hyperparameter = search_hyperparameter(
        data_pack, solver_class, slice_training)
    solver, training_loss, validation_loss = train(
        data_pack, hyperparameter, solver_class, slice_training, slice_validation)
    plot_learning_curve(training_loss, validation_loss)
    evaluation_outcome = evaluate(data_pack, solver, slice_testing)
    return 0
# }}}

# {{{ set_seed
def set_seed(seed) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
# }}}

# {{{ load_data_pack
def load_data_pack(data_pack_file_path: str) -> typing_utils.DataPack:
    return io_utils.load_array_dictionary(data_pack_file_path)
# }}}

# {{{ search_hyperparameter
def search_hyperparameter(
    data_pack: typing_utils.DataPack,
    solver_class: optimisation_utils.Solver,
    slice_training: slice,
) -> typing_utils.Hyperparameter:
    pipeline = pipeline_utils.SolverHyperparameterSearchPipeline(
        solver_class,
        data_pack['charge'],
        data_pack['ion_size'],
        data_pack['activity'][slice_training,:],
        data_pack['potential'][slice_training,:],
    )
    return pipeline.search()
# }}}

# {{{ train
def train(
    data_pack: typing_utils.DataPack,
    hyperparameter: typing_utils.Hyperparameter,
    solver_class: optimisation_utils.Solver,
    slice_training: slice,
    slice_validation: slice,
) -> tuple[optimisation_utils.Solver, list[float], list[float]]:
    pipeline = pipeline_utils.SolverTrainingPipeline(
        solver_class(data_pack['charge'], data_pack['ion_size']),
        data_pack['charge'],
        data_pack['ion_size'],
        data_pack['activity'][slice_training,:],
        data_pack['potential'][slice_training,:],
        data_pack['activity'][slice_validation,:],
        data_pack['potential'][slice_validation,:],
        hyperparameter['lr'],
        hyperparameter['optimizer'],
        hyperparameter['criterion'],
    )
    return pipeline.train()
# }}}

# {{{ plot_learning_curve
def plot_learning_curve(
    training_loss: list[float], validation_loss: list[float]) -> None:
    plt.plot([i for i in range(len(training_loss))], np.log10(training_loss))
    plt.plot([i for i in range(len(validation_loss))], np.log10(validation_loss))
    plt.show()
# }}}

# {{{ evaluate
def evaluate(
    data_pack: typing_utils.DataPack,
    solver: optimisation_utils.Solver,
    slice_testing: slice,
) -> collections.defaultdict:
    pipeline = pipeline_utils.SolverEvaluationPipeline(
        solver,
        data_pack['selectivity_coefficient'],
        data_pack['slope'],
        data_pack['drift'],
        data_pack['concentration'][slice_testing,:],
        data_pack['activity'][slice_testing,:],
        data_pack['potential'][slice_testing,:],
    )
    return pipeline.evaluate()
# }}}


if __name__ == '__main__':
    main()
