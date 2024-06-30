#! /usr/bin/env python
"""Script used to run multivariate ion analysis with optimisation.

Author: Weixun Luo
Date: 28/04/2024
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from NESolver.utils import io_utils
from NESolver.utils import optimisation_utils
from NESolver.utils import pipeline_utils
from NESolver.utils import typing_utils


# {{{ main
def main() -> int:
    argument = parse_argument()
    set_seed(argument.seed)
    data_pack = load_data_pack(argument.file_path)
    hyperparameter = search_hyperparameter(data_pack, argument.training_range)
    agent, training_loss, validation_loss = train_agent(
        data_pack,
        hyperparameter,
        argument.training_range,
        argument.validation_range,
    )
    plot_learning_curve(training_loss, validation_loss)
    evaluate_agent(data_pack, agent, argument.testing_range)
    return 0
# }}}

# {{{ parse_argument
def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = 'Script used to run multivariate ion analysis with optimisation',
    )
    parser.add_argument(
        '-f', '--file_path',
        type = str,
        required = True,
        help = 'file path of the data pack used for multivariate ion analysis',
    )
    parser.add_argument(
        '--training_range',
        default = slice(10),
        type = lambda x: slice(*map(int, x.strip('()').split(","))),
        help = 'the range of data used for training/calibration',
    )
    parser.add_argument(
        '--validation_range',
        default = slice(1000, 1100),
        type = lambda x: slice(*map(int, x.strip('()').split(","))),
        help = 'the range of data used for validation',
    )
    parser.add_argument(
        '--testing_range',
        default = slice(1100, 1200),
        type = lambda x: slice(*map(int, x.strip('()').split(","))),
        help = 'the range of data used for testing',
    )
    parser.add_argument(
        '--seed',
        default = 0,
        type = int,
        help = 'the random seed used for multivariate ion analysis',
    )
    return parser.parse_args()
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
    data_pack: typing_utils.DataPack, slice_training: slice,
) -> typing_utils.Hyperparameter:
    pipeline = pipeline_utils.HyperparameterSearchPipeline(
        data_pack['charge'],
        data_pack['ion_size'],
        data_pack['concentration'][slice_training,:],
        data_pack['response'][slice_training,:],
    )
    return pipeline.search()
# }}}

# {{{ train_agent
def train_agent(
    data_pack: typing_utils.DataPack,
    hyperparameter: typing_utils.Hyperparameter,
    slice_training: slice,
    slice_validation: slice,
) -> tuple[optimisation_utils.OptimisationAgent, list[float], list[float]]:
    pipeline = pipeline_utils.TrainingPipeline(
        data_pack['charge'],
        data_pack['ion_size'],
        data_pack['concentration'],
        data_pack['response'],
        slice_training,
        slice_validation,
        **hyperparameter,
    )
    return pipeline.train()
# }}}

# {{{ plot_learning_curve
def plot_learning_curve(
    training_loss: list[float], validation_loss: list[float]) -> None:
    plt.plot(np.log10(training_loss))
    plt.plot(np.log10(validation_loss))
    plt.show()
# }}}

# {{{ evaluate_agent
def evaluate_agent(
    data_pack: typing_utils.DataPack,
    agent: optimisation_utils.OptimisationAgent,
    slice_testing: slice,
) -> None:
    pipeline = pipeline_utils.EvaluationPipeline(
        agent,
        data_pack['response_intercept'],
        data_pack['response_slope'],
        data_pack['selectivity_coefficient'],
        data_pack['concentration'][slice_testing,:],
        data_pack['response'][slice_testing,:],
    )
    pipeline.evaluate()
# }}}


if __name__ == '__main__':
    main()
