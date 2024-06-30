#! /usr/bin/env python
"""Script used to run multivariate ion analysis with regression.

Author: Weixun Luo
Date: 24/05/2024
"""
import argparse
import collections
import sys
import typing

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
    agent = calibrate(data_pack, argument.method, argument.training_range)
    evaluate(data_pack, agent, argument.testing_range)
    return 0
# }}}

# {{{ parse_argument
def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = 'Script used to run multivariate ion analysis with regression',
    )
    parser.add_argument(
        '-f', '--file_path',
        type = str,
        required = True,
        help = 'file path of the data pack used for multivariate ion analysis',
    )
    parser.add_argument(
        '-m', '--method',
        type = lambda x: {
            'OLS': optimisation_utils.OrdinaryRegressionAgent,
            'PLS': optimisation_utils.PartialRegressionAgent,
            'BR': optimisation_utils.BayesianRegressionAgent,
        }[x],
        required = True,
        help = 'regression method used for multivariate ion analysis',
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
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
# }}}

# {{{ load_data_pack
def load_data_pack(data_pack_file_path: str) -> typing_utils.DataPack:
    return io_utils.load_array_dictionary(data_pack_file_path)
# }}}

# {{{ calibrate
def calibrate(
    data_pack: typing_utils.DataPack,
    agent_class: optimisation_utils.RegressionAgent,
    slice_training: slice,
) -> optimisation_utils.RegressionAgent:
    agent = agent_class(data_pack['charge'].shape[1])
    agent.calibrate(
        data_pack['concentration'][slice_training,:],
        data_pack['response'][slice_training,:],
    )
    return agent
# }}}

# {{{ evaluate
def evaluate(
    data_pack: typing_utils.DataPack,
    agent: optimisation_utils.RegressionAgent,
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
