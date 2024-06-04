#! /usr/bin/env python
"""Script used to calibrate and evaluate RegressionAgent.

Author: Weixun Luo
Date: 24/05/2024
"""
import collections
import sys
import typing

import numpy as np
import torch

from configuration import path_configuration
from utils import io_utils
from utils import optimisation_utils
from utils import pipeline_utils
from utils import typing_utils


AGENT_CLASS = optimisation_utils.OrdinaryRegressionAgent
DATA_PACK_FILE_PATH = f'{path_configuration.DATA_DIRECTORY_PATH}/Na-K-Cl_simulated_clean.npz'
SEED = 0  # Seed used for the control of randomness
SLICE_TRAINING = slice(10)
SLICE_VALIDATION = slice(1000,1100)
SLICE_TESTING = slice(1100,1200)


# {{{ main
def main(
    data_pack_file_path: str = DATA_PACK_FILE_PATH,
    agent_class: typing.Callable = AGENT_CLASS,
    slice_training: slice = SLICE_TRAINING,
    slice_validation: slice = SLICE_VALIDATION,
    slice_testing: slice = SLICE_TESTING,
    seed: int = SEED,
) -> int:
    set_seed(seed)
    data_pack = load_data_pack(data_pack_file_path)
    agent = calibrate(
        data_pack,
        agent_class(),
        slice_training,
    )
    evaluation_outcome = evaluate(data_pack, agent, slice_testing)
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

# {{{ calibrate
def calibrate(
    data_pack: typing_utils.DataPack,
    agent: optimisation_utils.RegressionAgent,
    slice_training: slice,
) -> optimisation_utils.RegressionAgent:
    agent.calibrate(
        data_pack['concentration'][slice_training,:],
        data_pack['potential'][slice_training,:],
    )
    return agent
# }}}

# {{{ evaluate
def evaluate(
    data_pack: typing_utils.DataPack,
    agent: optimisation_utils.RegressionAgent,
    slice_testing: slice,
) -> collections.defaultdict:
    pipeline = pipeline_utils.RegressionAgentEvaluationPipeline(
        agent,
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
