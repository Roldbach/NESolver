#! /usr/bin/env python
"""Script used to evaluate RegressionAgent performance with respect to training
sample number.

Author: Weixun Luo
Date: 08/05/2024
"""
import collections
import sys

import numpy as np
import torch

from configuration import path_configuration
from utils import io_utils
from utils import matrix_utils
from utils import optimisation_utils
from utils import pipeline_utils
from utils import typing_utils


DATA_PACK_FILE_PATH_ALL = (
    f'{path_configuration.DATA_DIRECTORY_PATH}/Na-K-Cl_simulated_clean.npz',
    f'{path_configuration.DATA_DIRECTORY_PATH}/Na-K-Mg-Ca-Cl_simulated_clean.npz',
)
PLOT_DATA_FILE_PATH_ALL = (
    f'{path_configuration.DATA_DIRECTORY_PATH}/regression_performance_vs_training_sample_size_Na-K-Cl.mat',
    f'{path_configuration.DATA_DIRECTORY_PATH}/regression_performance_vs_training_sample_size_Na-K-Mg-Ca-Cl.mat',
)
SEED_ALL = tuple(i for i in range(3))
SLICE_TRAINING_ALL = tuple(slice(i) for i in range(5,1001))
SLICE_VALIDATION = slice(1000,1100)
SLICE_TESTING = slice(1100,1200)


# {{{ main
def main(
    seed_all: tuple[int,...] = SEED_ALL,
    data_pack_file_path_all: tuple[str,...] = DATA_PACK_FILE_PATH_ALL,
    plot_data_file_path_all: tuple[str, ...] = PLOT_DATA_FILE_PATH_ALL,
    slice_training_all: tuple[slice,...] = SLICE_TRAINING_ALL,
    slice_validation: slice = SLICE_VALIDATION,
    slice_testing: slice = SLICE_TESTING,
) -> int:
    for data_pack_file_path, plot_data_file_path in zip(data_pack_file_path_all, plot_data_file_path_all):
        plot_data = {}
        plot_data['ordinary'] = run_pipeline_all(
            seed_all, data_pack_file_path, optimisation_utils.OrdinaryRegressionAgent,
            slice_training_all, slice_validation, slice_testing)
        plot_data['partial'] = run_pipeline_all(
            seed_all, data_pack_file_path, optimisation_utils.PartialRegressionAgent,
            slice_training_all, slice_validation, slice_testing)
        plot_data['bayesian'] = run_pipeline_all(
            seed_all, data_pack_file_path, optimisation_utils.BayesianRegressionAgent,
            slice_training_all, slice_validation, slice_testing)
        io_utils.save_array_dictionary_matlab(plot_data, plot_data_file_path)
    return 0
# }}}

# {{{ run_pipeline_all
def run_pipeline_all(
    seed_all: tuple[int,...],
    data_pack_file_path: str,
    agent_class: optimisation_utils.RegressionAgent,
    slice_training_all: tuple[slice,...],
    slice_validation: slice,
    slice_testing: slice,
) -> dict[str,float]:
    output = []
    for slice_training in slice_training_all:
        output.append(
            np.mean([
                run_pipeline_single(
                    seed,
                    data_pack_file_path,
                    agent_class,
                    slice_training,
                    slice_validation,
                    slice_testing,
                )
                for seed in seed_all
            ], axis=0)
        )
    output = np.vstack(output)
    return output
# }}}

# {{{ run_pipeline_single
def run_pipeline_single(
    seed: int,
    data_pack_file_path: str,
    agent_class: optimisation_utils.RegressionAgent,
    slice_training: slice,
    slice_validation: slice,
    slice_testing: slice,
) -> np.ndarray:
    set_seed(seed)
    data_pack = load_data_pack(data_pack_file_path)
    agent = agent_class(data_pack['charge'], data_pack['ion_size'])
    agent.calibrate(
        data_pack['concentration'][slice_training,:],
        data_pack['potential'][slice_training,:],
    )
    performance = evaluate(data_pack, agent, slice_testing)
    performance = matrix_utils.build_row_array([
        np.nanmean(parameter['sensor_wise_error'])
        for parameter in performance.values()
    ])
    return performance
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
