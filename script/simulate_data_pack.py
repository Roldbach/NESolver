#! /usr/bin/env python
"""Script used to simualte the concentration and activity of ions and
corresponding potentials from ISEs.

Author: Weixun Luo
Date: 01/04/2024
"""
import numpy as np

from configuration import path_configuration
from utils import chemistry_utils
from utils import data_simulation_utils
from utils import io_utils
from utils import matrix_utils
from utils import typing_utils


CHARGE_ALL = (
    matrix_utils.build_row_array((+1,+1,-1)),  # Na+, K+, Cl-
    matrix_utils.build_row_array((+1,+1,+2,-1)),  # Na+, K+, Ca2+, Cl-
    matrix_utils.build_row_array((+1,+1,+2,+2,-1)),  # Na+, K+, Mg2+, Ca2+, Cl-
)
DATA_PACK_FILE_PATH_ALL = (
    f'{path_configuration.DATA_DIRECTORY_PATH}/Na-K-Cl_simulated_clean.npz',
    f'{path_configuration.DATA_DIRECTORY_PATH}/Na-K-Ca-Cl_simulated_clean.npz',
    f'{path_configuration.DATA_DIRECTORY_PATH}/Na-K-Mg-Ca-Cl_simulated_clean.npz',
)
ION_SIZE_ALL = (
    matrix_utils.build_row_array((450,300,300)),
    matrix_utils.build_row_array((450,300,600,300)),
    matrix_utils.build_row_array((450,300,800,600,300)),
)  # Ion size (pm)
NOISE_PARAMETER_ALL = (
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
)
SCOPE_TO_SAMPLE_NUMBER = {
    (1e-6, 1e-4): 1200,  # Training: 1000, Validation: 100, Test (in-range): 100
    (1e-8, 1e-6): 50,  # Test (out-range): 50 + 50
    (1e-4, 1e-2): 50,
}  # The overall ionic strength must be < 0.1M due to the extended Debye-Huckel
   # equation
SEED = 0  # Seed used for the control of randomness
SELECTIVITY_COEFFICIENT_ALL = (
    matrix_utils.build_array((
        (1.0, np.power(10,-2.1), 0.0),  # Reference: Table 3, Na+ -2 (on Page 47)
        (np.power(10,-4.6), 1.0, 0.0),  # Reference: Table 4, K+ -1 (on Page 81)
        (0.0, 0.0, 1.0),  # Assume no interference from anion
    )),
    matrix_utils.build_array((
        (1.0, np.power(10,-2.1), np.power(10,-2.8), 0.0),  # Reference: Table 3, Na+ -2, Page 47
        (np.power(10,-4.6), 1.0, np.power(10,-5.15), 0.0),  # Reference: Table 4, K+ -1, Page 81
        (np.power(10,-4.05), np.power(10,-4.1), 1.0, 0.0),  # Reference: Table 9, Ca2+ -4 Page 149
        (0.0, 0.0, 0.0, 1.0),  # Assume no interference from Cl- ion
    )),
    matrix_utils.build_array((
        (1.0, np.power(10,-2.1), np.power(10,-4.7), np.power(10,-2.8), 0.0),  # Reference: Table 3, Na+ -2, Page 47
        (np.power(10,-4.6), 1.0, np.power(10,-5.1), np.power(10,-5.15), 0.0),  # Reference: Table 4, K+ -1, Page 81
        (np.power(10,-1.1), np.power(10,-0.6), 1.0, np.power(10,-0.2), 0.0),  # Reference: Table 8, Mg2+ -7, Page 127 
        (np.power(10,-4.05), np.power(10,-4.1), np.power(10,-3.3), 1.0, 0.0),  # Reference: Table 9, Ca2+ -4 Page 149
        (0.0, 0.0, 0.0, 0.0, 1.0),  # Assume no interference from Cl- ion
    )),
)


# {{{ main
def main(
    charge_all: tuple = CHARGE_ALL,
    ion_size_all: tuple = ION_SIZE_ALL,
    selectivity_coefficient_all: tuple = SELECTIVITY_COEFFICIENT_ALL,
    noise_parameter_all: tuple = NOISE_PARAMETER_ALL,
    scope_to_sample_number: dict = SCOPE_TO_SAMPLE_NUMBER,
    data_pack_file_path_all: tuple = DATA_PACK_FILE_PATH_ALL,
) -> int:
    set_seed()
    for i in range(len(CHARGE_ALL)):
        data_pack = simulate_data_pack(
            charge_all[i],
            ion_size_all[i],
            selectivity_coefficient_all[i],
            noise_parameter_all[i],
            scope_to_sample_number,
        )
        save_data_pack(data_pack, data_pack_file_path_all[i])
    return 0
# }}}

# {{{ set_seed
def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
# }}}

# {{{ simulate_data_pack
def simulate_data_pack(
    charge: np.ndarray,
    ion_size: np.ndarray,
    selectivity_coefficient: np.ndarray,
    noise_parameter: tuple[float, float],
    scope_to_sample_number: dict,
) -> typing_utils.DataPack:
    data_pack_simulator = data_simulation_utils.DataPackSimulator(
        charge, ion_size, selectivity_coefficient, noise_parameter)
    data_pack = data_pack_simulator.simulate(scope_to_sample_number)
    return data_pack
# }}}

# {{{ save_data_pack
def save_data_pack(data_pack: typing_utils.DataPack, file_path: str) -> None:
    """Save the simulated data to the specified file path."""
    io_utils.save_arrray_dictionary(data_pack, file_path)
# }}}

if __name__ == '__main__':
    main()
