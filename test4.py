import numpy as np

from utils import io_utils


data_pack = io_utils.load_array_dictionary('./data/Na-K-Mg-Ca-Cl_simulated_noisy_high.npz')
print(data_pack['drift'])
print(data_pack['slope'])
