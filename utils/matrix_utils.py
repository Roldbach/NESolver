"""A module that handles array and tensor-related operations in the project.

Author: Weixun Luo
Date: 29/03/2024
"""
from collections.abc import Iterable
import typing

import numpy as np
import torch
from torch import nn

from utils import chemistry_utils


FLOAT_NUMPY = np.float64
FLOAT_TORCH = torch.float64


"""----- Array -----"""
# {{{ build_array
def build_array(
    element: typing.Any, data_type: np.dtype = FLOAT_NUMPY) -> np.ndarray:
    return np.array(element, data_type)
# }}}

# {{{ build_column_array
def build_column_array(
    element: Iterable, data_type: np.dtype = FLOAT_NUMPY) -> np.ndarray:
    return np.array(element, data_type).reshape((-1,1))
# }}}

# {{{ build_row_array
def build_row_array(
    element: Iterable, data_type: np.dtype = FLOAT_NUMPY) -> np.ndarray:
    return np.array(element, data_type).reshape((1,-1))
# }}}

# {{{ build_ones_array
def build_ones_array(
    shape: tuple[int, ...], data_type: np.dtype = FLOAT_NUMPY) -> np.ndarray:
    return np.ones(shape, data_type)
# }}}

# {{{ build_zeros_array
def build_zeros_array(
    shape: tuple[int, ...], data_type: np.dtype = FLOAT_NUMPY) -> np.ndarray:
    return np.zeros(shape, data_type)
# }}}

# {{{ sample_uniform_distribution_array
def sample_uniform_distribution_array(
    shape: tuple[int,int], scope: tuple[float,float]) -> np.ndarray:
    return np.random.uniform(*scope, shape).astype(FLOAT_NUMPY)
# }}}


"""----- Tensor -----"""
# {{{ build_tensor
def build_tensor(
    element: typing.Any,
    data_type: torch.dtype = FLOAT_TORCH,
    requires_grad = False,
) -> torch.Tensor:
    return torch.tensor(element, dtype=data_type, requires_grad=requires_grad)
# }}}

# {{{ build_identity_tensor
def build_identity_tensor(
    length: int,
    data_type: torch.dtype = FLOAT_TORCH,
    requires_grad = False,
) -> torch.Tensor:
    return torch.eye(length, dtype=data_type, requires_grad=requires_grad)
# }}}

# {{{ build_zeros_tensor
def build_zeros_tensor(
    shape: tuple[int, ...],
    data_type: torch.dtype = FLOAT_TORCH,
    requires_grad = False,
) -> torch.Tensor:
    return torch.zeros(shape, dtype=data_type, requires_grad=requires_grad)
# }}}

# {{{ initialise_weight_tensor
def initialise_weight_tensor(
    shape: tuple[int,...], initialisation: str, **kwargs) -> torch.Tensor:
    match initialisation:
        case 'Ag/AgCl':
            weight = build_zeros_tensor(shape, requires_grad=True)
            return nn.init.constant_(weight, chemistry_utils.DRIFT_VALUE)
        case 'eye':
            assert len(shape) == 2
            return build_identity_tensor(shape[0], requires_grad=True)
        case 'nernst':
            return _initialise_Nernst_weight_tensor(shape, kwargs['charge'])
        case 'uniform':
            weight = build_zeros_tensor(shape, requires_grad=True)
            return nn.init.uniform_(weight, a=1e-14)  # Avoid error in log()
        case 'zeros':
            weight = build_zeros_tensor(shape, requires_grad=True)
            return nn.init.zeros_(weight)
        case _:
            raise ValueError(
                f'Cannot find the initialisation: {initialisation}')
# }}}

# {{{ _initialise_Nernst_weight_tensor
def _initialise_Nernst_weight_tensor(
    shape: tuple[int,...], charge: np.ndarray) -> torch.Tensor:
    weight = chemistry_utils.compute_Nernst_slope(charge)
    weight = weight.reshape(shape)
    weight = build_tensor(weight, requires_grad=True)
    return weight
# }}}
