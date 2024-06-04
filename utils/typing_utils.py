"""A module that contains customised types in the project.

Author: Weixun Luo
Date: 10/04/2024
"""
import typing

import numpy as np
import torch
from torch import nn
from torch import optim


class DataPack(typing.TypedDict):
    """A data pack that contains everything about the experiment."""
    charge: np.ndarray
    ion_size: np.ndarray
    selectivity_coefficient: np.ndarray
    slope: np.ndarray
    drift: np.ndarray
    concentration: np.ndarray
    activity: np.ndarray
    potential: np.ndarray

class Hyperparameter(typing.TypedDict):
    lr: float
    optimizer: optim.Optimizer
    criterion: nn.Module
