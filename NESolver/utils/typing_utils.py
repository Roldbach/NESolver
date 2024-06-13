"""A utility module that contains customised types used in the project.

Author: Weixun Luo
Date: 10/04/2024
"""
import typing

import numpy as np
import torch
from torch import nn
from torch import optim


class DataPack(typing.TypedDict):
    """A dict that contains all required data for multivariate ion analysis."""
    charge: np.ndarray
    ion_size: np.ndarray
    selectivity_coefficient: np.ndarray
    slope: np.ndarray
    drift: np.ndarray
    concentration: np.ndarray
    activity: np.ndarray
    potential: np.ndarray

class Hyperparameter(typing.TypedDict):
    """A dict that contains all required hyperparameters for training NESolver."""
    learning_rate: float
    optimiser_class: optim.Optimizer
    criterion_class: nn.Module
