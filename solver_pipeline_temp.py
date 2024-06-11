#! /usr/bin/env python
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_decomposition
from sklearn import linear_model
from sklearn import model_selection
from sklearn import svm
from skorch import net
import torch
from torch import nn
from torch import optim
import torchvision.models as models
from torchmetrics import regression

from utils import chemistry_utils
from utils import data_processing_utils
from utils import data_simulation_utils
from utils import evaluation_utils
from utils import io_utils
from utils import matrix_utils
from utils import optimisation_utils
from utils import pipeline_utils


data_pack = io_utils.load_array_dictionary('./data/Na-K-Cl_simulated_clean.npz')
pipeline = pipeline_utils.NeuralNetworkAgentTrainingPipeline(
    data_pack['charge'],
    data_pack['ion_size'],
    data_pack['concentration'],
    data_pack['potential'],
    slice(10),
    slice(1000,1100),
    1e-4,
    optim.AdamW,
    nn.MSELoss,
)
network, training_loss, validation_loss = pipeline.train()
plt.plot([i for i in range(len(training_loss))], np.log10(training_loss))
plt.plot([i for i in range(len(validation_loss))], np.log10(validation_loss))
plt.show()

pipeline = pipeline_utils.NeuralNetworkAgentEvaluationPipeline(
    network,
    data_pack['selectivity_coefficient'],
    data_pack['slope'],
    data_pack['drift'],
    data_pack['concentration'][1000:1100,:],
    data_pack['activity'][1000:1100,:],
    data_pack['potential'][1000:1100,:],
)
pipeline.evaluate()
