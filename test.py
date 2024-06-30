import numpy as np
from NESolver.utils import io_utils
from NESolver.utils import optimisation_utils
from NESolver.utils import pipeline_utils
from torch import optim
from torch import nn
import matplotlib.pyplot as plt


data_pack = io_utils.load_array_dictionary('./data/Na-K-Cl_simulated_clean.npz')
pipeline = pipeline_utils.TrainingPipeline(
    data_pack['charge'], data_pack['ion_size'], data_pack['concentration'], data_pack['response'], slice(10), slice(1000,1100), 0.001, optim.Adam, nn.MSELoss)
agent, training_loss, validation_loss = pipeline.train()

plt.plot(np.log10(training_loss))
plt.plot(np.log10(validation_loss))
plt.show()

pipeline = pipeline_utils.EvaluationPipeline(
    agent,
    data_pack['response_intercept'],
    data_pack['response_slope'],
    data_pack['selectivity_coefficient'],
    data_pack['concentration'][1100:1200,:],
    data_pack['response'][1100:1200,:],
)
pipeline.evaluate()
