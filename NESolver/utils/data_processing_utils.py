"""A utility module that handles data loading and processing for multivaraite ion
analysis in the project.

Author: Weixun Luo
Date: 13/04/2024
"""
import abc

import numpy as np
import torch
from torch.utils import data

from NESolver.utils import chemistry_utils
from NESolver.utils import matrix_utils


# {{{ AgentDataProcessor
class AgentDataProcessor(abc.ABC):
    """An interface that defines a data processor for Agent used in forward/
    backward solving.
    """

    # {{{ __init__
    @abc.abstractmethod
    def __init__(self) -> None:
        pass
    # }}}

    # {{{ pre_process_candidate_forward
    @abc.abstractmethod
    def pre_process_candidate_forward(
        self, candidate: np.ndarray) -> np.ndarray | torch.Tensor:
        pass
    # }}}

    # {{{ pre_process_candidate_backward
    @abc.abstractmethod
    def pre_process_candidate_backward(
        self, candidate: np.ndarray) -> np.ndarray | torch.Tensor:
        pass
    # }}}

    # {{{ pre_process_reference_forward
    @abc.abstractmethod
    def pre_process_reference_forward(
        self, reference: np.ndarray) -> np.ndarray | torch.Tensor:
        pass
    # }}}

    # {{{ pre_process_reference_backward
    @abc.abstractmethod
    def pre_process_reference_backward(
        self, reference: np.ndarray) -> np.ndarray | torch.Tensor:
        pass
    # }}}
# }}}

# {{{ TrainableAgentDataProcessor
class TrainableAgentDataProcessor(AgentDataProcessor):
    """An interface that defines a data processor for TrainableAgent used in
    forward/ backward solving.
    """

    # {{{ post_process_prediction_forward
    @abc.abstractmethod
    def post_process_prediction_forward(
        self, prediction: torch.Tensor) -> np.ndarray:
        pass
    # }}}

    # {{{ post_process_prediction_backward
    @abc.abstractmethod
    def post_process_prediction_backward(
        self, prediction: torch.Tensor) -> np.ndarray:
        pass
    # }}}
# }}}

# {{{ NeuralNetworkAgentDataProcessor
class NeuralNetworkAgentDataProcessor(TrainableAgentDataProcessor):
    """A class that can process data for NeuralNetworkAgent in forward/backward
    solving.
    """

    # {{{ __init__
    def __init__(self) -> None:
        pass
    # }}}

    # {{{ pre_process_candidate_forward
    def pre_process_candidate_forward(self) -> None:
        pass
    # }}}

    # {{{ pre_process_candidate_backward
    def pre_process_candidate_backward(
        self, candidate: np.ndarray) -> torch.Tensor:
        return matrix_utils.build_tensor(candidate)
    # }}}

    # {{{ pre_process_reference_forward
    def pre_process_reference_forward(self) -> None:
        pass
    # }}}

    # {{{ pre_process_reference_backward
    def pre_process_reference_backward(
        self, reference: np.ndarray) -> torch.Tensor:
        return matrix_utils.build_tensor(reference)
    # }}}

    # {{{ post_process_prediction_forward
    def post_process_prediction_forward(self) -> None:
        pass
    # }}}

    # {{{ post_process_prediction_backward
    def post_process_prediction_backward(
        self, prediction: torch.Tensor) -> np.ndarray:
        return prediction.numpy()
    # }}}
# }}}

# {{{ NumericalAgentDataProcessor
class NumericalAgentDataProcessor(TrainableAgentDataProcessor):
    """A processor that can process data for NumericalAgent in forward/backward
    solving.
    """

    # {{{ __init__
    def __init__(
        self,
        charge: np.ndarray,
        ion_size: np.ndarray,
    ) -> None:
        self._charge = self._construct_charge(charge)
        self._ion_size = self._construct_ion_size(ion_size)
        self._activity_power = self._construct_activity_power()
    # }}}

    # {{{ _construct_charge
    def _construct_charge(self, charge: np.ndarray) -> np.ndarray:
        return matrix_utils.build_array(charge)
    # }}}

    # {{{ _construct_ion_size
    def _construct_ion_size(self, ion_size: np.ndarray) -> np.ndarray:
        return matrix_utils.build_array(ion_size)
    # }}}

    # {{{ _construct_activity_power
    def _construct_activity_power(self) -> np.ndarray:
        return chemistry_utils.compute_activity_power(self._charge)
    # }}}

    # {{{ pre_process_candidate_forward
    def pre_process_candidate_forward(
        self, candidate: np.ndarray) -> torch.Tensor:
        candidate = chemistry_utils.convert_concentration_to_activity(
            candidate, self._charge, self._ion_size)
        for _ in range(2):
            candidate = np.expand_dims(candidate, 1)
        candidate = np.tile(candidate, (1,candidate.shape[3],1,1))
        candidate = np.power(candidate, self._activity_power)
        candidate = matrix_utils.build_tensor(candidate)
        return candidate
    # }}}

    # {{{ pre_process_candidate_backward
    def pre_process_candidate_backward(
        self,
        candidate: np.ndarray,
        slope: np.ndarray,
        drift: np.ndarray,
    ) -> np.ndarray:
        return np.power(10, (candidate-drift)/slope)
    # }}}

    # {{{ pre_process_reference_forward
    def pre_process_reference_forward(
        self, reference: np.ndarray) -> torch.Tensor:
        reference = matrix_utils.build_array(reference)
        for _ in range(2):
            reference = np.expand_dims(reference, 2)
        reference = matrix_utils.build_tensor(reference)
        return reference
    # }}}

    # {{{ pre_process_reference_backward
    def pre_process_reference_backward(self) -> None:
        pass
    # }}}

    # {{{ post_process_prediction_forward
    def post_process_prediction_forward(
        self, prediction: torch.Tensor) -> np.ndarray:
        return prediction.numpy().squeeze()
    # }}}

    # {{{ post_process_prediction_backward
    def post_process_prediction_backward(
        self, prediction: torch.Tensor) -> np.ndarray:
        pass
    # }}}
# }}}

# {{{ RegressionAgentDataProcessor
class RegressionAgentDataProcessor(AgentDataProcessor):
    """A class that can process data for RegressionAgent in forward/backward
    solving.
    """
    
    # {{{ __init__
    def __init__(self, slope: np.ndarray, drift: np.ndarray) -> None:
        self._slope = self._construct_slope(slope)
        self._drift = self._construct_drift(drift)
    # }}}

    # {{{ _construct_slope
    def _construct_slope(self, slope: np.ndarray) -> np.ndarray:
        return matrix_utils.build_array(slope)
    # }}}

    # {{{ _construct_drift
    def _construct_drift(self, drift: np.ndarray) -> np.ndarray:
        return matrix_utils.build_array(drift)
    # }}}

    # {{{ pre_process_candidate_forward
    def pre_process_candidate_forward(self, candidate: np.ndarray) -> np.ndarray:
        return matrix_utils.build_array(candidate)
    # }}}

    # {{{ pre_process_candidate_backward
    def pre_process_candidate_backward(
        self, candidate: np.ndarray) -> np.ndarray:
        return np.power(10, (candidate-self._drift)/self._slope)
    # }}}

    # {{{ pre_process_reference_forward
    def pre_process_reference_forward(self) -> None:
        pass
    # }}}

    # {{{ pre_process_reference_backward
    def pre_process_reference_backward(self) -> None:
        pass
    # }}}
# }}}


"""----- Dataset -----"""
# {{{ Dataset
class Dataset(abc.ABC, data.Dataset):
    """A class that can load and process data for multivariate ion analysis.
    """
    
    # {{{ __init__
    @abc.abstractmethod
    def __init__(self, candidate: np.ndarray, reference: np.ndarray) -> None:
        pass
    # }}}

    # {{{ __len__
    @abc.abstractmethod
    def __len__(self) -> int:
        pass
    # }}}

    # {{{ __getitem__
    @abc.abstractmethod
    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        pass
    # }}}

    # {{{ @property: candidate
    @property
    @abc.abstractmethod
    def candidate(self) -> np.ndarray:
        pass
    # }}}

    # {{{ @property: reference
    @property
    @abc.abstractmethod
    def reference(self) -> np.ndarray:
        pass
    # }}}
# }}}

# {{{ NumericalAgentForwardDataset
class NumericalAgentForwardDataset(Dataset):
    """A class that csan load and process data for multivariate ion analysis with
    numerical agents.
    """

    # {{{ __init__
    def __init__(
        self,
        candidate: np.ndarray, reference: np.ndarray) -> None:
    # }}}
# }}}
