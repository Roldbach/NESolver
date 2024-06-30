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

# {{{ OptimisationAgentForwardDataset
class OptimisationAgentForwardDataset(Dataset):
    """A class that can load and process data for multivariate ion analysis based
    on optimisation.
    """

    # {{{ __init__
    def __init__(
        self,
        charge: np.ndarray,
        ion_size: np.ndarray,
        concentration: np.ndarray,
        response: np.ndarray,
    ) -> None:
        self._charge = self._construct_charge(charge)
        self._ion_size = self._construct_ion_size(ion_size)
        self._activity_power = self._construct_activity_power()
        self._candidate = self._construct_candidate(concentration)
        self._reference = self._construct_reference(response)
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
        return chemistry_utils.compute_Nikolsky_Eisenman_activity_power(
            self._charge)
    # }}}

    # {{{ _construct_candidate
    def _construct_candidate(self, concentration: np.ndarray) -> torch.Tensor:
        candidate = chemistry_utils.convert_concentration_to_activity(
            concentration, self._charge, self._ion_size)
        for _ in range(2):
            candidate = np.expand_dims(candidate, 1)
        candidate = np.tile(candidate, (1,candidate.shape[3],1,1))
        candidate = np.power(candidate, self._activity_power)
        candidate = matrix_utils.build_tensor(candidate)
        return candidate
    # }}}

    # {{{ _construct_reference
    def _construct_reference(self, response: np.ndarray) -> torch.Tensor:
        reference = matrix_utils.build_array(response)
        for _ in range(2):
            reference = np.expand_dims(reference, 2)
        reference = matrix_utils.build_tensor(reference)
        return reference
    # }}}

    # {{{ __len__
    def __len__(self) -> int:
        return self._candidate.shape[0]
    # }}}

    # {{{ __getitem__
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._candidate[index,...], self._reference[index, ...]
    # }}}

    # {{{ @property: candidate
    @property
    def candidate(self) -> np.ndarray:
        return self._candidate.clone().numpy()
    # }}}

    # {{{ @property: reference
    @property
    def reference(self) -> np.ndarray:
        return self._reference.clone().numpy()
    # }}}
# }}}


"""----- Data Loader -----"""
# {{{ OptimisationAgentForwardDataLoader
class OptimisationAgentForwardDataLoader(data.DataLoader):
    """A class that can load and process data for the training of
    OptimisationAgnet in multivariate ion analysis.
    """

    # {{{ __init__
    def __init__(
        self,
        charge: np.ndarray,
        ion_size: np.ndarray,
        concentration: np.ndarray,
        response: np.ndarray,
    ) -> None:
        super().__init__(
            dataset = OptimisationAgentForwardDataset(
                charge, ion_size, concentration, response),
            batch_size = concentration.shape[0],
            shuffle = True,
            num_workers = 0,
        )
    # }}}
# }}}
