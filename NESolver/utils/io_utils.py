"""An utility module that handles file loading and saving in the project.

Author: Weixun Luo
Date: 01/04/2023
"""
import numpy as np
import scipy
import torch
from torch import nn


""" ----- File Loading -----"""
# {{{ load_array_dictionary
def load_array_dictionary(file_path: str) -> dict:
    return dict(np.load(file_path))
# }}}

# {{{ load_state_dictionary
def load_state_dictionary(file_path: str, module: nn.Module) -> nn.Module:
    module.load_state_dict(torch.load(file_path))
    return module
# }}}


""" ----- File Saving -----"""
# {{{ save_array_dictionary
def save_arrray_dictionary(file_content: dict, file_path: str) -> None:
    np.savez(file_path, **file_content)
# }}}

# {{{ save_state_dictionary
def save_state_dictionary(module: nn.Module, file_path: str) -> None:
    torch.save(module.state_dict(), file_path)
# }}}
