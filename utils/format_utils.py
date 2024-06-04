"""A module that can handle string formatting in the project.

Author: Weixun Luo
Date: 25/04/2024
"""
import numpy as np


SPECIFIER_FLOAT = '.4f'
SPECIFIER_SCIENTIFIC = '.4e'

"""----- Float -----"""
# {{{ format_float_value
def format_float_value(value: float, unit: str = '') -> str:
    return f'{value:{SPECIFIER_FLOAT}}{unit}'
# }}}

# {{{ format_scientific_value
def format_scientific_value(value: float, unit: str = '') -> str:
    return f'{value:{SPECIFIER_SCIENTIFIC}}{unit}'
# }}}


"""----- Array -----"""
# {{{ format_float_array
def format_float_array(array: np.ndarray, unit: str = '') -> str:
    return '[' + ', '.join([f'{value:{SPECIFIER_FLOAT}}{unit}' for value in array.flatten()]) +']'
# }}}

# {{{ format_scientific_array
def format_scientific_array(array: np.ndarray, unit: str = '') -> str:
    return '[' + ', '.join([f'{value:{SPECIFIER_SCIENTIFIC}}{unit}' for value in array.flatten()]) +']'
# }}}
