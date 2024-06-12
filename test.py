"""Script to set up the required virtual environment for the project.

Author: Weixun Luo
Date: 12/06/2024
"""
import os
import platform
import urllib
import shutil
import subprocess
import sys


ENVIRONMENT_NAME = "NESolver"
ENVIRONMENT_FILE_PATH = "environment.yml"


"""----- Conda Installation -----"""
# {{{ setup_conda
def setup_conda() -> None:
    try:
        subprocess.run(['conda', '--version'], check=True)
    except subprocess.CalledProcessError:
        return False
# }}}

# {{{ install_conda
def install_conda() -> None:
# }}}
