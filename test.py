"""Script to set up the required virtual environment for the project.

Author: Weixun Luo
Date: 12/06/2024
"""
import logging
import os
import platform
import urllib
import shutil
import subprocess
import sys


ENVIRONMENT_NAME = "NESolver"
ENVIRONMENT_FILE_PATH = "environment.yml"


"----- Conda Environment Setup -----"
# {{{ setup_conda_environment
def setup_conda_environment(
    environment_name: str = ENVIRONMENT_NAME,
    environment_file_path: str = ENVIRONMENT_FILE_PATH,
) -> None:
# }}}
