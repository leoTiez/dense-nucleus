#!/usr/bin/env python3
import os
from pathlib import Path


def validate_dir(rel_path=''):
    """
    Validates path and creates parent/child folders if path is not existant
    :param rel_path: Relative path from the current directory to the target directory
    :return: Path as a string
    """
    curr_dir = os.getcwd()
    Path('%s/%s/' % (curr_dir, rel_path)).mkdir(parents=True, exist_ok=True)
    return '%s/%s/' % (curr_dir, rel_path)

