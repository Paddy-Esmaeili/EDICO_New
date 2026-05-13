"""
utils.py
Shared utility functions used across the Edico pipeline.
"""

import os
import shutil


def prepare_directory(path: str):
    """
    Create a directory at `path`, or clear it if it already exists.

    WARNING: This permanently deletes all contents of the directory.
    The user is warned about this in the GUI before running.
    """
    if os.path.exists(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Warning: could not delete {file_path}: {e}")
    else:
        os.makedirs(path)