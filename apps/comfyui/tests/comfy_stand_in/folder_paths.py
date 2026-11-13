"""The stand-in for ComfyUI's folder_paths (0.45.3, folder_paths.py:70 and 143): the temp folder,
which a test sets (`temp_directory`) to a directory of its own."""

import tempfile

temp_directory = tempfile.gettempdir()


def get_temp_directory() -> str:
    return temp_directory
