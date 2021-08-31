import os

_files_dir = os.path.join(os.path.dirname(__file__), "files")

orig_file_path = os.path.join(_files_dir, "orig.wav")


def get_new_file_path(name: str) -> str:
    return os.path.join(_files_dir, f'{name}.wav')
