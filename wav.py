import numpy as np
from scipy.io import wavfile

import file


def read_orig() -> np.array:
    speech = wavfile.read(file.orig_file_path)
    return np.array(speech[1], dtype=float)


def save_result(name: str, amps: np.array):
    wavfile.write(file.get_new_file_path(name), 44100, amps)
