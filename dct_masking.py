import numpy as np
from scipy.fftpack import dct, idct


def stub(amps: np.array) -> np.array:
    return __dct_batching(amps, lambda x: x, 64)


def mask(amps: np.array, batch_size=64, batch_offset=2) -> np.array:
    def on_batch(spec: np.array) -> np.array:
        if batch_offset > 0:
            spec[:batch_size - batch_offset] = spec[batch_offset:]
        return spec

    return __dct_batching(amps, on_batch, batch_size)


def unmask(amps: np.array, batch_size=64, batch_offset=2) -> np.array:
    def on_batch(spec: np.array) -> np.array:
        if batch_offset > 0:
            spec[batch_offset:] = spec[:batch_size - batch_offset]
            spec[:batch_offset] = 0
        return spec

    return __dct_batching(amps, on_batch, batch_size)


def __dct_batching(amps: np.array, on_batch, batch_size=64):
    new_amps = np.zeros(amps.shape)
    for batch_i in range(amps.shape[0] // batch_size):
        st, end = batch_i * batch_size, (batch_i + 1) * batch_size
        batch = amps[st:end]
        spec = dct(batch, norm='ortho')

        new_spec = on_batch(spec)

        new_amps[st:end] = idct(new_spec, norm='ortho')
    return new_amps
