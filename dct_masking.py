import numpy as np
import masking_rules
from scipy.fftpack import dct, idct


def mask(amps: np.array, batch_size=64, masking_rule: masking_rules.MaskingRule = None) -> np.array:
    on_batch = lambda spec: masking_rule.mask(spec)
    return __dct_batching(amps, on_batch, batch_size)


def unmask(amps: np.array, batch_size=64, masking_rule: masking_rules.MaskingRule = None) -> np.array:
    on_batch = lambda spec: masking_rule.unmask(spec)
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
