import numpy as np


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def cut_peaks(y, percentiles=None):
    if not percentiles:
        percentiles = (2, 98)

    per_start, per_end = np.percentile(y, percentiles)
    y[y > per_end] = per_end
    y[y < per_start] = per_start
    return y.astype('int16')


def smoothen_batch_borders(amps: np.array, batch_size, radius=10) -> np.array:
    amps = np.array(amps)
    for batch_i in range(amps.shape[0] // batch_size):
        border = batch_size * (batch_i + 1)
        frm = amps[border - radius - 1]
        t = amps[border + radius + 1]
        amps[border - radius:border + radius] = np.linspace(frm, t, radius * 2)
    return amps
