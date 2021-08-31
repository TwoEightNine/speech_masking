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
