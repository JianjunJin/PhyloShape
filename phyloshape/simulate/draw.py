#!/usr/bin/env python

"""

TODO: find old code for randomly generating faces on these models?
"""

import k3d
import numpy as np


def draw_beak(landmarks: np.ndarray, **kwargs) -> k3d.Plot:
    """...

    """
    if not landmarks.dtype == np.float32:
        landmarks = landmarks.astype(np.float32)
    mid = int(landmarks.shape[1] / 2)
    qrt = int(landmarks.shape[1] / 4)
    plot = kwargs.get("plot", k3d.plot(**kwargs))
    # plot += k3d.points(landmarks, point_size=0.1, color=0x000000, opacity=0.35)
    plot += k3d.points(landmarks[:, :qrt], point_size=0.1, color=0x008b8b, opacity=0.8)
    plot += k3d.points(landmarks[:, -qrt:], point_size=0.1, color=0x008b8b, opacity=0.8)
    plot += k3d.points(landmarks[:, qrt:mid + qrt], point_size=0.1, color=0x8B008B, opacity=0.8)

    # plot += k3d.points(landmarks[:, mid:], point_size=0.15, color=0x008B8B)
    # plot += k3d.points(landmarks[:, 0], point_size=0.15, color=0x8B008B)
    # plot += k3d.points(landmarks[:, mid], point_size=0.15, color=0x008B8B)
    # plot += k3d.vectors(landmarks[:-1, 0], landmarks[1:, 0] - landmarks[:-1, 0], color=0x8B008B)
    # plot += k3d.vectors(landmarks[:-1, mid], landmarks[1:, mid] - landmarks[:-1, mid], color=0x008B8B)
    return plot


if __name__ == "__main__":

    import pandas as pd
    from landmarks import get_beak_landmarks

    # landmarks = get_beak_landmarks(num_intervals=20)
    landmarks = pd.read_csv("../../simbeak/datasets/beak_landmarks.csv", index_col=0)

    DSET = 16
    SIDX = 4

    landmarks = landmarks.loc[(landmarks.dataset == DSET) & (landmarks.sample_idx == SIDX), :]
    # print(landmarks)
    landmarks = landmarks.iloc[:, -3:]
    landmarks = landmarks.values.reshape((50, -1, 3))
    # landmarks = landmarks.values.reshape((49, -1, 3))
    plot = draw_beak(landmarks)

    with open('/tmp/test.html', 'w') as fp:
        fp.write(plot.get_snapshot())
    with open('/tmp/test.png', 'w') as fp:
        plot.display()
        plot.fetch_screenshot()
        fp.write(plot.screenshot)
