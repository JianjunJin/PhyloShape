#!/usr/bin/env python

"""Simulate an artifical beak dataset ...

"""

import itertools
import pandas as pd
import numpy as np
from landmarks import get_beak_landmarks

NAME = "beak_data4"
LABELS = ["length", "rotation", "curve_x", "curve_y", "beak_radius_start"]

# all parameters of interest
params = list(itertools.product(
    np.linspace(0.5, 7, 3),
    np.linspace(-2 * np.pi, 2 * np.pi, 5),
    np.linspace(-1, 1, 3),
    np.linspace(-1, 1, 3),
    np.linspace(0.5, 2.5, 3),
))

# mid point for each param
param_standard = [3.75, 0, 0, 0, 1.5]

# all ways of sampling 2 diff param settings
param_pairs = itertools.product([param_standard], params)

# store params
params = []
full = pd.DataFrame()

# iterate over sim datasets
for dataset, pair in enumerate(param_pairs):

    average = (np.array(pair[0]) + np.array(pair[1])) / 2
    data = pd.DataFrame(
        index=["A", "B", "M"],
        columns=LABELS,
        data=[pair[0], pair[1], average],
    )
    data["dataset"] = dataset
    params.append(data)

    for row in data.index:
        arr = get_beak_landmarks(
            num_intervals=40,
            num_disc_points=15,
            **{i: data.loc[row, i] for i in LABELS},
        )
        # data['x'] = arr[:, 0]
        # print(arr.shape)
        landmarks = arr.reshape((-1, 3))
        # print(landmarks.shape)

        df = pd.DataFrame({
            'dataset': dataset,
            'sample_idx': row,
            'landmark': range(landmarks.shape[0]),
            'x': landmarks[:, 0],
            'y': landmarks[:, 1],
            'z': landmarks[:, 2],
        })
        full = pd.concat([full, df])

# write simulation parameters
arr = pd.concat(params).reset_index(names="sample_idx")
arr = arr[["dataset", "sample_idx"] + LABELS]
arr.to_csv(f"./{NAME}_params.csv")
print(arr)

print(full.shape)
print(full.reset_index(drop=True))
full.to_csv(f"./{NAME}_landmarks.csv")
