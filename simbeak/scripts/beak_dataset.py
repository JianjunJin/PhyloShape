#!/usr/bin/env python


import toytree
import pandas as pd
import numpy as np
from landmarks import get_beak_landmarks

full = pd.DataFrame()
tree = toytree.rtree.unittree(ntips=10, seed=123, treeheight=1)
tree.write("./beak_landmarks.nwk")

for dataset in range(50):

    # choose values ...
    data = tree.pcm.simulate_continuous_brownian(
        rates=[1., 20., 1., 1., 1],
        root_states=[1., 1e-6, 0., 0., 0.5],
        seed=dataset,
    )
    data.columns = ["length", "rotation", "curve_x", "curve_y", "beak_radius_start"]

    # transform data values
    # print(data.rotation)
    data.rotation = (data.rotation / abs(data.rotation)) * np.sqrt(abs(data.rotation))
    data.length = np.exp(data.length)
    data.beak_radius_start = np.exp(data.beak_radius_start)

    # store to tree
    # tree = tree.set_node_data_from_dataframe(data)

    for row in data.index:
        arr = get_beak_landmarks(
            num_intervals=50,
            num_disc_points=20,
            **{i: data.loc[row, i] for i in data.columns},
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

# ...
# c, a, m = tree.draw(width=800)
# for cidx, trait in enumerate(data.columns):
#     tree.annotate.add_tip_markers(
#         axes=a,
#         size=10,
#         color=(trait, "BlueRed"),
#         xshift=40 + 20 * cidx,
#     )
# toytree.utils.show(c)

print(data.shape)
print(data.head())
print(full.shape)
print(full.reset_index(drop=True))
full.to_csv("./beak_landmarks.csv")
