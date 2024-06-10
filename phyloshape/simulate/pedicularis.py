#!/usr/bin/env python

"""Draw Pedicularis flowers.

This generates landmark CSV files with points forming a model that
looks similar to a number of Pedicularis flowers, at least according
to Deren. It uses the landmarks.py and draw.py modules to generate
landmarks and draw the model, respectively.
"""

import numpy as np
from draw import draw_beak
from landmarks import get_beak_landmarks


PEDICULARIS = {
    "groenlandica": dict(
        length=8,
        curve_radius_start=0.5,
        rotation=2.5 * np.pi,
        num_intervals=50,
    ),
    "anas": dict(
        length=3,
        rotation=np.pi * 3 / 2,
        curve_x=0,
        beak_radius_end=0.05,
        num_intervals=50,
    ),
    "fetisowii": dict(
        rotation=np.pi * 2.5,
        curve_radius_start=1,
        curve_radius_end=0.5,
        beak_radius_start=1,
        beak_radius_end=0.1,
        num_intervals=50,
    ),
    "cranolopha": dict(
        length=5,
        rotation=np.pi * 3 / 2,
        curve_radius_start=1.5,
        curve_radius_end=1,
        beak_radius_start=1.5,
        beak_radius_end=0.2,
        num_intervals=50,
    ),
    "densispica": dict(
        length=1,
        rotation=np.pi / 2,
        curve_x=0,
        curve_y=1,
        beak_radius_end=0.2,
    ),
    "integrifolia": dict(
        length=7,
        rotation=np.pi * 3.25,
        curve_x=0,
        curve_y=1,
        num_intervals=50,
        beak_radius_start=1.25,
    ),
    "davidii": dict(
        length=5,
        rotation=-np.pi * 3 / 2,
        curve_radius_start=1.5,
        curve_radius_end=1,
        beak_radius_start=1.5,
        beak_radius_end=0.2,
        num_intervals=50,
    )
}


if __name__ == "__main__":

    for spp, params in PEDICULARIS.items():
        landmarks = get_beak_landmarks(**params)
        plot = draw_beak(landmarks)
        with open(f"/tmp/{spp}.html", "w") as fp:
            fp.write(plot.get_snapshot())
