#!/usr/bin/env python

import numpy as np
from numpy.typing import ArrayLike


def rgb_to_hex(rgb_array: ArrayLike):
    """convert (2^8, 2^8, 2^8)-based rgb array to hex array"""
    rgb_array = np.asarray(rgb_array, dtype="uint32")
    return (rgb_array[:, 0] << 16) + (rgb_array[:, 1] << 8) + rgb_array[:, 2]


def rgb_to_hsv(rgb):
    """Convert RGB to HSV color space"""
    # TODO improve
    # 1. -> array
    # 2. -> simpler function

    # read R,G,B values
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0

    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn

    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    elif mx == b:
        h = (60 * ((r - g) / df) + 240) % 360

    if mx == 0:
        s = 0
    else:
        s = (df / mx) * 100

    v = mx * 100

    hsv = [int(h), int(s), int(v)]

    return hsv
