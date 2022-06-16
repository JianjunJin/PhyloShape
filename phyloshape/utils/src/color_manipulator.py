#!/usr/bin/env python

import numpy as np
from numpy.typing import ArrayLike


def rgb_to_hex(rgb_array: ArrayLike):
    """convert (2^8, 2^8, 2^8)-based rgb array to hex array"""
    rgb_array = np.asarray(rgb_array, dtype="uint32")
    return (rgb_array[:, 0] << 16) + (rgb_array[:, 1] << 8) + rgb_array[:, 2]

