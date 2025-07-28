#!/usr/bin/env python

import numpy as np
from numpy.typing import ArrayLike, NDArray


def rgb_to_hex(rgb_array: NDArray[np.uint8]) -> NDArray[np.uint32]:
    """
    Convert (2^8, 2^8, 2^8)-based rgb array to hex array.
    
    Parameters
    ----------
    rgb_array : NDArray[np.uint8]
        Array of RGB values, with shape (..., 3)
        
    Returns
    -------
    NDArray[np.uint32]
        Array of hexadecimal color values
    """
    if rgb_array.shape[-1] != 3:
        raise ValueError("Last dimension of input array must be 3; "
                         "shape {} was found.".format(rgb_array.shape))
    rgb_array = np.asarray(rgb_array, dtype="uint32")
    return (rgb_array[:, 0] << 16) + (rgb_array[:, 1] << 8) + rgb_array[:, 2]


# converted from: https://github.com/matplotlib/matplotlib/blob/v3.5.2/lib/matplotlib/colors.py
def rgb_to_hsv(rgb_array: NDArray[np.uint8]) -> NDArray[np.float32]:
    """
    Convert (2^8, 2^8, 2^8)-based rgb array to hsv array.

    Parameters
    ----------
    rgb_array : NDArray[np.uint8]
       Array of RGB values, with shape (..., 3)
       Values should be in the range [0, 255]
    
    Returns
    -------
    NDArray[np.float32]
       Colors converted to HSV values in range [0, 1]
    """
    rgb_array = np.array(rgb_array, dtype=np.float32) / 255.0

    # check length of the last dimension, should be _some_ sort of rgb
    if rgb_array.shape[-1] != 3:
        raise ValueError("Last dimension of input array must be 3; "
                         "shape {} was found.".format(rgb_array.shape))

    in_shape = rgb_array.shape
    rgb_array = np.array(
        rgb_array, copy=False,
        dtype=np.promote_types(rgb_array.dtype, np.float32),  # Don't work on ints.
        ndmin=2,  # In case input was 1D.
    )
    out = np.zeros_like(rgb_array)
    arr_max = rgb_array.max(-1)
    ipos = arr_max > 0
    delta = rgb_array.ptp(-1)
    s = np.zeros_like(delta)
    s[ipos] = delta[ipos] / arr_max[ipos]
    ipos = delta > 0
    # red is max
    idx = (rgb_array[..., 0] == arr_max) & ipos
    out[idx, 0] = (rgb_array[idx, 1] - rgb_array[idx, 2]) / delta[idx]
    # green is max
    idx = (rgb_array[..., 1] == arr_max) & ipos
    out[idx, 0] = 2. + (rgb_array[idx, 2] - rgb_array[idx, 0]) / delta[idx]
    # blue is max
    idx = (rgb_array[..., 2] == arr_max) & ipos
    out[idx, 0] = 4. + (rgb_array[idx, 0] - rgb_array[idx, 1]) / delta[idx]

    out[..., 0] = (out[..., 0] / 6.0) % 1.0
    out[..., 1] = s
    out[..., 2] = arr_max

    return out.reshape(in_shape)


# TODO: color to wave length distribution?


