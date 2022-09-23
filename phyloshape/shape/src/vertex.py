#!/usr/bin/env python

"""Core PhyloShape class object of the phyloshape package.

"""

from typing import Union, List

# from loguru import logger
import numpy as np
from numpy.typing import ArrayLike
from phyloshape.utils import COORD_TYPE, RGB_TYPE


class Vertices:
    """Vertex object for storing coordinates and color data.

    A Vertices object contains array data representing the coordinates
    of all vertices composing a 3D model object.

    Parameters
    ----------
    coords: ...
        Coordinates of shape=...
    colors: ...
        An array or list of colors of shape...
    """

    # TODO how to specify the dimension of the array/list
    def __init__(
        self,
        coords: Union[ArrayLike, List, None] = None,
        colors: Union[ArrayLike, List, None] = None,
    ):
        self.coords: np.ndarray = np.array(
            [] if coords is None else coords, dtype=COORD_TYPE
        )
        """: ..."""
        self.colors: np.ndarray = np.array(
            [] if colors is None else colors, dtype=RGB_TYPE
        )
        """: ..."""
        assert len(self.coords) == len(self.colors)

    def __getitem__(self, item):
        return self.coords[item], self.colors[item]

    def __bool__(self):
        return bool(self.coords)

    def __iter__(self):
        # if self.coords and self.colors:
        for coord, color in zip(self.coords and self.colors):
            yield coord, color
        # else:
        #     for coord in self.coords:
        #         yield coord

    def __len__(self):
        return len(self.coords)
