#!/usr/bin/env python

"""The Vertices class stores vertex coordinates and colors as arrays.

The Vertices and Faces classes are stored in Shape instances.
"""

from typing import Union, List, Tuple

from loguru import logger
import numpy as np
from numpy.typing import ArrayLike
from phyloshape.utils import COORD_TYPE, RGB_TYPE

logger = logger.bind(name="phyloshape")


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

    def __getitem__(self, item: Union[ArrayLike, int, slice, list]):
        if self.colors:
            return self.coords[item], self.colors[item]
        else:
            return self.coords[item]

    def __bool__(self):
        return bool(self.coords)

    def __iter__(self):
        if bool(self.colors):
            for coord, color in zip(self.coords, self.colors):
                yield coord, color
        else:
            raise ValueError("No colors found! Please iter Vertices.coord directly!")

    def __len__(self):
        return len(self.coords)

    def __delitem__(self, key: Union[List, Tuple, int, slice]):
        assert isinstance(key, (int, tuple, slice, list))
        self.coords = np.delete(self.coords, key, axis=0)
        if self.colors:
            self.colors = np.delete(self.colors, key, axis=0)

    def __repr__(self):
        return f"Vertices({self.coords.shape})"
        # return self.coords, self.colors


if __name__ == "__main__":

    COORDS = np.array([[0, 1], [2, 3], [4, 5]])
    COLORS = None
    verts = Vertices(COORDS, COLORS)