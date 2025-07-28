#!/usr/bin/env python

"""Core PhyloShape class object of the phyloshape package.

"""

from loguru import logger
import numpy as np
from phyloshape.utils import COORD_TYPE, RGB_TYPE
from numpy.typing import ArrayLike, NDArray
from typing import Union, List, Tuple
from loguru import logger
from copy import deepcopy
logger = logger.bind(name="phyloshape")


class Vertices:
    def __init__(self,
                 # Dimension of the array/list is specified in the type hints
                 coords: Union[NDArray[np.float32], List, None] = None,
                 colors: Union[NDArray[np.uint8], List, None] = None):
        self.coords = np.array([], dtype=COORD_TYPE) if coords is None else np.array(coords, dtype=COORD_TYPE)
        self.colors = np.array([], dtype=RGB_TYPE) if colors is None else np.array(colors, dtype=RGB_TYPE)
        if len(self.colors):
            assert len(self.coords) == len(self.colors)

    def __deepcopy__(self, memodict={}):
        return Vertices(deepcopy(self.coords), deepcopy(self.colors))

    def __getitem__(self, item: Union[NDArray[np.uint32], int, slice, list]):
        if isinstance(self.colors, np.ndarray):
            if self.colors.any():
                return self.coords[item], self.colors[item]
            else:
                return self.coords[item]
        else:
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

    # def __repr__(self):
    #     return self.coords, self.colors




