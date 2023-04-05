#!/usr/bin/env python

"""Core PhyloShape class object of the phyloshape package.

"""

from loguru import logger
import numpy as np
from phyloshape.utils import COORD_TYPE, RGB_TYPE
from numpy.typing import ArrayLike
from typing import Union, List
from loguru import logger
logger = logger.bind(name="phyloshape")


class Vertices:
    def __init__(self,
                 # TODO how to specify the dimension of the array/list
                 coords: Union[ArrayLike, List, None] = None,
                 colors: Union[ArrayLike, List, None] = None):
        self.coords = np.array([], dtype=COORD_TYPE) if coords is None else np.array(coords, dtype=COORD_TYPE)
        self.colors = np.array([], dtype=RGB_TYPE) if colors is None else np.array(colors, dtype=RGB_TYPE)
        if len(self.colors):
            assert len(self.coords) == len(self.colors)

    def __getitem__(self, item):
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
            raise ValueError("No colors found! Please iter the coord directly!")

    def __len__(self):
        return len(self.coords)

    # def __repr__(self):
    #     return self.coords, self.colors




