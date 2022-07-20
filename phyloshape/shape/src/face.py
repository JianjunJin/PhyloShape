#!/usr/bin/env python

"""Core PhyloShape class object of the phyloshape package.

"""

from loguru import logger
import numpy as np
from phyloshape.utils import ID_TYPE, COORD_TYPE
from phyloshape.shape.src.vertex import Vertices
from numpy.typing import ArrayLike
from typing import Union, List, Generator


class Faces:
    def __init__(self,
                 # TODO how to specify the dimension of the array/list
                 vertex_ids: Union[ArrayLike, List, None] = None,
                 vertices: Vertices = None,
                 texture_ids: Union[ArrayLike, List, None] = None,
                 texture_anchor_coords: Union[ArrayLike, List, None] = None,
                 texture_image_obj=None):
        self.vertex_ids = np.array([], dtype=ID_TYPE) if vertex_ids is None else np.array(vertex_ids, dtype=ID_TYPE)
        self.__vertices = Vertices() if vertices is None else vertices
        self.texture_ids = np.array([], dtype=ID_TYPE) if texture_ids is None else np.array(texture_ids, dtype=ID_TYPE)
        self.__texture_anchor_coords = np.array([], dtype=COORD_TYPE) if texture_anchor_coords is None \
            else np.array(texture_anchor_coords, dtype=COORD_TYPE)
        self.texture_image_obj = texture_image_obj

    def __getitem__(self, item):
        return self.vertex_ids[item]

    # def __iter__(self):
    #     for vertex_id in self.vertex_ids:
    #         yield vertex_id

    def iter_coords(self, coord_type: str = "vertex"):
        """
        Returns a generator that each time generates the tri-coordinates (coord_type==vertex) or
        bi-coordinates (coord_type==texture) of the three points of a face.

        :param coord_type: vertex (default) or texture
        :return: Generator[ArrayLike[ArrayLike, ArrayLike, ArrayLike]]
        """
        if coord_type == "vertex":
            for vertex_id in self.vertex_ids:
                return self.__vertices[vertex_id]
        elif coord_type == "texture":
            for texture_id in self.texture_ids:
                return self.__texture_anchor_coords[texture_id]
        else:
            raise ValueError(coord_type + " is not a valid input! coord_type must be either vertex or texture!")

