#!/usr/bin/env python

"""Core class types in phyloshape

Classes
-------
Vertex:
    stores coordinates and color data
Face:
    stores collection of Vertex objects composing a face.
Vector:
    stores an edge connecting 2 Vertex objects w/ a reference face
"""

from __future__ import annotations
from typing import Tuple, List, Sequence, Optional

import numpy as np
from phyloshape.vectors.rotate import get_unit_vector_from_face
from phyloshape.vectors.transform import (
    # transform_vector_to_absolute,
    transform_vector_to_relative,
)


class _Unique:
    def __init__(self, id: int):
        self.id = id


class Vertex(_Unique):
    def __init__(self, id: int, coords: Sequence[float], color: Sequence[int] = None):
        super().__init__(id)

        self._coords: np.ndarray = np.array(coords)
        """: The (x, y, z) coordinates in absolute coord space"""
        self._color: np.ndarray = None
        """: The (R, G, B) value of vertex color"""

    def __repr__(self) -> str:
        return f"Vertex(id={self.id}, coords={tuple(self._coords.round(3))})"

    @property
    def coords(self) -> np.ndarray:
        return self._coords

    @property
    def color(self) -> np.ndarray:
        if self._color is None:
            self._color = np.array([0, 0, 0])
        return self._color

    # def get_vector_from(self, vertex: Vertex, face: Optional[Face] = None) -> Vector:
    #     return Vector(self, vertex, face)

    # def get_relative_vector_from(self, vertex: Vertex, face: Face) -> Vector:
    #     vec = self.get_vector_from(vertex, face)
    #     return transform_vector_to_relative(vec, face)


class Face(tuple):
    def __init__(self, iterable=(), /):
        assert all(isinstance(i, Vertex) for i in iterable)
        assert len(iterable) == 3

    def __repr__(self):
        return f"Face({tuple(i.id for i in self)})"

    def coordinates(self) -> List[Tuple[float, float, float]]:
        return tuple(i.coords for i in self)

    @property
    def edges(self) -> Tuple[Tuple[int, int]]:
        return ((self[0], self[1]), (self[1], self[2]), (self[0], self[2]))

# class Texture:
#     def __init__(self, faces: Sequence[Face], image: np.ndarray):
#         self.face = face


class Vector(_Unique):
    """Vector between two vertices.

    In the simplest form Vectors represent edges on a 3D model with a
    start and end Vertex, and the distance between them. In more complex
    form, it also records the unit vector between them, and a reference
    face that allows for converting the vector between absolute and
    relative coordinate space.
    """
    def __init__(self, start: Vertex, end: Vertex, face: Face = None):
        self._start = start
        """: Vertex object at beginning of this vector"""
        self._end = end
        """: Vertex object at the end of this vector."""
        self._face = face
        """: The reference face for this vector, represented by 3 vertex IDs"""
        self._unit: np.ndarray = None
        """: Unit vector in a normed vector space (sphere) of length 1"""
        self._absolute: np.ndarray = None
        """: vector between vertices in absolute coordinate space"""
        self._relative: np.ndarray = None
        """: vector between vertices in relative coordinate space given face"""
        self._dist: float = None
        """: Euclidean distance between start and end vertices"""
        super().__init__(id=start.id)

    def __repr__(self):
        return f"Vector({self.start.id}, {self.end.id})"

    @property
    def start(self) -> Vertex:
        return self._start

    @property
    def end(self) -> Vertex:
        return self._end

    @property
    def face(self) -> Face:
        return self._face

    @property
    def unit(self) -> np.ndarray:
        """Return unit vector from face in normed space (sphere) of length 1"""
        if self._unit is None:
            if self._face is None:
                raise ValueError(
                    "Vector must have a .face to compute a unit vector")
            self._unit = get_unit_vector_from_face(*self.face.coordinates())
        return self._unit

    @property
    def dist(self) -> float:
        """Return Euclidean distance between start and end vertices"""
        if self._dist is None:
            diff = (self.end.coords - self.start.coords) ** 2
            self._dist = np.sqrt(np.sum(diff))
        return self._dist

    @property
    def absolute(self) -> np.ndarray:
        """Return absolute coordinates of vector."""
        if self._absolute is None:
            self._absolute = self.end.coords - self.start.coords
        return self._absolute

    @property
    def relative(self) -> np.ndarray:
        """Return relative coordinates of vector given a reference face"""
        if self._relative is None:
            self._relative = transform_vector_to_relative(
                vector=self.absolute, face=[i.coords for i in self.face]
            )
        return self._relative


if __name__ == "__main__":

    v0 = Vertex(id=0, coords=(0, 0, 1))
    v1 = Vertex(id=1, coords=(0, 1, 0))
    v2 = Vertex(id=2, coords=(1, 0, 0))
    v3 = Vertex(id=3, coords=(0, 1, 1))

    f0 = Face((v0, v1, v2))
    f1 = Face((v2, v3, v0))
    print(f0)

    V = Vector(v0, v1, f0)
    print(V)
    print(V.dist)

    print(V.absolute)
    print(V.relative)
    print(v0)
    print(Vertex(id=5, coords=(100.2341, 22.0, 0)))