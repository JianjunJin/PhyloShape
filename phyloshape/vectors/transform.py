#!/usr/bin/env python

"""Transformations on vectors between relative and absolute coordinates

TODO: Make functions broadcasting.

"A novel framework for ancestral reconstruction of 3D morphologies
that addresses limitations of GPA for evolutionary rate analyses"


old vector_manipulator.py
"""

from typing import Sequence
import numpy as np

# from phyloshape.shape.src.core import Vertex, Vector, Face
from phyloshape.vectors.rotate import (
    rotate_vector_on_single_axis,
    get_rotation_angles_from_face,
)


__all__ = [
    "transform_vector_to_relative",
    "transform_vector_to_absolute",
]


# trans_vector_to_relative(...)
def transform_vector_to_relative(
    vector: np.ndarray,
    face: Sequence[np.ndarray],
) -> np.ndarray:
    """Return vector transformed to a relative coordinate system.

    This function transforms a relative_vector from its original
    coordination to a new coordination, where the input reference face
    (t1, t2, t3) is on the xy plane (its perpendicular vector on the
    positive z axis), and where the vector (t1, t2) is on the positive
    x axis.

    Parameters
    ----------
    vector: ArrayLike[np.float32]
        Array of triangle vertices: float (x, y, z) coordinate triplets.
    face: List[ArrayLike[np.float32]]
        List of three coordinate triplets (t1, t2, t3), each is an
        array of np.float32 (x, y, z)

    Returns
    -------
    ArrayLike[np.float32]
        Array of triangle vertices: float (x, y, z) coordinate triplets.
    """
    # 1. calculate rotations to transform vector perpendicular to face
    angles = get_rotation_angles_from_face(*face)

    # 2. apply the Euler rotations to the input vertices
    relative_vector = rotate_vector_on_single_axis(
        vector,
        cos_theta=angles["cos_x_theta"],
        sin_theta=angles["sin_x_theta"],
        axis=0,
    )
    relative_vector = rotate_vector_on_single_axis(
        relative_vector,
        cos_theta=angles["cos_y_theta"],
        sin_theta=angles["sin_y_theta"],
        axis=1,
    )
    relative_vector = rotate_vector_on_single_axis(
        relative_vector,
        cos_theta=angles["cos_z_theta"],
        sin_theta=angles["sin_z_theta"],
        axis=2,
    )
    return relative_vector


def transform_vector_to_absolute(
    vector: np.ndarray,
    face_vertices: Sequence[np.ndarray]
) -> np.ndarray:
    """Return vector transformed to an absolute coordinate system.

    This function is the reverse of `transform_vector_to_relative`.

    Parameters
    ----------
    vector: ArrayLike[np.float32]
        Array of triangle vertices: float (x, y, z) coordinate triplets.
    face_vertices: List[ArrayLike[np.float32]]
        List of three coordinate triplets (t1, t2, t3), each is an
        array of np.float32 (x, y, z)

    Returns
    -------
    ArrayLike[np.float32]
        Array of triangle vertices: float (x, y, z) coordinate triplets.
    """
    # 1. calculate rotations to transform vector perpendicular to face
    angles = get_rotation_angles_from_face(*face_vertices)

    # 2. apply the rotations to the input vertices
    absolute_vector = rotate_vector_on_single_axis(
        vector,
        cos_theta=angles["cos_z_theta"],
        sin_theta=-angles["sin_z_theta"],
        axis=2,
    )
    absolute_vector = rotate_vector_on_single_axis(
        absolute_vector,
        cos_theta=angles["cos_y_theta"],
        sin_theta=-angles["sin_y_theta"],
        axis=1,
    )
    absolute_vector = rotate_vector_on_single_axis(
        absolute_vector,
        cos_theta=angles["cos_x_theta"],
        sin_theta=-angles["sin_x_theta"],
        axis=0,
    )
    return absolute_vector


if __name__ == "__main__":

    from phyloshape.shape.src.core import Vertex, Face, Vector
    vector = np.array([0.5, 0.9, 0.3], dtype=np.float32)
    vertex0 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    vertex1 = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    vertex2 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    face = [
        np.array([0.1, 0.2, 0.3], dtype=np.float32),
        np.array([0.5, 0.5, 0.5], dtype=np.float32),
        np.array([1.0, 1.0, 1.0], dtype=np.float32),
    ]

    v0 = Vertex(0, vertex0)
    v1 = Vertex(1, vertex1)
    v2 = Vertex(2, vertex2)
    f0 = Face((v0, v1, v2))
    V = Vector(v0, v1, f0)

    rV = transform_vector_to_relative(V.absolute, [i.coords for i in f0])

    # avector = transform_vector_to_absolute(rvector, face)
    print(V)
    print(rV)
    # print(avector)
