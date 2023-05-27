#!/usr/bin/env python

"""Rotation functions for vectors.

TODO: why use float32 here?
"""

from typing import Mapping, Tuple
import numpy as np

# dicts used to get reorder indices in rotate_vector_on_single_axis
ORDER1 = {0: [0, 1, 2], 1: [1, 0, 2], 2: [2, 0, 1]}
ORDER2 = {0: [0, 1, 2], 1: [1, 0, 2], 2: [1, 2, 0]}


# gen_unit_perpendicular_v(...)
def get_unit_vector_from_face(
    vertex0: np.ndarray,
    vertex1: np.ndarray,
    vertex2: np.ndarray,
) -> np.ndarray:
    """Return the unit vector of a plane defined by three points.

    A unit vector in a normed vector space is a vector (often a spatial
    vector) of length 1. Is is also called a 'direction vector',
    commonly denoted as d, to describe a unit vector being used to
    represent spatial direction and relative direction. In 3D spatial
    directions are numerically equivalent to points on the unit
    sphere (https://en.wikipedia.org/wiki/Unit_vector)

    Parameters
    ----------
    three_points: List[ArrayLike]
        Three elements of coordinate triplets, each is an array of
        np.float32 (x, y, z)

    Returns
    -------
    ArrayLike[np.float32, np.float32, np.float32]
        Array of np.float32 (x, y, z) coordinate triplet
    """
    a = vertex0 - vertex1
    b = vertex0 - vertex2
    perpendicular_v = np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])
    norm = np.linalg.norm(perpendicular_v)
    return perpendicular_v if norm == 0 else perpendicular_v / norm


# __find_rotation_angles(...)
def get_rotation_angles_from_face(
    vertex0: np.ndarray,
    vertex1: np.ndarray,
    vertex2: np.ndarray,
) -> Mapping[str, float]:
    """Return dict with cos, sin thetas for x, y and z for rotations.

    Example
    -------
    >>> face_points = np.array([
    >>>     [0, 0, 1],
    >>>     [0, 1, 0],
    >>>     [1, 0, 0],
    >>> ])
    >>> get_rotation_angles(face_points)
    >>> # ...
    """
    # get the direction vector given a face
    dvec = get_unit_vector_from_face(vertex0, vertex1, vertex2)

    # calculate rotations to transform dvec into (0, 0, 1)
    norm_yz = np.linalg.norm(dvec[1:])

    # dict of results for rotations on x and y axes
    xy_mapping = dict(
        cos_x_theta=dvec[2] / norm_yz if norm_yz else 1.,
        sin_x_theta=dvec[1] / norm_yz if norm_yz else 0.,
        cos_y_theta=norm_yz,
        sin_y_theta=dvec[0],
    )

    # calculate z-axis rotation for the first edge vector of face
    # of ref_face to the positive x-axis
    vector = vertex1 - vertex0  # [1] - vertex[0]
    cos_z_theta, sin_z_theta = get_z_axis_angle(vector, **xy_mapping)

    # return dict with all 6 angle values
    z_mapping = dict(cos_z_theta=cos_z_theta, sin_z_theta=sin_z_theta)
    return xy_mapping | z_mapping


# __find_z_axis_angle(...)
def get_z_axis_angle(
    first_edge_vector: np.ndarray,
    cos_x_theta: float,
    sin_x_theta: float,
    cos_y_theta: float,
    sin_y_theta: float,
) -> Tuple[float, float]:
    """Return Z-axis angle given a vector and its x, y angles

    """
    # 1.2.1 apply the rotations to the first edge vector
    first_edge_vector = rotate_vector_on_single_axis(
        first_edge_vector,
        cos_theta=cos_x_theta,
        sin_theta=sin_x_theta,
        axis=0,
    )
    first_edge_vector = rotate_vector_on_single_axis(
        first_edge_vector,
        cos_theta=cos_y_theta,
        sin_theta=sin_y_theta,
        axis=1,
    )

    # 1.2.2 the angle to rotate along the z axis
    norm_xyz = np.linalg.norm(first_edge_vector)
    sin_z_theta = -first_edge_vector[1] / norm_xyz
    cos_z_theta = first_edge_vector[0] / norm_xyz
    return cos_z_theta, sin_z_theta


# x_single_rotate(...) y_single_...
def rotate_vector_on_single_axis(
    vector: np.ndarray,
    cos_theta: float,
    sin_theta: float,
    axis: int = 0,
) -> np.ndarray:
    """Return vector rotated anticlockwise by theta on one axis

    """
    # shift order to ax, ax, static
    order1 = ORDER1[axis]
    _, a, b = vector[order1]

    # project rotation on other two axes
    new_a = a * cos_theta - b * sin_theta
    new_b = a * sin_theta + b * cos_theta

    # convert back to orig order
    order2 = ORDER2[axis]
    x, y, z = np.array([_, new_a, new_b])[order2]

    # return as vector format
    return np.array([x, y, z], dtype=vector.dtype)


# vertices_rotate(...)  # not currently used...
# def rotate_vertex_on_single_axis(
#     vertices: np.ndarray,
#     cos_theta: float,
#     sin_theta: float,
#     axis: int = 0,
# ) -> np.ndarray:
#     """Return coordinates of a vertex rotated counter-clockwise on axis.

#     :param vertices:
#     :param axis: 0 or 1 or 2
#     :param cos_theta:
#     :param sin_theta:
#     :return:
#     """
#     axis_a, axis_b = [0, 1, 2].pop(axis)
#     nv = np.array(vertices, dtype=vertices.dtype)
#     nv[:, axis_a] = vertices[:, axis_a] * cos_theta - vertices[:, axis_b] * sin_theta
#     nv[:, axis_b] = vertices[:, axis_a] * sin_theta + vertices[:, axis_b] * cos_theta
#     return nv


if __name__ == "__main__":

    vector = np.array([0.5, 0.9, 0.3], dtype=np.float32)
    
    vertex0 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    vertex1 = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    vertex2 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    face = [
        np.array([0.1, 0.2, 0.3], dtype=np.float32),
        np.array([0.5, 0.5, 0.5], dtype=np.float32),
        np.array([1.0, 1.0, 1.0], dtype=np.float32),
    ]

    print(get_rotation_angles_from_face(*face))
