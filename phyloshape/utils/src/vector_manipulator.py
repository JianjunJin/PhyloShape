#!/usr/bin/env python

"""

"""
import numpy as np
from numpy.typing import ArrayLike
from typing import List


#TODO how to indicate the elements should be [np.float32, np.float32, np.float32]?
def gen_unit_perpendicular_v(
        three_points: List[ArrayLike])\
        -> ArrayLike:
    """
    Find the unit perpendicular vector of a plane defined by three coordinate triplets.

    Parameters
    ----------
    three_points: List[ArrayLike]
        Three elements of coordinate triplets, each is an arrays of np.float32 (x, y, z)

    Returns
    -------
    ArrayLike[np.float32, np.float32, np.float32]
        Array of np.float32 (x, y, z) coordinate triplet
    """
    a = three_points[0] - three_points[1]
    b = three_points[0] - three_points[2]
    perpendicular_v = np.array([a[1]*b[2]-a[2]*b[1],
                                a[2]*b[0]-a[0]*b[2],
                                a[0]*b[1]-a[1]*b[0]])
    norm = np.linalg.norm(perpendicular_v)
    return perpendicular_v if norm == 0 else perpendicular_v/norm


#TODO how to indicate the elements should be [np.float32, np.float32, np.float32]?
def trans_vector_to_relative(
        absolute_vector: ArrayLike,
        ref_face_points: List[ArrayLike])\
        -> ArrayLike:
    """
    This function transforms a relative_vector from its original __texture_anchor_coords to a new __texture_anchor_coords,
    where the input reference face is on the xy plane and its perpendicular vector is on the positive z axis

    Parameters
    ----------
    absolute_vector: ArrayLike[np.float32]
        Array of triangle vertices: float (x, y, z) coordinate triplets.
    ref_face_points: List[ArrayLike[np.float32]]
        List of three coordinate triplets, each is an arrays of np.float32 (x, y, z)

    Returns
    -------
    ArrayLike[np.float32]
        Array of triangle vertices: float (x, y, z) coordinate triplets.
    """
    # new_vector = np.array(absolute_vector, dtype=absolute_vector.dtype)
    vx, vy, vz = absolute_vector
    perpendicular_v = gen_unit_perpendicular_v(ref_face_points)

    # 1. calculate rotations to transform the perpendicular_v into (0, 0, 1)
    norm_yz = np.linalg.norm(perpendicular_v[1:])
    # the angle to rotate along the x axis
    sin_x_theta = perpendicular_v[1] / norm_yz
    cos_x_theta = perpendicular_v[2] / norm_yz
    # the angle to rotate along the y axis
    sin_y_theta = perpendicular_v[0]
    cos_y_theta = norm_yz

    # 2. apply the rotations to the input absolute_vector
    # new_vector[1:] = \
    #     np.sum(absolute_vector[1:] * np.array([[cos_x_theta, -sin_x_theta], [sin_x_theta, cos_x_theta]]), axis=1)
    new_vy = vy * cos_x_theta - vz * sin_x_theta
    new_vz = vy * sin_x_theta + vz * cos_x_theta
    # new_vector[::2] = \
    #     np.sum(new_vector[::2] * np.array([[cos_y_theta, -sin_y_theta], [sin_y_theta, cos_y_theta]]), axis=1)
    new_vx = vx * cos_y_theta - new_vz * sin_y_theta
    new_vz = vx * sin_y_theta + new_vz * cos_y_theta

    # return new_vector
    return np.array([new_vx, new_vy, new_vz])


#TODO how to indicate the elements should be [np.float32, np.float32, np.float32]?
def trans_vector_to_absolute(
        relative_vector: ArrayLike,
        ref_face_points: List[ArrayLike])\
        -> ArrayLike:
    """
    This function is the reverse function of get_relative_vector_to_face.

    Parameters
    ----------
    relative_vector: ArrayLike[np.float32]
        Array of triangle vertices: float (x, y, z) coordinate triplets.
    ref_face_points: List[ArrayLike[np.float32]]
        List of three coordinate triplets, each is an arrays of np.float32 (x, y, z)

    Returns
    -------
    ArrayLike[np.float32]
        Array of triangle vertices: float (x, y, z) coordinate triplets.
    """
    vx, vy, vz = relative_vector
    perpendicular_v = gen_unit_perpendicular_v(ref_face_points)

    # 1. calculate rotations to transform (0, 0, 1) into the perpendicular_v
    norm_yz = np.linalg.norm(perpendicular_v[1:])
    # the angle to rotate along the y axis
    sin_y_theta = -perpendicular_v[0]
    cos_y_theta = norm_yz
    # the angle to rotate along the x axis
    sin_x_theta = -perpendicular_v[1] / norm_yz
    cos_x_theta = perpendicular_v[2] / norm_yz

    # 2. apply the rotations to the input relative_vector
    new_vx = vx * cos_y_theta - vz * sin_y_theta
    new_vz = vx * sin_y_theta + vz * cos_y_theta
    new_vy = vy * cos_x_theta - new_vz * sin_x_theta
    new_vz = vy * sin_x_theta + new_vz * cos_x_theta

    return np.array([new_vx, new_vy, new_vz])



