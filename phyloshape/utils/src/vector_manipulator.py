#!/usr/bin/env python

"""

"""
import numpy as np
from numpy.typing import ArrayLike
from typing import List
from loguru import logger
logger = logger.bind(name="phyloshape")


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
    # norm = np.linalg.norm(perpendicular_v)
    norm = (sum(perpendicular_v ** 2)) ** 0.5    # for compatible with sympy for testing
    return perpendicular_v if norm == 0 else perpendicular_v/norm


def x_single_rotate(vector, cos_theta, sin_theta):
    """Anticlockwise by theta

    :param vector:
    :param cos_theta:
    :param sin_theta:
    :return:
    """
    _x, _y, _z = vector
    _new_y = _y * cos_theta - _z * sin_theta
    _new_z = _y * sin_theta + _z * cos_theta
    return np.array([_x, _new_y, _new_z], dtype=vector.dtype)


def y_single_rotate(vector, cos_theta, sin_theta):
    """Anticlockwise by theta

    :param vector:
    :param cos_theta:
    :param sin_theta:
    :return:
    """
    _x, _y, _z = vector
    _new_x = _x * cos_theta - _z * sin_theta
    _new_z = _x * sin_theta + _z * cos_theta
    return np.array([_new_x, _y, _new_z], dtype=vector.dtype)


def z_single_rotate(vector, cos_theta, sin_theta):
    """Anticlockwise by theta

    :param vector:
    :param cos_theta:
    :param sin_theta:
    :return:
    """
    _x, _y, _z = vector
    _new_x = _x * cos_theta - _y * sin_theta
    _new_y = _x * sin_theta + _y * cos_theta
    return np.array([_new_x, _new_y, _z], dtype=vector.dtype)


def vertices_rotate(vertices, axis, cos_theta, sin_theta):
    """Anticlockwise by theta

    :param vertices:
    :param axis: 0 or 1 or 2
    :param cos_theta:
    :param sin_theta:
    :return:
    """
    axis_a, axis_b = [0, 1, 2].pop(axis)
    _new_vertices = np.array(vertices, dtype=vertices.dtype)
    # new_vector[1:] = \
    #     np.sum(vertices[1:] * np.array([[cos_x_theta, -sin_x_theta], [sin_x_theta, cos_x_theta]]), axis=1)
    _new_vertices[:, axis_a] = vertices[:, axis_a] * cos_theta - vertices[:, axis_b] * sin_theta
    # new_vector[::2] = \
    #     np.sum(new_vector[::2] * np.array([[cos_y_theta, -sin_y_theta], [sin_y_theta, cos_y_theta]]), axis=1)
    _new_vertices[:, axis_b] = vertices[:, axis_a] * sin_theta + vertices[:, axis_b] * cos_theta
    return _new_vertices


def __find_z_axis_angle(first_edge_vector, cos_x_theta, sin_x_theta, cos_y_theta, sin_y_theta):
    # 1.2.1 apply the rotations to the first edge vector
    first_edge_vector = x_single_rotate(first_edge_vector, cos_theta=cos_x_theta, sin_theta=sin_x_theta)
    first_edge_vector = y_single_rotate(first_edge_vector, cos_theta=cos_y_theta, sin_theta=sin_y_theta)
    # 1.2.2 the angle to rotate along the z axis
    # norm_xyz = np.linalg.norm(first_edge_vector)
    norm_xyz = (sum(first_edge_vector ** 2)) ** 0.5  # for compatible with sympy for testing
    sin_z_theta = -first_edge_vector[1] / norm_xyz
    cos_z_theta = first_edge_vector[0] / norm_xyz
    return cos_z_theta, sin_z_theta


def __find_rotation_angles(ref_face_points):
    perpendicular_v = gen_unit_perpendicular_v(ref_face_points)
    # 1.1. calculate rotations to transform the perpendicular_v into (0, 0, 1)
    # norm_yz = np.linalg.norm(perpendicular_v[1:])
    norm_yz = (sum(perpendicular_v[1:] ** 2)) ** 0.5  # for compatible with sympy for testing

    # the angle to rotate along the x axis
    sin_x_theta = perpendicular_v[1] / norm_yz if norm_yz else 0.
    cos_x_theta = perpendicular_v[2] / norm_yz if norm_yz else 1.
    # the angle to rotate along the y axis
    sin_y_theta = perpendicular_v[0]
    cos_y_theta = norm_yz
    # 1.2. calculate the z-axis rotation to transform the first_edge_vector of ref_face to the positive x-axis
    cos_z_theta, sin_z_theta = __find_z_axis_angle(ref_face_points[1] - ref_face_points[0],
                                                   cos_x_theta=cos_x_theta, sin_x_theta=sin_x_theta,
                                                   cos_y_theta=cos_y_theta, sin_y_theta=sin_y_theta)
    return cos_x_theta, sin_x_theta, cos_y_theta, sin_y_theta, cos_z_theta, sin_z_theta


# TODO how to indicate the elements should be [np.float32, np.float32, np.float32]?
def trans_vector_to_relative(
        absolute_vector: ArrayLike,
        ref_face_points: List[ArrayLike])\
        -> ArrayLike:
    """
    This function transforms a relative_vector from its original coordination to a new coordination,
    where the input reference face (t1, t2, t3) is on the xy plane (its perpendicular vector on the positive z axis),
    and where the vector (t1, t2) is on the positive x axis.

    Parameters
    ----------
    absolute_vector: ArrayLike[np.float32]
        Array of triangle vertices: float (x, y, z) coordinate triplets.
    ref_face_points: List[ArrayLike[np.float32]]
        List of three coordinate triplets (t1, t2, t3), each is an arrays of np.float32 (x, y, z)

    Returns
    -------
    ArrayLike[np.float32]
        Array of triangle vertices: float (x, y, z) coordinate triplets.
    """
    cos_x_theta, sin_x_theta, cos_y_theta, sin_y_theta, cos_z_theta, sin_z_theta = \
        __find_rotation_angles(ref_face_points)
    # # new_vector = np.array(vertices, dtype=vertices.dtype)
    # perpendicular_v = gen_unit_perpendicular_v(ref_face_points)
    #
    # # 1.1. calculate rotations to transform the perpendicular_v into (0, 0, 1)
    # norm_yz = np.linalg.norm(perpendicular_v[1:])
    # # the angle to rotate along the x axis
    # sin_x_theta = perpendicular_v[1] / norm_yz if norm_yz else 0.
    # cos_x_theta = perpendicular_v[2] / norm_yz if norm_yz else 1.
    # # the angle to rotate along the y axis
    # sin_y_theta = perpendicular_v[0]
    # cos_y_theta = norm_yz
    # # 1.2. calculate the z-axis rotation to transform the first_edge_vector of ref_face to the positive x-axis
    # cos_z_theta, sin_z_theta = __find_z_axis_angle(ref_face_points[1] - ref_face_points[0],
    #                                                cos_x_theta=cos_x_theta, sin_x_theta=sin_x_theta,
    #                                                cos_y_theta=cos_y_theta, sin_y_theta=sin_y_theta)

    # 2. apply the rotations to the input vertices
    relative_vector = x_single_rotate(absolute_vector, cos_theta=cos_x_theta, sin_theta=sin_x_theta)
    relative_vector = y_single_rotate(relative_vector, cos_theta=cos_y_theta, sin_theta=sin_y_theta)
    relative_vector = z_single_rotate(relative_vector, cos_theta=cos_z_theta, sin_theta=sin_z_theta)

    return relative_vector


# TODO how to indicate the elements should be [np.float32, np.float32, np.float32]?
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
    cos_x_theta, sin_x_theta, cos_y_theta, sin_y_theta, cos_z_theta, sin_z_theta = \
        __find_rotation_angles(ref_face_points)
    # perpendicular_v = gen_unit_perpendicular_v(ref_face_points)
    #
    # # 1.1. calculate rotations to transform (0, 0, 1) into the perpendicular_v
    # norm_yz = np.linalg.norm(perpendicular_v[1:])
    # # the angle to rotate along the y axis
    # sin_y_theta = -perpendicular_v[0]
    # cos_y_theta = norm_yz
    # # the angle to rotate along the x axis
    # sin_x_theta = -perpendicular_v[1] / norm_yz if norm_yz else 0.
    # cos_x_theta = perpendicular_v[2] / norm_yz if norm_yz else 1.
    # # 1.2. calculate the z-axis rotation to transform the first_edge_vector of ref_face to the positive x-axis
    # cos_z_theta, sin_z_theta = __find_z_axis_angle(ref_face_points[1] - ref_face_points[0],
    #                                                cos_x_theta=cos_x_theta, sin_x_theta=sin_x_theta,
    #                                                cos_y_theta=cos_y_theta, sin_y_theta=sin_y_theta)

    # 2. apply the rotations to the input relative_vector, differing from trans_vector_to_relative:
    # a. using minus sin
    # b. in the reverse order
    absolute_vector = z_single_rotate(relative_vector, cos_theta=cos_z_theta, sin_theta=-sin_z_theta)
    absolute_vector = y_single_rotate(absolute_vector, cos_theta=cos_y_theta, sin_theta=-sin_y_theta)
    absolute_vector = x_single_rotate(absolute_vector, cos_theta=cos_x_theta, sin_theta=-sin_x_theta)

    return absolute_vector



