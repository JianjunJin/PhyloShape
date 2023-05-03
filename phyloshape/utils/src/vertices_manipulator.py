#!/usr/bin/env python

"""

"""
import numpy as np
from numpy.typing import ArrayLike
from typing import List
from loguru import logger
from phyloshape.utils.src.vector_manipulator import gen_unit_perpendicular_v
logger = logger.bind(name="phyloshape")


def uniform_vertices(
        vertices: ArrayLike,
        ref_face_points: List[ArrayLike])\
        -> ArrayLike:
    """
    This function transforms ...

    Parameters
    ----------
    vertices: ArrayLike[np.float32]
        Array of triangle vertices: float (x, y, z) coordinate triplets.
    ref_face_points: List[ArrayLike[np.float32]]
        List of three coordinate triplets, each is an arrays of np.float32 (x, y, z)

    Returns
    -------
    ArrayLike[np.float32]
        Array of triangle vertices: float (x, y, z) coordinate triplets.
    """
    #######
    # I. set the first point to zero
    vertices = vertices - ref_face_points[0]
    ref_face_points = [ref_face_points[p_id] - ref_face_points[p_id] for p_id in range(3)]

    # II: Parallelize to the ref
    vx, vy, vz = vertices.T
    perpendicular_v = gen_unit_perpendicular_v(ref_face_points)

    # 1. calculate rotations to transform the perpendicular_v into (0, 0, 1)
    norm_yz = np.linalg.norm(perpendicular_v[1:])
    # the angle to rotate along the x axis
    sin_x_theta = perpendicular_v[1] / norm_yz if norm_yz else 0.
    cos_x_theta = perpendicular_v[2] / norm_yz if norm_yz else 1.
    # the angle to rotate along the y axis
    sin_y_theta = perpendicular_v[0]
    cos_y_theta = norm_yz

    # 2. apply the rotations to the input vertices
    new_vy = vy * cos_x_theta - vz * sin_x_theta
    new_vz = vy * sin_x_theta + vz * cos_x_theta
    new_vx = vx * cos_y_theta - new_vz * sin_y_theta
    new_vz = vx * sin_y_theta + new_vz * cos_y_theta

    # III. rotate around z-axis
    # apply the rotation to the second point of the ref
    ref_x, ref_y, ref_z = ref_face_points[1]
    ref_y = ref_y * cos_x_theta - ref_z * sin_x_theta
    ref_z = ref_y * sin_x_theta + ref_z * cos_x_theta
    ref_x = ref_x * cos_y_theta - ref_z * sin_y_theta
    ref_z = ref_x * sin_y_theta + ref_z * cos_y_theta

    norm_xy = np.linalg.norm([ref_x, ref_y])




    return np.array([new_vx, new_vy, new_vz]).T


def find_duplicates_in_vertices_list(_vertices_list: List):
    across_sample_duplicates = {}
    recorded_triplets = {}
    for go_t_, triplet_ in enumerate(_vertices_list[0]):
        triplet_ = tuple(triplet_)
        if triplet_ in recorded_triplets:
            across_sample_duplicates[go_t_] = recorded_triplets[triplet_]
        else:
            recorded_triplets[triplet_] = go_t_
    del recorded_triplets
    if across_sample_duplicates:
        for triplets_list in _vertices_list[1:]:
            for go_dup_, go_t_ in list(across_sample_duplicates.items()):
                if not np.array_equal(triplets_list[go_dup_], triplets_list[go_t_]):
                    del across_sample_duplicates[go_dup_]
            if not across_sample_duplicates:
                break
    # triplet_set = set([tuple(unique_ids) for triplet_ in triplets_list_list])
    # assert len(triplet_set) >= 3, "Insufficient valid points!"
    return across_sample_duplicates
