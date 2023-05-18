#!/usr/bin/env python

"""

"""
import numpy as np
from numpy.typing import ArrayLike
from typing import List
from loguru import logger
from phyloshape.utils.src.vector_manipulator import gen_unit_perpendicular_v
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
from copy import deepcopy
logger = logger.bind(name="phyloshape")
import random


# def uniform_vertices(
#         vertices: ArrayLike,
#         ref_face_points: List[ArrayLike])\
#         -> ArrayLike:
#     """
#     This function transforms ...
#
#     Parameters
#     ----------
#     vertices: ArrayLike[np.float32]
#         Array of triangle vertices: float (x, y, z) coordinate triplets.
#     ref_face_points: List[ArrayLike[np.float32]]
#         List of three coordinate triplets, each is an arrays of np.float32 (x, y, z)
#
#     Returns
#     -------
#     ArrayLike[np.float32]
#         Array of triangle vertices: float (x, y, z) coordinate triplets.
#     """
#     #######
#     # I. set the first point to zero
#     vertices = vertices - ref_face_points[0]
#     ref_face_points = [ref_face_points[p_id] - ref_face_points[p_id] for p_id in range(3)]
#
#     # II: Parallelize to the ref
#     vx, vy, vz = vertices.T
#     perpendicular_v = gen_unit_perpendicular_v(ref_face_points)
#
#     # 1. calculate rotations to transform the perpendicular_v into (0, 0, 1)
#     norm_yz = np.linalg.norm(perpendicular_v[1:])
#     # the angle to rotate along the x axis
#     sin_x_theta = perpendicular_v[1] / norm_yz if norm_yz else 0.
#     cos_x_theta = perpendicular_v[2] / norm_yz if norm_yz else 1.
#     # the angle to rotate along the y axis
#     sin_y_theta = perpendicular_v[0]
#     cos_y_theta = norm_yz
#
#     # 2. apply the rotations to the input vertices
#     new_vy = vy * cos_x_theta - vz * sin_x_theta
#     new_vz = vy * sin_x_theta + vz * cos_x_theta
#     new_vx = vx * cos_y_theta - new_vz * sin_y_theta
#     new_vz = vx * sin_y_theta + new_vz * cos_y_theta
#
#     # III. rotate around z-axis
#     # apply the rotation to the second point of the ref
#     ref_x, ref_y, ref_z = ref_face_points[1]
#     ref_y = ref_y * cos_x_theta - ref_z * sin_x_theta
#     ref_z = ref_y * sin_x_theta + ref_z * cos_x_theta
#     ref_x = ref_x * cos_y_theta - ref_z * sin_y_theta
#     ref_z = ref_x * sin_y_theta + ref_z * cos_y_theta
#
#     norm_xy = np.linalg.norm([ref_x, ref_y])
#
#
#
#
#     return np.array([new_vx, new_vy, new_vz]).T


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


def unscaled_procrustes(
        reference: ArrayLike,
        coordinates: ArrayLike,
        ):
    """
    Uses only translation, reflection, and orthogonal rotation.

    Parameters
    ----------
        reference (array-like of Vertices (n_points, n_dim)): reference Vertices to transform `coordinates`
        coordinates (array-like of Vertices (n_points, n_dim)): shape to align to `reference`

    Returns
    ----------
    coordinates: (np.ndarray of vertices (n_points, n_dim))
        transformed `coordinates` matrix
    reference: (np.ndarray of vertices (n_points, n_dim))
        0-centered `reference` matrix
    """
    # Convert inputs to np.ndarray types
    coordinates = np.array(coordinates, dtype=np.double, copy=True)
    reference = np.array(reference, dtype=np.double, copy=True)

    # Translate coordinates to the origin
    coordinates -= coordinates.mean(axis=0)
    reference -= reference.mean(axis=0)

    # Rotate / reflect coordinates to match reference
    # transform mtx2 to minimize disparity
    r_matrix, scale = orthogonal_procrustes(coordinates, reference)
    coordinates = coordinates @ r_matrix

    return reference, coordinates


class GeneralizedProcrustesAnalysis:
    """
    https://wikipedia.org/wiki/Generalized_Procrustes_analysis
    """
    def __init__(self,
                 coordinates_list,
                 max_iter=10,
                 tol=1e-4,
                 init="random",
                 scale: bool = True):
        """
        Parameters
        ----------
        coordinates_list: a deepcopy-ed object will be stored
        max_iter
        tol
        init
        scale
        """
        self.coords_list = np.array([coords.copy() for coords in coordinates_list])
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.scale = scale

        # to be generated
        self.ref = None

    def fit_transform(self):
        """In situ fit the model with shapes.

        Return
        ----------
        adj_shapes:
            new alignment
        adj_ref:
            adjusted reference coordinates
        """
        n_samples = len(self.coords_list)

        # 1. arbitrarily choose a reference shape (typically by selecting it among the available instances)
        if self.init == 'random':
            random.seed(12345)  # for test
            ref_id = random.randint(0, n_samples - 1)
            self.ref = self.coords_list[ref_id].copy()
        elif self.init == 'mean':
            self.ref = self.coords_list.mean(axis=0)
        else:
            raise ValueError("init method must be one of ('random', 'mean')")

        iter_idx = -1
        for iter_idx in range(self.max_iter):
            # 2. superimpose all instances to current reference shape
            for sample_id in range(n_samples):
                if self.scale:
                    sd_ref, self.coords_list[sample_id], _ = procrustes(self.ref, self.coords_list[sample_id])
                else:
                    sd_ref, self.coords_list[sample_id] = unscaled_procrustes(self.ref, self.coords_list[sample_id])

            # 3. compute the mean shape of the current set of superimposed shapes
            mean_vts = self.coords_list.mean(axis=0)

            # 4. if the Procrustes distance between the mean shape and the reference is above a certain threshold,
            # set the reference to mean shape and continue
            procrustes_distance = np.linalg.norm(self.ref - mean_vts)
            if procrustes_distance <= self.tol:
                break
            else:
                self.ref = mean_vts

        logger.debug("{} procrustes iterations".format(iter_idx + 1))


