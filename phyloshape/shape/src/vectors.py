#!/usr/bin/env python

"""Core PhyloShape class object of the phyloshape package.

"""

from typing import Tuple, List, Union
from collections import OrderedDict
from copy import deepcopy
import random
import numpy as np
from numpy.typing import ArrayLike
from phyloshape.utils import trans_vector_to_relative, trans_vector_to_absolute
from phyloshape.utils.src.vertices_manipulator import find_duplicates_in_vertices_list
# from phyloshape.utils.src.stats import mean_without_outliers
from phyloshape.shape.src.shape import ShapeAlignment
from loguru import logger
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.stats import norm
logger = logger.bind(name="phyloshape")


class _VectorHandler:
    """
    Used in FaceVectorMapper.
    Unit class recording a single map between the vertices-face system and the vector system.

    """
    def __init__(
            self,
            from_id: int,
            to_id: int,
            ref_plane_triplet: Union[Tuple[int, int, int], None]):
        """
        Parameters
        ----------
        from_id: int
        to_id: int
        ref_plane_triplet: (int, int, int)
        """
        self.from_id = from_id
        self.to_id = to_id
        self.ref_plane = ref_plane_triplet

    def __repr__(self):
        if self.ref_plane:
            face_with_id_marked = list(self.ref_plane)
            try:
                face_with_id_marked[self.ref_plane.index(self.from_id)] = [self.from_id]
            except ValueError:
                # mark from_id out of the ref_plane scope
                face_with_id_marked.append([self.from_id])
        else:
            face_with_id_marked = [[self.from_id]]
        return f"{face_with_id_marked} -> {self.to_id}"

    def __str__(self):
        return self.__repr__()
#
#
# class _VectorHandlerList:
#     """
#     list of _VectorHandler
#     """
#     def __init__(self, *vh: _VectorHandler):
#         self.__list = list(vh)
#


class _VertexTree:
    """
    Used in FaceVectorMapper/VertexVectorMapper for tracing and k3d plotting.
    Tree-like class recording the vertices connection map via the vectors.

    """
    def __init__(self, root_id: int = None):
        self.__vertex_children = {}
        self.__vertex_parent = {}
        self.__root = None
        if not (root_id is None):
            self.set_root(root_id)
        # self.vertex_id = vertex_id

    def set_root(self, vertex_id: int):
        self.__root = vertex_id
        self.__vertex_children = {vertex_id: []}
        self.__vertex_parent = {vertex_id: None}

    def add_link(self, from_id, to_id):
        if from_id not in self.__vertex_children:
            self.__vertex_children[from_id] = []
        self.__vertex_children[from_id].append(to_id)
        self.__vertex_parent[to_id] = from_id

    def get_lines_for_k3d_plot(self, start_id: int = None):
        # TODO: usable but bugs to be fixed, last extra idx
        """
        For k3d line plotting.

        Parameters
        ----------
        start_id: int
            start from the root if None by default
        """
        # was written recursively, but the maximum recursion depth will be reached

        if start_id is None:
            start_id = self.__root

        path = [start_id]
        if start_id in self.__vertex_children:
            next_id = self.__vertex_children[start_id][0]
            path.append(next_id)
        else:
            raise TypeError("At least two points are required for a line!")
        # branching_node_visited and this_branch_path should be co-changed
        branching_node_visited = [[start_id, 0]]
        this_branch_path = [[start_id]]
        while branching_node_visited:
            go_id = path[-1]
            if go_id in self.__vertex_children:
                # elongation
                next_id = self.__vertex_children[go_id][0]
                path.append(next_id)
                # make a branch
                if len(self.__vertex_children[go_id]) > 1:
                    branching_node_visited.append([go_id, 0])
                    this_branch_path.append([next_id])
                else:
                    this_branch_path[-1].append(next_id)
            else:
                last_node_id, last_br = branching_node_visited[-1]
                # traverse back to the latest branching point
                traverse_back_path = this_branch_path[-1][-2::-1] + [last_node_id]
                path.extend(traverse_back_path)
                # both branches traversed
                while last_br == len(self.__vertex_children[last_node_id]) - 1:
                    # traverse back deeper
                    del branching_node_visited[-1]
                    del this_branch_path[-1]
                    if branching_node_visited:
                        last_node_id, last_br = branching_node_visited[-1]
                        traverse_back_path = this_branch_path[-1][-2::-1] + [last_node_id]
                        path.extend(traverse_back_path)
                    else:
                        break
                # criterion still stands
                if branching_node_visited:
                    # switch to next branch
                    next_id = self.__vertex_children[last_node_id][last_br + 1]
                    path.append(next_id)
                    branching_node_visited[-1][1] = last_br + 1
                    this_branch_path[-1] = [next_id]
        return path

    def get_parent(self, child_id):
        return self.__vertex_parent[child_id]

    def trace(self, from_id, n_steps: int = 1):
        id_list = [from_id]
        for foo in range(n_steps):
            parent = self.__vertex_parent[id_list[-1]]
            if parent is None:
                return tuple(id_list)
            else:
                id_list.append(parent)
        return tuple(id_list)


class _VertexFlow:
    """
    Used in FaceVectorMapper/VertexVectorMapper for tracing and k3d plotting.
    Network-like class recording the vertices connection map via the vectors.

    """

    def __init__(self, root_id: int = None):
        self.__vertex_children = {}
        self.__vertex_parent = {}
        self.__root = None
        if not (root_id is None):
            self.set_root(root_id)
        # self.vertex_id = vertex_id

    def set_root(self, vertex_id: int):
        self.__root = vertex_id
        self.__vertex_children = {vertex_id: []}
        self.__vertex_parent = {vertex_id: None}

    def add_link(self, from_id, to_id):
        if from_id not in self.__vertex_children:
            self.__vertex_children[from_id] = []
        self.__vertex_children[from_id].append(to_id)
        if to_id not in self.__vertex_parent:
            self.__vertex_parent[to_id] = []
        self.__vertex_parent[to_id].append(from_id)

    def get_lines_for_k3d_plot(self, start_id: int = None):
        """
        For k3d line plotting.

        Parameters
        ----------
        start_id: int
            start from the root if None by default
        """
        # TODO: to be updated for network
        raise NotImplementedError("to be done")

        if start_id is None:
            start_id = self.__root

        path = [start_id]
        if start_id in self.__vertex_children:
            next_id = self.__vertex_children[start_id][0]
            path.append(next_id)
        else:
            raise TypeError("At least two points are required for a line!")
        # branching_node_visited and this_branch_path should be co-changed
        branching_node_visited = [[start_id, 0]]
        this_branch_path = [[start_id]]
        while branching_node_visited:
            go_id = path[-1]
            if go_id in self.__vertex_children:
                # elongation
                next_id = self.__vertex_children[go_id][0]
                path.append(next_id)
                # make a branch
                if len(self.__vertex_children[go_id]) > 1:
                    branching_node_visited.append([go_id, 0])
                    this_branch_path.append([next_id])
                else:
                    this_branch_path[-1].append(next_id)
            else:
                last_node_id, last_br = branching_node_visited[-1]
                # traverse back to the latest branching point
                traverse_back_path = this_branch_path[-1][-2::-1] + [last_node_id]
                path.extend(traverse_back_path)
                # both branches traversed
                while last_br == len(self.__vertex_children[last_node_id]) - 1:
                    # traverse back deeper
                    del branching_node_visited[-1]
                    del this_branch_path[-1]
                    if branching_node_visited:
                        last_node_id, last_br = branching_node_visited[-1]
                        traverse_back_path = this_branch_path[-1][-2::-1] + [last_node_id]
                        path.extend(traverse_back_path)
                    else:
                        break
                # criterion still stands
                if branching_node_visited:
                    # switch to next branch
                    next_id = self.__vertex_children[last_node_id][last_br + 1]
                    path.append(next_id)
                    branching_node_visited[-1][1] = last_br + 1
                    this_branch_path[-1] = [next_id]
        return path

    def trace(self, from_id, n_steps: int = 1):
        raise NotImplementedError("to be done")
        id_list = [from_id]
        for foo in range(n_steps):
            parent = self.__vertex_parent[id_list[-1]]
            if parent is None:
                return tuple(id_list)
            else:
                id_list.append(parent)
        return tuple(id_list)


class VMapperOld:
    """
    Base Class recording all maps between the coordination system and the vector system.
    The representation swifts are reversible with S(Vectors) = S(vertices) - 1
    """

    def __init__(self, random_seed: int = 0):
        self._vh_list = []
        self._id_to_triplets = {}  # record the triplets (of the ref_plane) where the id is from
        self._vertex_tree = _VertexTree()  # for plotting and tracing
        self.random_seed = random_seed

    def _append(self, vh: _VectorHandler):
        self._vh_list.append(vh)

    def to_vectors(self, vertices) -> ArrayLike:
        """Based on the mapping information, it takes vertices of a shape to generate representative vectors

        Parameters
        ----------
        vertices:
            Array of triangle vertices: float (x, y, z) coordinate triplets.

        Returns
        -------
        vectors
            ArrayLike
        """
        assert len(vertices) == len(self._vh_list) + 1, \
            "The length of the vertices ({}) must be 1-unit larger than the length of vector handlers ({})!".format(
                len(vertices), len(self._vh_list))

        # initialize with the first vectors
        vh_first = self._vh_list[0]
        space_v1 = vertices[vh_first.to_id] - vertices[vh_first.from_id]
        norm_v1 = np.linalg.norm(space_v1)
        relative_v1 = np.array([norm_v1, 0, 0])
        vectors = [relative_v1]

        # initialize with the second vectors
        # vh_next = self._vh_list[1]
        # space_v2 = vertices[vh_next.to_id] - vertices[vh_first.from_id]
        # norm_v2 = np.linalg.norm(space_v2)
        # dot_product_v12 = sum(space_v1 * space_v2)
        # cos_theta2 = dot_product_v12 / (norm_v1 * norm_v2)
        # sin_theta2 = (1 - cos_theta2 ** 2) ** 0.5
        # vectors.append(np.array([cos_theta2 * norm_v2, sin_theta2 * norm_v2, 0]) - relative_v1)

        # do the following vectors
        for vh in self._vh_list[1:]:
            # logger.trace("vector handler: {}".format(vh))
            vector_in_space = vertices[vh.to_id] - vertices[vh.from_id]
            # logger.trace("vector_in_space: %s" % str(vector_in_space))
            # logger.trace("face points: {}".format(vertices[list(vh.ref_plane_triplet)]))
            relative_vector = trans_vector_to_relative(vector_in_space, vertices[list(vh.ref_plane)])
            # logger.trace("relative vector: {}".format(relative_vector))
            vectors.append(relative_vector)
        return np.array(vectors)

    def to_vertices(self, vectors) -> ArrayLike:
        assert len(vectors) == len(self._vh_list), \
            "The length of the vectors ({}) must equals the length of vector handlers ({})!".format(
                len(vectors), len(self._vh_list))
        vertices = np.full((len(vectors) + 1, 3), np.nan, dtype=np.float64)
        vertices[self._vh_list[0].from_id] = [0., 0., 0.]
        vertices[self._vh_list[0].to_id] = vectors[0]
        vertices[self._vh_list[1].to_id] = vectors[0] + vectors[1]
        for go_vct, vh in enumerate(self._vh_list[2:]):
            relative_vector = vectors[go_vct + 2]
            vector_in_space = trans_vector_to_absolute(relative_vector, vertices[list(vh.ref_plane)])
            if np.isnan(vertices[vh.from_id]).any():
                raise ValueError(
                    f"While building Vtx {vh.to_id}, Vtx {vh.from_id} is invalid {vertices[vh.from_id]}!")
            else:
                vertices[vh.to_id] = vertices[vh.from_id] + vector_in_space
            # logger.trace("goid:{}, vh:{}, vertices[vh.from_id]:{}, vertices[vh.to_id:{}]:{}".
            #              format(go_vct, vh, vertices[vh.from_id], vh.to_id, vertices[vh.to_id]))
        return np.array(vertices, dtype=np.float64)

    def vh_list(self):
        return deepcopy(self._vh_list)

    def get_lines_for_k3d_plot(self):
        return self._vertex_tree.get_lines_for_k3d_plot()


class VMapper:
    """
    Base Class recording all maps between the coordination system and the vector system.
    The representation swifts can be irreversible with S(Vectors) = S(vertices) * N - N*(N+1)/2, when S(vertices) > N
    """

    def __init__(self,
                 num_vs: int = 10,
                 num_vt_iter: int = 5,
                 random_seed: int = 0):
        """
         Parameters
         ----------
         num_vs:
             number of vertices as local topology.
         num_vt_iter:
             number of iterations for updating vertices.
         random_seed: int
             0: closest-neighbor traverse
             other values: random
         """
        self.num_vs = num_vs
        self.num_vt_iter = num_vt_iter
        self._vh_bundle_list = []
        self._shape = []
        # _id_to_triplets* are recording the triplets (of the ref_plane) where the id is from
        # no need to store as class variable, but here stored for debugging
        self._id_to_triplets = {}  # for going back to the starting vertices and more

        # self._vertex_tree = _VertexFlow()  # for plotting and tracing
        self._vertex_tree = _VertexTree()  # for plotting and tracing
        self.random_seed = random_seed
        self._debug_vertices = []

    def _append(self, vh_bundle: List[Union[_VectorHandler]]):
        # TODO: create _VectorHandlerList to record the weights, the shared to_id, size, and other attributes
        #       the weights can be equal (?), or average-v-length (?),
        #       or predictability given from_id (inverse of variation or angle-variation among samples)
        #       try "inverse of variation among samples" first, i.e highest probability point shared by distributions
        self._vh_bundle_list.append(vh_bundle)
        self._shape.append(len(vh_bundle))

    def to_vectors(self, vertices: ArrayLike) -> ArrayLike:
        """
        Based on the vh_bundle_list information, it takes vertices of a shape to generate representative vectors.

        Parameters
        ----------
        vertices:
            Array of triangle vertices: float (x, y, z) coordinate triplets.

        Returns
        -------
        vectors
            ArrayLike
        """
        assert len(vertices) == len(self._shape) - self.num_vs + 1, \
            "The length of the vertices ({}) must be (num_vs-1)-units i.e. {}-units " \
            "smaller than the length of vh bundles ({})!".format(
                len(vertices), self.num_vs - 1, len(self._shape))

        # initialize with the first vectors
        vh_first_bundle = self._vh_bundle_list[0]
        # relative_v1 = []
        vectors = []
        for vh_first in vh_first_bundle:  # size of 1
            # if vh_first is None:
            #     relative_v1.append(np.array([None, None, None]))
            # else:
            space_v1 = vertices[vh_first.to_id] - vertices[vh_first.from_id]
            norm_v1 = np.linalg.norm(space_v1)
            # relative_v1.append(np.array([norm_v1, 0, 0]))
            vectors.append(np.array([norm_v1, 0, 0]))
        # vectors = [relative_v1]

        # initialize with the second vectors
        # vh_next = self._vh_bundle_list[1]
        # space_v2 = vertices[vh_next.to_id] - vertices[vh_first.from_id]
        # norm_v2 = np.linalg.norm(space_v2)
        # dot_product_v12 = sum(space_v1 * space_v2)
        # cos_theta2 = dot_product_v12 / (norm_v1 * norm_v2)
        # sin_theta2 = (1 - cos_theta2 ** 2) ** 0.5
        # vectors.append(np.array([cos_theta2 * norm_v2, sin_theta2 * norm_v2, 0]) - relative_v1)

        # do the following vectors
        for vh_bundle_list in self._vh_bundle_list[1:]:
            # relative_vectors = []
            for vh in vh_bundle_list:
                # if vh is None:
                #     relative_vectors.append(np.array([None, None, None]))
                # else:
                # logger.trace("vector handler: {}".format(vh))
                vector_in_space = vertices[vh.to_id] - vertices[vh.from_id]
                # logger.trace("vector_in_space: %s" % str(vector_in_space))
                # logger.trace("face points: {}".format(vertices[list(vh.ref_plane_triplet)]))
                # if np.linalg.norm(vector_in_space) < 0.00005:
                #     print("problematic ref plant: {}".format(vh.ref_plane))
                # relative_vectors.append(trans_vector_to_relative(vector_in_space, vertices[list(vh.ref_plane)]))
                vectors.append(trans_vector_to_relative(vector_in_space, vertices[list(vh.ref_plane)]))
                # logger.trace("relative vector: {}".format(relative_vector))
            # vectors.append(relative_vectors)
        # TODO: remove the redundant half vectors, not influencing the reconstruction given PCA though
        return np.array(vectors, dtype=np.float64)

    def to_vertices(self, vectors) -> ArrayLike:
        assert len(vectors) == sum(self._shape), \
            "The length of the vectors ({}) must equal the length of vector handlers ({})!".format(
                len(vectors), sum(self._shape))
        len_vertices = len(self._shape) - self.num_vs + 1
        vertices = np.full((len_vertices, 3), np.nan, dtype=np.float64)

        # first vh bundle only has one valid handler
        go_v = 0
        first_vh_bundle = self._vh_bundle_list[0]
        first_vectors = vectors[go_v]
        vertices[first_vh_bundle[0].from_id] = [0., 0., 0.]
        vertices[first_vh_bundle[0].to_id] = first_vectors[0]  # only has one vector too
        go_v += self._shape[0]

        # second vh bundle has two valid handlers
        # TODO, weights can be calculated from vector variations
        second_vh_bundle = self._vh_bundle_list[1]
        vertices[second_vh_bundle[0].to_id] = np.median([vertices[vh.from_id] + vectors[go_v + go_h]
                                                         for go_h, vh in enumerate(second_vh_bundle)], axis=0)

        # using mean without outliers does not work!
        # here_vertices = [vertices[vh.from_id] + vectors[go_v + go_h] for go_h, vh in enumerate(second_vh_bundle)]
        # here_vertices = np.array(here_vertices, dtype=np.float64)
        # vertices[second_vh_bundle[0].to_id] = [mean_without_outliers(here_vertices[:, i]) for i in range(3)]
        go_v += self._shape[1]

        # following vh bundle requires converting relative vectors to absolute vectors
        self._debug_vertices = []
        for go_b, vh_bundle in enumerate(self._vh_bundle_list[2:-self.num_vs]):
            v_len = self._shape[2 + go_b]
            relative_vectors = vectors[go_v: go_v + v_len]
            go_v += v_len
            self.__update_vertex_with_vh_bundle(
                vh_bundle=vh_bundle, relative_vectors=relative_vectors, vertices=vertices)

        self._debug_vertices.append(vertices)
        new_vertices = self.__updating_vertices(vertices=vertices, vectors=vectors)

        return new_vertices

    def __updating_vertices(self, vertices, vectors):
        # TODO probably using GPA to verify convergence
        # iteratively refine the vertices
        len_vertices = len(vertices)
        vt_status = []
        new_vertices = vertices.copy()
        vt_status.append(new_vertices.copy())
        # convergence = False
        # while not convergence:
        # for iter_n in range(3):
        for iter_n in range(self.num_vt_iter):
            # the last self.num_vs bundles support the first num_vs points of the new round
            # len(new_bundle_list) equals num of vertices & new_vertices
            new_bundle_list = self._vh_bundle_list[-self.num_vs:] + self._vh_bundle_list[self.num_vs - 1: -self.num_vs]
            # TODO if to_vectors was modified to remove the last half vector, the new_vectors shall be modified
            new_vectors = vectors[-len_vertices * self.num_vs:]
            for go_b, vh_bundle in enumerate(new_bundle_list):
                relative_vectors = new_vectors[go_b * self.num_vs: (go_b + 1) * self.num_vs]
                self.__update_vertex_with_vh_bundle(
                    vh_bundle=vh_bundle, relative_vectors=relative_vectors, vertices=new_vertices)
            vt_status.append(new_vertices.copy())
            self._debug_vertices.append(vt_status[-1])
            # TODO better convergence approach and verify that it always converges
            # the following code is not working
            # if max(np.linalg.norm(vt_status[-2] - new_vertices, axis=1)) < 0.01:
            #     convergence = True
        logger.debug("total num of iterations: {}".format(len(vt_status) - 1))
        return new_vertices

    def __update_vertex_with_vh_bundle(self, vh_bundle, relative_vectors, vertices):
        """ Update one vertex per vh_bundle
        :param vh_bundle:
        :param relative_vectors:
        :param vertices:
        :return:
        """
        new_triplets = []
        for go_h, vh in enumerate(vh_bundle):
            vector_in_space = trans_vector_to_absolute(relative_vectors[go_h], vertices[list(vh.ref_plane)])
            if np.isnan(vertices[vh.from_id]).any():
                raise ValueError(
                    f"While building Vtx {vh.to_id}, Vtx {vh.from_id} is invalid {vertices[vh.from_id]}!")
            new_triplets.append(vertices[vh.from_id] + vector_in_space)
        vertices[vh_bundle[0].to_id] = np.median(new_triplets, axis=0)
        # using mean without outliers does not work!
        # new_triplets = np.array(new_triplets, dtype=np.float64)
        # vertices[vh_bundle[0].to_id] = [mean_without_outliers(new_triplets[:, i]) for i in range(3)]
        # self._debug_vertices.append(new_triplets)

    def vh_list(self):
        return deepcopy(self._vh_bundle_list)

    def get_lines_for_k3d_plot(self):
        return self._vertex_tree.get_lines_for_k3d_plot()


# class FaceVectorMapperOld(VMapperOld):
#     """
#     Main Class recording all maps between the face-vertices system and the vector system
#
#     """
#     def __init__(self, input_obj, random_seed: int = 0):
#         """
#
#         Parameters
#         ----------
#         input_obj:
#             ArrayLike or Faces (with vertex_ids) or Shape (with Faces.vertex_ids).
#             All faces must be from a single connected object.
#         random_seed: int
#             0: pre-order-like traverse
#             -1: post-order-like traverse
#             other values: random
#         """
#         super().__init__(random_seed)
#         self.__fixed_vertex_ids = set()
#         self.__waiting_triplets = OrderedDict()  # to remove random effect outside the random.choice
#         self.__checking_triplet = None
#         self.__checked_triplets = set()
#         # if is Shape
#         if "faces" in dir(input_obj) and "vertex_ids" in dir(input_obj.faces):
#             vts = input_obj.faces.vertex_ids
#         # if is Faces
#         elif "vertex_ids" in dir(input_obj):
#             vts = input_obj.vertex_ids
#         else:
#             vts = input_obj
#         self.__update(vts)
#
#     def __update(
#             self,
#             triplets_list: ArrayLike or List[ArrayLike]):
#         """
#         Build the maps between the face-vertices system and the vector system.
#         We do it with a single start point to ensure self.to_vertices works properly.
#
#         Parameters
#         ----------
#         triplets_list: ArrayLike or List[ArrayLike]
#             Must be faces from single connected object.
#         """
#         random.seed(self.random_seed)
#         triplets_set = set([tuple(tl) for tl in triplets_list])
#         len_triplets = len(triplets_set)
#         logger.info(f"{len_triplets} triplets input.")
#
#         for tri_ids in triplets_list:
#             for single_id in tri_ids:
#                 if single_id not in self._id_to_triplets:
#                     self._id_to_triplets[single_id] = set()
#                     # a list of triplets will be used later to determine the the_reference_triplet:
#                     #   the_reference_triplet = id_related_triplets & self.__checked_triplets
#                     # so we have to use set here to store the id_related_triplets
#                 self._id_to_triplets[single_id].add(tuple(tri_ids))
#         logger.info(f"{len(self._id_to_triplets)} vertices indexed.")
#
#         # have the first point id fixed in space
#         if self.random_seed not in {0, -1}:
#             # TODO I don't know why the random mode will be significantly smaller.
#             #      Should find out where it is and optimize it.
#             self.__checking_triplet = id_1, id_2, id_3 = random.choice(triplets_list)
#         else:
#             self.__checking_triplet = id_1, id_2, id_3 = triplets_list[0]
#         self.__checking_triplet = tuple(self.__checking_triplet)
#         self.__fixed_vertex_ids.add(id_1)
#         self.__checked_triplets.add(self.__checking_triplet)
#         for triplets in self._id_to_triplets[id_1]:
#             if triplets not in self.__checked_triplets:
#                 self.__waiting_triplets[triplets] = None
#         self._vertex_tree.set_root(id_1)
#         # record the first and second vector
#         # Different from downstream, the reference faces here are uniquely `None` and "the same face of the vector".
#         self._append(_VectorHandler(id_1, id_2, ref_plane_triplet=None))
#         self.__to_id_update(id_2)
#         self._vertex_tree.add_link(id_1, id_2)
#         self._append(_VectorHandler(id_2, id_3, ref_plane_triplet=self.__checking_triplet))
#         self.__to_id_update(id_3)
#         self._vertex_tree.add_link(id_1, id_3)
#         # previous_triplet = this_triplet
#
#         # start building
#         # while len(self.__checked_triplets) < len_triplets:
#         while self.__waiting_triplets:
#             # other searching proposals may be applied if necessary
#             if self.random_seed not in {0, -1}:
#                 this_triplet = id_1, id_2, id_3 = random.choice(list(self.__waiting_triplets.keys()))
#                 del self.__waiting_triplets[this_triplet]
#             else:
#                 this_triplet = id_1, id_2, id_3 = self.__waiting_triplets.popitem(bool(self.random_seed))[0]
#
#             if id_1 in self.__fixed_vertex_ids:
#                 if id_2 in self.__fixed_vertex_ids:
#                     if id_3 in self.__fixed_vertex_ids:
#                         pass
#                     else:
#                         self.__build_vector(id_1, id_3)
#                 else:
#                     if id_3 in self.__fixed_vertex_ids:
#                         self.__build_vector(id_1, id_2)
#                     else:
#                         self.__build_vector(id_1, id_2)
#                         self.__build_vector(id_1, id_3)
#             else:
#                 if id_2 in self.__fixed_vertex_ids:
#                     if id_3 in self.__fixed_vertex_ids:
#                         self.__build_vector(id_2, id_1)
#                     else:
#                         self.__build_vector(id_2, id_1)
#                         self.__build_vector(id_2, id_3)
#                 else:
#                     if id_3 in self.__fixed_vertex_ids:
#                         self.__build_vector(id_3, id_2)
#                         self.__build_vector(id_3, id_1)
#                     else:
#                         raise TypeError(f"({id_1}, {id_2}, {id_3})"
#                                         "Isolated face is not allowed! "
#                                         "All faces must be from a single connected object!")
#             self.__checked_triplets.add(this_triplet)
#             # previous_triplet = this_triplet
#         logger.info(f"{len(self.__checked_triplets)} triplets checked.")
#
#     def __build_vector(self, from_id, to_id):
#         candidate_triplets = self._id_to_triplets[from_id] & self.__checked_triplets
#         if self.random_seed not in {0, -1}:
#             reference_triplet = random.choice(sorted(candidate_triplets))
#         else:
#             reference_triplet = sorted(candidate_triplets)[0]
#         self._append(_VectorHandler(from_id, to_id, reference_triplet))
#         self._vertex_tree.add_link(from_id, to_id)
#         self.__to_id_update(to_id=to_id)
#
#     def __to_id_update(self, to_id):
#         self.__fixed_vertex_ids.add(to_id)
#         for triplets in self._id_to_triplets[to_id]:
#             if triplets not in self.__checked_triplets and triplets != self.__checking_triplet:
#                 self.__waiting_triplets[triplets] = None


class VertexVectorMapperOld(VMapperOld):
    """
    Main Class recording all maps between the dispersed vertices system and the vector system

    """
    def __init__(self, input_obj, random_seed: int = 0):
        """
        Parameters
        ----------
        input_obj:
            ArrayLike or Vertices (with coords).
        random_seed: int
            0: closest-neighbor traverse
            other values: random
        """
        super().__init__(random_seed)
        self.__fixed_vertex_ids = OrderedDict()
        self.__unfixed_vertex_ids = OrderedDict()
        # if is Vertices
        if "coords" in dir(input_obj):
            vts = input_obj.coords
        else:
            vts = input_obj
        self.__update(vts)

    def __update(
            self,
            triplets_list: ArrayLike or List[ArrayLike]):
        """
        Build the maps between the dispersed vertices system and the vector system.

        Parameters
        ----------
        triplets_list: ArrayLike or List[ArrayLike]
            Must be coordinates of Vertices.
        """
        self.__unfixed_vertex_ids = OrderedDict([(_id, None) for _id in range(len(triplets_list))])
        triplet_set = set([tuple(triplet_) for triplet_ in triplets_list])
        assert len(triplet_set) >= 3, "Insufficient valid points!"
        if self.random_seed != 0:
            # TODO I don't know why the random mode will be significantly smaller.
            #      Should find out where it is and optimize it.
            random.seed(self.random_seed)
            # TODO unfinished
        else:
            # TODO: use k-d tree to speed up
            assert len(triplets_list) < 2000, "larger number of points not implemented yet!"
            diffs = triplets_list[:, np.newaxis, :] - triplets_list[np.newaxis, :, :]
            pairwise_distances = np.sqrt(np.sum(diffs ** 2, axis=-1))
            pairwise_distances_m = np.ma.masked_array(pairwise_distances, mask=pairwise_distances == 0)
            # np.where looks for multiple values
            # min_dist = np.min(pairwise_distances[pairwise_distances > 0])
            # min_v1_ids, min_v2_ids = np.where(pairwise_distances == min_dist)
            # id_1 = min_v1_ids[0]
            # id_2 = min_v2_ids[0]
            id_1, id_2 = np.unravel_index(pairwise_distances_m.argmin(), pairwise_distances_m.shape)
            self._vertex_tree.set_root(id_1)
            self._append(_VectorHandler(id_1, id_2, ref_plane_triplet=None))
            self.__to_id_update(id_1)
            self.__to_id_update(id_2)
            self._vertex_tree.add_link(id_1, id_2)
            # when a row contains zero, the coordination of that id overlaps with the id_1 or id_2,
            # meaning that row should be masked
            search_distances = pairwise_distances[:, id_2]
            search_mask = pairwise_distances[:, (id_1, id_2)].min(axis=1) == 0
            id_3 = np.ma.masked_array(search_distances, mask=search_mask).argmin()
            initial_ref_plane_ids = (id_1, id_2, id_3)
            self._append(_VectorHandler(id_2, id_3, ref_plane_triplet=initial_ref_plane_ids))
            self.__to_id_update(id_3)
            self._vertex_tree.add_link(id_2, id_3)
            for initial_id in initial_ref_plane_ids:
                self._id_to_triplets[initial_id] = initial_ref_plane_ids

            # add more points
            while self.__unfixed_vertex_ids:
                # # id_1 and id_2 are not considered to extend for easier conversion between two systems
                # fixed_id_list = list(self.__fixed_vertex_ids)[2:]
                fixed_id_list = list(self.__fixed_vertex_ids)
                unfixed_id_list = list(self.__unfixed_vertex_ids)
                search_distances = pairwise_distances[np.ix_(unfixed_id_list, fixed_id_list)]
                # search_mask = np.zeros(search_distances.shape[0], dtype=bool)
                # search_mask[fixed_id_list] = True
                # search_distances_m = np.ma.masked_array(search_distances, mask=search_mask[:, np.newaxis])
                to_id_id, from_id_id = np.unravel_index(search_distances.argmin(), search_distances.shape)
                to_id = unfixed_id_list[to_id_id]
                from_id = fixed_id_list[from_id_id]
                self._append(_VectorHandler(from_id, to_id, ref_plane_triplet=self._id_to_triplets[from_id]))
                self.__to_id_update(to_id)
                self._vertex_tree.add_link(from_id, to_id)
                if search_distances[to_id_id, from_id_id] == 0:
                    # the current point superposition a previous point,
                    # it can not be taken to form a valid reference plane, use the previous reference plane instead
                    self._id_to_triplets[to_id] = self._id_to_triplets[from_id]
                else:
                    traced_back = self._vertex_tree.trace(to_id, n_steps=2)
                    if len(traced_back) == 3:
                        self._id_to_triplets[to_id] = traced_back
                    else:  # from_id belongs to (id_1, id_2)
                        self._id_to_triplets[to_id] = self._id_to_triplets[from_id]

    def __to_id_update(self, to_id):
        del self.__unfixed_vertex_ids[to_id]
        self.__fixed_vertex_ids[to_id] = None


class VertexVectorMapper(VMapper):
    """
    Main Class recording all maps between the dispersed vertices system and the vector system

    """
    def __init__(self,
                 input_obj_list: List,  # TODO: should also accept ShapeAlignment for convenience
                 mode: str = "linear-variation",
                 random_seed: int = 0,
                 num_vs: int = 10,
                 num_vt_iter: int = 5):
        """
        Parameters
        ----------
        input_obj_list:
            A list of ArrayLike or Vertices (with coords) objects.
        mode: str
            mode of building the mapper. Under test and development, so it was not fixed yet.
        random_seed: int
            0: closest-neighbor traverse
            other values: random
        num_vs: int
            number of vertices as local topology.
        num_vt_iter: int
             number of iterations for updating vertices.
        """
        self.__fixed_vertex_ids = OrderedDict()
        self.__unfixed_vertex_ids = OrderedDict()

        if not mode.startswith("linear"):
            self._vertex_tree = _VertexFlow()

        # if is Vertices
        assert isinstance(input_obj_list, list) and len(input_obj_list) > 0, "please input valid list of vertices!"
        if "coords" in dir(input_obj_list[0]):
            vts_list = [vt_obj.coords for vt_obj in input_obj_list]
        else:
            vts_list = input_obj_list
        self.tri_ll = vts_list
        self.len_samples = len(self.tri_ll)
        self.len_vertices = len(self.tri_ll[0])
        if num_vs >= self.len_vertices:
            num_vs = self.len_vertices - 1
            logger.warning("Resetting num_vs to {} according to the number of alignment points!".format(num_vs))
        super().__init__(num_vs=num_vs, num_vt_iter=num_vt_iter, random_seed=random_seed)
        self.kd_trees = None
        self.__update(mode=mode)

    def __update(self, mode):
        """
        Build the maps between the dispersed vertices system and the vector system.
        """
        self.check_duplicates()
        assert self.len_vertices > 3
        # TODO: if using variation order, this is not usable
        self.__unfixed_vertex_ids = OrderedDict([(_id, None) for _id in range(self.len_vertices)])

        if mode.endswith("-random"):
            if mode.startswith("linear-"):
                # TODO I don't know why the random mode will be significantly smaller.
                #      Should find out where it is and optimize it.
                construct_v_orders = list(range(self.len_vertices))
                random.seed(self.random_seed)
                random.shuffle(construct_v_orders)
                self.__update_linear_build(construct_v_orders)
            else:  # mode == "linear-variation":
                raise ValueError("not implemented mode: {}".format(mode))

        elif mode.endswith("-local") or mode.endswith("-variation"):
            # TODO: efficiency can be improved
            # TODO: use the distance to mimic the changes to find the most conserved part to start
            #       the real conserved part should be calculated from sequential vectors? probably too compute-expensive
            # TODO: phylogenetic variation instead of all-to-all
            # TODO: can be also model-based to accommodate to scale-change
            # TODO: some points may still be duplicate in a few samples after deduplication, extra care for recons

            # TODO: try variation weighted by distance
            self.kd_trees = []
            for triplets_list in self.tri_ll:
                self.kd_trees.append(KDTree(triplets_list))
            construct_v_orders, variation_orders, variations = self.gen_variation()
            if mode.startswith("linear-"):
                self.__update_linear_build(construct_v_orders)
            elif mode.startswith("network-"):
                if mode == "network-local":
                    self.__update_network_build(construct_v_orders, variation_orders, variations)
        else:
            raise ValueError("invalid mode: {}".format(mode))

    def __update_network_build(self, construct_v_orders, variation_orders, variations):
        # not working well under current method
        # first plane
        # TODO
        raise NotImplementedError("iteratively updating the starting points under development")

        id_1 = construct_v_orders[0]
        id_2, id_3 = variation_orders[id_1][1:3]
        self._vertex_tree.set_root(id_1)
        self._append([_VectorHandler(id_1, id_2, ref_plane_triplet=None)])
        self.__to_id_update(id_1)
        self.__to_id_update(id_2)
        self._vertex_tree.add_link(id_1, id_2)
        initial_ref_plane_ids = (id_1, id_2, id_3)
        self._append([_VectorHandler(id_1, id_3, ref_plane_triplet=initial_ref_plane_ids),
                      _VectorHandler(id_2, id_3, ref_plane_triplet=initial_ref_plane_ids)])  # +
        self.__to_id_update(id_3)
        self._vertex_tree.add_link(id_1, id_3)
        self._vertex_tree.add_link(id_2, id_3)
        for initial_id in initial_ref_plane_ids:
            self._id_to_triplets[initial_id] = initial_ref_plane_ids

        # add more points
        # TODO: improve efficiency
        while self.__unfixed_vertex_ids:
            candidate_ids = {}
            for fixed_id in self.__fixed_vertex_ids:
                for ordered_id, variation in zip(variation_orders[fixed_id], variations[fixed_id]):
                    if ordered_id in self.__unfixed_vertex_ids:
                        if ordered_id not in candidate_ids:
                            candidate_ids[ordered_id] = (variation, fixed_id)
                        elif candidate_ids[ordered_id][0] > variation:
                            candidate_ids[ordered_id] = (variation, fixed_id)
            if not candidate_ids:
                raise NotImplementedError("not candidate ids")
                # # TODO: consider variation here too, use shortest distance to fixed ids in the first kd_tree for now
                # distances, indices = self.kd_trees[0].query(point, k=self.num_vs * 10)
            candidate_id_sorted = sorted(candidate_ids, key=lambda x: candidate_ids[x][0])
            to_id = candidate_id_sorted[0]
            supporting_ids = [spi for spi in variation_orders[to_id] if spi in self.__fixed_vertex_ids]
            min_num_sp = min(len(self.__fixed_vertex_ids), self.num_vs)
            if len(supporting_ids) < min_num_sp:
                # TODO: use all kd_trees later, use the first kd_tree for now
                # TODO: is the distance in order?
                distances, indices = self.kd_trees[0].query(self.tri_ll[0][to_id], k=self.len_vertices-1)
                indices = list(indices)
                supporting_ids_set = set(supporting_ids)
                while len(supporting_ids) < min_num_sp:
                    potential_id = indices.pop(0)
                    if potential_id not in supporting_ids_set and potential_id in self.__fixed_vertex_ids:
                        supporting_ids.append(potential_id)
            vh_list = []
            for from_id in supporting_ids[:min_num_sp]:
                self._vertex_tree.add_link(from_id, to_id)
                # TODO when reconstruct, if the plane is collapsed due to duplicate points, skip the vh
                vh_list.append(_VectorHandler(from_id, to_id, ref_plane_triplet=self._id_to_triplets[from_id]))
            self._append(vh_list)
            self.__to_id_update(to_id)
            self._id_to_triplets[to_id] = tuple([to_id] + supporting_ids[:2])

        # TODO going back to the start points

    def __update_linear_build(self, construct_v_orders):
        # start from the vertex
        id_1, id_2, id_3 = construct_v_orders[:3]
        self._vertex_tree.set_root(id_1)
        self._append([_VectorHandler(id_1, id_2, ref_plane_triplet=None)])  # +
        # [None] * (self.num_vs - 1))
        self.__to_id_update(id_1)
        self.__to_id_update(id_2)
        self._vertex_tree.add_link(id_1, id_2)
        initial_ref_plane_ids = (id_1, id_2, id_3)
        self._append([_VectorHandler(id_1, id_3, ref_plane_triplet=initial_ref_plane_ids),
                      _VectorHandler(id_2, id_3, ref_plane_triplet=initial_ref_plane_ids)])  # +
        # [None] * (self.num_vs - 2))
        self.__to_id_update(id_3)
        # self._vertex_tree.add_link(id_1, id_3)
        self._vertex_tree.add_link(id_2, id_3)
        for initial_id in initial_ref_plane_ids:
            self._id_to_triplets[initial_id] = initial_ref_plane_ids

        # add more points
        supporting_ids = list(initial_ref_plane_ids)
        for id_id, to_id in enumerate(construct_v_orders):
            if id_id < 3:
                continue
            vh_list = []
            for from_id in supporting_ids:
                # self._vertex_tree.add_link(from_id, to_id)
                # TODO when reconstruct, if the plane is collapsed due to duplicate points, skip the vh
                vh_list.append(_VectorHandler(from_id, to_id, ref_plane_triplet=self._id_to_triplets[from_id]))
            # vh_list += [None] * (self.num_vs - len(vh_list))  # to avoid making ragged array in downstream analysis
            self._append(vh_list)
            self.__to_id_update(to_id)
            self._vertex_tree.add_link(supporting_ids[-1], to_id)
            self._id_to_triplets[to_id] = tuple(construct_v_orders[id_id - 2: id_id + 1])
            supporting_ids = construct_v_orders[max(id_id + 1 - self.num_vs, 0): id_id + 1]
            # traced_back = self._vertex_tree.trace(from_id[-1], n_steps=2)
            # if len(traced_back) == 3:
            #     self._id_to_triplets[to_id] = traced_back
            # else:  # from_id belongs to (id_1, id_2)
            #     self._id_to_triplets[to_id] = self._id_to_triplets[from_id]

        # goes back to the starting points
        # for id_id, to_id in enumerate(construct_v_orders[:min(self.num_vs, len(construct_v_orders) - 1)]):
        supporting_ids = list(supporting_ids)
        for id_id, to_id in enumerate(construct_v_orders[:self.num_vs]):
            vh_list = []
            for from_id in supporting_ids:
                # self._vertex_tree.add_link(from_id, to_id)
                # TODO when reconstruct, if the plane is collapsed due to duplicate points, skip the vh
                vh_list.append(_VectorHandler(from_id, to_id, ref_plane_triplet=self._id_to_triplets[from_id]))
            self._append(vh_list)
            # rolling the supporting_ids
            supporting_ids.pop(0)
            supporting_ids.append(to_id)
            # updating the ref_plane of the num_vs starting points
            if to_id < 3:
                self._id_to_triplets[to_id] = tuple(supporting_ids[-3:])

    def __to_id_update(self, to_id):
        del self.__unfixed_vertex_ids[to_id]
        self.__fixed_vertex_ids[to_id] = None

    def gen_variation(self):
        """
        """
        # generate the variation statistics and find the order of the construction
        variation_orders = []  # used now and later
        variations = []  # used now and later
        top_k_var_sum = []  # used to find the start vertex with the smallest neighboring variation
        for go_v in range(self.len_vertices):
            indices_set = set()
            for go_sample, kd_tree in enumerate(self.kd_trees):
                distances, indices = kd_tree.query(self.tri_ll[go_sample][go_v],
                                                   k=min(self.num_vs * 10, self.len_vertices - 1))
                indices_set |= set(indices)
            indices_list = np.array(sorted(indices_set))
            distances_array = []
            for go_sample in range(self.len_samples):
                # TODO, cdist is doing pairwise, probably no need
                distances_array.append(cdist([self.tri_ll[go_sample][go_v]],
                                              self.tri_ll[go_sample][indices_list])[0])
            standard_dev = np.array(distances_array).std(axis=0)
            arg_sort_ids = standard_dev.argsort()
            variation_orders.append(indices_list[arg_sort_ids])
            variations.append(standard_dev[arg_sort_ids])
            top_k_var_sum.append(sum(variations[-1][:self.num_vs * 10]))  # Apr 21 just fix a bug
        top_k_var_sum = np.array(top_k_var_sum)
        construct_v_orders = top_k_var_sum.argsort()
        return construct_v_orders, variation_orders, variations

    def check_duplicates(self):
        """
        check if there are duplicated points for each samples

        """
        # TODO: allow missing but not allow duplicates
        across_sample_duplicates = find_duplicates_in_vertices_list(self.tri_ll)
        # check if number of across-sample duplicated-free points is sufficient
        logger.info("{} ouf of {} sample-wide unique points".format(
            self.len_vertices - len(across_sample_duplicates),
            self.len_vertices))
        assert len(across_sample_duplicates) == 0, "sample-wide duplicates exist!"


if __name__ == "__main__":

    # shapes is a ShapeAlignment
    ali = ShapeAlignment()

    # VertexVectorMapper... need to understand it.
    v_translator = VertexVectorMapper(
        [vt.coords for lb, vt in ali.shapes],
        mode="linear-variation",
        num_vs=10,
        num_vt_iter=5,
    )
