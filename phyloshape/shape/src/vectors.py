#!/usr/bin/env python

"""Core PhyloShape class object of the phyloshape package.

"""

from typing import (Tuple, List)
from collections import OrderedDict
from copy import deepcopy
import random
import numpy as np
from numpy.typing import ArrayLike
from phyloshape.utils import trans_vector_to_relative, trans_vector_to_absolute
from loguru import logger
logger = logger.bind(name="phyloshape")


class VectorHandler:
    """
    Used in VertexVectorMapper.
    Unit class recording a single map between the vertices-face system and the absolute_vector system.

    """
    def __init__(
            self,
            from_id: int,
            to_id: int,
            from_face: Tuple[int, int, int] or None):
        self.from_id = from_id
        self.to_id = to_id
        self.from_face = from_face

    def __repr__(self):
        if self.from_face:
            face_with_id_marked = list(self.from_face)
            face_with_id_marked[self.from_face.index(self.from_id)] = [self.from_id]
        else:
            face_with_id_marked = [[self.from_id]]
        return f"{face_with_id_marked} -> {self.to_id}"

    def __str__(self):
        return self.__repr__()


class VertexTree:
    """
    Used in VertexVectorMapper and used for k3d plotting.
    Tree-like class recording the vertices connection map via the vectors.

    """
    def __init__(self, root_id: int = None):
        self.__vertex_map = {}
        self.__root = None
        if not (root_id is None):
            self.set_root(root_id)
        # self.vertex_id = vertex_id
        # self.children = []

    def set_root(self, vertex_id: int):
        self.__root = vertex_id
        self.__vertex_map = {vertex_id: []}

    #TODO how to set the type of vn as VectorNode, which is not defined yet
    def add_link(self, from_id, to_id):
        if from_id not in self.__vertex_map:
            self.__vertex_map[from_id] = []
        self.__vertex_map[from_id].append(to_id)

    def get_lines_for_k3d_plot(self, start_id: int = None):
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
        if start_id in self.__vertex_map:
            next_id = self.__vertex_map[start_id][0]
            path.append(next_id)
        else:
            raise TypeError("At least two points are required for a line!")
        # branching_node_visited and this_branch_path should be co-changed
        branching_node_visited = [[start_id, 0]]
        this_branch_path = [[start_id]]
        while branching_node_visited:
            go_id = path[-1]
            if go_id in self.__vertex_map:
                # elongation
                next_id = self.__vertex_map[go_id][0]
                path.append(next_id)
                # make a branch
                if len(self.__vertex_map[go_id]) > 1:
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
                while last_br == len(self.__vertex_map[last_node_id]) - 1:
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
                    next_id = self.__vertex_map[last_node_id][last_br + 1]
                    path.append(next_id)
                    branching_node_visited[-1][1] = last_br + 1
                    this_branch_path[-1] = [next_id]
        return path


class VertexVectorMapper:
    """
    Main Class recording all maps between the vertices-face system and the vector system

    """
    def __init__(self,
                 vertices_ids_in_faces: ArrayLike = None,
                 random_seed: int = 0):
        """

        Parameters
        ----------
        vertices_ids_in_faces:
            All faces must be from a single connected object.
        random_seed: int
            0: pre-order-like traverse
            -1: post-order-like traverse
            other values: random
        """
        self.__vh_list = []
        self.__fixed_vertex_ids = set()
        self.__waiting_triplets = OrderedDict()  # to remove random effect outside the random.choice
        self.__checking_triplet = None
        self.__checked_triplets = set()
        self.__id_to_triplets = {}
        self.__vertex_tree = VertexTree()  # for plotting
        #TODO I don't know why the random mode will be significantly smaller.
        # Should find out where it is and optimize it.
        self.random_seed = random_seed
        self.__update(vertices_ids_in_faces)

    def __update(
            self,
            triplets_list: ArrayLike or List[ArrayLike]):
        """
        Build the maps between the vertices-face system and the vector system.
        We do it with a single start point to ensure self.to_vertices works properly.

        Parameters
        ----------
        triplets_list: ArrayLike or List[ArrayLike]
            Must be faces from single connected object.
        """
        # # using a checking_sf_indices and a pending_sf_indices to make sure all surface indices are checked eventually
        # checking_sf_indices = OrderedDict([(tuple(f_points), None) for f_points in triplets_list])
        # pending_sf_indices = OrderedDict([])

        random.seed(self.random_seed)
        triplets_set = set([tuple(tl) for tl in triplets_list])
        len_triplets = len(triplets_set)
        logger.info(f"{len_triplets} triplets input.")

        for tri_ids in triplets_list:
            for single_id in tri_ids:
                if single_id not in self.__id_to_triplets:
                    self.__id_to_triplets[single_id] = set()
                    # a list of triplets will be used later to determine the the_reference_triplet:
                    #   the_reference_triplet = id_related_triplets & self.__checked_triplets
                    # so we have to use set here to store the id_related_triplets
                self.__id_to_triplets[single_id].add(tuple(tri_ids))
        logger.info(f"{len(self.__id_to_triplets)} vertices indexed.")

        # have the first point id fixed in space
        if self.random_seed not in {0, -1}:
            self.__checking_triplet = id_1, id_2, id_3 = random.choice(triplets_list)
        else:
            self.__checking_triplet = id_1, id_2, id_3 = triplets_list[0]
        self.__checking_triplet = tuple(self.__checking_triplet)
        self.__fixed_vertex_ids.add(id_1)
        self.__checked_triplets.add(self.__checking_triplet)
        for triplets in self.__id_to_triplets[id_1]:
            if triplets not in self.__checked_triplets:
                self.__waiting_triplets[triplets] = None
        self.__vertex_tree.set_root(id_1)
        # record the first and second vector
        # Different from downstream, the reference faces here are uniquely `None` and "the same face of the vector".
        self.__append(VectorHandler(id_1, id_2, None))
        self.__to_id_update(id_2)
        self.__vertex_tree.add_link(id_1, id_2)
        self.__append(VectorHandler(id_2, id_3, self.__checking_triplet))
        self.__to_id_update(id_3)
        self.__vertex_tree.add_link(id_1, id_3)
        # previous_triplet = this_triplet

        # start building
        # while len(self.__checked_triplets) < len_triplets:
        while self.__waiting_triplets:
            # other searching proposals may be applied if necessary
            if self.random_seed not in {0, -1}:
                this_triplet = id_1, id_2, id_3 = random.choice(list(self.__waiting_triplets.keys()))
                del self.__waiting_triplets[this_triplet]
            else:
                this_triplet = id_1, id_2, id_3 = self.__waiting_triplets.popitem(bool(self.random_seed))[0]

            if id_1 in self.__fixed_vertex_ids:
                if id_2 in self.__fixed_vertex_ids:
                    if id_3 in self.__fixed_vertex_ids:
                        pass
                    else:
                        self.__build_vector(id_1, id_3)
                else:
                    if id_3 in self.__fixed_vertex_ids:
                        self.__build_vector(id_1, id_2)
                    else:
                        self.__build_vector(id_1, id_2)
                        self.__build_vector(id_1, id_3)
            else:
                if id_2 in self.__fixed_vertex_ids:
                    if id_3 in self.__fixed_vertex_ids:
                        self.__build_vector(id_2, id_1)
                    else:
                        self.__build_vector(id_2, id_1)
                        self.__build_vector(id_2, id_3)
                else:
                    if id_3 in self.__fixed_vertex_ids:
                        self.__build_vector(id_3, id_2)
                        self.__build_vector(id_3, id_1)
                    else:
                        raise TypeError(f"({id_1}, {id_2}, {id_3})"
                                        "Isolated face is not allowed! "
                                        "All faces must be from a single connected object!")
            self.__checked_triplets.add(this_triplet)
            # previous_triplet = this_triplet
            # checking_sf_indices = list(pending_sf_indices)
            # pending_sf_indices = []
        logger.info(f"{len(self.__checked_triplets)} triplets checked.")

    def __build_vector(self, from_id, to_id):
        candidate_triplets = self.__id_to_triplets[from_id] & self.__checked_triplets
        if self.random_seed not in {0, -1}:
            reference_triplet = random.choice(sorted(candidate_triplets))
        else:
            reference_triplet = sorted(candidate_triplets)[0]
        self.__append(VectorHandler(from_id, to_id, reference_triplet))
        self.__vertex_tree.add_link(from_id, to_id)
        self.__to_id_update(to_id=to_id)

    def __to_id_update(self, to_id):
        self.__fixed_vertex_ids.add(to_id)
        for triplets in self.__id_to_triplets[to_id]:
            if triplets not in self.__checked_triplets and triplets != self.__checking_triplet:
                self.__waiting_triplets[triplets] = None

    def __append(self, vh: VectorHandler):
        self.__vh_list.append(vh)

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
        # initialize with the first & second vectors
        norm_v1 = np.linalg.norm(vertices[self.__vh_list[0].target_id] - vertices[self.__vh_list[0].from_id])
        vectors = [np.array([norm_v1, 0, 0])]
        # do the following vectors
        for vh in self.__vh_list[1:]:
            vector_in_space = vertices[vh.target_id] - vertices[vh.from_id]
            relative_vector = trans_vector_to_relative(vector_in_space, vertices[list(vh.from_face)])
            vectors.append(relative_vector)
        return np.array(vectors)

    def to_vertices(self, vectors) -> ArrayLike:
        assert len(vectors) == len(self.__vh_list), \
            "The length of the vectors must equals the length of absolute_vector handlers!"
        vertices = np.array([np.array([None, None, None])] * (len(vectors) + 1), dtype=np.float32)
        vertices[self.__vh_list[0].from_id] = [0., 0., 0.]
        vertices[self.__vh_list[0].target_id] = vectors[0]
        vertices[self.__vh_list[1].target_id] = vectors[0] + vectors[1]
        for go_vct, vh in enumerate(self.__vh_list[2:]):
            relative_vector = vectors[go_vct]
            vector_in_space = trans_vector_to_absolute(relative_vector, vertices[list(vh.from_face)])
            if np.isnan(vertices[vh.from_id]).any():
                raise ValueError(
                    f"While building Vtx {vh.target_id}, Vtx {vh.from_id} is invalid {vertices[vh.from_id]}!")
            else:
                vertices[vh.target_id] = vertices[vh.from_id] + vector_in_space
        return np.array(vertices)

    def vh_list(self):
        return deepcopy(self.__vh_list)

    def get_lines_for_k3d_plot(self):
        return self.__vertex_tree.get_lines_for_k3d_plot()
