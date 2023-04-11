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
from loguru import logger
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
            face_with_id_marked[self.ref_plane.index(self.from_id)] = [self.from_id]
        else:
            face_with_id_marked = [[self.from_id]]
        return f"{face_with_id_marked} -> {self.to_id}"

    def __str__(self):
        return self.__repr__()


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


class VMapper:
    """
    Base Class recording all maps between the coordination system and the vector system
    """

    def __init__(self, random_seed: int = 0):
        self._vh_list = []
        self._id_to_triplets = {}
        self._vertex_tree = _VertexTree()  # for plotting
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
            "The length of the vectors ({}) must equals the length of vertices handlers ({})!".format(
                len(vectors), len(self._vh_list))
        vertices = np.array([np.array([None, None, None])] * (len(vectors) + 1), dtype=np.float32)
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
        return np.array(vertices)

    def vh_list(self):
        return deepcopy(self._vh_list)

    def get_lines_for_k3d_plot(self):
        return self._vertex_tree.get_lines_for_k3d_plot()


class FaceVectorMapper(VMapper):
    """
    Main Class recording all maps between the face-vertices system and the vector system

    """
    def __init__(self, input_obj, random_seed: int = 0):
        """

        Parameters
        ----------
        input_obj:
            ArrayLike or Faces (with vertex_ids) or Shape (with Faces.vertex_ids).
            All faces must be from a single connected object.
        random_seed: int
            0: pre-order-like traverse
            -1: post-order-like traverse
            other values: random
        """
        super().__init__(random_seed)
        # self._vh_list = []
        self.__fixed_vertex_ids = set()
        self.__waiting_triplets = OrderedDict()  # to remove random effect outside the random.choice
        self.__checking_triplet = None
        self.__checked_triplets = set()
        # self._id_to_triplets = {}
        # self._vertex_tree = _VertexTree()  # for plotting
        # self.random_seed = random_seed
        # if is Shape
        if "faces" in dir(input_obj) and "vertex_ids" in dir(input_obj.faces):
            vts = input_obj.faces.vertex_ids
        # if is Faces
        elif "vertex_ids" in dir(input_obj):
            vts = input_obj.vertex_ids
        else:
            vts = input_obj
        self.__update(vts)

    def __update(
            self,
            triplets_list: ArrayLike or List[ArrayLike]):
        """
        Build the maps between the face-vertices system and the vector system.
        We do it with a single start point to ensure self.to_vertices works properly.

        Parameters
        ----------
        triplets_list: ArrayLike or List[ArrayLike]
            Must be faces from single connected object.
        """
        random.seed(self.random_seed)
        triplets_set = set([tuple(tl) for tl in triplets_list])
        len_triplets = len(triplets_set)
        logger.info(f"{len_triplets} triplets input.")

        for tri_ids in triplets_list:
            for single_id in tri_ids:
                if single_id not in self._id_to_triplets:
                    self._id_to_triplets[single_id] = set()
                    # a list of triplets will be used later to determine the the_reference_triplet:
                    #   the_reference_triplet = id_related_triplets & self.__checked_triplets
                    # so we have to use set here to store the id_related_triplets
                self._id_to_triplets[single_id].add(tuple(tri_ids))
        logger.info(f"{len(self._id_to_triplets)} vertices indexed.")

        # have the first point id fixed in space
        if self.random_seed not in {0, -1}:
            # TODO I don't know why the random mode will be significantly smaller.
            #      Should find out where it is and optimize it.
            self.__checking_triplet = id_1, id_2, id_3 = random.choice(triplets_list)
        else:
            self.__checking_triplet = id_1, id_2, id_3 = triplets_list[0]
        self.__checking_triplet = tuple(self.__checking_triplet)
        self.__fixed_vertex_ids.add(id_1)
        self.__checked_triplets.add(self.__checking_triplet)
        for triplets in self._id_to_triplets[id_1]:
            if triplets not in self.__checked_triplets:
                self.__waiting_triplets[triplets] = None
        self._vertex_tree.set_root(id_1)
        # record the first and second vector
        # Different from downstream, the reference faces here are uniquely `None` and "the same face of the vector".
        self._append(_VectorHandler(id_1, id_2, ref_plane_triplet=None))
        self.__to_id_update(id_2)
        self._vertex_tree.add_link(id_1, id_2)
        self._append(_VectorHandler(id_2, id_3, ref_plane_triplet=self.__checking_triplet))
        self.__to_id_update(id_3)
        self._vertex_tree.add_link(id_1, id_3)
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
        logger.info(f"{len(self.__checked_triplets)} triplets checked.")

    def __build_vector(self, from_id, to_id):
        candidate_triplets = self._id_to_triplets[from_id] & self.__checked_triplets
        if self.random_seed not in {0, -1}:
            reference_triplet = random.choice(sorted(candidate_triplets))
        else:
            reference_triplet = sorted(candidate_triplets)[0]
        self._append(_VectorHandler(from_id, to_id, reference_triplet))
        self._vertex_tree.add_link(from_id, to_id)
        self.__to_id_update(to_id=to_id)

    def __to_id_update(self, to_id):
        self.__fixed_vertex_ids.add(to_id)
        for triplets in self._id_to_triplets[to_id]:
            if triplets not in self.__checked_triplets and triplets != self.__checking_triplet:
                self.__waiting_triplets[triplets] = None

    # def _append(self, vh: _VectorHandler):
    #     self._vh_list.append(vh)
    #
    # def to_vectors(self, vertices) -> ArrayLike:
    #     """Based on the mapping information, it takes vertices of a shape to generate representative vectors
    #
    #     Parameters
    #     ----------
    #     vertices:
    #         Array of triangle vertices: float (x, y, z) coordinate triplets.
    #
    #     Returns
    #     -------
    #     vectors
    #         ArrayLike
    #     """
    #     # initialize with the first vectors
    #     vh_first = self._vh_list[0]
    #     space_v1 = vertices[vh_first.to_id] - vertices[vh_first.from_id]
    #     norm_v1 = np.linalg.norm(space_v1)
    #     relative_v1 = np.array([norm_v1, 0, 0])
    #     vectors = [relative_v1]
    #
    #     # initialize with the second vectors
    #     # vh_next = self._vh_list[1]
    #     # space_v2 = vertices[vh_next.to_id] - vertices[vh_first.from_id]
    #     # norm_v2 = np.linalg.norm(space_v2)
    #     # dot_product_v12 = sum(space_v1 * space_v2)
    #     # cos_theta2 = dot_product_v12 / (norm_v1 * norm_v2)
    #     # sin_theta2 = (1 - cos_theta2 ** 2) ** 0.5
    #     # vectors.append(np.array([cos_theta2 * norm_v2, sin_theta2 * norm_v2, 0]) - relative_v1)
    #
    #     # do the following vectors
    #     for vh in self._vh_list[1:]:
    #         # logger.trace("vector handler: {}".format(vh))
    #         vector_in_space = vertices[vh.to_id] - vertices[vh.from_id]
    #         # logger.trace("vector_in_space: %s" % str(vector_in_space))
    #         # logger.trace("face points: {}".format(vertices[list(vh.ref_plane_triplet)]))
    #         relative_vector = trans_vector_to_relative(vector_in_space, vertices[list(vh.ref_plane)])
    #         # logger.trace("relative vector: {}".format(relative_vector))
    #         vectors.append(relative_vector)
    #     return np.array(vectors)
    #
    # def to_vertices(self, vectors) -> ArrayLike:
    #     assert len(vectors) == len(self._vh_list), \
    #         "The length of the vectors ({}) must equals the length of vertices handlers ({})!".format(
    #             len(vectors), len(self._vh_list))
    #     vertices = np.array([np.array([None, None, None])] * (len(vectors) + 1), dtype=np.float32)
    #     vertices[self._vh_list[0].from_id] = [0., 0., 0.]
    #     vertices[self._vh_list[0].to_id] = vectors[0]
    #     vertices[self._vh_list[1].to_id] = vectors[0] + vectors[1]
    #     for go_vct, vh in enumerate(self._vh_list[2:]):
    #         relative_vector = vectors[go_vct + 2]
    #         vector_in_space = trans_vector_to_absolute(relative_vector, vertices[list(vh.ref_plane)])
    #         if np.isnan(vertices[vh.from_id]).any():
    #             raise ValueError(
    #                 f"While building Vtx {vh.to_id}, Vtx {vh.from_id} is invalid {vertices[vh.from_id]}!")
    #         else:
    #             vertices[vh.to_id] = vertices[vh.from_id] + vector_in_space
    #         # logger.trace("goid:{}, vh:{}, vertices[vh.from_id]:{}, vertices[vh.to_id:{}]:{}".
    #         #              format(go_vct, vh, vertices[vh.from_id], vh.to_id, vertices[vh.to_id]))
    #     return np.array(vertices)
    #
    # def vh_list(self):
    #     return deepcopy(self._vh_list)
    #
    # def get_lines_for_k3d_plot(self):
    #     return self._vertex_tree.get_lines_for_k3d_plot()

# TODO: Vectors for recording points?


class VertexVectorMapper(VMapper):
    """
    Main Class recording all maps between the dispersed vertices system and the vector system

    """
    def __init__(self, input_obj, random_seed: int = 0):
        """
        Parameters
        ----------
        input_obj:
            ArrayLike or Vertices (with coords).
            All faces must be from a single connected object.
        random_seed: int
            0: closest-neighbor traverse
            other values: random
        """
        super().__init__(random_seed)
        # self._vh_list = []
        self.__fixed_vertex_ids = OrderedDict()
        self.__unfixed_vertex_ids = OrderedDict()
        # self._id_to_triplets = {}
        # self._vertex_tree = _VertexTree()  # for plotting
        # self.random_seed = random_seed
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
        # TODO do not use duplicated points as the reference face, both in rooting and extending
        self.__unfixed_vertex_ids = OrderedDict([(_id, None) for _id in range(len(triplets_list))])
        triplet_set = set([tuple(triplet_) for triplet_ in triplets_list])
        assert len(triplet_set) >= 3, "Insufficient valid points!"
        if self.random_seed != 0:
            # TODO I don't know why the random mode will be significantly smaller.
            #      Should find out where it is and optimize it.
            random.seed(self.random_seed)
        else:
            # TODO: use k-d tree to speed up
            # TODO: without k-d tree, scipy.spatial.distance.pdist+squareform may be a faster alternative
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
                    traced_back = self._vertex_tree.trace(from_id, n_steps=2)
                    if len(traced_back) == 3:
                        self._id_to_triplets[to_id] = traced_back
                    else:  # from_id belongs to (id_1, id_2)
                        self._id_to_triplets[to_id] = self._id_to_triplets[from_id]

    # def _append(self, vh: _VectorHandler):
    #     self._vh_list.append(vh)

    def __to_id_update(self, to_id):
        del self.__unfixed_vertex_ids[to_id]
        self.__fixed_vertex_ids[to_id] = None


