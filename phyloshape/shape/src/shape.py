#!/usr/bin/env python

"""Core PhyloShape class object of the phyloshape package.

"""
from loguru import logger
from plyfile import PlyData, PlyElement
from PIL import Image
import numpy as np
from copy import deepcopy
from typing import Union, List, Tuple
from numpy.typing import ArrayLike, NDArray
from phyloshape.shape.src.face import Faces
from phyloshape.shape.src.vertex import Vertices
from phyloshape.shape.src.network import IdNetwork
from phyloshape.utils import PSIOError, find_image_file, ID_TYPE, COORD_TYPE, RGB_TYPE
from phyloshape.utils.src.vertices_manipulator import find_duplicates_in_vertices_list
logger = logger.bind(name="phyloshape")


class Shape:
    """Shape class for manipulating shapes.

    """
    def __init__(self,
                 file_name: str = None,
                 texture_image_file: str = None):
        """Initialize a Shape from a file.

        The core object

        :param file_name: ply/obj
            PLY see https://pypi.org/project/plyfile/
            OBJ see ??
        :param texture_image_file: jpg/png/tif

        :return Shape object
        """
        self.label = self.file_name = str(file_name) if file_name else ""
        if texture_image_file:
            self.texture_image_file = texture_image_file
        elif self.file_name.endswith(".obj"):
            self.texture_image_file = find_image_file(file_name)
        else:
            self.texture_image_file = None
        self.vertices = Vertices()
        self.faces = Faces()
        self.network = IdNetwork()
        self.texture_image_obj = None
        self.texture_image_data = None
        if file_name:
            # TODO check the existence of files if applicable
            if file_name.endswith(".ply"):
                self.parse_ply()
                self.update_network()
            elif file_name.endswith(".obj"):
                self.parse_obj()
                self.update_network()
            else:
                raise TypeError("PhyloShape currently only support *.ply/*.obj files!")

    def __repr__(self):
        # TODO
        raise NotImplementedError("TODO")

    def __str__(self):
        # TODO
        raise NotImplementedError("TODO")

    def __eq__(self, other):
        # TODO
        raise NotImplementedError("TODO")

    def parse_ply(self, from_external_file: str = None):
        """
        :param from_external_file: optionally from outside file
        """
        file_name = from_external_file if from_external_file else self.file_name
        logger.trace("parsing {}".format(file_name))
        obj = PlyData.read(file_name)
        # read the coordinates
        vertex_coords = np.stack([obj["vertex"]["x"], obj["vertex"]["y"], obj["vertex"]["z"]], axis=1)
        try:
            # read the vertex_colors as rgb
            vertex_colors = np.stack([obj["vertex"]["red"], obj["vertex"]["green"], obj["vertex"]["blue"]], axis=1)
        except ValueError:
            vertex_colors = None
        # self.vertex_colors = rgb_to_hex(self.vertex_colors)
        self.vertices = Vertices(coords=vertex_coords, colors=vertex_colors)  # self.faces.vertices =
        # read the face indices
        self.faces.vertex_ids = np.array(np.vstack(obj["face"]["vertex_indices"]), dtype=ID_TYPE)
        logger.trace("parsing {} finished.".format(file_name))

    def parse_obj(self, from_external_file: str = None, from_external_image: str = None):
        file_name = from_external_file if from_external_file else self.file_name
        image_file = from_external_image if from_external_image else self.texture_image_file
        logger.trace("parsing {}".format(file_name))
        vertex_coords = []  # store vertices coordinates
        vertex_colors = []  # store vertices color
        texture_anchor_percent_coords = []  # store texture coordinates
        face_v_indices = []  # vertices index triplet
        face_t_indices = []  # texture index triplet
        with open(file_name) as input_handler:
            go_l = 0
            for line in input_handler:
                line = line.strip().split(" ")
                go_l += 1
                if line[0] == "v":
                    if len(line) == 4:
                        vertex_coords.append([float(i) for i in line[1:4]])
                        # vertex_colors.append([None] * 3)
                    elif len(line) == 7:
                        vertex_coords.append([float(i) for i in line[1:4]])
                        vertex_colors.append([float(i) for i in line[4:]])
                    else:
                        raise PSIOError("invalid line " + str(go_l) + " at " + self.file_name)
                elif line[0] == "vt":
                    texture_anchor_percent_coords.append([float(i) for i in line[1:3]])
                elif line[0] == "f":
                    this_v_indices = []
                    this_t_indices = []
                    for v_t_pair in line[1:]:
                        if "/" in v_t_pair:
                            v_, t_ = v_t_pair.split("/")
                            this_v_indices.append(int(v_))
                            this_t_indices.append(int(t_))
                        else:
                            this_v_indices.append(int(v_t_pair))
                    face_v_indices.append(this_v_indices)
                    face_t_indices.append(this_t_indices)
        # start with 1->0
        face_v_indices = np.array(face_v_indices, dtype=ID_TYPE) - 1
        face_t_indices = np.array(face_t_indices, dtype=ID_TYPE) - 1
        # read image obj
        if image_file:
            self.texture_image_obj = Image.open(image_file)
            self.texture_image_data = np.asarray(self.texture_image_obj)
        self.vertices = Vertices(coords=vertex_coords,
                                 # TODO check vertex_colors if it's None
                                 colors=np.round(np.array(vertex_colors) * 255) if vertex_colors else None)
        self.faces = Faces(vertex_ids=face_v_indices,
                           # vertices=self.vertices,
                           texture_ids=face_t_indices,
                           texture_anchor_percent_coords=texture_anchor_percent_coords,
                           texture_image_data=self.texture_image_data)
        logger.trace("parsing {} finished.".format(file_name))

    def update_network(self):
        logger.trace("constructing network")
        # generate the connection from edges of faces
        nw_pairs = np.unique(np.concatenate((self.faces[:, 0:2], self.faces[:, 1:3], self.faces[:, :3:2])), axis=0)
        # euclidean distance
        nw_weights = \
            np.sum((self.vertices.coords[nw_pairs[:, 1]] - self.vertices.coords[nw_pairs[:, 0]]) ** 2, axis=1) ** 0.5
        self.network = IdNetwork(pairs=nw_pairs, edge_lens=nw_weights)
        logger.trace("constructing network finished.")

    def extract_component(self, component_id: Union[slice, int] = 0):
        """Extract connected component(s) from a shape object

        The connected components were sorted decreasingly by the number of vertices,
        so that the first component (component_id=0) is the largest component.

        Parameters
        ----------
        component_id

        Returns
        -------
        new_shape
        """
        sorted_components = sorted(self.network.find_unions(), key=lambda x: (-len(x), x))
        if isinstance(component_id, int):
            chosen_v_ids = sorted(sorted_components[component_id])
        else:
            chosen_v_ids = sorted(set.union(*sorted_components[component_id]))
        new_shape = self.grab_a_piece(grab_v_ids=chosen_v_ids)
        return new_shape

    def grab_a_piece(self, grab_v_ids: NDArray[np.uint32]):
        """Extract a piece off the original mesh shape.

        Parameters
        ----------
        grab_v_ids: NDArray[np.uint32]
            Array of vertices ids in the new mesh shape.
        Returns
        -------
        new_shape
        """
        new_shape = Shape()
        # only keep the chosen vertex ids
        if self.vertices.colors is None or not len(self.vertices.colors):
            new_shape.vertices.coords = self.vertices[grab_v_ids]
        else:
            new_shape.vertices.coords, new_shape.vertices.colors = self.vertices[grab_v_ids]
        # deepcopy the original faces object
        new_shape.faces = deepcopy(self.faces)
        # find the faces that contains any unwanted vertex ids, and assign the ids to rm_f_ids, then delete them
        face_v_ids = new_shape.faces.vertex_ids
        rm_f_ids = np.isin(face_v_ids, grab_v_ids, invert=True).any(axis=1)
        face_v_ids = np.delete(face_v_ids, rm_f_ids, axis=0)
        # modify the v ids in the faces, because some vertices were removed
        # using the np.unique() approach here will be much faster than using np.vectorize(dict.__getitem__)()
        v_id_translator = {old_id: new_id for new_id, old_id in enumerate(grab_v_ids)}
        uniq, inv = np.unique(face_v_ids, return_inverse=True)
        new_shape.faces.vertex_ids = np.array([v_id_translator[o_i] for o_i in uniq])[inv].reshape(face_v_ids.shape)
        # also delete associated texture ids
        if len(new_shape.faces.texture_ids):
            new_shape.faces.texture_ids = np.delete(new_shape.faces.texture_ids, rm_f_ids, axis=0)
        # TODO: check other new_shape.faces.texture_* attributes
        #  may or may not in need of updating after extraction
        return new_shape


class ShapeAlignment:
    """Shape Alignment class for comparative analysis of Shapes

    Note: each shape must be single connected component TODO: node code to guarantee yet
    TODO: is there a standard file format to record shape alignments?
    """
    def __init__(self):
        self._vertices_list = []
        self.__n_vertices = None
        self.__labels = []
        self._label_to_sample_id = {}
        self.faces = Faces()
        # self.network = IdNetwork()

    def __deepcopy__(self, memodict={}):
        new_shapes = ShapeAlignment()
        for label, vertices in self:
            new_shapes.append(label=label, sample=deepcopy(vertices))
        new_shapes.faces = deepcopy(self.faces)
        return new_shapes

    def __contains__(self, item):
        if isinstance(item, str):
            return item in self._label_to_sample_id
        else:
            return item in self._vertices_list  # TODO Shape.__eq__()

    def __delitem__(self, item):
        if isinstance(item, str):
            del_id = self._label_to_sample_id[item]
            del self.__labels[del_id]
            del self._vertices_list[del_id]
            for lb in self.__labels[del_id:]:
                self._label_to_sample_id[lb] -= 1
        elif isinstance(item, slice):
            del self.__labels[item]
            del self._vertices_list[item]
            self._update_index()
        elif isinstance(item, int):
            del self.__labels[item]
            del self._vertices_list[item]
            for lb in self.__labels[item:]:
                self._label_to_sample_id[lb] -= 1
            self._update_index()
        else:
            raise TypeError(type(item))

    def __getitem__(self, item):
        if isinstance(item, str):
            return item, self._vertices_list[self._label_to_sample_id[item]]
        else:
            if isinstance(item, slice):
                return list(zip(self.__labels[item], self._vertices_list[item]))
            else:
                return self.__labels[item], self._vertices_list[item]

    def __iter__(self):
        for go_s, label in enumerate(self.__labels):
            yield label, self._vertices_list[go_s]

    def __len__(self):
        return len(self._vertices_list)

    def _update_index(self):
        self._label_to_sample_id = {}
        for go_s, label in enumerate(self.__labels):
            self._label_to_sample_id[label] = go_s

    def append(self, label: str, sample: Vertices):
        # check duplicate label
        assert label not in self._label_to_sample_id, "Label %s existed in the alignment!" % label
        self._label_to_sample_id[label] = len(self.__labels)  #TODO fix the bug by removing -1
        self.__labels.append(label)
        # check vertices shape
        if self.__n_vertices:
            assert len(sample) == self.__n_vertices, "Unmatched Vertices dimension!"
        else:
            if self.faces:
                assert len(sample) > np.amax(self.faces.vertex_ids), "Face ids out of range!"
            self.__n_vertices = len(sample)
        self._vertices_list.append(sample)

    def get_labels(self):
        return list(self.__labels)

    def iter_labels(self):
        for label in self.__labels:
            yield label

    def remove(self, labels):
        del_names = set(labels)
        go_to = 0
        while go_to < len(self.__labels):
            if self.__labels[go_to] in del_names:
                del_names.remove(self.__labels[go_to])
                del self._vertices_list[go_to]
                del self.__labels[go_to]
            else:
                go_to += 1
        self._update_index()
        if del_names:
            logger.warning("label(s) " + ",".join(sorted(del_names)) + " not found!\n")

    def del_vertices(self, item: Union[List, Tuple, int, slice]):
        if isinstance(item, (list, tuple, slice, int)):
            for vertices in self._vertices_list:
                del vertices[item]
        else:
            raise TypeError(type(item))

    def n_faces(self):
        return len(self.faces.vertex_ids)

    def n_vertices(self):
        return self.__n_vertices

    def n_samples(self):
        return len(self._vertices_list)

    def find_duplicate(self):
        across_sample_duplicates = find_duplicates_in_vertices_list([vts.coords for vts in self._vertices_list])
        logger.info("{} ouf of {} sample-wide unique points".format(
            self.__n_vertices - len(across_sample_duplicates),
            self.__n_vertices))
        return tuple(sorted(across_sample_duplicates))

    def deduplicate(self):
        if not self.faces:
            across_sample_duplicates = self.find_duplicate()
            self.del_vertices(list(across_sample_duplicates))
            self.__n_vertices -= len(across_sample_duplicates)
        else:
            # TODO
            logger.error("deduplication with faces were not implemented yet")

    def get_vertices_list(self):
        return list(self._vertices_list)




