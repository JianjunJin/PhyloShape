#!/usr/bin/env python

"""Core PhyloShape class object of the phyloshape package.

"""
from loguru import logger
from plyfile import PlyData, PlyElement
from PIL import Image
import numpy as np
from phyloshape.shape.src.face import Faces
from phyloshape.shape.src.vertex import Vertices
from phyloshape.shape.src.network import IdNetwork
from phyloshape.utils import PSIOError, find_image_file, ID_TYPE, COORD_TYPE, RGB_TYPE
logger = logger.bind(name="phyloshape")


class Shape:
    """Shape class for manipulating shapes.

    Parameters
    ----------
    file_name
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
        self.file_name = file_name
        if texture_image_file:
            self.texture_image_file = texture_image_file
        elif file_name.endswith(".obj"):
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
                self.__update_network()
            elif file_name.endswith(".obj"):
                self.parse_obj()
                self.__update_network()
            else:
                raise TypeError("PhyloShape currently only support *.ply/*.obj files!")

    def parse_ply(self, from_external_file: str = None):
        """
        :param from_external_file: optionally from outside file
        """
        file_name = from_external_file if from_external_file else self.file_name
        obj = PlyData.read(file_name)
        # read the coordinates
        vertex_coords = np.stack([obj["vertex"]["x"], obj["vertex"]["y"], obj["vertex"]["z"]], axis=1)
        # read the vertex_colors as rgb
        vertex_colors = np.stack([obj["vertex"]["red"], obj["vertex"]["green"], obj["vertex"]["blue"]], axis=1)
        # self.vertex_colors = rgb_to_hex(self.vertex_colors)
        self.vertices = self.faces.vertices = Vertices(coords=vertex_coords, colors=vertex_colors)
        # read the face indices
        self.faces.vertex_ids = np.array(np.vstack(obj["face"]["vertex_indices"]), dtype=ID_TYPE)

    def parse_obj(self, from_external_file: str = None, from_external_image: str = None):
        file_name = from_external_file if from_external_file else self.file_name
        image_file = from_external_image if from_external_image else self.texture_image_file
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
                        vertex_colors.append([None] * 3)
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
                        v_, t_ = v_t_pair.split("/")
                        this_v_indices.append(int(v_))
                        this_t_indices.append(int(t_))
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
                                 colors=np.round(np.array(vertex_colors) * 255))
        self.faces = Faces(vertex_ids=face_v_indices,
                           vertices=self.vertices,
                           texture_ids=face_t_indices,
                           texture_anchor_percent_coords=texture_anchor_percent_coords,
                           texture_image_data=self.texture_image_data)

    def __update_network(self):
        # generate the connection from edges of faces
        nw_pairs = np.unique(np.concatenate((self.faces[:, 0:2], self.faces[:, 1:3], self.faces[:, :3:2])), axis=0)
        # euclidean distance
        nw_weights = np.sum((self.vertices[nw_pairs[:, 1]] - self.vertices[nw_pairs[:, 0]]) ** 2, axis=1) ** 0.5
        self.network = IdNetwork(pairs=nw_pairs, edge_lens=nw_weights)

    # #TODO multiple objects
    # def update_vertex_clusters(self):
    #     self.vertex_clusters = []
    #     vertices = sorted(self.vertex_info)
    #     for this_vertex in vertices:
    #         connecting_those = set()
    #         for connected_set in self.vertex_info[this_vertex].connections.values():
    #             for next_v, next_d in connected_set:
    #                 for go_to_set, cluster in enumerate(self.vertex_clusters):
    #                     if next_v in cluster:
    #                         connecting_those.add(go_to_set)
    #         if not connecting_those:
    #             self.vertex_clusters.append({this_vertex})
    #         elif len(connecting_those) == 1:
    #             self.vertex_clusters[connecting_those.pop()].add(this_vertex)
    #         else:
    #             sorted_those = sorted(connecting_those, reverse=True)
    #             self.vertex_clusters[sorted_those[-1]].add(this_vertex)
    #             for go_to_set in sorted_those[:-1]:
    #                 for that_vertex in self.vertex_clusters[go_to_set]:
    #                     self.vertex_clusters[sorted_those[-1]].add(that_vertex)
    #                 del self.vertex_clusters[go_to_set]
