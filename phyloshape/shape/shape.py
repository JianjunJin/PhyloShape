#!/usr/bin/env python

"""Core PhyloShape class object of the phyloshape package.

"""
import os.path

from loguru import logger
from plyfile import PlyData, PlyElement
from PIL import Image
import numpy as np
from phyloshape.utils import rgb_to_hex, PSIOError, find_image_file
logger = logger.bind(name="phyloshape")

INT_TYPE = np.uint32
FLOAT_TYPE = np.float32


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
        self.texture_image_file = texture_image_file if texture_image_file else find_image_file(file_name)
        self.vertex_coords = np.array([], dtype=FLOAT_TYPE)  # 3*l
        self.vertex_colors = np.array([], dtype=INT_TYPE)  # 3*l
        self.face_v_indices = np.array([], dtype=INT_TYPE)  # 3*m
        self.face_t_indices = np.array([], dtype=INT_TYPE)  # 3*m
        self.texture_coords = np.array([], dtype=FLOAT_TYPE)  # 2*n
        self.texture_image_obj = None
        if file_name:
            # TODO check the existence of files if applicable
            if file_name.endswith(".ply"):
                self.parse_ply()
            elif file_name.endswith(".obj"):
                self.parse_obj()
            else:
                raise TypeError("PhyloShape currently only support *.ply/*.obj files!")

    def parse_ply(self, from_external_file: str = None):
        """
        :param from_external_file: optionally from outside file
        :return:
        """
        obj = PlyData.read(self.file_name)
        # read the coordinates
        self.vertex_coords = np.stack([obj["vertex"]["x"], obj["vertex"]["y"], obj["vertex"]["z"]], axis=1)
        # read the vertex_colors as rgb, then convert it into hex
        self.vertex_colors = np.stack([obj["vertex"]["red"], obj["vertex"]["green"], obj["vertex"]["blue"]], axis=1)
        self.vertex_colors = rgb_to_hex(self.vertex_colors)
        # read the face indices
        self.face_v_indices = np.array(np.vstack(obj["face"]["vertex_indices"]), dtype=INT_TYPE)

    def parse_obj(self, from_external_file: str = None, from_external_image: str = None):
        file_name = from_external_file if from_external_file else self.file_name
        image_file = from_external_image if from_external_image else self.texture_image_file
        vertex_coords = []  # store vertex coordinates
        vertex_colors = []  # store vertex color
        texture_coords = []  # store texture coordinates
        face_v_indices = []  # vertex index triplet
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
                    texture_coords.append([float(i) for i in line[1:3]])
                elif line[0] == "f":
                    this_v_indices = []
                    this_t_indices = []
                    for v_t_pair in line[1:]:
                        v_, t_ = v_t_pair.split("/")
                        this_v_indices.append(int(v_))
                        this_t_indices.append(int(t_))
                    face_v_indices.append(this_v_indices)
                    face_t_indices.append(this_t_indices)
        if image_file:
            self.texture_image_obj = Image.open(image_file)
        self.vertex_coords = np.array(vertex_coords, dtype=FLOAT_TYPE)
        self.vertex_colors = np.array(vertex_colors, dtype=FLOAT_TYPE)
        self.face_v_indices = np.array(face_v_indices, dtype=INT_TYPE)
        self.face_t_indices = np.array(face_t_indices, dtype=INT_TYPE)
        self.texture_coords = np.array(texture_coords, dtype=FLOAT_TYPE)

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

