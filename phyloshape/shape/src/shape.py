#!/usr/bin/env python

"""Core PhyloShape class object of the phyloshape package.

"""
from typing import Optional, Union
from copy import deepcopy
from pathlib import Path
from loguru import logger
from plyfile import PlyData, PlyElement
from PIL import Image
import numpy as np
from phyloshape.shape.src.face import Faces
from phyloshape.shape.src.vertex import Vertices
from phyloshape.shape.src.network import IdNetwork
from phyloshape.utils import PSIOError, find_image_path, ID_TYPE, COORD_TYPE, RGB_TYPE

logger = logger.bind(name="phyloshape")


class Shape:
    """Shape class for manipulating shapes.

    Parameters
    ----------
    file_name: str or Path
        Filepath to a .obj or .ply 3D object.
    texture_image_file: str, Path, or None
        Optional image file with textures for the 3D object faces.
    """

    def __init__(self, file_name: str, texture_image_file: Optional[str] = None):

        # load the object and image files.
        self.opath = self._get_object_path(file_name)
        """: Path of the input 3D object file (.ply or .obj)."""
        self.tpath = self._get_image_path(texture_image_file)
        """: Path of the input image/texture file."""

        # init data objects empty
        self.vertices: Vertices = Vertices()
        """: phyloshape.Vertices object containing vertex data."""
        self.faces: Faces = Faces()
        """: phyloshape.Faces object containing faces data."""
        self.network: IdNetwork = IdNetwork()
        """: phyloshape.IdNetwork object containing network data."""
        self.texture_image_obj: Image = None
        """: PIL.Image object parsed from the texture file."""
        self.texture_image_data: np.ndarray = None
        """: Image data stored as an array."""

        # parse the object file and update the network.
        if not self.opath.exists():
            raise IOError(f"file {self.opath} does not exist.")
        if self.opath.suffix == ".ply":
            self.parse_ply()
            self.__update_network()
        elif self.opath.suffix == ".obj":
            self.parse_obj()
            self.__update_network()
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

    def _get_object_path(self, fname: Union[str, Path, None]) -> Path:
        """Return a path to a 3D object file.

        Parameters
        ----------
        fname: str, Path
            Path to a PLY or OBJ file.

        If a fname is entered then it is returned. If fname is None
        then the current object path of self is returned. If self
        does not have a current object path then an exception is
        raised.
        """
        if fname is None:
            if self.opath.exists():
                return self.opath
            raise IOError("You must enter a path to a 3D object file.")
        return Path(fname)

    def _get_image_path(self, fname: Union[str, Path, None]) -> Union[Path, None]:
        """Return a path to a image/texture file.

        Parameters
        ----------
        fname: str, Path
            Path to a JPG, TIF, PNG, etc. type file.

        If a fname is entered then it is returned. If fname is None
        then the current texture path of self is returned. If self
        does not have a current texture path then one is searched for
        with the same pathname as the object file, but allowing for
        flexible suffices. Finally, if no file is found then a warning
        is logged but we proceed and return None.
        """
        if fname is None:
            if self.tpath.exists():
                return self.tpath
            return find_image_path(self.opath)
        return Path(fname)

    def parse_ply(self, from_external_file: Optional[str] = None) -> None:
        """Parse vertex data from a PLY formatted 3D model file.

        :param from_external_file: optionally from outside file
        """
        # get path to new object file, or use existing self.opath.
        opath = self._get_object_path(from_external_file)
        logger.trace(f"parsing {opath}")

        # parse PLY object from the file.
        obj = PlyData.read(opath)

        # read the coordinates as an array
        vertex_coords = np.stack(
            [obj["vertex"]["x"], obj["vertex"]["y"], obj["vertex"]["z"]], axis=1
        )

        # read the vertex_colors as an rgb array
        vertex_colors = np.stack(
            [obj["vertex"]["red"], obj["vertex"]["green"], obj["vertex"]["blue"]],
            axis=1,
        )
        # self.vertex_colors = rgb_to_hex(self.vertex_colors)
        self.vertices = Vertices(coords=vertex_coords, colors=vertex_colors)
        # self.faces.vertices

        # read the face indices
        self.faces.vertex_ids = np.array(
            np.vstack(obj["face"]["vertex_indices"]), dtype=ID_TYPE
        )
        logger.trace(f"parsing {opath} finished.")

    def parse_obj(
        self,
        from_external_file: Optional[str] = None,
        from_external_image: Optional[str] = None,
        ):
        """Parse an OBJ file to fill the Shape object data.

        OBJ files contain tabular data formatted into different
        sections corresponding to different data elements of a 3D
        model object: v=vertices, vt=..., f=faces.

        Parameters
        ----------
        from_external_file:
            ...
        from_external_image: Optional
            ...
        """
        # get paths to the object and image files.
        opath = self._get_object_path(from_external_file)
        tpath = self._get_image_path(from_external_image)
        logger.trace(f"parsing {opath}")

        # store data from the files.
        vertex_coords = []  # store vertices coordinates
        vertex_colors = []  # store vertices color
        texture_anchor_percent_coords = []  # store texture coordinates
        face_v_indices = []  # vertices index triplet
        face_t_indices = []  # texture index triplet

        # parse file line by line
        with open(opath, "r", encoding="utf-8") as input_handler:
            go_l = 0
            for line in input_handler:
                line = line.strip().split(" ")
                go_l += 1

                # v = ...
                if line[0] == "v":
                    if len(line) == 4:
                        vertex_coords.append([float(i) for i in line[1:4]])
                        vertex_colors.append([None] * 3)
                    elif len(line) == 7:
                        vertex_coords.append([float(i) for i in line[1:4]])
                        vertex_colors.append([float(i) for i in line[4:]])
                    else:
                        raise PSIOError(f"invalid line {go_l} at {opath}.")

                # vt = vertex ...
                elif line[0] == "vt":
                    texture_anchor_percent_coords.append([float(i) for i in line[1:3]])

                # f = face
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

        # make the arrays zero-indexed instead of 1-indexed
        face_v_indices = np.array(face_v_indices, dtype=ID_TYPE) - 1
        face_t_indices = np.array(face_t_indices, dtype=ID_TYPE) - 1

        # read image obj and update the image data stored to self.
        if tpath:
            self.texture_image_obj = Image.open(tpath)
            self.texture_image_data = np.asarray(self.texture_image_obj)

        # load Vertices object to store coordinates and color 
        self.vertices = Vertices(
            coords=vertex_coords,
            # TODO check vertex_colors if it's None
            colors=np.round(np.array(vertex_colors) * 255),
        )
        self.faces = Faces(
            vertex_ids=face_v_indices,
            # vertices=self.vertices,
            texture_ids=face_t_indices,
            texture_anchor_percent_coords=texture_anchor_percent_coords,
            texture_image_data=self.texture_image_data,
        )
        logger.trace(f"parsing {opath} finished.")

    def __update_network(self):
        """Hidden function to update the .network ..."""
        logger.trace("constructing network")
        # generate the connection from edges of faces
        nw_pairs = np.unique(
            np.concatenate(
                (self.faces[:, 0:2], self.faces[:, 1:3], self.faces[:, :3:2])
            ),
            axis=0,
        )
        # euclidean distance
        nw_weights = (
            np.sum(
                (
                    self.vertices.coords[nw_pairs[:, 1]]
                    - self.vertices.coords[nw_pairs[:, 0]]
                )
                ** 2,
                axis=1,
            )
            ** 0.5
        )
        self.network = IdNetwork(pairs=nw_pairs, edge_lens=nw_weights)
        logger.trace("constructing network finished.")

    def extract_component(self, component_id: Union[slice, int] = 0):
        """Extract connected component(s) from a shape object

        The connected components were sorted decreasingly by the number of vertices,
        so that the first component (component_id=0) is the largest component.

        :param component_id:
        :return: Shape
        """
        sorted_components = sorted(
            self.network.find_unions(), key=lambda x: (-len(x), x)
        )
        if isinstance(component_id, int):
            chosen_v_ids = sorted(sorted_components[component_id])
        else:
            chosen_v_ids = sorted(set.union(*sorted_components[component_id]))
        
        # create a new Shape object ...
        new_shape = Shape()
        # only keep the chosen vertex ids
        new_shape.vertices.coords, new_shape.vertices.colors = self.vertices[
            chosen_v_ids
        ]
        # deepcopy the original faces object
        new_shape.faces = deepcopy(self.faces)
        # find the faces that contains any unwanted vertex ids, and assign the ids to rm_f_ids, then delete them
        face_v_ids = new_shape.faces.vertex_ids
        rm_f_ids = np.isin(face_v_ids, chosen_v_ids, invert=True).any(axis=1)
        face_v_ids = np.delete(face_v_ids, rm_f_ids, axis=0)
        # modify the v ids in the faces, because some vertices were removed
        # using the np.unique() approach here will be much faster than using np.vectorize(dict.__getitem__)()
        v_id_translator = {old_id: new_id for new_id, old_id in enumerate(chosen_v_ids)}
        uniq, inv = np.unique(face_v_ids, return_inverse=True)
        new_shape.faces.vertex_ids = np.array([v_id_translator[o_i] for o_i in uniq])[
            inv
        ].reshape(face_v_ids.shape)
        # also delete associated texture ids
        new_shape.faces.texture_ids = np.delete(
            new_shape.faces.texture_ids, rm_f_ids, axis=0
        )
        # TODO: check other new_shape.faces.texture_* attributes
        #  may or may not in need of updating after extraction
        return new_shape


class ShapeAlignment:
    """Shape Alignment class for comparative analysis of Shapes

    Note: each shape must be single connected component TODO: node code to guarantee yet
    TODO: is there a standard file format to record shape alignments?
    """

    def __init__(self):
        self.__vertices_list = []
        self.__n_vertices = None
        self.__labels = []
        self.__label_to_sample_id = {}
        self.faces = Faces()
        # self.network = IdNetwork()

    def __contains__(self, item):
        if isinstance(item, str):
            return item in self.__label_to_sample_id
        else:
            return item in self.__vertices_list  # TODO Shape.__eq__()

    def __delitem__(self, item):
        if isinstance(item, str):
            del_id = self.__label_to_sample_id[item]
            del self.__labels[del_id]
            del self.__vertices_list[del_id]
            for lb in self.__labels[del_id:]:
                self.__label_to_sample_id[lb] -= 1
        elif isinstance(item, slice):
            del self.__labels[item]
            del self.__vertices_list[item]
            self._update_index()
        elif isinstance(item, int):
            del self.__labels[item]
            del self.__vertices_list[item]
            for lb in self.__labels[item:]:
                self.__label_to_sample_id[lb] -= 1
            self._update_index()
        else:
            raise TypeError(type(item))

    def __getitem__(self, item):
        if isinstance(item, str):
            return item, self.__vertices_list[self.__label_to_sample_id[item]]
        else:
            return list(zip(self.__labels[item], self.__vertices_list[item]))

    def __iter__(self):
        for go_s, label in enumerate(self.__labels):
            yield label, self.__vertices_list[go_s]

    def __len__(self):
        return len(self.__vertices_list)

    def _update_index(self):
        self.__label_to_sample_id = {}
        for go_s, label in enumerate(self.__labels):
            self.__label_to_sample_id[label] = go_s

    def append(self, label: str, sample: Vertices):
        # check duplicate label
        assert label not in self.__label_to_sample_id, (
            "Label %s existed in the alignment!" % label
        )
        self.__label_to_sample_id[label] = len(self.__labels) - 1
        self.__labels.append(label)
        # check vertices shape
        if self.__n_vertices:
            assert len(Vertices) == self.__n_vertices, "Unmatched Vertices dimension!"
        else:
            self.__n_vertices = len(Vertices)
        self.__vertices_list.append(sample)

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
                del self.__vertices_list[go_to]
                del self.__labels[go_to]
            else:
                go_to += 1
        self._update_index()
        if del_names:
            logger.warning("label(s) " + ",".join(sorted(del_names)) + " not found!\n")

    def n_faces(self):
        return len(self.faces.vertex_ids)

    def n_vertices(self):
        return self.__n_vertices

    def n_samples(self):
        return len(self.__vertices_list)


if __name__ == "__main__":

    MODEL_FILE = "..."
    IMAGE_FILE = "..."

    shape_ = Shape(MODEL_FILE, IMAGE_FILE)
    print(shape_)
