#!/usr/bin/env python

"""Shape class is the main 3D model object in phyloshape.

A Shape class instance should be loaded from a file (OBJ or PLY) or
from an array of landmark coordinates. Depending on how it is constructed
it can vary in the type of data available. For example, some models have
only vertex info, while others have faces (and thus also edges), and
others have color or texture data as well.

Methods
-------
>>> # parse Shape from data
>>> shape = phyloshape.io.load_shape_from_coordinates(...)
>>> print(shape)
>>> # ...

>>> # generate an interactive k3d plot
>>> shape.draw()

>>> # measure shortest distance between two vertices
>>> shape.distance.get_shortest_path()
>>> shape.distance.get_shortest_paths_from(cutoff=)

"""

from __future__ import annotations
from typing import Union, Optional
from copy import deepcopy

from loguru import logger
import numpy as np
import k3d
from phyloshape.shape.src.face import Faces
from phyloshape.shape.src.vertex import Vertices
# from phyloshape.shape.src.network import Network
# from phyloshape.utils import PSIOError, find_image_path, ID_TYPE, COORD_TYPE, RGB_TYPE

logger = logger.bind(name="phyloshape")


class Shape:
    """Shape class container for 3D model data.

    A shape class is flexible for supporting 3D data in a variety of
    formats, from only points (landmarks) to a full mesh containing
    points, lines, and polygons.

    Note
    ----
    A Shape object should be initialized by parsing data from a 3D
    file using the function `phyloshape.io()...`
    """
    def __init__(self, vertices: Vertices, faces: Faces = None):

        # data objects filled by `parse_object_file()`
        self.vertices: Vertices = vertices
        """: phyloshape.Vertices object containing vertex data."""
        self.faces: Faces = faces
        """: phyloshape Faces object containing faces data."""

        # filled by _update_network call
        self.network: Network = None
        """: phyloshape.IdNetwork object containing network data."""

        # ...
        if self.faces:
            self._update_network()

    def __repr__(self):
        nverts = self.vertices.coords.shape[0]
        nfaces = 0 if self.faces is None else self.faces.vertex_ids.shape[0]
        return f"Shape(nverts={nverts}, nfaces={nfaces})"

    def __eq__(self, other: Shape) -> bool:
        raise NotImplementedError("TODO")

    def _update_network(self):
        """Private function to update the network."""
        logger.trace("constructing network")

        # generate all edges from the connections in faces
        edge_dim = self.faces.vertex_ids.shape[1] - 1
        edges = [self.faces[:, i:(i + 2)] for i in range(edge_dim)]
        edges = np.unique(np.concatenate(edges), axis=0)

        # euclidean distance
        dim0 = self.vertices.coords[edges[:, 1]]
        dim1 = self.vertices.coords[edges[:, 0]]
        edge_weights = np.sum((dim0 - dim1) ** 2, axis=1) ** 0.5
        self.network = Network(pairs=edges, edge_lens=edge_weights)
        logger.trace("constructing network finished.")

    def extract_component(self, component_id: Union[slice, int] = 0) -> Shape:
        """Extract connected component(s) from a shape object

        The connected components are sorted decreasingly by the number
        of vertices so that the first component (component_id=0) is the
        largest. This is used to exclude model junk disconnected from
        the main model mesh.

        :param component_id:
        :return: Shape
        """
        # sort for longest first, and if tied, then lowest vertex id
        clusters = sorted(
            self.network.get_vertex_clusters(), key=lambda x: (-len(x), x)
        )

        # get vertices from selected component(s) sorted
        if isinstance(component_id, int):
            chosen_v_ids = sorted(clusters[component_id])
            logger.debug(
                f"{len(clusters)} cluster of vertices; selecting largest "
                f"({len(chosen_v_ids)}) vertices")
        else:
            chosen_v_ids = sorted(set.union(*clusters[component_id]))
            logger.debug(
                f"{len(clusters)} cluster of vertices; selecting slice of "
                f"({len(chosen_v_ids)} vertices")

        # subset vertices
        verts = Vertices(
            coords=self.vertices.coords[chosen_v_ids],
            colors=self.vertices.colors[chosen_v_ids],
        )

        # subset faces by removing faces containing missing vertices
        faces = deepcopy(self.faces)

        # find faces that contain any unwanted vertex ids, and assign
        # the ids to rm_f_ids, then delete them.
        vidxs = faces.vertex_ids
        rm_f_ids = np.isin(vidxs, chosen_v_ids, invert=True).any(axis=1)
        face_v_ids = np.delete(vidxs, rm_f_ids, axis=0)

        # modify the v ids in the faces, because some vertices were removed
        # using the np.unique() approach here will be much faster than
        # using np.vectorize(dict.__getitem__)()
        v_id_translator = {old: new for new, old in enumerate(chosen_v_ids)}
        uniq, inv = np.unique(face_v_ids, return_inverse=True)

        # ...
        faces.vertex_ids = (
            np.array([v_id_translator[o_i] for o_i in uniq])[inv]
            .reshape(face_v_ids.shape)
        )

        # also delete associated texture ids
        if len(faces.texture_ids):
            faces.texture_ids = np.delete(faces.texture_ids, rm_f_ids, axis=0)
        return Shape(verts, faces)

    def draw(
        self,
        plot: Optional[k3d.Plot] = None,
        point_size: int = None,
        xbaseline: int = 0,
        ybaseline: int = 0,
        zbaseline: int = 0,
    ) -> k3d.Plot:
        """Return K3D plot as points or mesh.
        """
        plot = plot if plot is not None else k3d.plot()

        # if only vertices then plot points
        if self.faces is None:
            coords = self.vertices.coords.copy().astype(np.float32)
            coords[:, 0] += xbaseline
            coords[:, 1] += ybaseline
            coords[:, 2] += zbaseline
            size = (coords[:, 0].max() - coords[:, 0].min()) / 20
            plot += k3d.points(
                positions=coords,
                point_size=point_size if point_size is not None else size,
                shader="flat",
            )

        # if faces then draw mesh
        else:
            plot += k3d.mesh(...)
        return plot


if __name__ == "__main__":

    import phyloshape
    phyloshape.set_log_level("TRACE")

    MODEL_FILE = "/usr/share/inkscape/extensions/Poly3DObjects/cube.obj"
    IMAGE_FILE = None
    obj = phyloshape.io.load_shape_from_obj(MODEL_FILE, IMAGE_FILE)
    print(obj)


    # MODEL_FILE = "../../../../PhyloShapeTest/data/cranolopha_DE183.ply"
    # IMAGE_FILE = "../../../../PhyloshapeTest/data/cranolopha_DE183.jpg"

    # from loguru import logger
    # logger.error("TEST")


