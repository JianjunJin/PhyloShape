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
from typing import Sequence, Optional, Dict, Iterator, Tuple

from loguru import logger
import numpy as np
import k3d
from phyloshape.shape.src.core import Vertex, Face, Vector
from phyloshape.shape.src.graph import Graph

logger = logger.bind(name="phyloshape")


class Model:
    """The core phyloshape class object for representing 3D models.

    Note
    ----
    vertices and faces are ...
    """
    def __init__(
        self,
        vertices: Sequence[Vertex],
        faces: Sequence[Face],
    ):
        # simple data storage
        self.vertices = vertices
        self.faces = faces

        # complex data storage
        self._graph: Graph = None
        """: Graph of edges between vertices on a mesh (Mapping[int, Edge])"""
        self._vectors: Dict[Tuple[int, int], Vector] = {}
        """: Vectors between any two vertices"""

    def __repr__(self) -> str:
        return f"Model(nverts={len(self.vertices)}, nfaces={len(self.faces)})"

    ##################################################################
    # graph/distance methods
    ##################################################################
    def _build_graph_from_faces(self) -> None:
        """Extract all edges from faces to construct a mesh graph"""
        if not self.faces:
            raise ValueError("Cannot build graph without Face/Edge data.")

        # iterate over faces extracting each edge and getting its dist
        data = {}
        for face in self.faces:
            for edge in face.edges:
                # sort so that keys can be checked, makes it faster
                v0, v1 = sorted(edge, key=lambda x: x.id)
                key = (v0.id, v1.id)
                if key not in data:
                    data[key] = Vector(v0, v1).dist

        # Graph will create bidirectional edges from pairs data
        edges = np.array(list(data.keys()))
        dists = np.array(list(data.values()))
        self._graph = Graph(edges, dists)

    ##################################################################
    # get vectors from vertices
    ##################################################################
    def _iter_vector_path(self, start: int = 0) -> Iterator[Vector]:
        """Yield ordered vectors along a path traversing all vertices"""
        for v0, v1 in self._graph._iter_path_traversal(start=start):
            # TODO: how to select the reference face?
            yield Vector(self.vertices[v0], self.vertices[v1], face=None)

    ##################################################################
    # get vertices from vectors
    ##################################################################
    # def _iter_transformed_vertices(self, start: int = 0) -> Iterator[Vector]:
    #     """..."""
    #     for v0, v1 in self._graph._iter_path_traversal(start=start):
    #         # TODO: how to select the reference face?
    #         yield ...

    ##################################################################
    # drawing functions
    ##################################################################
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
        plot = plot if plot is not None else k3d.plot(grid_visible=False, antialias=5)

        # if only vertices then plot points
        if not self.faces:
            coords = np.array([i.coords for i in self.vertices], dtype=np.float32)
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
    obj = phyloshape.io.load_model_from_obj(MODEL_FILE, IMAGE_FILE)
    print(obj)
    print(list(obj._iter_vector_path()))

    # MODEL_FILE = "../../../../PhyloShapeTest/data/cranolopha_DE183.ply"
    # IMAGE_FILE = "../../../../PhyloshapeTest/data/cranolopha_DE183.jpg"

    # model = phyloshape.io.load_model_from_ply(MODEL_FILE)
    # model._build_graph_from_faces()
    # print(model)
    # print(model._graph._get_vertex_clusters())

    # from loguru import logger
    # logger.error("TEST")
