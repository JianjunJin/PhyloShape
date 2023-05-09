#!/usr/bin/env python

"""Vector tree class.

"""

from typing import List, Optional

import numpy as np
from phyloshape.shape.src.model import Model
from phyloshape.shape.src.core import Vertex, Vector, Face


class VectorMapper:
    def __init__(
        self,
        vertices: List[Vertex],
        linear: bool = True,
        random_seed: Optional[int] = None,
        num_neighbors: int = 10,
        num_iterations: int = 5,
    ):

        self.vertices = vertices
        self.linear = linear
        self.rng = np.random.default_rng(random_seed)
        self.num_neighbors = num_neighbors
        self.num_iterations = num_iterations

    def to_vectors(self) -> np.ndarray:
        pass

    def to_vertices(self) -> np.ndarray:
        pass

    def _update_vertices_by_rotation(self):
        pass

    


if __name__ == "__main__":

    import phyloshape
    phyloshape.set_log_level("DEBUG")
    MODEL_FILE = "/usr/share/inkscape/extensions/Poly3DObjects/cube.obj"
    model = phyloshape.io.load_model_from_obj(MODEL_FILE)
    print(model)

    # select a first vertex
    v0 = model.vertices[2]
    print(v0)
