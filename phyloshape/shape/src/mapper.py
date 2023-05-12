#!/usr/bin/env python

"""VectorMapper class.

Given a set of Models a Vector path will be proposed that is shared by
all models, representing a traversal over all vertices. Each vector
is defined by homologous Vertex IDs in each model. Each Vertex is
also associated with a number of neighbor edges, which are used to
constrain its change...
"""

from typing import List, Optional, Mapping, Dict
import itertools

from loguru import logger
import numpy as np
from phyloshape.shape.src.model import Model
from phyloshape.shape.src.core import Vector, Face

logger = logger.bind(name="phyloshape")


class VectorMapper:
    """...

    Note the 10th vertex in verts does not necessarily correspond to
    Model.vertex[10] b/c some vertices may be dropped.
    """
    def __init__(
        self,
        models: Mapping[str, Model],
        random_seed: Optional[int] = None,
        num_neighbors: int = 10,
        num_iterations: int = 5,
        # local_path: bool = True,
        # linear_path: bool = True,
    ):

        self.models = models
        self.labels = sorted(models)
        self.nvertices = len(self.models[self.labels[0]].vertices)
        self.nmodels = len(models)
        self.rng = np.random.default_rng(random_seed)
        self.num_neighbors = min(num_neighbors, self.nvertices - 1)
        self.num_iterations = num_iterations
        # self.linear_path = linear_path
        # self.local_path = local_path

        # Values filled by ._set functions
        self.vertex_ids = np.arange(self.nvertices)
        """: Order of Vertex traversal. Updated in _set funcs."""
        self.verts: np.ndarray = None
        """: all vertices (nmodels, nvertices, 3)"""
        self.vectors: Dict[str, Dict[int, List[Vector]]] = {}
        """: {model_index: {vertex_index: Vector}}"""
        self.vector_weights: Dict[str, Dict[int, np.ndarray]] = {}
        """: {model_index: {vertex_index: Vector}}"""

        # set all vertices into one large array
        self._set_vertices()
        # set vectors excluding any duplicate landmarks
        self._set_init_vectors()
        # set vector faces and order as sorted list by dist variation
        self._set_ordered_vectors()

    def _set_vertices(self):
        """Fill the .verts array with Vertex data from all Models.

        This array can be indexed by [model, vertex_id]. Beware that
        some vertex_ids may be excluded from the analysis if they are
        identical, but they will not be dropped from this array, such
        that the axis=1 index will always match the vertex_id. Use the
        list self.vertex_ids to iterate over non-excluded vertex ids.
        """
        self.verts = np.zeros(
            shape=(self.nmodels, self.nvertices, 3),
            dtype=np.float32,
        )
        for midx, label in enumerate(self.labels):
            model = self.models[label]
            for vidx, vertex in enumerate(model.vertices):
                self.verts[midx, vidx, :] = vertex.coords

    def _set_init_vectors(self, min_dist: float = 1e-12) -> None:
        """Store initial vectors and record duplicate vertices

        The initial vectors for each model are one-directional from
        lower vertex_idx to higher vertex_id, including between
        vertices that are marked for exclusion as duplicates.
        """
        # for each model a dict mapping a vertex_id to Set of Vectors
        # Note .vectors will later be converted from set to a sorted list.
        for label in self.labels:
            self.vectors[label] = {i: {} for i in range(self.nvertices)}

        # to store vertex ID pairs with 0 dist to identify duplicates
        duplicates = []

        # iterate over all pairs of vertex IDs
        for i, j in itertools.combinations(range(self.nvertices), 2):

            # to fill with paired vertex distances
            identical = np.zeros(self.nmodels, dtype=np.bool_)

            # get pairwise distance between all vertices
            for midx, label in enumerate(self.labels):
                model = self.models[label]
                v0 = Vector(model.vertices[i], model.vertices[j])
                v1 = Vector(model.vertices[j], model.vertices[i])
                self.vectors[label][i][j] = v0
                self.vectors[label][j][i] = v1

                # store if dist is near 0
                if v0.dist < min_dist:
                    identical[midx] = True

            # store vertex as duplicate if identical in all models
            if all(identical):
                duplicates.append((i, j))

        # keep just one at each vertex given >=2 stacked vertices
        remove = set()
        keep = set()
        for i, j in duplicates:
            if i not in remove:
                if j not in keep:
                    remove.add(i)
                    keep.add(j)
                else:
                    remove.add(i)
                    keep.discard(i)
            else:
                if j not in keep:
                    keep.add(j)
                else:
                    keep.discard(j)
                    remove.discard(i)

        # subset array of vertex_ids to only retained vertices
        self.vertex_ids = np.array([i for i in self.vertex_ids if i not in remove])
        self.nvertices = self.vertex_ids.shape[0]
        self.num_neighbors = min(self.num_neighbors, self.nvertices - 1)

        # remove vectors starting or ending in excluded vertices
        for label in self.labels:
            # remove item: [ridx: {Vector, Vector, Vector}]
            for ridx in remove:
                self.vectors[label].pop(ridx)
            # remove Vectors ending in rm IDs: [vidx: {Vector, x, Vector, ...}]
            for vidx in self.vertex_ids:
                self.vectors[label][vidx] = {
                    i: j for (i, j) in self.vectors[label][vidx].items()
                    if j.end.id not in remove
                }

        # finally, add a vector to itself for every sample and assert len
        for vidx in self.vertex_ids:
            for label in self.labels:
                model = self.models[label]
                self.vectors[label][vidx][vidx] = Vector(
                    model.vertices[vidx], model.vertices[vidx]
                )
                assert len(self.vectors[label][vidx]) == len(self.vertex_ids)

        # log report discarded vertices
        if remove:
            remove = sorted(remove)
            logger.warning(
                f"{len(remove)} identical vertices will be excluded: {remove}")

    def _set_ordered_vectors(self) -> None:
        """Set .vectors for each model to a sorted list instead of set.

        Each vector set will be [optionally] subset to only vectors to
        the num_neighbors with most stable distances to the vertex,
        sorted from lowest to hightest distance variation.
        """
        # store summed variation in neighbor distances among vertices
        sum_var_ndists = {}

        # iterate over vertex ids
        # NOTE: not range(self.nvertices) b/c some verts may been dropped
        # for vidx, nidx in itertools.permutations(self.vertex_ids, 2):
        # logger.info(f"vertex_ids = {self.vertex_ids}")
        for vertex_id in self.vertex_ids:

            # iterate over models getting ordered (vidx, nidx) dists
            dists = np.zeros((self.nmodels, len(self.vertex_ids)))
            for lidx, label in enumerate(self.labels):
                vectors = self.vectors[label][vertex_id]
                dists[lidx] = [vectors[i].dist for i in self.vertex_ids]

                # create empty weights to be filled {str: {int: ndarray}}
                if label not in self.vector_weights:
                    self.vector_weights[label] = {vertex_id: None}

            # get variation in dist vidx -> all other vertex_ids across models
            vstds = np.std(dists, axis=0)

            # store how variable this vertex is to ALL neighbors
            sum_var_ndists[vertex_id] = sum(vstds)

            # get order of vstds ordered by distance variance from vidx
            neighbor_stability_order = np.argsort(vstds)

            # sort vectors for each vertex the same way for each model:
            # by their variance in distances from vidx
            for label in self.labels:
                model = self.models[label]

                # sort vectors into list: [Vertex, Vertex, Vertex, ...]
                self.vectors[label][vertex_id] = [
                    self.vectors[label][vertex_id][i]
                    for i in self.vertex_ids[neighbor_stability_order]
                ][1:]

                # ...
                self.vector_weights[label][vertex_id] = (
                    1 / vstds[neighbor_stability_order][1:]
                )

                # slice list to first N neighbors
                slx = slice(0, self.num_neighbors)
                if self.num_neighbors < self.nvertices - 1:
                    self.vectors[label][vertex_id] = (
                        self.vectors[label][vertex_id][slx])
                    self.vector_weights[label][vertex_id] = (
                        self.vector_weights[label][vertex_id][slx])

                # convert vector stds to weights: [0.08, 0.01, 0.001, ...]
                self.vector_weights[label][vertex_id] /= (
                    self.vector_weights[label][vertex_id].sum())

                # store reference face as (vidx, 2 least variable neighbors
                # that are not the vector target)
                top3 = self.vectors[label][vertex_id][:4]
                for vector in self.vectors[label][vertex_id]:
                    top2 = [i.end.id for i in top3 if i.end.id != vector.end.id]
                    vector._face = Face((
                        model.vertices[vertex_id],
                        model.vertices[top2[0]],
                        model.vertices[top2[1]],
                    ))

        # get best vertex visit order that visits each vertex in the
        # order from least variable to neighbors to most variable.
        self.vertex_ids = sorted(sum_var_ndists, key=lambda x: sum_var_ndists[x])
        logger.info(f"best vertex path: {self.vertex_ids[:5]}, ... {self.vertex_ids[-5:]}")

    def get_vectors(self, label: str, vertex_id: int) -> np.ndarray:
        """Return array of vectors sorted from N most stable neighbors"""
        return np.vstack([i.relative for i in self.vectors[label][vertex_id]])

    def get_vector_ids(self, label: str, vertex_id: int) -> np.ndarray:
        """Return array of vectors sorted from N most stable neighbors"""
        return np.vstack([(i.start.id, i.end.id) for i in self.vectors[label][vertex_id]])

    def get_vector_weights(self, label: str, vertex_id: int) -> np.ndarray:
        """Return array of vectors sorted from N most stable neighbors"""
        return self.vector_weights[label][vertex_id]

    def to_vectors(self) -> np.ndarray:
        """Return vectors among vertices."""

    def to_vertices(self) -> np.ndarray:
        """Return vertices from a set of vectors."""

    def _update_vertices_by_rotation(self):
        pass


if __name__ == "__main__":

    import phyloshape
    phyloshape.set_log_level("DEBUG")

    models = phyloshape.data.get_gesneriaceae_models()
    vm = VectorMapper(models, num_neighbors=10)
