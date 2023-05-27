#!/usr/bin/env python

"""VectorMapper class.

Given a set of Models a Vector path will be proposed that is shared by
all models, representing a traversal over all vertices. Each vector
is defined by homologous Vertex IDs in each model. Each Vertex is
also associated with a number of neighbor edges, which are used to
constrain its change...

0. Clean Vertex data given N input Vertex datasets (Models).
1. Convert Vertices -> Vectors
2. Find most stable path to visit Vertices.
3. Infer PCs from Vectors for tip Nodes.
4. Reconstruct ancestral PCs under an Evolutionary Model (e.g., BM).
5. Convert PCs back into Vectors for ancestral Nodes using path vectors.
6. Convert Vectors back into Vertices for ancestral Nodes.
"""

from typing import List, Optional, Mapping, Dict, Tuple, Sequence, Optional
import itertools
from loguru import logger
import k3d
import numpy as np
from toytree import ToyTree
from sklearn.decomposition import PCA
from phyloshape.shape.src.model import Model
from phyloshape.shape.src.core import Vector, Face, Vertex

MIN_DIST = 1e-9
logger = logger.bind(name="phyloshape")


class VectorMapper:
    """Converts from Vertices -> Vectors -> PCs and back.

    A shape is represented by an array of Vertex coordinates. This func
    takes a dict containing multiple Model objects with homologous
    landmarks. These data are analyzed in the following way:
        1. Clean models to remove duplicate (uninformative) vertices.
        2. Create Vectors for each Model between all pairs of Vertices.
        3. Find the order of Vertices with the most stable vectors
        across all models. This is the vertex_ids path.
        4. Set a reference face to each Vector composed of previous
        vertices in the path, creating a 'relative' orientation for
        each vectors that will allow us to project into a new coordinate
        space.
        5. ...
    """
    def __init__(
        self,
        models: Mapping[str, Model],
        random_seed: Optional[int] = None,
        num_neighbors: Optional[int] = 10,
        num_iterations: int = 5,
    ):

        self.models = models
        self.labels = sorted(models)

        # counters here are updated if any duplicates are excluded.
        self.num_vertices = len(self.models[self.labels[0]].vertices)
        self.num_models = len(models)
        self.num_iterations = max(1, num_iterations)
        self.num_neighbors = min(
            self.num_vertices - 1 if None else num_neighbors,
            self.num_vertices - 1)

        # log init of VectorMapper
        logger.info(
            f"VectorMapper num_models={self.num_models} "
            f"num_vertices={self.num_vertices}")

        # Values filled by ._set functions
        self.vertex_ids = np.arange(self.num_vertices)
        """: Order of Vertex traversal. Updated in _set funcs."""
        self.vectors: Dict[str, Dict[Tuple[int, int], Vector]] = {}
        """: Vector objects with (Vertex, Vertex) info in each model"""
        self.vector_weights: Dict[Tuple[int, int], float] = {}
        """: Phylo and variance weighted stability of vectors."""
        self._neighbors: Dict[int, np.ndarray] = {}
        # """: store N most stable neighbors of a vertex. Not used currently."""

        # remove duplicate vertices (updates .vertex_ids and .num_[...])
        self._remove_duplicates()

        # set .vectors between all vertices for all models
        self._set_vectors()

        # set .vertex_ids vertex ordered by dist-variance
        self._set_vertex_path()

        # set .faces on each vector in .vectors based on ordered v path
        self._set_vector_faces()

    def _get_duplicate_vertex_pairs(self) -> List[int]:
        """Return List of vertex pairs with zero dist across all models.
        """
        # initially compare all pairs of verts
        pairs = itertools.combinations(range(self.num_vertices), 2)

        # iterate over models
        for midx, label in enumerate(self.labels):
            model = self.models[label]

            # store tuples of duplicate vertex IDs
            duplicates = []

            # compare verts appeared as dups in all previous models
            for i, j in pairs:
                vec = Vector(model.vertices[i], model.vertices[j])
                # store if dist is near 0
                if vec.dist < MIN_DIST:
                    duplicates.append((i, j))

            # set pairs to only test further the pairs in dups
            pairs = duplicates
        return duplicates

    def _remove_duplicates(self) -> None:
        """...
        """
        duplicates = self._get_duplicate_vertex_pairs()

        # keep just one at each vertex pair given >=2 stacked vertices
        # e.g., (0, 1), (0, 2), (1, 2). Remove 0 and 1, keep 2.
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
        self.num_vertices = self.vertex_ids.shape[0]
        self.num_neighbors = min(self.num_neighbors, self.num_vertices - 1)

        # logger report discarded vertices
        if remove:
            rm_s = sorted(remove)
            rm_n = len(rm_s)
            logger.warning(f"{rm_n} identical vertices will be excluded: {rm_s}")

    def _set_vectors(self, labels: Optional[Sequence[str]] = None) -> None:
        """Set Vector objects between all Vertices in one or more models.
        """
        labels = self.labels if labels is None else labels

        # set Vectors between pairs of (v0, v1) for each model.
        for label in labels:
            self.vectors[label] = {}
            model = self.models[label]
            for vidx, nidx in itertools.combinations(self.vertex_ids, 2):
                v0 = model.vertices[vidx]
                v1 = model.vertices[nidx]
                self.vectors[label][(vidx, nidx)] = Vector(v0, v1)
                self.vectors[label][(nidx, vidx)] = Vector(v1, v0)
                self.vectors[label][(vidx, vidx)] = Vector(v0, v0)
            self.vectors[label][(nidx, nidx)] = Vector(v1, v1)

    def _set_vertex_path(self) -> None:
        """Find optimal path to visit Vertices and store to .vertex_ids.

        Also sets initial vector weights.
        """
        logger.debug("finding optimal stable path visiting all Vertices")

        # dict to store {vertex_id: std of its vector dists}
        sum_var_ndists = {}

        # iterate over all vertices
        for vidx, vertex_id in enumerate(self.vertex_ids):

            # fill array of dists from vertex to all others across models
            dists = np.zeros((self.num_models, self.num_vertices))
            for lidx, label in enumerate(self.labels):
                dists[lidx] = [
                    self.vectors[label][(vertex_id, neighbor_id)].dist
                    for neighbor_id in self.vertex_ids
                ]

            # get variation in dist vidx -> all other vertex_ids across models
            vstds = np.std(dists, axis=0)

            # store how variable this vertex pair is
            sum_var_ndists[vertex_id] = vstds.sum()

            # temporarily store stds, trimmed and normalized below.
            self.vector_weights[vertex_id] = vstds

        # store translator of old vertex order given duplicates removed
        dtrans = dict(zip(self.vertex_ids.copy(), range(len(self.vertex_ids))))

        # get best vertex visit order that visits each vertex in the
        # order from least variable to neighbors to most variable.
        self.vertex_ids = sorted(sum_var_ndists, key=lambda x: sum_var_ndists[x])
        logger.info(f"vertex path: {self.vertex_ids[:5]}, ... {self.vertex_ids[-5:]}")

        # sort vertex_weights by new order and only store from 0-vidx
        for vidx, vertex_id in enumerate(self.vertex_ids):
            # get the variation in vertices ordered 0, 1, 2...
            vstds = self.vector_weights[vertex_id]

            # reorder w/ new vertex order while accounting for dups rm
            odx = [dtrans[i] for i in self.vertex_ids]
            vstds = vstds[odx]

            # trim to only 0-vidx
            vstds = vstds[:vidx]

            # convert to weights
            if not vidx:
                self.vector_weights[vertex_id] = []
            else:
                assert 0. not in vstds, "dups cannot exist here."
                weights = 1 / vstds
                self.vector_weights[vertex_id] = weights / weights.sum()

    def _set_vector_faces(self, labels: Optional[Sequence[str]] = None) -> None:
        """Set .face on each Vector in each Model.

        """
        logger.debug("setting reference faces to all Vectors along path")

        # get one or more models to update faces on
        labels = self.labels if labels is None else labels

        # iterate over models setting faces in path order
        for label in labels:
            model = self.models[label]
            for pair, vector in self.vectors[label].items():
                # get index of vector start ID
                vstart_idx = self.vertex_ids.index(pair[0])
                vend_idx = self.vertex_ids.index(pair[1])

                # get last three ordered vertex IDs to form a face
                if vstart_idx > 2 | vend_idx > 2:
                    trio = [
                        self.vertex_ids[i] for i in
                        (vstart_idx, vstart_idx - 1, vstart_idx - 2)
                    ]

                # or if one of first two vectors then use first three as face
                else:
                    trio = self.vertex_ids[:3]

                # set Model vertices to the face
                vector._face = Face(tuple(model.vertices[i] for i in trio))
                # logger.info([vector, vector.face])

    def _get_phylo_vector_weights(self, tree: ToyTree, label: str) -> np.ndarray:
        """Return vector weights for a tip or ancestral sample.
        """
        # ...

    def get_model_from_vectors(
        self,
        label: str,
        vectors: Mapping[Tuple[int, int], Vector],
        weighted: bool = True,
    ) -> Model:
        """Return an array of vertex coords inferred from weighted vectors.

        Vertex coordinates are generated in a new R3 space by inferring
        the position of each vertex one at a time in the order previously
        inferred to be the most stable across models. At each subsequent
        Vertex its position is inferred relative to the already placed
        Vertices relative to their reference face.
        """
        logger.debug("inferring Vertex coords from mean path accumulated relative Vectors")

        # TODO: get updated weights given the phylo position of this model
        # weights = self._get_vector_weights(tree, label)

        # build an empty vertex coords array to fill
        vcoords = np.zeros(shape=(self.num_vertices, 3))

        # iterate to refine model coordinates
        for rep in range(self.num_iterations):
            # track difference from previous iteration
            diff = 0.

            # iterate over vertices to serve as vector endpoints
            for idx, vec_end in enumerate(self.vertex_ids):

                # iterate over a subset of vertices to serve as vector
                # start points, using only those that already have
                # inferred positions in the new coordinate space.
                subset = self.vertex_ids[:idx]

                # get coordinates from accumulated relative vectors so far
                arr = np.zeros((idx, 3))
                for aidx, vec_start in enumerate(subset):
                    pair = (vec_start, vec_end)
                    arr[aidx, :] = vcoords[aidx] + vectors[pair].relative

                # get mean weighted coordinate
                if idx:
                    if weighted:
                        weights = self.vector_weights[vec_end]
                        new_pos = np.average(arr, axis=0, weights=weights)
                    else:
                        new_pos = arr.mean(axis=0)
                    diff += np.linalg.norm(vcoords[idx] - new_pos)
                    vcoords[idx] = new_pos

            # log improvement
            logger.info(f"refining iteration {rep}; sum diff = {diff:.5g}")

            # add reconstructed model to .models and get its vectors
            new_label = f"{label}-x"
            self.add_model(label=new_label, coords=vcoords)
            vectors = self.vectors[new_label]
        return self.models[new_label]

    def add_model(self, label: str, coords: np.ndarray) -> None:
        """Create a new Model from Vertex coordinates and add to .models
        """
        vertices = []

        # copy vertex ids from the first model
        flabel = self.labels[0]
        for vertex in self.models[flabel].vertices:
            if vertex.id in self.vertex_ids:
                # get its index in the vertex_ids list
                vidx = self.vertex_ids.index(vertex.id)
                # create a new Vertex with updated coords
                vertices.append(Vertex(id=vertex.id, coords=coords[vidx]))
            else:
                # placeholder for removed duplicate vertices
                vertices.append(Vertex(id=-9, coords=(0, 0, 0)))

        # store a new Model with the new set of Vertices
        self.models[label] = Model(vertices, faces=[])

        # set its Vectors and faces
        self._set_vectors(labels=[label])
        self._set_vector_faces(labels=[label])

    ###################################################################
    # ...
    ###################################################################

    def get_PCs(self) -> np.ndarray:
        """

        IDEA: visualize vector co-variances
            1. get relative vector covariances from PCA
            2. user selects a module (e.g., beak) of the 3D model made
               up of an array of vertex IDs.
            3. color all other vertices by avg covariance with the
               selected vertices making up the module.
            4. e.g., tube length may co-vary with beak lengths.

        """
        # store orig shape of relative vectors arr (nverts, nneighbors. 3)
        shape = (len(self.vertex_ids), self.num_neighbors, 3)

        # create arr for running PCA of shape (nmodels, nverts*nneigh*3)
        fdat = np.zeros((self.num_models, np.product(shape)), dtype=np.float64)

        # fill arr with flattened vector array data for each sample
        for lidx, label in enumerate(self.labels):
            fdat[lidx, :] = np.array([
                self.get_vectors_relative(self.labels[lidx], i)
                for i in self.vertex_ids
            ]).flatten()
        return fdat
        # decompose into PC axes
        # pc_tool = PCA(n_components=None, svd_solver="full")
        # pc_tool.fit(fdat)
        # logger.info(pc_tool.explained_variance_ratio_)

    # def draw_...

    def draw_vectors(self, label: str, vertex_id: int, invert: bool = False, **kwargs):
        """..."""
        model = self.models[label]
        plot = model.draw(point_size=10)
        vectors = self.get_vectors_absolute(label, vertex_id)
        if invert:
            origins = self.get_vector_vertex_start_coords(label, vertex_id)
        else:
            origins = self.get_vector_vertex_end_coords(label, vertex_id)
            vectors *= -1

        kwargs["color"] = kwargs.get("color", 0xde49a1)
        kwargs["line_width"] = kwargs.get("line_width", 1)
        kwargs["head_size"] = kwargs.get("head_size", 50)
        kwargs["use_head"] = kwargs.get("use_head", True)
        plot += k3d.vectors(
            origins=origins.astype(np.float32),
            vectors=vectors.astype(np.float32),
            **kwargs,
        )
        return plot


if __name__ == "__main__":

    import phyloshape
    phyloshape.set_log_level("DEBUG")

    models = phyloshape.data.get_gesneriaceae_models()
    vm = VectorMapper(models, num_neighbors=10)
    vectors = vm.vectors["34_HC3403-3_17"]
    vm.get_model_from_vectors("test", vectors)

    # vm.get_PCs()
    # print(vm.get_vector_face_coordinates("34_HC3403-3_17", 3))
