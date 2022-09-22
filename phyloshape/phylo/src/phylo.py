#!/usr/bin/env python

"""Reconstruct ancestral shape

"""
import numpy as np
from scipy.optimize import minimize, basinhopping
from phyloshape.shape.src.vectors import VertexVectorMapper
from phyloshape.shape.src.shape import ShapeAlignment
from phyloshape.phylo.src.models import *
from typing import Union, Dict, TypeVar
from sympy import Symbol
Url = TypeVar("Url")
from pathlib import Path
from loguru import logger
from toytree.io.src.treeio import tree


# TODO
class Tree:
    """
    A simplified tree object for ad hoc phylogenetic analysis
    """
    def __init__(self):
        pass
# temporarily use toytree instead


class PhyloShape:
    """

    """
    def __init__(self,
                 tree_obj,
                 shape_alignments: ShapeAlignment):
        """
        :param tree_obj: TODO can be str/path/url
        :param shape_alignments: shape labels must match tree labels
        """
        self.tree = tree_obj
        self.shapes = shape_alignments
        self.faces = shape_alignments.faces
        self.__map_shape_to_tip()
        # self.vv_translator is a VertexVectorMapper, making conversion between vertices and vectors
        self.vv_translator = None

    def __map_shape_to_tip(self):
        # set vertices as features of tip Nodes
        for node_id in range(self.tree.ntips):
            leaf_node = self.tree[node_id]
            assert leaf_node.name in self.shapes, f"data must include all tips in the tree. Missing={leaf_node.name}"
            label, leaf_node.vertices = self.shapes[leaf_node.name]

    def build_vv_translator(self):
        # TODO: VertexVectorMapper.__init__(): auto detect face and face_v_ids
        self.vv_translator = VertexVectorMapper(self.shapes.faces.vertex_ids)

    def build_tip_vectors(self):
        for node_id in range(self.tree.ntips):
            leaf_node = self.tree[node_id]
            leaf_node.vectors = self.vv_translator.to_vectors(leaf_node.vertices)

    def symbolize_ancestral_vectors(self):
        for node_id in range(self.tree.ntips, self.tree.nnodes):
            ancestral_node = self.tree[node_id]
            ancestral_node.vectors = \
                ancestral_node.vectors_symbols = \
                np.array([[Symbol("%s_%s_%s" % (_dim, node_id, go_v))
                           for _dim in ("x", "y", "z")]
                          for go_v in range(self.shapes.n_vertices())])

    def form_log_like(self, model):
        # TODO: more models to think about
        if model.lower() == "brownian":
            model_like_func = Brownian().form_log_like
        else:
            raise ValueError
        log_like = 0
        for node in self.tree.traverse():
            if not node.is_root():
                log_like += model_like_func(time=node._dist, from_states=node.up.vectors, to_states=node.vectors)
        return log_like

    def reconstruct_ancestral_shapes(self, model):
        self.build_vv_translator()
        self.build_tip_vectors()
        self.symbolize_ancestral_vectors()
        self.form_log_like(model)





