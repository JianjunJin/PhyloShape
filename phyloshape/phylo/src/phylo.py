#!/usr/bin/env python

"""Reconstruct ancestral shape

"""
import numpy as np
from scipy.optimize import minimize, basinhopping
from phyloshape.shape.src.vectors import VertexVectorMapper
from phyloshape.shape.src.shape import ShapeAlignment
from phyloshape.shape.src.vertex import Vertices
from phyloshape.phylo.src.models import *
from phyloshape.utils.src.vertices_manipulator import GeneralizedProcrustesAnalysis
from typing import Union, Dict, TypeVar
from symengine import Symbol, lambdify, expand
from symengine import log as s_log
from copy import deepcopy
Url = TypeVar("Url")
from pathlib import Path
# from toytree.io.src.treeio import tree
from loguru import logger
logger = logger.bind(name="phyloshape")


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
                 shape_alignments: ShapeAlignment,
                 model=None,
                 dim_transform=None,
                 dim_inverse_transform=None):
        """
        Parameters
        __________
        tree_obj: TODO can be str/path/url
        shape_alignments: shape labels must match tree labels
        model
        dim_transform: e.g. PCA.transform
        dim_inverse_transform: e.g. PCA.inverse_transform
        """
        self.tree = tree_obj
        self.shapes = shape_alignments
        self.faces = shape_alignments.faces
        self.__map_shape_to_tip()
        self._weights = None
        # TODO: more models to think about
        self.model = model if model else MultivariateBrownian()
        assert isinstance(self.model, MotionModel)
        assert (dim_transform is None) == (dim_inverse_transform is None), \
            "dim_transform and dim_inverse_transform must be used together!"
        self.dim_transform = dim_transform
        self.dim_inverse_transform = dim_inverse_transform

        self.__variables = []
        # self.vv_translator is a FaceVectorMapper, making conversion between vertices and vectors
        self.vv_translator = None
        self.loglike_form = 0
        self.negloglike_func = None  # for point estimation
        self.negloglike_func_s = None  # for scipy optimization
        self.__result = None

        self.gpa_obj = None

    def __map_shape_to_tip(self):
        # set vertices as features of tip Nodes
        for node_id in range(self.tree.ntips):
            leaf_node = self.tree[node_id]
            assert leaf_node.name in self.shapes, f"data must include all tips in the tree. Missing={leaf_node.name}"
            label, leaf_node.vertices = self.shapes[leaf_node.name]
            # logger.trace(label + str(leaf_node.vertices.coords))

    def build_vv_translator(self, mode="network-local", num_vs=20, num_vt_iter=5):
        """
        """
        self.vv_translator = None
        # if self.faces is None or len(self.faces) == 0:
        # build the translator using vertices only
        # TODO use alignment information rather than using the vertices of the first sample
        if mode == "old":
            label, vertices = self.shapes[0]
            from phyloshape.shape.src.vectors import VertexVectorMapperOld
            self.vv_translator = VertexVectorMapperOld(vertices)
            logger.info(f"using {label} to construct the vector system ..")
        else:
            self.vv_translator = VertexVectorMapper(
                [vt.coords for lb, vt in self.shapes],
                mode=mode,
                num_vs=num_vs,
                num_vt_iter=num_vt_iter,
                )
        # else:
        #     self.vv_translator = FaceVectorMapper(self.faces.vertex_ids)
        len_vt = len(self.vv_translator.vh_list())
        logger.info("Vertex:Vector ({}:{}) translator built.".format(len(self.shapes[0][0]), len_vt))

    def build_weights(self, vector_list: list, mode: str = "uni"):
        if mode == "phylo-uni":
            # TODP
            pass
        elif mode == "phylo-relaxed":
            pass
        else:  # "uni": using universal rate (variation?) as weights
            # vector_list = [np.array([[1., 2., 3., ], [4., 5., 6., ], [7., 8., 9.], [1., 2., 3.]]),
            #                np.array([[1.1, 2.2, 3.3, ], [4.4, 5.5, 6.6, ], [7.7, 8.7, 9.1], [4.4, 5.5, 6.6, ]])]
            vector_wise_mean = np.mean(vector_list, axis=0)
            vector_wise_centered = vector_list - vector_wise_mean
            variations = np.sum(vector_wise_centered ** 2, axis=2)
            variations_across_vectors = np.sum(variations, axis=0)
            # TODO variation or std?
            # add axis and make repeats
            per_sample_weight = 1. / variations_across_vectors
            # Repeat the array along the first axis to create an array of shape (n_nodes, n_vectors)
            self._weights = np.tile(per_sample_weight, (len(vector_list) * 2 + 1, 1))

    def build_tip_vectors(self, weighted=True):
        if self.dim_transform is None or self.dim_inverse_transform is None:
            for node_id in range(self.tree.ntips):
                leaf_node = self.tree[node_id]
                leaf_node.vectors = self.vv_translator.to_vectors(leaf_node.vertices)
        else:
            vectors_list = []
            for node_id in range(self.tree.ntips):
                vectors_list.append(self.vv_translator.to_vectors(self.tree[node_id].vertices))
            self.build_weights(vectors_list)
            # TODO: transform the original vectors to modified (e.g. weighted-PCA) ones
            for node_id, trans_vectors in enumerate(self.dim_transform(vectors_list, weights=self._weights)):
                self.tree[node_id].vectors = trans_vectors
            logger.info("Dimension {} -> {}".format(vectors_list[0].shape, self.tree[0].vectors.shape))
        logger.info("Vectors for {} tips built.".format(self.tree.ntips))

    def sym_ancestral_state(self, attribute):
        vectors_shape = self.tree[0].__getattribute__(attribute).shape
        n_vals = np.prod(vectors_shape)
        for node_id in range(self.tree.ntips, self.tree.nnodes):
            ancestral_node = self.tree[node_id]
            state = np.array([Symbol("%s_%s" % (node_id, go_v)) for go_v in range(n_vals)]).reshape(vectors_shape)
            ancestral_node.__setattr__(attribute, state)
            ancestral_node.__setattr__(attribute + "_symbols", deepcopy(state))
            # ancestral_node.vectors = \
            #     ancestral_node.vectors_symbols = \
            #     np.array([Symbol("%s_%s" % (node_id, go_v)) for go_v in range(n_vals)]).reshape(vectors_shape)

                # np.array([[Symbol("%s_%s_%s" % (_dim, node_id, go_v))
                #            for _dim in ("x", "y", "z")]
                #           for go_v in range(n_vectors)])
        logger.info("{} for {} ancestral nodes symbolized.".format(
            attribute.capitalize(), self.tree.nnodes - self.tree.ntips))

    def formularize_log_like(self, attribute, log_func, v_id: int = None):
        """
        Parameters
        ----------
        attribute:
        v_id:
            vector id, for multivariate mode
        log_func:
             input symengine.log for maximum likelihood analysis using scipy,
             input tt.log for bayesian analysis using pymc3
        """
        self.loglike_form = 0
        if v_id is None:
            for node in self.tree.traverse():
                if not node.is_root():
                    new_term = self.model.form_log_like(
                        time=node._dist,
                        from_states=node.up.__getattribute__(attribute),
                        to_states=node.__getattribute__(attribute),
                        log_f=log_func)
                    self.loglike_form += new_term
                    logger.trace("Term {} -> {} {}: {}".format(node.up.idx, node.idx, node.name, new_term))
        else:
            for node in self.tree.traverse():
                if not node.is_root():
                    new_term = self.model.form_log_like(
                        time=node._dist,
                        from_states=node.up.__getattribute__(attribute)[v_id],
                        to_states=node.__getattribute__(attribute)[v_id],
                        log_f=log_func)
                    self.loglike_form += new_term
                    logger.trace("Term {} -> {} {}: {}".format(node.up.idx, node.idx, node.name, new_term))
        self.__update_variables(attribute=attribute, v_id=v_id)
        if v_id is None:
            logger.info("Num of variables: %i" % len(self.__variables))
            logger.info("Log-likelihood formula constructed.")
        else:
            logger.debug("Num of variables: %i" % len(self.__variables))
            logger.debug("Log-likelihood formula constructed.")
        logger.trace("Formula: {}".format(self.loglike_form))

    def functionalize_log_like(self):
        self.negloglike_func = lambdify(args=self.__variables,
                                        exprs=[-self.loglike_form])
        self.negloglike_func_s = lambda x: self.negloglike_func(*tuple(x))
        logger.trace("Log-likelihood formula functionalized.")

    def __update_variables(self, attribute: str, v_id: int = None):
        self.__variables = []
        self.__variables.extend(self.model.get_parameters())
        if v_id is None:
            for ancestral_node_id in range(self.tree.ntips, self.tree.nnodes):
                self.__variables.extend(
                    self.tree[ancestral_node_id].__getattribute__(attribute + "_symbols").flatten().tolist())
        else:
            for ancestral_node_id in range(self.tree.ntips, self.tree.nnodes):
                here_symbols = self.tree[ancestral_node_id].__getattribute__(attribute + "_symbols")[v_id]
                if isinstance(here_symbols, Symbol):
                    self.__variables.append(here_symbols)
                else:  # np.array
                    self.__variables.extend(here_symbols.flatten().tolist())

    # def get_variables(self):
    #     return list(self.__variables)

    # TODO
    # def compute_point_negloglike(self, model_params, ancestral_shapes):
    #     assert self.loglike_form
    #     assert self.negloglike_func_s
    #     # some conversion
    #     return self.negloglike_func(converted_model_params + linearized_shapes)

    def __summarize_ml_result(self, attribute: str, v_id: int = None):
        # 1. assign first part of the result.x to model parameters
        model_params_signs = self.model.get_parameters()
        go_p = len(model_params_signs)
        model_params = self.__result.x[:go_p]
        logger.info(", ".join(["%s=%f" % (_s, _v) for _s, _v in zip(model_params_signs, model_params)]))
        state_shape = self.tree[0].__getattribute__(attribute).shape
        if v_id is None:
            # 2. assign second part of the result.x to ancestral shape vectors
            # n_vts = self.shapes.n_vertices() - 1
            n_vals = int(np.prod(state_shape))
            for anc_node_id in range(self.tree.ntips, self.tree.nnodes):
                to_p = go_p + n_vals
                self.tree[anc_node_id].__setattr__(attribute, np.array(self.__result.x[go_p: to_p]).reshape(state_shape))
                go_p = to_p
        else:
            # 2. assign second part of the result.x to ancestral shape vectors
            # n_vts = self.shapes.n_vertices() - 1
            # print("state_shape", state_shape)
            n_vals = int(np.prod(state_shape[1:]))
            # print("n_vals", n_vals)
            for anc_node_id in range(self.tree.ntips, self.tree.nnodes):
                to_p = go_p + n_vals
                if not hasattr(self.tree[anc_node_id], attribute):
                    self.tree[anc_node_id].__setattr__(attribute, np.empty(state_shape))
                anc_state = self.tree[anc_node_id].__getattribute__(attribute)
                # print("go_p, to_p", go_p, to_p)
                anc_state[v_id] = np.array(self.__result.x[go_p: to_p]).reshape(state_shape[1:])
                go_p = to_p

    def minimize_negloglike(
            self,
            attribute: str,
            v_id: int = None,
            num_proc: int = 1):
        # constraints = # use constraints to avoid self-collision
        # verbose = False
        # other_optimization_options = {"disp": verbose, "maxiter": 5000, "ftol": 1.0e-6, "eps": 1.0e-10}
        count_run = 0
        all_runs = []
        success = False
        logger.info("Searching for the best solution ..")
        while count_run < 200:
            # propose the initials as the average of the values
            if v_id is None:
                state_mean = np.average([self.tree[_n_id].__getattribute__(attribute)
                                         for _n_id in range(self.tree.ntips)], axis=0).flatten()
            else:
                state_mean = np.average([self.tree[_n_id].__getattribute__(attribute)[v_id]
                                         for _n_id in range(self.tree.ntips)], axis=0).flatten()
            logger.trace("state_means: " + str(state_mean))
            len_vt = len(state_mean)
            initials = np.random.random(len(self.__variables))
            # last len_vt are states
            initials[-len_vt:] = state_mean
            # initials[-len_vt:] = (1 - (initials[-len_vt:] - 0.5) * 0.01)
            logger.trace("initials: " + str(initials))
            #
            # result = minimize(
            #     fun=self.negloglike_func_s,
            #     x0=initials,
            #     method="L-BFGS-B",
            #     tol=1e-6,
            #     # constraints=constraints,
            #     jac=False,
            #     options=other_optimization_options)
            minimizer_kwargs = {"method": "Nelder-Mead", "options": {"maxiter": 1000}, "tol": 1e-4}
            # using method='Nelder-Mead' to avoid 'Desired error not necessarily achieved due to precision loss'
            # if num_proc == 1:
            result = basinhopping(self.negloglike_func_s,
                                  x0=initials,
                                  minimizer_kwargs=minimizer_kwargs,
                                  niter=10,
                                  seed=12345678)

            # multiprocessing not working yet
            # else:
            #     from optimparallel import minimize_parallel
            #     result = minimize_parallel(self.negloglike_func_s,
            #                                x0=initials, parallel={"max_workers": num_proc})
            all_runs.append(result)
            if result.success:
                success = True
                break
                # TODO: test if more success runs are necessary
                # if len(success_runs) > 5:
                #     break
            count_run += 1
        if not success:
            logger.warning("optimization failed!")  # TODO: find the error object from scipy
        # TODO: test if more success runs are necessary
        # result = sorted(success_runs, key=lambda x: x.fun)[0]
        result = all_runs[-1]
        logger.info("Loglikelihood: %s" % result.fun)
        self.__result = result
        self.__summarize_ml_result(attribute=attribute, v_id=v_id)

    def build_ancestral_vertices(self):
        if self.dim_inverse_transform is None:
            for anc_node_id in range(self.tree.ntips, self.tree.nnodes):
                self.tree[anc_node_id].vertices = \
                    Vertices(self.vv_translator.to_vertices(self.tree[anc_node_id].vectors,
                                                            weights=self._weights[anc_node_id]))
        else:
            for anc_node_id in range(self.tree.ntips, self.tree.nnodes):
                real_vectors = self.dim_inverse_transform(self.tree[anc_node_id].vectors)
                self.tree[anc_node_id].vertices = \
                    Vertices(self.vv_translator.to_vertices(real_vectors,
                                                            weights=self._weights[anc_node_id]))

    def reconstruct_ancestral_shapes_using_ml(
            self,
            mode="network-local",
            num_vs: int = 20,
            num_vt_iter: int = 5,
            weighted: bool = True,
            num_proc: int = 1,
            multivariate: bool = True):
        """
        maximum likelihood approach
        """
        self.build_vv_translator(mode, num_vs=num_vs, num_vt_iter=num_vt_iter)
        self.build_tip_vectors(weighted=weighted)
        self.sym_ancestral_state(attribute="vectors")
        if multivariate:
            for go_v in range(len(self.tree[0].vectors)):
                self.formularize_log_like(attribute="vectors", v_id=go_v, log_func=s_log)
                self.functionalize_log_like()
                self.minimize_negloglike(attribute="vectors", v_id=go_v, num_proc=1)  # multiprocessing not working yet
        else:
            self.formularize_log_like(attribute="vectors", log_func=s_log)
            self.functionalize_log_like()
            self.minimize_negloglike(attribute="vectors", num_proc=1)  # multiprocessing not working yet
        self.build_ancestral_vertices()

    def perform_gpa(self, scale: bool):
        coords_list = [self.tree[node_id].vertices.coords for node_id in range(self.tree.ntips)]
        self.gpa_obj = GeneralizedProcrustesAnalysis(coords_list, scale=scale)
        self.gpa_obj.fit_transform()
        for node_id in range(self.tree.ntips):
            self.tree[node_id].gpa_vertices = Vertices(self.gpa_obj.coords_list[node_id])

    def build_tip_moves(self):
        ref_coords = self.gpa_obj.ref
        if self.dim_transform is None or self.dim_inverse_transform is None:
            for node_id in range(self.tree.ntips):
                leaf_node = self.tree[node_id]
                # _, vts = self.gpa_obj.shapes[leaf_node.name]
                # leaf_node.gpa_vertices = vts
                leaf_node.moves = leaf_node.gpa_vertices.coords - ref_coords
        else:
            moves_list = []
            for node_id in range(self.tree.ntips):
                leaf_node = self.tree[node_id]
                # _, vts = self.gpa_obj.shapes[leaf_node.name]
                # leaf_node.gpa_vertices = vts
                # moves_list.append(vts.coords - ref_coords)
                moves_list.append(leaf_node.gpa_vertices.coords - ref_coords)
            # transform the original moves to modified (e.g. PCA-transformed) ones
            for node_id, trans_moves in enumerate(self.dim_transform(moves_list)):
                self.tree[node_id].moves = trans_moves
            logger.info("Dimension {} -> {}".format(moves_list[0].shape, self.tree[0].moves.shape))
        logger.info("Post-GPA moves for {} tips assigned.".format(self.tree.ntips))

    def build_ancestral_moves(self):
        ref_coords = self.gpa_obj.ref
        if self.dim_inverse_transform is None:
            for anc_node_id in range(self.tree.ntips, self.tree.nnodes):
                self.tree[anc_node_id].gpa_vertices = \
                    Vertices(ref_coords + self.tree[anc_node_id].moves)
        else:
            for anc_node_id in range(self.tree.ntips, self.tree.nnodes):
                real_moves = self.dim_inverse_transform(self.tree[anc_node_id].moves)
                self.tree[anc_node_id].gpa_vertices = \
                    Vertices(ref_coords + real_moves)

    def reconstruct_ancestral_shapes_using_gpa(
            self,
            scale: bool = True,
            multivariate: bool = True):
        self.perform_gpa(scale=scale)
        self.build_tip_moves()
        self.sym_ancestral_state(attribute="moves")
        if multivariate:
            for go_v in range(len(self.tree[0].moves)):
                self.formularize_log_like(attribute="moves", v_id=go_v, log_func=s_log)
                self.functionalize_log_like()
                self.minimize_negloglike(attribute="moves", v_id=go_v, num_proc=1)  # multiprocessing not working yet
        else:
            self.formularize_log_like(attribute="moves", log_func=s_log)
            self.functionalize_log_like()
            self.minimize_negloglike(attribute="moves", num_proc=1)  # multiprocessing not working yet
        self.build_ancestral_moves()








