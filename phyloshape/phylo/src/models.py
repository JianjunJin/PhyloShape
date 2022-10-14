#!/usr/bin/env python

"""models describing how vertices moving through time, particularly how neighboring vertices covariate"""
from symengine import Symbol
# from sympy import Symbol
import numpy as np
from loguru import logger
logger = logger.bind(name="phyloshape")


class MotionModel:
    """Base class for all models

    """
    def __init__(self):
        # universal rate
        self.rate = Symbol("sigma^2")

    def get_parameters(self):
        return [self.rate]


class Brownian(MotionModel):
    """The simplest model

    """
    def __init__(self):
        super(Brownian, self).__init__()

    def form_log_like(self, time, from_states, to_states, log_f):
        return np.sum(-1 / 2 * log_f(self.rate * time) - 1 / (2 * self.rate * time) * (to_states - from_states) ** 2)


class PotentialOtherModels(MotionModel):
    def __init__(self):
        super(PotentialOtherModels).__init__()

    def form_log_like(self, time, from_states, to_states, topology_info, log_f):
        # TODO: something related to the topology_info
        pass

    def get_parameters(self):
        pass


# TODO: think about what the sample size is
