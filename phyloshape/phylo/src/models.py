#!/usr/bin/env python

"""models describing how vertices moving through time, particularly how neighboring vertices covariate"""
from sympy import Symbol
from sympy import log as s_log
import numpy as np


class MotionModel:
    """Base class for all models

    """
    def __init__(self):
        # universal rate
        self.rate = Symbol("sigma^2")


class Brownian(MotionModel):
    """The simplest model

    """
    def __init__(self):
        super().__init__()

    def form_log_like(self, time, from_states, to_states):
        return np.sum(-1 / 2 * s_log(self.rate * time) - 1 / (2 * self.rate * time) * (to_states - from_states) ** 2)


class PotentialOtherModels(MotionModel):
    def __init__(self):
        super().__init__()

    def form_log_like(self, time, from_states, to_states, topology_info):
        # TODO: something related to the topology_info
        pass
