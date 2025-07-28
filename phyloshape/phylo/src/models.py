#!/usr/bin/env python

"""models describing how vertices moving through time, particularly how neighboring vertices covariate"""
from symengine import Symbol, symbols
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


class MultivariateBrownian(Brownian):
    def __init__(self, n_var: int = None):
        super(MultivariateBrownian, self).__init__()
        if n_var is None:
            self.rate = None
        else:
            self.rate = np.array(symbols("sigma[:{}]^2".format(n_var)))

    def form_log_like(self, time, from_states, to_states, log_f):
        if self.rate is None:
            # initialized rate list
            assert len(from_states) == len(to_states)
            self.rate = np.array(symbols("sigma[:{}]^2".format(len(from_states))))
        else:
            assert len(from_states) == len(to_states) == len(self.rate)
        return super(MultivariateBrownian, self).form_log_like(
            time=time, from_states=from_states, to_states=to_states, log_f=lambda x: np.array([log_f(_x) for _x in x]))

    def get_parameters(self):
        if self.rate is None:
            raise ValueError("rate is None!")
        else:
            return self.rate


class PotentialOtherModels(MotionModel):
    def __init__(self):
        super(PotentialOtherModels).__init__()

    def form_log_like(self, time, from_states, to_states, topology_info, log_f):
        # TODO: something related to the topology_info
        pass

    def get_parameters(self):
        pass


# TODO: think about what the sample size is
