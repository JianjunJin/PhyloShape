#!/usr/bin/env python

"""Simulate a random beak shape.

This randomly samples simulation parameters from a range of supplied
values and generates a simulated 'beak-like' shape. A correlation
structure (MVN) is described among several parameters.

This is just a test script currently.
"""

from scipy import stats
import numpy as np
# import pandas


corr = np.array([
    [1.0,   0.7,  0.5,  0.5, -0.5, -0.5],
    [0.7,   1.0,  0.5,  0.5,  0.0, -0.1],
    [0.5,   0.5,  1.0,  0.2,  0.0, -0.2],
    [0.5,   0.5,  0.2,  1.0,  0.0, -0.2],
    [-0.5,  0.0,  0.0,  0.0,  1.0,  0.5],
    [-0.5, -0.1, -0.2, -0.2,  0.5,  1.0],
])

# convert corr matrix to cov matrix
std_deviations = np.sqrt(np.diag(corr))
std_matrix = np.diag(std_deviations)
covariance_matrix = std_matrix @ corr @ std_matrix
covariance_matrix

mean = np.array([5, 0, 0, 0, 2, 1.5])
rvs = stats.multivariate_normal.rvs(mean, covariance_matrix)
rvs

# random
rvs = stats.multivariate_normal.rvs(mean, covariance_matrix)
kwargs = dict(zip(["length", "rotation", "curve_x", "curve_y", "beak_radius_start"], rvs))
print(kwargs)
