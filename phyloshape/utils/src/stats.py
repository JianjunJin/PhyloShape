#!/usr/bin/env python

import numpy as np


# Define a function to remove outliers
def remove_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def mean_without_outliers(data):
    return np.mean(remove_outliers(data))


