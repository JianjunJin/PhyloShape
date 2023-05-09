#!/usr/bin/env python

"""Alignment of 2 or more Models.

"""

from typing import Sequence, List
# import numpy as np
from phyloshape.shape.src.model import Model
from phyloshape.shape.src.core import Vector


class Alignment(dict):
    """A container class for multiple Model objects.

    An alignment container is used to measure variance statistics
    on a set of Models prior to an analysis as a pretext to design
    a vector traversal order. The models in an Alignment always have
    homologous vector paths.
    """
    def __init__(self, models: Sequence[Model], labels: Sequence[str] = None):

        for label, model in zip(labels, models):
            self[label] = model
        self.vectors: List[Vector] = None
        # self.nlandmarks: int = len(models[0].vertices)

    def _find_duplicate(self):
        """..."""

    def _deduplicate(self):
        """..."""


if __name__ == "__main__":
    pass
