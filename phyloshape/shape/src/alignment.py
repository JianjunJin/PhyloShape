#!/usr/bin/env python

"""Alignment of 2 or more Models.

NOTE: I'm not sure how this object is useful...

- select models by index
- asserts models are same dimensions
- stores vectors for each model
- stores vertex inference order
"""

from typing import Sequence, List
# import numpy as np
from phyloshape.shape.src.model import Model
from phyloshape.shape.src.core import Vector


class Alignment(list):
    """A container class for multiple Model objects.

    An alignment container is used to measure variance statistics
    on a set of Models prior to an analysis as a pretext to design
    a vector traversal order. The models in an Alignment always have
    homologous vector paths.
    """
    def __init__(self, models: Sequence[Model], labels: Sequence[str] = None):    
        self._models = models
        self._labels = labels

    def _find_duplicate(self):
        """..."""

    def _deduplicate(self):
        """..."""


if __name__ == "__main__":

    from pathlib import Path
    import numpy as np
    import phyloshape
    phyloshape.set_log_level("DEBUG")

    from phyloshape.data import get_gesneriaceae_models

    data = get_gesneriaceae_models()
    print(data)

    # # path to directory with landmark CSVs
    # GIGA_DIR = Path("/home/deren/Documents/PhyloShapeTest/data/Gesneriaceae.Gigascience.2020/")
    # CSVS = list(GIGA_DIR.glob("[0-9]*.csv"))[:5]

    # # get number of landmarks
    # with open(CSVS[0], 'r') as indat:
    #     nmarks = int(len(indat.readline().split(",")) / 3)

    # # load all models and reshape landmarks to (x, y, z)
    # models = {}
    # for csv in CSVS:
    #     sample = csv.name.rsplit(".", 1)[0]
    #     with open(csv, 'r') as indata:
    #         data = indata.readline()
    #         try:
    #             arr = np.array(data.split(",")).reshape((nmarks, 3), order="F").astype(float)
    #         except ValueError:
    #             arr = np.array(data.split()).reshape((nmarks, 3), order="F").astype(float)

    #         # store as a Shape object
    #         model = phyloshape.io.load_model_from_coordinates(arr - 1)
    #         models[sample] = model

    # models = phyloshape.io.load_model_from_coordinates()

    # m = models["34_HC3403-3_17"]
    # verts = np.array([v.coords for v in m.vertices])
    # print(verts)

    # Alignment()
