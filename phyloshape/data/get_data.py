#!/usr/bin/env python


from pathlib import Path
import numpy as np
import phyloshape


def get_gesneriaceae_models():
    # path to directory with landmark CSVs
    GIGA_DIR = Path("/home/deren/Documents/PhyloShapeTest/data/Gesneriaceae.Gigascience.2020/")
    CSVS = list(GIGA_DIR.glob("[0-9]*.csv"))[:5]

    # get number of landmarks
    with open(CSVS[0], 'r') as indat:
        nmarks = int(len(indat.readline().split(",")) / 3)

    # load all models and reshape landmarks to (x, y, z)
    models = {}
    for csv in CSVS:
        sample = csv.name.rsplit(".", 1)[0]
        with open(csv, 'r') as indata:
            data = indata.readline()
            try:
                arr = np.array(data.split(",")).reshape((nmarks, 3), order="F").astype(float)
            except ValueError:
                arr = np.array(data.split()).reshape((nmarks, 3), order="F").astype(float)

            # store as a Shape object
            model = phyloshape.io.load_model_from_coordinates(arr - 1)
            models[sample] = model
    return models


if __name__ == "__main__":

    print(get_gesneriaceae_models())
