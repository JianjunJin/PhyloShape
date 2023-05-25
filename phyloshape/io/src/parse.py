#!/usr/bin/env python

"""...

"""

from typing import Optional, Iterator, List, Union
from pathlib import Path
from collections import deque

import numpy as np
from loguru import logger
from PIL import Image
from plyfile import PlyData
from phyloshape.io.src.paths import get_image_path
from phyloshape.shape.src.core import Vertex, Face
from phyloshape.shape.src.model import Model
from phyloshape.utils import PSIOError, ID_TYPE
logger = logger.bind(name="phyloshape")


__all__ = [
    "load_model_from_ply",
    "load_model_from_obj",
    "load_model_from_coordinates",
]


def load_model_from_ply(
    path: Path,
    components: Optional[int] = 0,
) -> Model:
    """Return Model parsed from data in a PLY formatted 3D model file.

    Parameters
    ----------
    path: str or Path
        File path to a .ply 3D object.
    components: int, Slice, or None
        If model contains multiple sets of vertices not connected by
        face edges the default option (0) is to select only the largest
        set as the main model. Multiple compnents can be selected with
        a slice, or None retains all components.
    """
    # parse PLY object from file using plyfile lib
    logger.info(f"parsing PLY file: {path}")
    obj = PlyData.read(path)

    # load vertex coordinates and face ID sets from ply data
    coords = np.stack([obj["vertex"][i] for i in "xyz"], axis=1)
    fidxs = np.vstack(obj["face"]["vertex_indices"]).astype(ID_TYPE)

    # load RGB values as int array or set all to 0
    try:
        colors = np.stack([
            obj["vertex"][i] for i in ('red', 'green', 'blue')
        ], axis=1).astype(np.uint8)
    except ValueError:
        colors = np.zeros(coords.shape, dtype=np.uint8)

    # store point data as Vertex objects
    verts = [Vertex(i, coords[i], colors[i]) for i in range(coords.shape[0])]

    # store trios of Vertex objects as Face objects
    faces = [Face((verts[i], verts[j], verts[k])) for i, j, k in fidxs]

    # store vertices and faces to a Model object
    model = Model(verts, faces)

    # extract main cluster from disconnected artifacts
    return extract_components(model, 0)


def load_model_from_obj(
    path: Path,
    texture: Optional[Path] = None,
    components: Optional[int] = 0,
) -> Model:
    """Parse an OBJ file to fill the Shape object data.

    OBJ files contain tabular data formatted into different
    sections corresponding to different data elements of a 3D
    model object: v=vertices, vt=..., f=faces.

    Parameters
    ----------
    obj: str or Path
        File path to a .obj 3D object.
    texture: str, Path, or None
        Optional image file with textures for the 3D object faces. If
        None an image file will be found if it is in the same directory
        as the object file, same name prefix, and has a supported
        suffix (e.g., '.jpg')

    References
    ----------
    https://en.wikipedia.org/wiki/Wavefront_.obj_file
    """
    # store data from the files.
    coords = []
    colors = []
    faces_v = []
    faces_t = []
    textures = []

    # iterate over lines of file parsing v, vt, or f elements
    with open(path, "r", encoding="utf-8") as obj_io:
        for lidx, line in enumerate(obj_io):
            line = line.strip().split()
            if not line:
                continue

            # v = vertex in raw coordinates:
            # List of geometric vertices, with (x, y, z, [w]) coordinates,
            # w is optional and defaults to 1.0.
            if line[0] == "v":
                # coordinates only data
                if len(line) == 4:
                    coords.append([float(i) for i in line[1:4]])
                    colors.append((0, 0, 0))
                # coordinate and RGB data
                elif len(line) == 7:
                    coords.append([float(i) for i in line[1:4]])
                    colors.append([float(i) for i in line[4:]])
                else:
                    raise PSIOError(f"invalid line {lidx} in {path}")

            # vt = vertex texture coordinates:
            # List of texture coordinates, in (u, [v, w]) coordinates.
            # these vary between 0 and 1. v, w are optional with default 0
            elif line[0] == "vt":
                textures.append([float(i) for i in line[1:3]])

            # Polygonal face element (see below)
            # f 1 2 3               # vertex indices
            # f 3/1 4/2 5/3         # vertex/texture indices
            # f 1 2 3 4             # >3 polygon
            elif line[0] == "f":
                # force spliting larger faces into triangles.
                for tri in iter_triangles(line[1:]):
                    tri_v = []
                    tri_t = []
                    for vt in tri:
                        if "/" in vt:
                            v_, t_ = vt.split("/")
                            tri_v.append(int(v_) - 1)
                            tri_t.append(int(t_) - 1)
                        else:
                            tri_v.append(int(vt) - 1)
                    faces_v.append(tri_v)
                    faces_t.append(tri_t)

    # convert image file to a Path, search for it by name, or None
    # ipath = get_image_path(texture, path)
    # read image obj and update the image data stored to self.
    # texture_image_data = None
    # if ipath:
    #     texture_image_obj = Image.open(ipath)
    #     texture_image_data = np.asarray(texture_image_obj)

    # load Vertices object to store coordinates and color
    vertices = [
        Vertex(id=i, coords=coords[i], color=colors[i])
        for i in range(len(coords))
    ]

    # creat Face objects as triplets of Vertex objects
    faces = []
    for i, j, k in faces_v:
        faces.append((Face((vertices[i], vertices[j], vertices[k]))))

    # return as a Shape object
    model = Model(vertices=vertices, faces=faces)
    return extract_components(model, 0)


def load_model_from_coordinates(coordinates: np.ndarray) -> Model:
    """Return Model loaded from an array of vertex x,y,z coordinates.

    This function is used to load a Model composing points in coordinate
    space from landmark data.
    """
    colors = np.zeros(coordinates.shape, dtype=np.uint8)
    vertices = [
        Vertex(id=i, coords=coordinates[i], color=colors[i])
        for i in range(coordinates.shape[0])
    ]
    return Model(vertices=vertices, faces=[])


#####################################################################
# convenience functions
#####################################################################

def extract_components(model: Model, components: Union[slice, int, None] = 0) -> Model:
    """Extract connected component(s) from a shape object

    The connected components are sorted decreasingly by the number
    of vertices so that the first component (component_id=0) is the
    largest. This is used to exclude model junk disconnected from
    the main model mesh.

    :param component_id:
    :return: Shape
    """
    # requires graph information
    if model._graph is None:
        model._build_graph_from_faces()

    # sort for longest first, and if tied, then lowest vertex id
    clusters = model._graph._get_vertex_clusters()
    clusters = sorted(clusters, key=lambda x: (-len(x), x))

    # return Model if no extra components present
    if len(clusters) == 1 or components is None:
        return model

    # get vertices from selected component(s) sorted
    if isinstance(components, int):
        chosen_v_ids = set(clusters[components])
        logger.debug(
            f"found {len(clusters)} components of connected vertices; "
            f"selecting largest ({len(chosen_v_ids)} vertices)")
    else:
        chosen_v_ids = set.union(*clusters[components])
        logger.debug(
            f"found {len(clusters)} components of connected vertices; "
            f"selecting slice of ({len(chosen_v_ids)} vertices")

    # subset vertices
    verts = [i for i in model.vertices if i.id in chosen_v_ids]

    # remove any faces that contain removed vertex IDs
    faces = [i for i in model.faces if all(v.id in chosen_v_ids for v in i)]

    # relabel vertices to 0-nverts which similarly relabels linked faces
    for vidx, vert in enumerate(verts):
        vert.id = vidx

    # re-build Model and graph with only selected component data
    model = Model(verts, faces)
    model._build_graph_from_faces()
    return model


def iter_triangles(face: np.ndarray) -> Iterator[List[int]]:
    """Return face as triplets of vertex IDs.

    This is used to split faces of larger polygons, such as faces of
    4 or more vertices, into smaller triangles of 3 vertices.
    """
    tri = []
    f = deque(face)
    while f:
        tri.append(f.popleft())
        if len(tri) == 3:
            yield tri
            f.append(tri[0])
            tri = [tri[-1]]


if __name__ == "__main__":

    import phyloshape
    phyloshape.set_log_level("DEBUG")

    MODEL_FILE = "/usr/share/inkscape/extensions/Poly3DObjects/cube.obj"
    # test = load_shape_from_obj(MODEL_FILE)
    # print(test.faces)

    test = load_model_from_obj(MODEL_FILE)
    print(test)

    MODEL_FILE = "/home/deren/Documents/PhyloShapeTest/data/Gesneriaceae.Gigascience.2020/12_K039105_04.ply"
    MODEL_FILE = "/home/deren/Documents/PhyloShapeTest/data/cranolopha_DE183.ply"

    # test = load_shape_from_ply(MODEL_FILE)
    # print(test)

    # test = load_model_from_ply(MODEL_FILE)
    # print(test)
    # print(test._graph.get_all_paths_from(1))
