#!/usr/bin/env python

"""
The core subpackage contains modules for manipulating and visualizing Shape objects.

Examples
--------
>>> import phyloshape

Parse shape data to a Shape instance:
>>> shape = phyloshape.Shape("...")

"""

from phyloshape.shape.src.shape import Shape, ShapeAlignment
from phyloshape.shape.src.face import Faces
from phyloshape.shape.src.vertex import Vertices
from phyloshape.shape.src.vectors import VertexVectorMapper
from phyloshape.shape.src.network import IdNetwork
