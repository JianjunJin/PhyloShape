#!/usr/bin/env python

"""
The core subpackage contains modules for manipulating and visualizing
Shape objects.

Examples
--------
>>> import phyloshape

Parse shape data to a Shape instance:
>>> shape = phyloshape.Shape("...")

"""

from phyloshape.shape.shape import Shape
from phyloshape.shape.vectors import VertexVectorMapper
