#!/usr/bin/env python

"""
The subpackage contains modules for color profiling

Examples
--------
>>> from phyloshape import Shape
>>> from phyloshape.portrait import ColorProfile

Parse shape data to a Shape instance:

>>> shape = Shape("...")

Create a ColorProfile instance with the shape:

>>> color_p = ColorProfile(shape=shape)

Generate color variation distrubtion:

>>> color_p.color_variation_across_vertices(dist_values=[0.01, 0.03, 0.05], n_start_vertices=500)

"""

from phyloshape.portrait.src.color import ColorProfile
