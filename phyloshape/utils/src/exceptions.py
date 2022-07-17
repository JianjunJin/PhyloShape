#!/usr/bin/env python

"""
Custom exception classes
"""


class PSIOError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)

