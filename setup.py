#!/usr/bin/env python

"""
    pip install -e . --no-deps
"""

import re
from setuptools import setup, find_packages


# get version from __init__.py
INIT_FILE = "phyloshape/__init__.py"
CUR_VERSION = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                        open(INIT_FILE, "r").read(),
                        re.M).group(1)

# run setup
setup(
    name="phyloshape",
    version=CUR_VERSION,
    url="https://github.com/Kinggerm/phyloshape",
    author="JianJun Jin; Yue Yang; Deren Eaton",
    author_email="jianjun.jin@columbia.edu",
    description="ancestral shape reconstruction toolkit",
    long_description=open('README.md').read(),
    # long_description_content_type='text/x-rst',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "symengine",
        "python-symengine",
        "k3d",
        "loguru",
        "plyfile",
        "toytree",  # temporarily
    ],
    entry_points={},
    license='GPLv3',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
