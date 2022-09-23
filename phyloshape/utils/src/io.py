#!/usr/bin/env python

from typing import Optional, Union
import os
from pathlib import Path
from loguru import logger


def find_image_file(shape_file_name):
    image_file = ""
    if shape_file_name.endswith(".obj"):
        if os.path.isfile(shape_file_name[:-3] + "jpg"):
            image_file = shape_file_name[:-3] + "jpg"
        elif os.path.isfile(shape_file_name[:-3] + "jpeg"):
            image_file = shape_file_name[:-3] + "jpeg"
        elif os.path.isfile(shape_file_name[:-3] + "png"):
            image_file = shape_file_name[:-3] + "png"
        elif os.path.isfile(shape_file_name[:-3] + "tiff"):
            image_file = shape_file_name[:-3] + "tiff"
        elif os.path.isfile(shape_file_name[:-3] + "tif"):
            image_file = shape_file_name[:-3] + "tif"
        else:
            logger.warning("No concomitant jpeg file found!")
    else:
        if os.path.isfile(shape_file_name + ".jpg"):
            image_file = shape_file_name + ".jpg"
        elif os.path.isfile(shape_file_name + ".jpeg"):
            image_file = shape_file_name + ".jpeg"
        elif os.path.isfile(shape_file_name + ".png"):
            image_file = shape_file_name + ".png"
        elif os.path.isfile(shape_file_name + ".tiff"):
            image_file = shape_file_name + ".tiff"
        elif os.path.isfile(shape_file_name + ".tif"):
            image_file = shape_file_name + ".tif"
        else:
            logger.warning("No concomitant jpeg file found!")
    return image_file


def find_image_path(fname: Union[str, Path]) -> Optional[Path]:
    """Return path to an image/texture file associated with an object file.

    This is a convenience function. It checks for the existence of a
    texture image file (e.g., .jpg, .png) that shares the same filepath
    basename as a 3D model object (e.g., .ply, .obj). This function 
    tries to detect this associated file automatically without 
    requiring the user to enter the full path or file suffix. If 
    detected, it returns a Path object, else it returns None.
    """
    fpath = Path(fname)
    for suffix in (".jpg", ".jpeg", ".png", ".tiff", ".tif"):
        if fpath.with_suffix(suffix).exists():
            return fpath.with_suffix(suffix)
    logger.warning(
        f"No image/texture file found associated with {fname}. "
        "Expecting to find a .jpg, .tiff, etc. in the same file path.")
    return None
