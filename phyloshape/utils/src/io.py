#!/usr/bin/env python

import os
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

