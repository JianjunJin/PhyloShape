#!/usr/bin/env python

"""...

"""

from typing import Union, Optional
from pathlib import Path
from loguru import logger

logger = logger.bind(name="phyloshape")


def _find_image_path(fname: Union[str, Path]) -> Optional[Path]:
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
        f"No image/texture file (e.g., jpg, tif) found associated with {fpath.name}")
    return None


def get_image_path(texture: Union[str, Path, None], obj: Path) -> Union[Path, None]:
    """Return a path to a image/texture file.

    Parameters
    ----------
    fname: str, Path
        Path to a JPG, TIF, PNG, etc. type file.

    If a fname is entered then it is returned. If fname is None
    then the current texture path of self is returned. If self
    does not have a current texture path then one is searched for
    with the same pathname as the object file, but allowing for
    flexible suffices. Finally, if no file is found then a warning
    is logged but we proceed and return None.
    """
    # if fname then return as a Path object
    if texture is not None:
        return Path(texture)
    # else search for a matched image name from opath basename
    searched_path = _find_image_path(obj)
    if searched_path is not None:
        return Path(searched_path)
    # no image file found
    return None
