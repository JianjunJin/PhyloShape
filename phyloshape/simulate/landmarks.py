#!/usr/bin/env python

"""...

"""

import numpy as np
from phyloshape.shape.src.core import Vertex, Vector


def _generate_horn_path(
    length: float = 5,
    curve_radius_start: float = 1,
    curve_radius_end: float = None,
    curve_x: float = 1,
    curve_y: float = 1,
    rotation: float = 2 * np.pi,
    vector: np.ndarray = None,
    num_intervals: int = 20,
) -> np.ndarray:
    """Generate points along a spiral in x, y, z forming the multi-vector path that
    the beak will follow.

    Parameters
    ----------
    length: float
        length of vector beak will follow (rotate around)
    curve_radius_start: float
        radius of circle around which beak is rotating at start
    curve_radius_end: float
        radius of circle around which beak is rotating at end
    curve_x: float
        multiplier of cosine rotation (0 - 1)
    curve_y: float
        multiplier of sin rotation (0 - 1)
    rotation: ndarray
        rotation of beak around circle in radians
    vector: ndarray
        unit vector path of beak; default=[0, 0, 1]
    num_intervals: int
        number of intervals at which points are sampled along vector
    """
    # defaults
    curve_radius_end = curve_radius_start if curve_radius_end is None else curve_radius_end
    vector = np.array([0, 0, 1]) if vector is None else vector

    # curve_[x,y] scale the cos vs sin function between -1 and 1 magnitudes
    curve_x = min(1, max(-1, curve_x))
    curve_y = min(1, max(-1, curve_y))

    # generate points for the surface
    theta = np.linspace(0, rotation, num_intervals)
    curve_radii = np.linspace(curve_radius_start, curve_radius_end, num_intervals)
    x = curve_x * curve_radii * np.cos(theta)
    y = curve_y * curve_radii * np.sin(theta)
    z = np.linspace(0, length, num_intervals)

    # Create the point cloud and center first point on [0, 0, 0]
    point_cloud = np.column_stack((x, y, z))
    point_cloud -= point_cloud[0]

    # todo: rotate by vector ...
    # ...
    return point_cloud


def _get_distance_ratio(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
) -> float:
    """get ratio of rx and ry of circle"""
    verts = [Vertex(i, j) for i, j in enumerate([p0, p1, p2, p3])]
    v0 = Vector(verts[0], verts[2])
    v1 = Vector(verts[1], verts[3])
    return v0.dist / v1.dist


def _generate_points_on_circle(
    radius: float,
    num_points: int,
    vector: np.ndarray,
    origin: np.ndarray,
    correct=False,
) -> np.ndarray:
    """Generate N points along the radius of a circle oriented in 3D space
    by a vector, with the specified origin position.

    Parameters
    -----------
    radius (float):
        Radius of the circle.
    num_points (int):
        Number of points to generate.
    vector (ndarray):
        3D vector representing the orientation of the circle.
    origin (ndarray):
        3D vector representing the origin position of the circle.
    """
    theta_test = np.array([0, np.pi / 2, np.pi, np.pi * 3 / 2])

    # get rotation matrix
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    u = np.cross(vector, [1, 0, 0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(vector, [0, 1, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(vector, u)

    # ensure it is a circle instead of ellipse
    if correct:
        points = (
            origin + radius * np.outer(np.cos(theta_test), u)
            + radius * np.outer(np.sin(theta_test), v)
        )
        ratio = _get_distance_ratio(*points)
        v *= ratio

    # get points along circumference of circle
    points = (
        origin
        + radius * np.outer(np.cos(theta), u)
        + radius * np.outer(np.sin(theta), v)
    )
    return points


def get_beak_landmarks(
    length: float = 5,                 #
    rotation: float = 2 * np.pi,       #
    curve_radius_start: float = 1.0,
    curve_radius_end: float = 1.0,
    curve_x: float = 1.0,              #
    curve_y: float = 1.0,              #
    beak_radius_start: float = 1.0,    #
    beak_radius_end: float = 0.1,
    num_intervals: int = 20,
    num_disc_points: int = 20,
) -> np.ndarray:
    """Return array of 3D landmarks on a beak shape.

    The beak shape is generated according to the parameters affecting
    component lengths, radii, and curvatures, and with additional
    parameters setting the number of landmark points to sample.
    """
    # point cloud to fill
    points = np.zeros((num_intervals, num_disc_points, 3), dtype=np.float32)

    # get path of beak
    horn_path = _generate_horn_path(
        length=length,
        curve_radius_start=curve_radius_start,
        curve_radius_end=curve_radius_end,
        curve_x=curve_x,
        curve_y=curve_y,
        rotation=rotation,
        num_intervals=num_intervals + 1,
    )

    # get radii of discs
    radii = np.geomspace(beak_radius_start, beak_radius_end, num_intervals)

    # get points on path as Vertex objects
    vertices = [Vertex(i, horn_path[i]) for i in range(horn_path.shape[0])]

    # fill point cloud with disc data
    for i in range(len(vertices) - 1):
        v = Vector(vertices[i], vertices[i + 1])

        points[i] = _generate_points_on_circle(
            radius=radii[i],
            num_points=num_disc_points,
            vector=v.absolute,
            origin=vertices[i].coords,
            correct=True,
        ).astype(np.float32)
    return points


if __name__ == "__main__":
    arr = get_beak_landmarks()
    print(arr.shape)
    print(arr)
