"""This module contains tools for sampling points on a 3D sphere.

..moduleauthor:: Marius Thorre, Rohit Yadav
"""

import numpy as np

from graph_matching_tools.utils.von_mises import sample_von_mises


def make_sphere(radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get mesh for a sphere of a given radius.

    :param float radius: the radius of the sphere.
    :return: tuple which contains the sphere coordinate
    :rtype: tuple
    """
    phi, theta = np.mgrid[0.0 : np.pi : 100j, 0.0 : 2.0 * np.pi : 100j]
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return x, y, z


def random_coordinate_sampling(
    nb_samples: int,
    mu: np.ndarray,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample random coordinate on a spherical surface (using von Mises - Fisher distribution).

    :param int nb_samples: number of samples.
    :param np.ndarray mu: mean of the vMF distribution.
    :param float kappa: variance vMF distribution.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    samples = sample_von_mises(mu, kappa, nb_samples)
    return samples[:, 0], samples[:, 1], samples[:, 2]


def random_sampling(
    vertex_number: int, radius: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a sphere with random sampling.

    :param int vertex_number: number of vertices in the output spherical mesh.
    :param float radius: radius of the output sphere.
    :return: a tuple with the coordinates of each point.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    coords = np.zeros((vertex_number, 3))
    for i in range(vertex_number):
        M = np.random.normal(size=(3, 3))
        Q, R = np.linalg.qr(M)
        coords[i, :] = Q[:, 0].transpose() * np.sign(R[0, 0])
    coords = radius * coords
    return coords[:, 0], coords[:, 1], coords[:, 2]


def regular_sampling(
    nb_point: int, radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate regular points on the sphere (using Fibonacci).

    :param int nb_point: the number of points to generate.
    :param float radius: the radius of the sphere.
    :return: the generated points.
    :rtype: tuple[np.ndarray]
    """
    inc = np.pi * (3 - np.sqrt(5))
    off = 2.0 / nb_point
    k = np.arange(0, nb_point)
    y = k * off - 1.0 + 0.5 * off
    r = np.sqrt(1 - y * y)
    phi = k * inc
    x = np.cos(phi) * r
    z = np.sin(phi) * r
    return x * radius, y * radius, z * radius
