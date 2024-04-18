"""
MSync method for graph matching synchronization.

.. moduleauthor:: François-Xavier Dupé
"""

from typing import Optional

import numpy as np

import graph_matching_tools.algorithms.multiway.utils as utils


def msync(
    x: np.ndarray, sizes: list[int], rank: int, choice: Optional[int] = None
) -> np.ndarray:
    """The projector on the universe of nodes of a given rank (or dimension).

    :param np.ndarray x: the input permutation matrix.
    :param list[int] sizes: the sizes of the graphs.
    :param int rank: the rank of the universe of nodes.
    :param Optional[int] choice: the reference graph.
    :return: the projection of the nodes.
    :rtype: np.ndarray

    Here an example using NetworkX and some utils:

    .. doctest:

    >>> node_kernel = kern.create_gaussian_node_kernel(0.1, "weight")  # Load a Gaussian kernel
    >>> knode = utils.create_full_node_affinity_matrix(graphs, node_kernel)  # Get the affinity matrix between nodes
    >>> perm = msync.msync(knode, [2, 2, 3], 3, choice = 0)  # Compute the universe of nodes
    >>> perm  # Show the permutation matrix
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 1., 0.],
           [1., 0., 0.],
           [0., 0., 0.],
           [1., 0., 0.],
           [0., 1., 0.]])
    """
    if choice is None:
        choice = 0

    s, u = np.linalg.eig(x)
    s = np.real(s)
    idx = np.argsort(s)
    u = np.real(u[:, idx[idx.shape[0] - rank :]])
    u = utils.permutation_projector(u / np.sqrt(len(sizes)), sizes, choice)
    return u
