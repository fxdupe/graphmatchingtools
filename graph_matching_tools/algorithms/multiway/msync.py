"""
MSync method for graph matching synchronization

.. moduleauthor:: François-Xavier Dupé
"""
import numpy as np

import graph_matching_tools.algorithms.multiway.utils as utils


def _default_choice(_):
    """Default choice function for reference graph

    :return: 0 (aka the first graph)
    """
    return 0


def msync(x, sizes, rank, choice=None):
    """The projector on the universe of nodes of a given rank (or dimension)

    :param np.ndarray x: the input permutation matrix
    :param list[int] sizes: the sizes of the graphs
    :param int rank: the rank of the universe of nodes
    :param callable choice: the choosing function for the reference graph
    :return: the projection of the nodes
    """
    if choice is None:
        choice = _default_choice

    s, u = np.linalg.eig(x)
    s = np.real(s)
    idx = np.argsort(s)
    u = np.real(u[:, idx[idx.shape[0] - rank:]])
    u = utils.permutation_projector(u / np.sqrt(len(sizes)), sizes, choice)
    return u
