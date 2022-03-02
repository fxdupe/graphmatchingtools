"""
HiPPI algorithm as described in ICCV 2019 paper

.. moduleauthor:: François-Xavier Dupé
"""
import numpy as np

import graph_matching_tools.algorithms.multiway.utils as utils


def hippi_multiway_matching(s, sizes, knode, u_dim, iterations=100, tolerance=1e-6, init=None):
    """
    HiPPI method for multi-graph matching based on a power method
    :param np.ndarray s: the matrix with all the pairwise assignment
    :param list sizes: the number of nodes of the different graphs (in order)
    :param np.ndarray knode: the node affinity matrix
    :param int u_dim: the dimension of the universe of nodes
    :param int iterations: the maximal number of iterations
    :param float tolerance: the tolerance for convergence
    :param np.ndarray init: the initialization, random if None
    :return: the universe of node projection for all the nodes
    """
    if init is None:
        u = np.ones((s.shape[0], u_dim)) / u_dim + 1e-3 * np.random.randn(s.shape[0], u_dim)
    else:
        u = init

    w = knode.T @ s @ knode
    vi = w @ u @ u.T @ w @ u
    fu = np.trace(u.T @ vi)

    for i in range(iterations):
        u = utils.u_projector(vi, sizes)
        vi = w @ u @ u.T @ w @ u
        n_fu = np.trace(u.T @ vi)
        if np.abs(n_fu - fu) < tolerance:
            break
        fu = n_fu

    return u
