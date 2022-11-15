""""
This code is from the paper
"Sparse Quadratic Optimisation over the Stiefel Manifold with Application to Permutation Synchronisation" (NeurIPS 2021)

.. moduleauthor:: François-Xavier Dupé
"""
import numpy as np

import graph_matching_tools.algorithms.multiway.utils as utils


def sparse_stiefel_manifold_sync(x, rank, sizes, iterations=100, power=3):
    """Sparse quadratic optimization over the Stiefel manifold for permutation synchronization

    :param np.ndarray x: the input "noisy" permutation matrix
    :param int rank: the size of the universe of nodes
    :param list[int] sizes: the size of the different graphs
    :param int iterations: the maximal number of iterations
    :param int power: the sparsity constraints (see paper)
    :return: the universe of nodes
    """
    u = np.random.normal(0, 1, size=(x.shape[0], rank))

    for i in range(iterations):
        tmp1 = u.T @ (u ** (power - 1))
        alpha = np.max(np.abs(tmp1 - tmp1.T))
        if alpha < 1e-3:
            break
        tmp2 = np.eye(rank) + (tmp1 - tmp1.T) / alpha
        q, r = np.linalg.qr(x @ u @ tmp2)
        u = q @ np.diag(np.sign(np.diag(r)))

    u = utils.u_projector(u, sizes)
    return u
