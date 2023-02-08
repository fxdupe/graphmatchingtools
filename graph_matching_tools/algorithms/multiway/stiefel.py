""""
This code is from the paper
"Sparse Quadratic Optimisation over the Stiefel Manifold with Application to Permutation Synchronisation"
(NeurIPS 2021).

.. moduleauthor:: François-Xavier Dupé
"""
import numpy as np

import graph_matching_tools.algorithms.multiway.utils as utils


def sparse_stiefel_manifold_sync(
    x: np.ndarray, rank: int, sizes: list[int], iterations: int = 100, power: int = 3
) -> np.ndarray:
    """Sparse quadratic optimization over the Stiefel manifold for permutation synchronization.

    :param np.ndarray x: the input "noisy" permutation matrix.
    :param int rank: the size of the universe of nodes.
    :param list[int] sizes: the size of the different graphs.
    :param int iterations: the maximal number of iterations.
    :param int power: the sparsity constraints (see paper).
    :return: the universe of nodes.
    :rtype: np.ndarray

    Here an example using NetworkX and some utils:

    .. doctest:

    >>> x_base = np.eye(4)
    >>> x_base[3, 0] = 1
    >>> x_base[0, 3] = 1
    >>> x_base[2, 1] = 1
    >>> x_base[1, 2] = 1
    >>> p = stiefel.sparse_stiefel_manifold_sync(x_base, 3, [2, 2, 3])
    >>> p
    array([[0., 1., 0.],
           [1., 0., 0.],
           [1., 0., 0.],
           [0., 1., 0.]])
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
