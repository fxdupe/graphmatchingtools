"""
This code is directly from the paper
"Robust Multi-Object Matching via Iterative Reweighting of the Graph Connection Laplacian" (NeurIPS 2020).

.. moduleauthor:: François-Xavier Dupé
"""
from typing import Optional, Callable

import numpy as np

import graph_matching_tools.algorithms.multiway.msync as msync
import graph_matching_tools.algorithms.multiway.stiefel as stiefel


def _beta_t(t: int) -> int:
    """Example of beta(t).

    :param int t: the iteration number.
    :return: beta at t.
    :rtype: int
    """
    return np.minimum(2**t, 40)


def _alpha_t(t: int) -> int:
    """Example of alpha(t).

    :param int t: the iteration number.
    :return: alpha at t.
    :rtype: int
    """
    return np.minimum(1.2 ** (t - 1), 40)


def _lambda_t(t: int) -> float:
    """Example of lambda(t).

    :param int t: the iteration number.
    :return: lambda at t.
    :rtype: float
    """
    return t / (t + 1)


def _block_scalar_product(
    x: np.ndarray, y: np.ndarray, node_number: int, graph_number: int
) -> np.ndarray:
    """Block scalar product (one scalar by graph).

    :param np.ndarray x: the left matrix.
    :param np.ndarray y: the right matrix.
    :param int node_number: the number of nodes.
    :param int graph_number: the number of graphs.
    :return: the matrix with the scalar product for each block.
    :rtype: np.ndarray
    """
    res = np.zeros((graph_number, graph_number))
    index1 = 0
    for g1 in range(graph_number):
        index2 = 0
        for g2 in range(graph_number):
            res[g1, g2] = np.trace(
                x[index1 : index1 + node_number, index2 : index2 + node_number].T
                @ y[index1 : index1 + node_number, index2 : index2 + node_number]
            )
            index2 += node_number
        index1 += node_number

    return res


def _cemp(
    x: np.ndarray,
    adj: np.ndarray,
    t0: int,
    beta: Callable[[int], int],
    node_number: int,
    graph_number: int,
) -> np.ndarray:
    """CEMP algorithm.

    :param np.ndarray x: the initial bulk permutation matrix.
    :param np.ndarray adj: the adjacency matrix between the graph.
    :param int t0: the number of iterations.
    :param Callable[[int], int] beta: the increasing beta(t).
    :param int node_number: the number of nodes.
    :param int graph_number: the number of graphs.
    :return: the initial weight matrix.
    :rtype: np.ndarray
    """
    one_m = np.ones((node_number, node_number))
    w_init = adj
    a_init = None
    for t in range(t0):
        s_init = np.kron(w_init, one_m) * x
        tmp = (
            1
            / node_number
            * (s_init @ s_init)
            / (np.kron(w_init @ w_init, one_m) + 1e-3)
        )  # Avoid division by 0
        a_init = _block_scalar_product(tmp, x, node_number, graph_number)
        w_init = np.exp(beta(t) * a_init)

    return a_init


def irgcl(
    x: np.ndarray,
    beta: Callable[[int], int],
    alpha: Callable[[int], int],
    lbd: Callable[[int], float],
    node_number: int,
    graph_number: int,
    t0: int = 5,
    t_max: int = 100,
    choice: Optional[int] = None,
) -> np.ndarray:
    """The Iteratively Reweighted Graph Connection Laplacian (IRGCL).

    :param np.ndarray x: the estimated bulk permutation matrix.
    :param Callable[[int], int] beta: the increasing beta(t) for initialization.
    :param Callable[[int], int] alpha: the increasing alpha(t).
    :param Callable[[int], float] lbd: the increasing lambda(t) for the equilibrium at update step.
    :param int node_number: the number of node inside one graph (also the rank).
    :param int graph_number: the number of graphs.
    :param int t0: number of iterations for initialization.
    :param int t_max: maximal number of iterations.
    :param Optional[int] choice: the reference graph.
    :return: the synchronized permutation matrix.
    :rtype: np.ndarray

    Note: if choice is None it uses Stiefel method.

    Here an example using NetworkX and some utils:

    .. doctest:

    >>> x_base = np.eye(4)
    >>> x_base[3, 0] = 1
    >>> x_base[0, 3] = 1
    >>> x_base[2, 1] = 1
    >>> x_base[1, 2] = 1
    >>> p = irgcl.irgcl(x_base, irgcl._beta_t, irgcl._alpha_t, irgcl._lambda_t, 2, 2, choice = 1)
    >>> p
    array([[0., 1.],
           [1., 0.],
           [1., 0.],
           [0., 1.]])
    """
    if choice is None:
        choice = 0

    one_m = np.ones((node_number, node_number))
    adj = np.ones((graph_number, graph_number)) - np.eye(graph_number)
    a_t = _cemp(x, adj, t0, beta, node_number, graph_number)
    w_t = a_t
    if choice is not None:
        p_t = msync.msync(
            np.kron(w_t, one_m) * x,
            [
                node_number,
            ]
            * graph_number,
            node_number,
            choice,
        )
    else:
        p_t = stiefel.sparse_stiefel_manifold_sync(
            np.kron(w_t, one_m) * x,
            node_number,
            [
                node_number,
            ]
            * graph_number,
        )
    for t in range(t_max):
        x_t = p_t @ p_t.T
        a1_t = _block_scalar_product(x_t, x, node_number, graph_number) / graph_number
        w1_t = np.exp(alpha(t) * a1_t)
        s_t = np.kron(w1_t, one_m) * x
        a2_t = _block_scalar_product(
            (s_t @ s_t) / (np.kron(w1_t @ w1_t, one_m)), x, node_number, graph_number
        )
        a_t = (1 - lbd(t)) * a1_t + lbd(t) * a2_t
        w_t = a_t

        if choice is not None:
            p_tt = msync.msync(
                np.kron(w_t, one_m) * x,
                [
                    node_number,
                ]
                * graph_number,
                node_number,
                choice,
            )
        else:
            p_tt = stiefel.sparse_stiefel_manifold_sync(
                np.kron(w_t, one_m) * x,
                node_number,
                [
                    node_number,
                ]
                * graph_number,
            )
        if np.linalg.norm(p_tt - p_t) < 1e-3:
            break
        p_t = p_tt

    return p_t
