"""
This code is directly from the paper
"Factorized multi-graph matching" by Zhu et al (Pattern Recognition 2023).

.. moduleauthor:: François-Xavier Dupé
"""
from typing import Callable

import numpy as np
import scipy.optimize as sco
import networkx as nx

import graph_matching_tools.algorithms.kernels.utils as ku


def create_pairwise_gradient(
    inc_g1: np.ndarray, inc_g2: np.ndarray, knode: np.ndarray, kedge: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Create the gradient function for pairwise graph matching

    :param np.ndarray inc_g1: the incident matrix of the first graph.
    :param np.ndarray inc_g2: the incident matrix of the second graph.
    :param np.ndarray knode: the node affinity matrix between the two graphs.
    :param np.ndarray kedge: the edge affinity matrix between the two graphs.
    :return: the gradient function.
    :rtype: Callable[[np.ndarray], np.ndarray]
    """
    v1 = np.hstack((inc_g1, np.identity(inc_g1.shape[0])))
    v2 = np.hstack((inc_g2, np.identity(inc_g2.shape[0])))

    d_mat = np.zeros(
        (inc_g1.shape[0] + inc_g1.shape[1], inc_g2.shape[0] + inc_g2.shape[1])
    )
    d_mat[0 : kedge.shape[0], 0 : kedge.shape[1]] = kedge
    d_mat[0 : kedge.shape[0], kedge.shape[1] :] = -kedge @ inc_g2.T
    d_mat[kedge.shape[0] :, 0 : kedge.shape[1]] = -inc_g1 @ kedge
    d_mat[kedge.shape[0] :, kedge.shape[1] :] = inc_g1 @ kedge @ inc_g2.T + knode

    def gradient(x: np.ndarray) -> np.ndarray:
        """The gradient at a given point.

        :param np.ndarray x: the current permutation matrix.
        :return: the gradient at x.
        :rtype: np.ndarray
        """
        tmp = v1.T @ x @ v2
        return 2 * v1 @ (d_mat * tmp) @ v2.T

    return gradient


def rrwm(
    gradient: Callable[[np.ndarray], np.ndarray],
    size: tuple[int, int],
    alpha: float = 0.2,
    beta: float = 30,
    iterations: int = 100,
    tolerance: float = 1e-3,
) -> np.ndarray:
    """The reweighted random walks for graph matching method.

    :param Callable[[np.ndarray], np.ndarray] gradient: the gradient of the current objective function.
    :param tuple[int, int] size: the size the permutation matrix.
    :param float alpha: the mixing parameter for convergence.
    :param float beta: the scaling parameter when reweighting.
    :param int iterations: the maximal number of iterations.
    :param float tolerance: the tolerance for convergence.
    :return: the permutation matrix.
    :rtype: np.ndarray
    """
    x = np.ones(size)
    grad = gradient(x)
    y = np.exp(beta * grad / np.max(grad))
    for iter1 in range(iterations):
        for iter2 in range(iterations):
            y = y / y.sum(axis=1).reshape(-1, 1)
            y = y / y.sum(axis=0)
        x_new = alpha * grad + (1 - alpha) * y
        if np.linalg.norm(x - x_new) < tolerance:
            break
        x = x_new

    # Discretization using Hungarian method
    res = np.zeros(size)
    r, c = sco.linear_sum_assignment(-x)
    for j in range(r.shape[0]):
        res[r[j], c[j]] = 1

    return res


def factorized_rrwm(
    source_graph: nx.Graph,
    target_graph: nx.Graph,
    node_kernel: Callable[[nx.Graph, int, nx.Graph, int], float],
    edge_kernel: Callable[
        [nx.Graph, nx.Graph, tuple[int, int], tuple[int, int]], float
    ],
    iterations: int = 100,
    tolerance: float = 1e-2,
) -> np.ndarray:
    """Factorized RRWM

    :param nx.Graph source_graph: the source graph.
    :param nx.Graph target_graph: the target graph.
    :param Callable[[nx.Graph, nx.Graph, int, int], float] node_kernel: the kernel between node attributes.
    :param Callable[[nx.Graph, nx.Graph, tuple[int, int], tuple[int, int]], float] edge_kernel: the kernel between edge attributes.
    :param int iterations: the maximal number of iterations.
    :param float tolerance: the tolerance for convergence.
    :return: the permutation matrix.
    :rtype: np.ndarray

    Here an example using NetworkX and some utils:

    .. doctest:

    >>> node_kernel = kern.create_gaussian_node_kernel(2.0, "weight")
    >>> def edge_kernel(g1, g2, e1, e2):
    ...     w1 = g1.edges[e1[0], e1[1]]["weight"]
    ...     w2 = g2.edges[e2[0], e2[1]]["weight"]
    ...     return w1 * w2
    >>> res = rrwm.factorized_rrwm(graph1, graph2, node_kernel, edge_kernel)
    >>> res
    array([[0., 1.],
           [1., 0.]])
    """
    # Get the node affinity matrices for all pairs
    knode = ku.compute_knode(source_graph, target_graph, node_kernel)

    # Get the individual incidence matrices
    inc_source = np.zeros(
        (nx.number_of_nodes(source_graph), nx.number_of_edges(source_graph))
    )
    i_e = 0
    for e in source_graph.edges:
        inc_source[e[0], i_e] = 1
        inc_source[e[1], i_e] = 1
        i_e += 1

    inc_target = np.zeros(
        (nx.number_of_nodes(target_graph), nx.number_of_edges(target_graph))
    )
    i_e = 0
    for e in source_graph.edges:
        inc_target[e[0], i_e] = 1
        inc_target[e[1], i_e] = 1
        i_e += 1

    # Compute the edge kernel
    kedge = np.zeros(
        (nx.number_of_edges(source_graph), nx.number_of_edges(target_graph))
    )
    i_e1 = 0
    for e1 in source_graph.edges:
        i_e2 = 0
        for e2 in target_graph.edges:
            kedge[i_e1, i_e2] = edge_kernel(source_graph, target_graph, e1, e2)
            i_e2 += 1
        i_e1 += 1

    grad = create_pairwise_gradient(
        inc_source,
        inc_target,
        knode,
        kedge,
    )
    pair_perm = rrwm(
        grad,
        (knode.shape[0], knode.shape[1]),
        iterations=iterations,
        tolerance=tolerance,
    )

    return pair_perm
