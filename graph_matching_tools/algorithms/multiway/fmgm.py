"""
This code is directly from the paper
"Factorized multi-graph matching" by Zhu et al (Pattern Recognition 2023).

.. moduleauthor:: François-Xavier Dupé
"""
from typing import Callable

import numpy as np
import networkx as nx

import graph_matching_tools.algorithms.kernels.utils as ku
import graph_matching_tools.algorithms.pairwise.rrwm as rrwm


def create_multiway_gradient(
    n_graph: int,
    perms: list[np.ndarray],
    incs: list[np.ndarray],
    knodes: list[np.ndarray],
    kedges: list[np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    """Create the gradient function for multiway graph matching

    :param int n_graph: the current graph (the gradient depends on this graph).
    :param list[np.ndarray] perms: the current permutation matrices.
    :param list[np.ndarray] incs: the incident matrices.
    :param list[np.ndarray] knodes: the node affinity matrices (against the current graph).
    :param list[np.ndarray] kedges: the edge affinity matrices (against the current graph).
    :return: the gradient function.
    :rtype: Callable[[np.ndarray], np.ndarray]
    """
    v = []
    for inc in incs:
        v1 = np.hstack((inc, np.identity(inc.shape[0])))
        v.append(v1)

    d_mats = []
    for i_inc in range(len(incs)):
        d_mat = np.zeros(
            (
                incs[n_graph].shape[0] + incs[n_graph].shape[1],
                incs[i_inc].shape[0] + incs[i_inc].shape[1],
            )
        )
        d_mat[0 : kedges[i_inc].shape[0], 0 : kedges[i_inc].shape[1]] = kedges[i_inc]
        d_mat[0 : kedges[i_inc].shape[0], kedges[i_inc].shape[1] :] = (
            -kedges[i_inc] @ incs[i_inc].T
        )
        d_mat[kedges[i_inc].shape[0] :, 0 : kedges[i_inc].shape[1]] = (
            -incs[n_graph] @ kedges[i_inc]
        )
        d_mat[kedges[i_inc].shape[0] :, kedges[i_inc].shape[1] :] = (
            incs[n_graph] @ kedges[i_inc] @ incs[i_inc].T + knodes[i_inc]
        )
        d_mats.append(d_mat)

    def gradient(x: np.ndarray) -> np.ndarray:
        """The gradient at a given point.

        :param np.ndarray x: the current permutation matrix.
        :return: the gradient at x.
        :rtype: np.ndarray
        """
        res = np.zeros(x.shape)
        for i_g in range(len(incs)):
            if i_g == n_graph:
                continue
            tmp = v[n_graph].T @ x @ perms[i_g].T @ v[i_g]
            res += v[n_graph] @ (d_mats[i_g] * tmp) @ v[i_g].T @ perms[i_g]
        return 2 * res

    return gradient


def compute_multiway_objective(
    perms: list[np.ndarray],
    incs: list[np.ndarray],
    knodes: list[list[np.ndarray]],
    kedges: list[list[np.ndarray]],
) -> float:
    """Compute the objective function at a given point

    :param list[np.ndarray] perms: the current permutation matrices.
    :param list[np.ndarray] incs: the incidence matrices.
    :param list[list[np.ndarray]] knodes: the node affinity matrices.
    :param list[list[np.ndarray]] kedges: the edge affinity matrices.
    :return:
    """
    v = []
    for inc in incs:
        v1 = np.hstack((inc, np.identity(inc.shape[0])))
        v.append(v1)

    value = 0.0
    for i_g1 in range(len(incs)):
        for i_g2 in range(len(incs)):
            if i_g1 == i_g2:
                continue

            d_mat = np.zeros(
                (
                    incs[i_g1].shape[0] + incs[i_g1].shape[1],
                    incs[i_g2].shape[0] + incs[i_g2].shape[1],
                )
            )
            d_mat[
                0 : kedges[i_g1][i_g2].shape[0], 0 : kedges[i_g1][i_g2].shape[1]
            ] = kedges[i_g1][i_g2]
            d_mat[0 : kedges[i_g1][i_g2].shape[0], kedges[i_g1][i_g2].shape[1] :] = (
                -kedges[i_g1][i_g2] @ incs[i_g2].T
            )
            d_mat[kedges[i_g1][i_g2].shape[0] :, 0 : kedges[i_g1][i_g2].shape[1]] = (
                -incs[i_g1] @ kedges[i_g1][i_g2]
            )
            d_mat[kedges[i_g1][i_g2].shape[0] :, kedges[i_g1][i_g2].shape[1] :] = (
                incs[i_g1] @ kedges[i_g1][i_g2] @ incs[i_g2].T + knodes[i_g1][i_g2]
            )

            tmp = (v[i_g1].T @ perms[i_g1] @ perms[i_g2].T @ v[i_g2]) ** 2.0
            value += d_mat.reshape(1, -1) @ tmp.reshape(-1, 1)

    return value


def create_multiway_local_gradient(
    source_graph: int,
    target_graph: int,
    perms: list[np.ndarray],
    incs: list[np.ndarray],
    knodes: list[np.ndarray],
    kedges: list[np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    """Create the local gradient function for multiway graph matching

    :param int source_graph: the source graph (the gradient depends on this graph).
    :param int target_graph: the target graph (the gradient depends on this graph).
    :param list[np.ndarray] perms: the current permutation matrices.
    :param list[np.ndarray] incs: the incident matrices.
    :param list[np.ndarray] knodes: the node affinity matrices (against the source graph).
    :param list[np.ndarray] kedges: the edge affinity matrices (against the source graph).
    :return: the gradient function.
    :rtype: Callable[[np.ndarray], np.ndarray]
    """
    v = []
    for inc in incs:
        v1 = np.hstack((inc, np.identity(inc.shape[0])))
        v.append(v1)

    d_mat = np.zeros(
        (
            incs[source_graph].shape[0] + incs[source_graph].shape[1],
            incs[target_graph].shape[0] + incs[target_graph].shape[1],
        )
    )
    d_mat[
        0 : kedges[target_graph].shape[0], 0 : kedges[target_graph].shape[1]
    ] = kedges[target_graph]
    d_mat[0 : kedges[target_graph].shape[0], kedges[target_graph].shape[1] :] = (
        -kedges[target_graph] @ incs[target_graph].T
    )
    d_mat[kedges[target_graph].shape[0] :, 0 : kedges[target_graph].shape[1]] = (
        -incs[source_graph] @ kedges[target_graph]
    )
    d_mat[kedges[target_graph].shape[0] :, kedges[target_graph].shape[1] :] = (
        incs[source_graph] @ kedges[target_graph] @ incs[target_graph].T
        + knodes[target_graph]
    )

    def gradient(x: np.ndarray) -> np.ndarray:
        """The gradient at a given point.

        :param np.ndarray x: the current permutation matrix.
        :return: the gradient at x.
        :rtype: np.ndarray
        """
        tmp = v[source_graph].T @ x @ perms[target_graph].T @ v[target_graph]
        res = v[source_graph] @ (d_mat * tmp) @ v[target_graph].T @ perms[target_graph]
        return 2 * res

    return gradient


def factorized_multigraph_matching(
    graphs: list[nx.Graph],
    reference: int,
    node_kernel: Callable[[nx.Graph, int, nx.Graph, int], float],
    edge_kernel: Callable[
        [nx.Graph, nx.Graph, tuple[int, int], tuple[int, int]], float
    ],
    iterations: int = 100,
    tolerance: float = 1e-2,
) -> np.ndarray:
    """Factorized Multi-Graph Matching

    :param list[nx.Graph] graphs: the list of graphs.
    :param int reference: the index of the reference graph.
    :param Callable[[nx.Graph, int, nx.Graph, int], float] node_kernel: the kernel between node attributes.
    :param Callable[[nx.Graph, nx.Graph, tuple[int, int], tuple[int, int]], float] edge_kernel: the kernel between edge attributes.
    :param int iterations: the maximal number of iterations.
    :param float tolerance: the tolerance for convergence.
    :return: the bulk permutation matrix.
    :rtype: np.ndarray

    Here an example using NetworkX and some utils:

    .. doctest:

    >>> node_kernel = kern.create_gaussian_node_kernel(2.0, "weight")
    >>> def edge_kernel(g1, g2, e1, e2):
    ...     w1 = g1.edges[e1[0], e1[1]]["weight"]
    ...     w2 = g2.edges[e2[0], e2[1]]["weight"]
    ...     return w1 * w2
    >>> res = fmgm.factorized_multigraph_matching(graphs, 2, node_kernel, edge_kernel)
    >>> res
    array([[1., 0., 0., 1., 0., 1., 0.],
           [0., 1., 1., 0., 0., 0., 1.],
           [0., 1., 1., 0., 0., 0., 1.],
           [1., 0., 0., 1., 0., 1., 0.],
           [0., 0., 0., 0., 1., 0., 0.],
           [1., 0., 0., 1., 0., 1., 0.],
           [0., 1., 1., 0., 0., 0., 1.]])
    """
    # Get the node affinity matrices for all pairs
    knodes = []
    for i_g1 in range(len(graphs)):
        line_knode = []
        for i_g2 in range(len(graphs)):
            knode = ku.compute_knode(graphs[i_g1], graphs[i_g2], node_kernel)
            line_knode.append(knode)
        knodes.append(line_knode)

    # Get the individual incidence matrices
    incidence_mat = []
    for i_g in range(len(graphs)):
        # Incidence matrix
        i_mat = np.zeros(
            (nx.number_of_nodes(graphs[i_g]), nx.number_of_edges(graphs[i_g]))
        )
        i_e = 0
        for e in graphs[i_g].edges:
            i_mat[e[0], i_e] = 1
            i_mat[e[1], i_e] = 1
            i_e += 1
        incidence_mat.append(i_mat)

    # Get the edge affinity matrix for all pairs
    kedges = []
    for i_g1 in range(len(graphs)):
        line_kedge = []
        for i_g2 in range(len(graphs)):
            kedge = np.zeros(
                (nx.number_of_edges(graphs[i_g1]), nx.number_of_edges(graphs[i_g2]))
            )
            i_e1 = 0
            for e1 in graphs[i_g1].edges:
                i_e2 = 0
                for e2 in graphs[i_g2].edges:
                    kedge[i_e1, i_e2] = edge_kernel(graphs[i_g1], graphs[i_g2], e1, e2)
                    i_e2 += 1
                i_e1 += 1
            line_kedge.append(kedge)
        kedges.append(line_kedge)

    # Pairwise initialization
    u = []
    for i_g in range(len(graphs)):
        if i_g == reference:
            u.append(np.identity(knodes[i_g][0].shape[0]))
            continue

        grad = rrwm.create_pairwise_gradient(
            incidence_mat[i_g],
            incidence_mat[reference],
            knodes[i_g][reference],
            kedges[i_g][reference],
        )
        pair_perm = rrwm.rrwm(
            grad,
            (knodes[i_g][0].shape[0], knodes[i_g][reference].shape[1]),
            iterations=iterations,
            tolerance=tolerance,
        )
        u.append(pair_perm)

    # Global updating
    prev_val = compute_multiway_objective(u, incidence_mat, knodes, kedges)
    for g_iter in range(iterations):
        modification = False
        for i_g in range(len(graphs)):
            if i_g == reference:
                continue

            grad = create_multiway_gradient(
                i_g, u, incidence_mat, knodes[i_g], kedges[i_g]
            )
            new_perm = rrwm.rrwm(
                grad,
                (knodes[i_g][0].shape[0], knodes[i_g][reference].shape[1]),
                iterations=iterations,
                tolerance=tolerance,
            )

            old_perm = u[i_g]
            u[i_g] = new_perm
            new_val = compute_multiway_objective(u, incidence_mat, knodes, kedges)
            if new_val < prev_val:
                prev_val = new_val
                modification = True
            else:
                u[i_g] = old_perm

        # No modification means convergence
        if not modification:
            break

    # Local updating
    for l_iter in range(iterations):
        modification = False
        for i_g1 in range(len(graphs)):
            if i_g1 == reference:
                continue

            for i_g2 in range(len(graphs)):
                if i_g2 == i_g1:
                    continue

                grad = create_multiway_local_gradient(
                    i_g1, i_g2, u, incidence_mat, knodes[i_g1], kedges[i_g1]
                )
                new_perm = rrwm.rrwm(
                    grad,
                    (knodes[i_g1][0].shape[0], knodes[i_g1][reference].shape[1]),
                    iterations=iterations,
                    tolerance=tolerance,
                )

                old_perm = u[i_g1]
                u[i_g1] = new_perm
                new_val = compute_multiway_objective(u, incidence_mat, knodes, kedges)
                if new_val < prev_val:
                    prev_val = new_val
                    modification = True
                else:
                    u[i_g1] = old_perm

        # No modification means convergence
        if not modification:
            break

    # Final result (removing reference dependency)
    res = np.concatenate(u, axis=0)
    return res @ res.T
