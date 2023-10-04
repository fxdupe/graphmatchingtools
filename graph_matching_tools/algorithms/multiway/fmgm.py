"""
This code is directly from the paper
"Factorized multi-graph matching" by Zhu et al (Pattern Recognition 2023).

.. moduleauthor:: François-Xavier Dupé
"""
import numpy as np
import scipy.optimize as sco
import networkx as nx

import graph_matching_tools.algorithms.kernels.utils as ku


def create_pairwise_gradient(
    inc_g1: np.ndarray, inc_g2: np.ndarray, knode: np.ndarray, kedge: np.ndarray
) -> callable:
    """Create the gradient function for pairwise graph matching

    :param np.ndarray inc_g1: the incident matrix of the first graph.
    :param np.ndarray inc_g2: the incident matrix of the second graph.
    :param np.ndarray knode: the node affinity matrix between the two graphs.
    :param np.ndarray kedge: the edge affinity matrix between the two graphs.
    :return: the gradient function.
    :rtype: callable
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


def create_multiway_gradient(
    n_graph: int,
    perms: list[np.ndarray],
    incs: list[np.ndarray],
    knodes: list[np.ndarray],
    kedges: list[np.ndarray],
) -> callable:
    """Create the gradient function for multiway graph matching

    :param int n_graph: the current graph (the gradient depends on this graph).
    :param list[np.ndarray] perms: the current permutation matrices.
    :param list[np.ndarray] incs: the incident matrices.
    :param list[np.ndarray] knodes: the node affinity matrices (against the current graph).
    :param list[np.ndarray] kedges: the edge affinity matrices (against the current graph).
    :return: the gradient function.
    :rtype: callable
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


def rrwm(
    gradient: callable,
    size: tuple[int, int],
    alpha: float = 0.2,
    beta: float = 30,
    iterations: int = 100,
    tolerance: float = 1e-3,
) -> np.ndarray:
    """The reweighted random walks for graph matching method.

    :param callable gradient: the gradient of the current objective function.
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


def factorized_multigraph_matching(
    graphs: list[nx.Graph],
    reference: int,
    node_kernel: callable,
    edge_kernel: callable,
    iterations: int = 100,
    tolerance: float = 1e-3,
):
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

        grad = create_pairwise_gradient(
            incidence_mat[i_g],
            incidence_mat[reference],
            knodes[i_g][reference],
            kedges[i_g][reference],
        )
        pair_perm = rrwm(
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
            new_perm = rrwm(
                grad,
                (knodes[i_g][0].shape[0], knodes[i_g][reference].shape[1]),
                iterations=iterations,
                tolerance=tolerance,
            )

            old_perm = u[i_g]
            u[i_g] = new_perm
            new_val = compute_multiway_objective(u, incidence_mat, knodes, kedges)
            if np.linalg.norm(new_perm - u[i_g]) > 1e-3 and new_val < prev_val:
                prev_val = new_val
                modification = True
            else:
                u[i_g] = old_perm

        # No modification means convergence
        if not modification:
            break

    # Local updating

    # Final result (removing reference dependency)
    res = np.concatenate(u, axis=0)
    return res @ res.T
