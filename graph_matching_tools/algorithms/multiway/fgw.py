"""Direct multi-graphs version of FGW

.. moduleauthor:: François-Xavier Dupé
"""

import numpy as np

import graph_matching_tools.algorithms.pairwise.fgw as fgw


def multi_pairwise_fgw(
    costs: list[np.ndarray],
    mus: list[np.ndarray],
    alpha: float,
    gamma: float,
    iterations: int,
    inner_iterations: int,
    cross_costs: dict[str, np.ndarray],
) -> np.ndarray:
    """Multi-graph version of FGW using only pairwise matching.

    :param list[np.ndarray] costs: the list of cost matrices for each graph.
    :param list[np.ndarray] mus: the list of mu (node probabilities) for each graph.
    :param float alpha: the equilibrium between node data and structure for the transport.
    :param float gamma: the equilibrium between entropy and data for the Sinkhorn-Knopp step.
    :param int iterations: the global number of iterations.
    :param int inner_iterations: the number of iterations for the OT step.
    :param dict[str, np.ndarray] cross_costs: the dictionary with the cross cost matrices.
    :return: the full permutation matrix.
    :rtype: np.ndarray

    Here an example using NetworkX and some utils:

    .. doctest:

    >>> node_kernel = kern.create_gaussian_node_kernel(1.0, "weight")
    >>> mus = [wb._get_degree_distributions(g) for g in graphs]
    >>> costs = [(1.0 - utils.compute_knode(g, g, node_kernel)) for g in graphs]
    >>> cross_costs = dict()
    >>> for i_s in range(len(graphs)):
    ...     for i_t in range(i_s + 1, len(graphs)):
    ...         cost_st = 1.0 - utils.compute_knode(graphs[i_s], graphs[i_t], node_kernel)
    ...         cross_costs["{},{}".format(i_s, i_t)] = cost_st
    >>> res = fgw.multi_pairwise_fgw(costs, mus, 0.5, 1.0, 10, 20, cross_costs)
    >>> res
    array([[1., 0., 0., 1., 0., 1., 0.],
           [0., 1., 1., 0., 0., 0., 1.],
           [0., 1., 1., 0., 0., 0., 1.],
           [1., 0., 0., 1., 0., 1., 0.],
           [0., 0., 0., 0., 1., 0., 0.],
           [1., 0., 0., 1., 0., 1., 0.],
           [0., 1., 1., 0., 0., 0., 1.]])
    """
    sizes = [c.shape[0] for c in costs]
    full_size = int(np.sum(sizes))
    full_perm = np.eye(full_size, full_size)

    index_s = 0
    for g_s in range(len(costs) - 1):
        index_t = index_s + sizes[g_s]
        for g_t in range(g_s + 1, len(costs)):
            cost_st = cross_costs["{},{}".format(g_s, g_t)]
            t_m = fgw.fgw_direct_matching(
                costs[g_s],
                costs[g_t],
                mus[g_s],
                mus[g_t],
                cost_st,
                alpha,
                iterations,
                gamma,
                inner_iterations,
            )

            matchs = np.zeros((costs[g_s].shape[0],)) - 1.0
            for i in range(t_m.shape[0]):
                matchs[i] = np.argmax(t_m[i, :])

            for i in range(matchs.shape[0]):
                if matchs[i] > -1:
                    full_perm[index_s + i, index_t + int(matchs[i])] = 1
                    full_perm[index_t + int(matchs[i]), index_s + i] = 1
            index_t += sizes[g_t]
        index_s += sizes[g_s]

    return full_perm
