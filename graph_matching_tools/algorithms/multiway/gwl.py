"""Direct multi-graphs version of GWL

.. moduleauthor:: François-Xavier Dupé
"""
from typing import Optional

import numpy as np

import graph_matching_tools.algorithms.pairwise.gwl as gwl


def multi_pairwise_gwl(
    costs: list[np.ndarray],
    mus: list[np.ndarray],
    beta: float,
    gamma: float,
    node_dim: int,
    outer_iterations: int,
    inner_iterations: int,
    embed_iterations: int,
    embed_step: float,
    use_cross_cost: bool = False,
    cross_costs: Optional[dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """Multi-graph version of GWL using only pairwise matching.

    :param list[np.ndarray] costs: the list of cost matrices for each graph.
    :param list[np.ndarray] mus: the list of mu (node probabilities) for each graph.
    :param float beta: the strength of the regularization for embedding updates.
    :param float gamma: the equilibrium between entropy and data for the Sinkhorn-Knopp step.
    :param int node_dim: the size of the node embedding space.
    :param int outer_iterations: the global number of iterations.
    :param int inner_iterations: the number of iterations for the OT step.
    :param int embed_iterations: the number of iterations for updating the embeddings.
    :param float embed_step: the descent step for embedding update (gradient descent).
    :param bool use_cross_cost: use the cross cost matrices.
    :param Optional[dict[str, np.ndarray]] cross_costs: the dictionary with the cross cost matrices.
    :return: the full permutation matrix.
    :rtype: np.ndarray

    Here an example using NetworkX and some utils:

    .. doctest:

    >>> node_kernel = kern.create_gaussian_node_kernel(1.0, "weight")
    >>> mus = [wb._get_degree_distributions(g) for g in graphs]
    >>> costs = [(1.0 - utils.compute_knode(g, g, node_kernel) * nx.to_numpy_array(g, weight=None)) for g in graphs]
    >>> cross_costs = dict()
    >>> for i_s in range(len(graphs)):
    ...     for i_t in range(i_s + 1, len(graphs)):
    ...         cost_st = 1.0 - utils.compute_knode(graphs[i_s], graphs[i_t], node_kernel)
    ...         cross_costs["{},{}".format(i_s, i_t)] = cost_st
    >>> res = gwl.multi_pairwise_gwl(costs, mus, 1.0, 2.0, 5, 20, 20, 2, 0.1,
    ...     use_cross_cost=True, cross_costs=cross_costs)
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
            cost_st = None
            if use_cross_cost:
                cost_st = cross_costs["{},{}".format(g_s, g_t)]

            matchs = gwl.gromov_wasserstein_learning(
                costs[g_s],
                costs[g_t],
                mus[g_s],
                mus[g_t],
                beta,
                gamma,
                node_dim,
                outer_iterations,
                inner_iterations,
                embed_iterations,
                embed_step,
                cost_st=cost_st,
                use_cross_cost=use_cross_cost,
            )
            for i in range(matchs.shape[0]):
                if matchs[i] > -1:
                    full_perm[index_s + i, index_t + int(matchs[i])] = 1
                    full_perm[index_t + int(matchs[i]), index_s + i] = 1
            index_t += sizes[g_t]
        index_s += sizes[g_s]

    return full_perm
