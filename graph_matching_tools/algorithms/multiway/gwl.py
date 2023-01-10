"""Direct multi-graphs version of GWL

.. moduleauthor:: François-Xavier Dupé
"""
import numpy as np

import graph_matching_tools.algorithms.pairwise.gwl as gwl


def multi_pairwise_gwl(costs, mus, beta, gamma, node_dim, outer_iterations,
                       inner_iterations, embed_iterations, embed_step,
                       use_cross_cost=False, cross_costs=None):
    """Multi-graph version of GWL using only pairwise matching

    :param costs: the list of cost matrices for each graph.
    :param mus: the list of mu (node probabilities) for each graph.
    :param beta: the strength of the regularization for embedding updates.
    :param gamma: the equilibrium between entropy and data for the Sinkhorn-Knopp step.
    :param node_dim: the size of the node embedding space.
    :param outer_iterations: the global number of iterations.
    :param inner_iterations: the number of iterations for the OT step.
    :param embed_iterations: the number of iterations for updating the embeddings.
    :param embed_step: the descent step for embedding update (gradient descent).
    :param use_cross_cost: use the cross cost matrices.
    :param cross_costs: the dictionary with the cross cost matrices.
    :return: the full permutation matrix.
    """
    sizes = [c.shape[0] for c in costs]
    full_size = int(np.sum(sizes))
    full_perm = np.eye(full_size, full_size)

    index_s = 0
    for g_s in range(len(costs)-1):
        index_t = index_s + sizes[g_s]
        for g_t in range(g_s+1, len(costs)):
            cost_st = None
            if use_cross_cost:
                cost_st = cross_costs["{},{}".format(g_s, g_t)]

            matchs = gwl.gromov_wasserstein_learning(costs[g_s], costs[g_t], mus[g_s], mus[g_t], beta, gamma,
                                                     node_dim, outer_iterations, inner_iterations,
                                                     embed_iterations, embed_step, cost_st=cost_st,
                                                     use_cross_cost=use_cross_cost)
            for i in range(matchs.shape[0]):
                if matchs[i] > -1:
                    full_perm[index_s+i, index_t+int(matchs[i])] = 1
                    full_perm[index_t+int(matchs[i]), index_s+i] = 1
            index_t += sizes[g_t]
        index_s += sizes[g_s]

    return full_perm
