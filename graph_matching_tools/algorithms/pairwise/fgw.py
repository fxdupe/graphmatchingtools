"""This module contains the matching algorithm between a pair of graph

Implementation of Vayer, T., Chapel, L., Flamary, R., Tavenard, R., & Courty, N.(2019, May).
 Optimal transport for structured data with application on graphs.
 In International Conference on Machine Learning (pp. 6275-6284). PMLR.

Note: this using the L2 loss (i.e. q=2)

.. moduleauthor:: François-Xavier Dupé
"""
import numpy as np

import graph_matching_tools.algorithms.pairwise.kergm as kergm


def _compute_c_constant(cost_s, cost_t, mu_s, mu_t):
    """
    Compute the constant matrix that appear in the loss function

    :param cost_s: the cost matrix of the source graph.
    :param cost_t: the cost matrix of the target graph.
    :param mu_s: the starting probabilities of the source nodes.
    :param mu_t: the starting probabilities of the target nodes.
    :return: the constant matrix.
    """
    const = (cost_s ** 2) @ mu_s @ np.ones((1, cost_t.shape[0]))
    const += np.ones((cost_s.shape[0], 1)) @ mu_t.T @ (cost_t ** 2).T
    return const


def _line_search_l2_loss(c_const, distances, cost_s, cost_t, transport, new_transport, alpha):
    """
    Line search computation for step size.

    :param c_const: the constant matrix for loss computation.
    :param distances: the matrix of distances between the node features of the two graphs.
    :param cost_s: the cost matrix of the source graph.
    :param cost_t: the cost matrix of the target graph.
    :param transport: the current transportation map.
    :param new_transport: the new estimated transportation map.
    :param alpha: the equilibrium between cost and distances.
    :return: the new step size.
    """
    a = -2.0*alpha*np.trace(new_transport.T @ cost_s @ new_transport @ cost_t)

    b = np.trace(new_transport.T @ ((1 - alpha) * distances + alpha * c_const))
    b -= 2.0 * alpha * np.trace(transport.T @ cost_s @ new_transport @ cost_t)
    b += np.trace(new_transport.T @ cost_s @ transport @ cost_t)

    # The following line is present in the paper but not used
    # c = np.trace(transport.T @ ((1.0 - alpha) * (distances ** 2.0) + alpha *
    #                             (c_const - cost_s @ transport @ (2.0 * cost_t).T)))

    tau = 0
    if a > 0:
        tau = np.min([1, np.max([0, -b / (2.0 * a)])])
    else:
        if a + b < 0:
            tau = 1

    return tau


def fgw_direct_matching(cost_s, cost_t, mu_s, mu_t, distances, alpha, iterations, gamma=1.0, inner_iterations=1000):
    """
    Fused Gromov-Wasserstein method for graph matching using optimal transport.

    :param cost_s: the cost matrix of the source graph.
    :param cost_t: the cost matrix of the target graph.
    :param mu_s: the starting probabilities of the source nodes.
    :param mu_t: the starting probabilities of the target nodes.
    :param distances: the matrix of distances between the node features of the two graphs.
    :param alpha: the equilibrium between cost and distances.
    :param iterations: the number of iterations for convergence.
    :param gamma: the strength of the regularization for the OT solver.
    :param inner_iterations: the number of inner iterations for Sinkhorn-Knopp.
    :return: the transport map.
    """
    # Ensure that we are using vectors
    mu_s = mu_s.reshape((-1, 1))
    mu_t = mu_t.reshape((-1, 1))

    transport = mu_s @ mu_t.T
    q = 2  # Set as a variable but mostly a constant
    distances_q = distances ** q
    c_const = _compute_c_constant(cost_s, cost_t, mu_s, mu_t)

    for iteration in range(iterations):
        # 1 - Gradient computation
        tmp = c_const - cost_s @ transport @ (2.0 * cost_t).T
        grad = (1.0 - alpha) * distances_q + 2 * alpha * tmp
        # 2 - Apply OT constraints
        new_transport = kergm.sinkhorn_method(grad, mu_s=np.squeeze(mu_s), mu_t=np.squeeze(mu_t),
                                              gamma=gamma, iterations=inner_iterations)
        # 3 - Line-search
        tau = _line_search_l2_loss(c_const, distances, cost_s, cost_t, transport, new_transport, alpha)
        # 4 - Update
        transport = (1 - tau) * transport + tau * new_transport

    return transport
