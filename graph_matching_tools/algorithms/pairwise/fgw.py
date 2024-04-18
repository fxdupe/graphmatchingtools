"""This module contains the matching algorithm between a pair of graph

Implementation of Vayer, T., Chapel, L., Flamary, R., Tavenard, R., & Courty, N.(2019, May).
 Optimal transport for structured data with application on graphs.
 In International Conference on Machine Learning (pp. 6275-6284). PMLR.

With correction from thesis of Cédric Vincent-Cuaz.

Note: this using the L2 loss (i.e. q=2)

.. moduleauthor:: François-Xavier Dupé
"""

import numpy as np

import graph_matching_tools.algorithms.pairwise.kergm as kergm


def _compute_c_constant(
    cost_s: np.ndarray, cost_t: np.ndarray, mu_s: np.ndarray, mu_t: np.ndarray
) -> np.ndarray:
    """Compute the constant matrix that appear in the loss function.

    :param np.ndarray cost_s: the cost matrix of the source graph.
    :param np.ndarray cost_t: the cost matrix of the target graph.
    :param np.ndarray mu_s: the starting probabilities of the source nodes.
    :param np.ndarray mu_t: the starting probabilities of the target nodes.
    :return: the constant matrix.
    :rtype: np.ndarray
    """
    const = (cost_s**2) @ mu_s @ np.ones((1, cost_t.shape[0]))
    const += np.ones((cost_s.shape[0], 1)) @ mu_t.T @ (cost_t**2).T
    return const


def _line_search_l2_loss(
    c_const: np.ndarray,
    distances: np.ndarray,
    cost_s: np.ndarray,
    cost_t: np.ndarray,
    transport: np.ndarray,
    new_transport: np.ndarray,
    alpha: float,
) -> float:
    """Line search computation for step size.

    :param np.ndarray c_const: the constant matrix for loss computation.
    :param np.ndarray distances: the matrix of distances between the node features of the two graphs.
    :param np.ndarray cost_s: the cost matrix of the source graph.
    :param np.ndarray cost_t: the cost matrix of the target graph.
    :param np.ndarray transport: the current transportation map.
    :param np.ndarray new_transport: the new estimated transportation map.
    :param float alpha: the equilibrium between cost and distances.
    :return: the new step size.
    :rtype: float
    """
    delta_transport = new_transport - transport
    a = -2.0 * alpha * np.trace(delta_transport.T @ cost_s @ delta_transport @ cost_t)

    b = np.trace(delta_transport.T @ ((1 - alpha) * distances + alpha * c_const))
    b -= 2.0 * alpha * np.trace(transport.T @ cost_s @ new_transport @ cost_t)
    b -= 2.0 * alpha * np.trace(new_transport.T @ cost_s @ transport @ cost_t)

    tau = 0.0
    if a > 0:
        tau = np.min([1.0, np.max([0.0, -b / (2.0 * a)])])
    else:
        if a + b < 0:
            tau = 1.0

    return tau


def fgw_direct_matching(
    cost_s: np.ndarray,
    cost_t: np.ndarray,
    mu_s: np.ndarray,
    mu_t: np.ndarray,
    distances: np.ndarray,
    alpha: float,
    iterations: int,
    gamma: float = 1.0,
    inner_iterations: int = 1000,
) -> np.ndarray:
    """Fused Gromov-Wasserstein method for graph matching using optimal transport.

    :param np.ndarray cost_s: the cost matrix of the source graph.
    :param np.ndarray cost_t: the cost matrix of the target graph.
    :param np.ndarray mu_s: the starting probabilities of the source nodes.
    :param np.ndarray mu_t: the starting probabilities of the target nodes.
    :param np.ndarray distances: the matrix of distances between the node features of the two graphs.
    :param float alpha: the equilibrium between cost and distances.
    :param int iterations: the number of iterations for convergence.
    :param float gamma: the strength of the regularization for the OT solver.
    :param int inner_iterations: the number of inner iterations for Sinkhorn-Knopp.
    :return: the transport map.
    :rtype: np.ndarray

    Here an example using NetworkX and some utils:

    .. doctest:

    >>> node_kernel = kern.create_gaussian_node_kernel(1.0, "weight")
    >>> cost_s = 1.0 - utils.create_full_node_affinity_matrix([graph1, ], node_kernel)
    >>> cost_t = 1.0 - utils.create_full_node_affinity_matrix([graph2, ], node_kernel)
    >>> mu_s = np.ones((nx.number_of_nodes(graph1), )) / nx.number_of_nodes(graph1)
    >>> mu_t = np.ones((nx.number_of_nodes(graph2), )) / nx.number_of_nodes(graph2)
    >>> distances = 1.0 - utils.compute_knode(graph1, graph2, node_kernel)
    >>> transport = fgw_pairwise.fgw_direct_matching(cost_s, cost_t, mu_s, mu_t, distances, 0.1, 50, gamma=0.1)
    >>> transport
    array([[1.06489980e-05, 4.99989351e-01],
           [4.99989351e-01, 1.06489980e-05]])
    """
    # Ensure that we are using vectors
    mu_s = mu_s.reshape((-1, 1))
    mu_t = mu_t.reshape((-1, 1))

    transport = mu_s @ mu_t.T
    q = 2  # Set as a variable but mostly a constant
    distances_q = distances**q
    c_const = _compute_c_constant(cost_s, cost_t, mu_s, mu_t)

    for iteration in range(iterations):
        # 1 - Gradient computation
        tmp = c_const - cost_s @ transport @ (2.0 * cost_t).T
        grad = (1.0 - alpha) * distances_q + 2.0 * alpha * tmp
        # 2 - Apply OT constraints
        new_transport = kergm.sinkhorn_method(
            grad,
            mu_s=np.squeeze(mu_s),
            mu_t=np.squeeze(mu_t),
            gamma=gamma,
            iterations=inner_iterations,
        )
        # 3 - Line-search
        tau = _line_search_l2_loss(
            c_const, distances, cost_s, cost_t, transport, new_transport, alpha
        )
        # 4 - Update
        transport = (1 - tau) * transport + tau * new_transport

    return transport
