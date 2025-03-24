"""This module contains a computation of the Wasserstein barycenter

Implementation of Vayer, T., Chapel, L., Flamary, R., Tavenard, R., & Courty, N.(2019, May).
 Optimal transport for structured data with application on graphs.
 In International Conference on Machine Learning (pp. 6275-6284). PMLR.

Note: this using the L2 loss (i.e. q=2)

.. moduleauthor:: François-Xavier Dupé
"""

from typing import Optional

import numpy as np
import networkx as nx

import graph_matching_tools.algorithms.pairwise.fgw as fgw
import graph_matching_tools.algorithms.kernels.gaussian as kern
import graph_matching_tools.algorithms.kernels.utils as ku


def _get_degree_distributions(g: nx.Graph) -> np.ndarray:
    """Create probability vector from the weights on the node.

    :param nx.Graph g: the graph.
    :return: the probability vector for each node.
    :rtype: np.ndarray
    """
    degrees = nx.degree(g)
    mu = np.array([degrees[i] for i in range(nx.number_of_nodes(g))]) + 1
    mu = mu / np.sum(mu)
    return mu


def _compute_distance_matrix(
    graph_data: np.ndarray, mean_data: np.ndarray, sigma: float
) -> np.ndarray:
    """Compute the distance between of the data on the mean graph and another graph.

    :param np.ndarray graph_data: a graph data matrix.
    :param np.ndarray mean_data: the mean data matrix.
    :param float sigma: the sigma for the Gaussian kernel used for distance between node data.
    :return: the distance matrix.
    :rtype: np.ndarray
    """
    distances = np.ones((graph_data.shape[0], mean_data.shape[0]))
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            dist = np.linalg.norm(graph_data[i, :] - mean_data[j, :])
            distances[i, j] = 1.0 - np.exp(-(dist**2) / (2.0 * sigma**2.0))
    return distances


def _get_data_matrix(graph: nx.Graph) -> np.ndarray:
    """Retrieve the data matrix for a given graph.

    :param nx.Graph graph: the input graph.
    :return: the data matrix.
    :rtype: np.ndarray
    """
    try:
        data_dim = graph.nodes[0]["weight"].shape[0]
    except AttributeError:
        data_dim = 1
    data = np.ones((nx.number_of_nodes(graph), data_dim))
    for n in range(data.shape[0]):
        data[n, :] = np.array(graph.nodes[n]["weight"])
    return data


def get_adjacency_matrix_from_costs_with_valuation(
    cost: np.ndarray, threshold: float
) -> tuple[np.ndarray, float]:
    """Compute the adjacency matrix from a cost matrix with a given threshold and compute a quality value.
    The value is computed as the norm of the difference between the cost matrix and the resulting shortest path
    matrix.

    :param np.ndarray cost: the cost matrix.
    :param float threshold: the threshold (only value lower than it create edges).
    :return: the adjacency matrix with its valuation.
    :rtype: tuple[np.ndarray, float]
    """
    adjacency = np.array(cost < threshold, dtype="i")
    adjacency -= np.diag(np.diag(adjacency))
    graph = nx.from_numpy_array(adjacency, create_using=nx.Graph)

    sp_matrix = np.zeros(adjacency.shape)
    sp_length = dict(nx.shortest_path_length(graph))

    for (i, j), _ in np.ndenumerate(adjacency):
        if i in sp_length and j in sp_length[i]:
            sp_matrix[i, j] = sp_length[i][j]

    return adjacency, np.linalg.norm(cost - sp_matrix)


def fgw_wasserstein_barycenter_direct(
    costs: list[np.ndarray],
    mus: list[np.ndarray],
    data: list[np.ndarray],
    node_sigma: float = 1.0,
    alpha: float = 0.5,
    weights: np.ndarray = None,
    iterations: int = 10,
    fgw_iterations: int = 10,
    gamma: float = 0.8,
    rho: float = 1.0,
    graph_init: int = 0,
    barycenter_size: int = None,
    inner_iterations_step1: int = 100,
    inner_iterations_step2: int = 100,
    random_state=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fused Gromov-Wasserstein + Frechet mean for Wasserstein barycenter.
    This code is closer to the one in the paper.

    Note: this version assumes vector of data on node only.

    :param costs: The set of costs matrices.
    :param mus: The probability vector for each node.
    :param data: The data associated to each node.
    :param float node_sigma: the sigma for the Gaussian kernel used for distance between node data.
    :param alpha: The equilibrium between structure and attribute.
    :param weights: The weight of each graph.
    :param iterations: The number of iterations.
    :param fgw_iterations: The number of FGW iterations.
    :param gamma: The percentage of kept values in Hessian for sns method.
    :param rho: The rho parameter.
    :param int graph_init: the index of the graph used as initialization (default: 0).
    :param int barycenter_size: the size of the barycenter (using the first graph by default).
    :param int inner_iterations_step1: the number of iterations for the classical steps in OT solver (sns).
    :param int inner_iterations_step2: the number of iterations for the Newton's steps in OT solver (sns).
    :param int random_state: fix the randomness (for reproductibility).
    :return: the computed barycenter and the data on the nodes.
    :rtype: a tuple with the barycenter and the data on the nodes.
    """
    if weights is None:
        weights = np.ones((len(costs),)) / len(costs)
    else:
        weights = np.array(weights)
        weights /= np.sum(weights)  # Must sum to one

    if barycenter_size is None:
        barycenter_size = costs[0].shape[0]

    if graph_init:
        mean_cost = costs[graph_init]
        mean_data = data[graph_init]
        mean_mu = mus[graph_init]
    else:
        gen = np.random.default_rng(seed=random_state)
        mean_data = gen.random(size=(barycenter_size, data[0].shape[1])) * np.max(
            [np.max(d) for d in data]
        )
        mean_cost = _compute_distance_matrix(mean_data, mean_data, node_sigma)
        mean_mu = np.ones((barycenter_size, 1)) / barycenter_size

    for iteration in range(iterations):
        transports = list()

        # 1 - Update the transport maps
        for i_g in range(len(costs)):
            distances = _compute_distance_matrix(costs[i_g], mean_data, node_sigma)
            transport = fgw.fgw_direct_matching(
                costs[i_g],
                mean_cost,
                mus[i_g],
                mean_mu,
                distances,
                alpha,
                fgw_iterations,
                gamma=gamma,
                rho=rho,
                inner_iterations_step1=inner_iterations_step1,
                inner_iterations_step2=inner_iterations_step2,
            )
            transports.append(transport)

        # 2 - Update the mean_cost and mean data
        mean_cost = mean_cost * 0.0
        mean_data = mean_data * 0.0
        for i_g in range(len(costs)):
            mean_cost += weights[i_g] * transports[i_g].T @ costs[i_g] @ transports[i_g]
            mean_data += (
                weights[i_g]
                * data[i_g].T
                @ transports[i_g]
                @ np.diag(1.0 / np.squeeze(mean_mu))
            ).T
        mean_cost /= mean_mu @ mean_mu.T

    return mean_cost, mean_data


def fgw_wasserstein_barycenter(
    graphs: list[nx.Graph],
    alpha: float,
    iterations: int,
    fgw_iterations: int,
    barycenter_size: int = None,
    node_sigma: float = 1.0,
    weights: Optional[np.ndarray] = None,
    gamma: float = 1.0,
    rho: float = 0.8,
    graph_init: int = None,
    inner_iterations_step1: int = 100,
    inner_iterations_step2: int = 100,
    random_state=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fused Gromov-Wasserstein + Frechet mean for Wasserstein barycenter.

    Note: this version assumes vector of data on node only.

    :param list[nx.Graph] graphs: the list of graphs (in NetworkX format).
    :param float alpha: the equilibrium between the data distance and Wasserstein loss.
    :param int iterations: the number of iterations.
    :param int fgw_iterations: the number of iterations for each matching.
    :param float node_sigma: the sigma for the Gaussian kernel used for distance between node data.
    :param Optional[np.ndarray] weights: the weight associated to each graph.
    :param float gamma: the strength of thr regularization for the OT solver.
    :param float rho: percentage of remaining value while computing Hessian in sns method.
    :param int graph_init: the index of the graph used as initialization (default: 0).
    :param int barycenter_size: the size of the barycenter (using the first graph by default).
    :param int inner_iterations_step1: the number of iterations for the classical steps in OT solver (sns).
    :param int inner_iterations_step2: the number of iterations for the Newton's steps in OT solver (sns).
    :param int random_state: fix the randomness (for reproducibility).
    :return: the cost and data of the barycenter.
    :rtype: tuple[np.ndarray, np.ndarray]

    Here an example using NetworkX and some utils:

    .. doctest:

    >>> mean_cost, mean_data = fgw_barycenter.fgw_wasserstein_barycenter(graphs, 0.5, 100,
    ... 200, barycenter_size=2, gamma=0.01, random_state=42)
    >>> mean_cost
    array([[0.37161864, 0.6076793 ],
           [0.6076793 , 0.42187286]])
    >>> mean_data
    array([[3.06670722],
           [3.86662611]])
    """
    node_kernel = kern.create_gaussian_node_kernel(node_sigma, "weight")

    mus = [_get_degree_distributions(g) for g in graphs]
    costs = [(1.0 - ku.compute_knode(g, g, node_kernel)) for g in graphs]
    datas = [_get_data_matrix(g) for g in graphs]

    return fgw_wasserstein_barycenter_direct(
        costs,
        mus,
        datas,
        node_sigma,
        alpha,
        weights,
        iterations,
        fgw_iterations,
        gamma,
        rho,
        graph_init,
        barycenter_size,
        inner_iterations_step1,
        inner_iterations_step2,
        random_state=random_state,
    )
