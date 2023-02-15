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
    mu = np.array([degrees[i] for i in range(nx.number_of_nodes(g))])
    mu = (1 + mu) / (np.sum(mu) + nx.number_of_nodes(g))
    return mu


def _compute_distance_matrix(
    mean_data: np.ndarray, graph: nx.Graph, sigma: float
) -> np.ndarray:
    """Compute the distance between of the data on the mean graph and another graph.

    :param np.ndarray mean_data: the mean data matrix.
    :param nx.Graph graph: a graph (in NetworkX format).
    :param float sigma: the sigma for the Gaussian kernel used for distance between node data.
    :return: the distance matrix.
    :rtype: np.ndarray
    """
    distances = np.ones((mean_data.shape[1], nx.number_of_nodes(graph)))
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            dist = np.linalg.norm(mean_data[:, i] - np.array(graph.nodes[j]["weight"]))
            distances[i, j] = np.exp(-(dist**2) / (2.0 * sigma**2.0))
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
    data = np.ones((data_dim, nx.number_of_nodes(graph)))
    for n in range(data.shape[1]):
        data[:, n] = np.array(graph.nodes[n]["weight"])
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


def fgw_wasserstein_barycenter(
    graphs: list[nx.Graph],
    alpha: float,
    iterations: int,
    fgw_iterations: int,
    node_sigma: float = 1.0,
    weights: Optional[np.ndarray] = None,
    gamma: float = 1.0,
    ot_iterations: int = 1000,
    graph_init: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fused Gromov-Wasserstein + Frechet mean for Wasserstein barycenter.

    Note: this version assumes vector of data on node only.

    Warning: the graphs must have the same number of nodes.

    :param list[nx.Graph] graphs: the list of graphs (in NetworkX format).
    :param float alpha: the equilibrium between the data distance and Wasserstein loss.
    :param int iterations: the number of iterations.
    :param int fgw_iterations: the number of iterations for each matching.
    :param float node_sigma: the sigma for the Gaussian kernel used for distance between node data.
    :param Optional[np.ndarray] weights: the weight associated to each graph.
    :param float gamma: the strength of thr regularization for the OT solver.
    :param int ot_iterations: the maximal number of iteration for the OT solver.
    :param int graph_init: the index of the graph used as initialization (default: 0).
    :return: the cost and data of the barycenter.
    :rtype: tuple[np.ndarray, np.ndarray]

    Here an example using NetworkX and some utils:

    .. doctest:

    >>> g1 = nx.Graph()
    >>> g1.add_node(0, weight=np.array((3.0,)))
    >>> g1.add_node(1, weight=np.array((4.0,)))
    >>> g1.add_node(2, weight=np.array((1.0,)))
    >>> g1.add_node(3, weight=np.array((1.0,)))
    >>> g1.add_edge(0, 1, weight=3.0)
    >>> g2 = nx.Graph()
    >>> g2.add_node(0, weight=np.array((5.0,)))
    >>> g2.add_node(1, weight=np.array((1.0,)))
    >>> g2.add_node(2, weight=np.array((2.0,)))
    >>> g2.add_node(3, weight=np.array((1.0,)))
    >>> g2.add_edge(0, 1, weight=1.0)
    >>> g2.add_edge(0, 2, weight=4.0)
    >>> g3 = nx.Graph()
    >>> g3.add_node(0, weight=np.array((1.0,)))
    >>> g3.add_node(1, weight=np.array((4.0,)))
    >>> g3.add_node(2, weight=np.array((2.0,)))
    >>> g3.add_node(3, weight=np.array((1.0,)))
    >>> g3.add_edge(0, 1, weight=2.0)
    >>> g3.add_edge(0, 3, weight=2.0)
    >>> g3.add_edge(1, 2, weight=3.0)
    >>> graphs = [g1, g2, g3]
    >>> mean_cost, mean_data = fgw_barycenter.fgw_wasserstein_barycenter(graphs, 5.0, 10, 30, node_sigma=0.1, gamma=0.1)
    >>> mean_cost
    array([[0.40811535, 0.3500874 , 0.24854876, 0.18845473],
           [0.3500874 , 0.31681839, 0.21378793, 0.14433069],
           [0.24854876, 0.21378793, 0.15633357, 0.09752863],
           [0.18845473, 0.14433069, 0.09752863, 0.09992307]])
    >>> mean_data
    array([[3.52954563, 2.98772953, 1.00000001, 1.00000001]])
    """
    if weights is None:
        weights = np.ones((len(graphs),)) / len(graphs)
    else:
        weights = np.array(weights)
        weights /= np.sum(weights)  # Must sum to one

    node_kernel = kern.create_gaussian_node_kernel(node_sigma, "weight")

    mus = [_get_degree_distributions(g) for g in graphs]
    costs = [
        (1.0 - ku.compute_knode(g, g, node_kernel) * nx.to_numpy_array(g, weight=None))
        for g in graphs
    ]
    datas = [_get_data_matrix(g) for g in graphs]

    mean_cost = costs[graph_init]
    mean_data = datas[graph_init]
    mean_mu = mus[graph_init]

    for iteration in range(iterations):
        transports = list()

        # 1 - Update the transport maps
        for i_g in range(len(graphs)):
            distances = _compute_distance_matrix(mean_data, graphs[i_g], node_sigma)
            transport = fgw.fgw_direct_matching(
                mean_cost,
                costs[i_g],
                mean_mu,
                mus[i_g],
                distances,
                alpha,
                fgw_iterations,
                gamma=gamma,
                inner_iterations=ot_iterations,
            )
            transports.append(transport)

        # 2 - Update the mean_cost and mean data
        mean_cost = mean_cost * 0.0
        mean_data = mean_data * 0.0
        for i_g in range(len(graphs)):
            mean_cost += weights[i_g] * transports[i_g].T @ costs[i_g] @ transports[i_g]
            mean_data += (
                weights[i_g]
                * datas[i_g]
                @ transports[i_g].T
                @ np.diag(1.0 / np.squeeze(mean_mu))
            )
        mean_cost /= mean_mu @ mean_mu.T

    return mean_cost, mean_data
