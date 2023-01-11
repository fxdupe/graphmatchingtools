"""This module contains a computation of the Wasserstein barycenter

Implementation of Vayer, T., Chapel, L., Flamary, R., Tavenard, R., & Courty, N.(2019, May).
 Optimal transport for structured data with application on graphs.
 In International Conference on Machine Learning (pp. 6275-6284). PMLR.

Note: this using the L2 loss (i.e. q=2)

.. moduleauthor:: François-Xavier Dupé
"""
import numpy as np
import networkx as nx

import graph_matching_tools.algorithms.pairwise.fgw as fgw
import graph_matching_tools.algorithms.kernels.gaussian as kern
import graph_matching_tools.algorithms.kernels.utils as ku


def _get_degree_distributions(g):
    """
    Create probability vector from the weights on the node
    :param g: the graph
    :return: the probability vector for each node
    """
    degrees = nx.degree(g)
    mu = np.array([degrees[i] for i in range(nx.number_of_nodes(g))])
    mu = (1+mu) / (np.sum(mu) + nx.number_of_nodes(g))
    return mu


def _compute_distance_matrix(mean_data, graph, sigma):
    """
    Compute the distance between of the data on the mean graph and another graph.

    :param mean_data: the mean data matrix.
    :param graph: a graph (in NetworkX format).
    :param sigma: the sigma for the Gaussian kernel used for distance between node data.
    :return: the distance matrix.
    """
    distances = np.ones((mean_data.shape[1], nx.number_of_nodes(graph)))
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            dist = np.linalg.norm(mean_data[:, i] - np.array(graph.nodes[j]["weight"]))
            distances[i, j] = np.exp(-dist**2 / (2.0 * sigma**2.0))
    return distances


def _get_data_matrix(graph):
    """
    Retrieve the data matrix for a given graph.

    :param graph: the input graph.
    :return: the data matrix.
    """
    data = np.ones((graph.nodes[0]["weight"].shape[0], nx.number_of_nodes(graph)))
    for n in range(data.shape[1]):
        data[:, n] = np.array(graph.nodes[n]["weight"])
    return data


def fgw_wasserstein_barycenter(graphs, alpha, iterations, fgw_iterations, node_sigma=1.0,
                               weights=None, gamma=1.0, ot_iterations=1000):
    """
    Fused Gromov-Wasserstein + Frechet mean for Wasserstein barycenter.

    Note: this version assumes vector of data on node only.

    :param graphs: the list of graphs (in NetworkX format).
    :param alpha: the equilibrium between the data distance and Wasserstein loss.
    :param iterations: the number of iterations.
    :param fgw_iterations: the number of iterations for each matching.
    :param node_sigma: the sigma for the Gaussian kernel used for distance between node data.
    :param weights: the weight associated to each graph.
    :param gamma: the strength of thr regularization for the OT solver.
    :param ot_iterations: the maximal number of iteration for the OT solver.
    :return: the cost and data of the barycenter.
    """
    weights = np.array(weights)
    weights /= np.sum(weights)  # Must sum to one

    node_kernel = kern.create_gaussian_node_kernel(node_sigma, "weight")

    mus = [_get_degree_distributions(g) for g in graphs]
    costs = [(1.0 - ku.compute_knode(g, g, node_kernel) * nx.adjacency_matrix(g).todense()) for g in graphs]
    datas = [_get_data_matrix(g) for g in graphs]

    # Create the initial graph (use the size of the first graph)
    mean_cost = np.ones(costs[0].shape) / costs[0].shape[0]
    mean_data = np.ones((graphs[0].nodes[0]["weight"].shape[0], costs[0].shape[0]))
    mean_mu = np.ones((mean_cost.shape[0], 1)) / mean_cost.shape[0]

    for iteration in range(iterations):
        transports = list()

        # 1 - Update the transport maps
        for i_g in range(len(graphs)):
            distances = _compute_distance_matrix(mean_data, graphs[i_g], node_sigma)
            transport = fgw.fgw_direct_matching(mean_cost, costs[i_g], mean_mu, mus[i_g], distances,
                                                alpha, fgw_iterations, gamma=gamma, inner_iterations=ot_iterations)
            transports.append(transport)

        # 2 - Update the mean_cost and mean data
        mean_cost = mean_cost * 0.0
        mean_data = mean_data * 0.0
        for i_g in range(len(graphs)):
            mean_cost += weights[i_g] * transports[i_g].T @ costs[i_g] @ transports[i_g]
            mean_data += weights[i_g] * datas[i_g] @ transports[i_g].T @ np.diag(1.0 / mean_mu)
        mean_cost /= mean_mu @ mean_mu.T

    return mean_cost, mean_data
