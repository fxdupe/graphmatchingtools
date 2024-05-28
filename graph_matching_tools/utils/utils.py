"""
Utility functions

..moduleauthor:: François-Xavier Dupé
"""

import numpy as np
import networkx as nx
import random

import graph_matching_tools.solvers.ot.sns as sns


def get_dim_data_edges(graph: nx.Graph, data_edge: str) -> int:
    """Get the dimension of the data on edges.

    :param nx.Graph graph: the graph.
    :param str data_edge: the name of the data vector on edges.
    :return: the dimension of the data on edges.
    :rtype: int
    """
    for u, v, data in graph.edges.data(data_edge):
        if np.isscalar(data):
            return 1
        elif isinstance(data, np.ndarray):
            return data.shape[0]
    return 0


def create_full_adjacency_matrix(graphs: list[nx.Graph]) -> np.ndarray:
    """Create the full adjacency matrix with the matrices on the diagonal.

    :param list[nx.Graph] graphs: the list of graphs.
    :return: The bulk adjacency matrix.
    :rtype: np.ndarray
    """
    sizes = []
    full_size = 0
    for g in graphs:
        size = nx.number_of_nodes(g)
        sizes.append(size)
        full_size += size

    a = np.zeros((full_size, full_size))

    index = 0
    for i in range(len(graphs)):
        adj = nx.to_numpy_array(graphs[i], weight=None)
        a[index : index + sizes[i], index : index + sizes[i]] = adj
        index += sizes[i]

    return a


def create_full_weight_matrix(
    graphs: list[nx.Graph], edge_data: str, sigma: float = 1.0
) -> np.ndarray:
    """Create the full weighted matrix with the matrices on the diagonal using Gaussian weights.

    :param list[nx.Graph] graphs: the list of graphs.
    :param str edge_data: the name of the scalar data on edges.
    :param float sigma: the variance of the data.
    :return: The weighted adjacency matrix.
    :rtype: np.ndarray
    """
    sizes = []
    full_size = 0
    for g in graphs:
        size = nx.number_of_nodes(g)
        sizes.append(size)
        full_size += size

    w = np.zeros((full_size, full_size))

    index = 0
    for i in range(len(graphs)):
        for n1, n2, data in graphs[i].edges.data(edge_data):
            w[index + n1, index + n2] = np.exp(-(data**2.0) / (2.0 * sigma**2.0))
            w[index + n2, index + n1] = w[index + n1, index + n2]
        index += sizes[i]

    return w


def randomize_nodes_position(
    graphs: list[nx.Graph],
) -> tuple[list[nx.Graph], list[list[int]]]:
    """Randomize the node position inside a list of graph.

    :param list[nx.Graph] graphs: a list of graph (networkx format).
    :return: the list of new graphs and the new index.
    :rtype: tuple[list[nx.Graph], list[list[int]]]
    """
    res = []
    new_graphs = []
    for g in graphs:
        nb_nodes = nx.number_of_nodes(g)
        nidx = list(range(nb_nodes))
        random.shuffle(nidx)
        n_g = nx.relabel_nodes(g, dict(zip(g.nodes(), nidx)))
        res.append(nidx)
        new_graphs.append(n_g)
    return new_graphs, res


def normalized_softperm_matrix(
    x: np.ndarray, sizes: list[int], entropy: float = 1.0
) -> np.ndarray:
    """Normalize a bulk matrix of several graphs (lines and column sum to 1.0 for each block).

    :param np.ndarray x: the matrix to normalize.
    :param list[int] sizes: the sizes of the different graphs.
    :param float entropy: the entropy (regularization).
    :return: the normalized matrix.
    """
    res = x * 0.0
    i_index = 0  # The global index
    for i_x in range(len(sizes)):
        j_index = i_index
        for i_y in range(i_x, len(sizes)):
            x_ij = x[i_index : i_index + sizes[i_x], j_index : j_index + sizes[i_y]]
            mu_s = np.ones((x_ij.shape[0], 1)) / x_ij.shape[0]
            mu_t = np.ones((x_ij.shape[1], 1)) / x_ij.shape[1]

            x_ij = sns.sinkhorn_newton_sparse_method(x_ij, mu_s, mu_t, eta=1 / entropy)

            res[i_index : i_index + sizes[i_x], j_index : j_index + sizes[i_y]] = x_ij
            res[j_index : j_index + sizes[i_y], i_index : i_index + sizes[i_x]] = x_ij.T
            j_index += sizes[i_y]
        i_index += sizes[i_x]

    return res
