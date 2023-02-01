"""
Utility functions

..moduleauthor:: François-Xavier Dupé
"""
import numpy as np
import networkx as nx
import random


def get_dim_data_edges(graph, data_edge):
    """Get the dimension of the data on edges

    :param graph: the graph
    :param data_edge: the name of the data vector on edges
    :return: the dimension of the data on edges
    """
    for u, v, data in graph.edges.data(data_edge):
        if np.isscalar(data):
            return 1
        elif isinstance(data, np.ndarray):
            return data.shape[0]
    return 0


def create_full_adjacency_matrix(graphs):
    """Create the full adjacency matrix with the matrices on the diagonal.

    :param list graphs: the list of graphs.
    :return: The bulk adjacency matrix.
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


def create_full_weight_matrix(graphs, edge_data, sigma=1.0):
    """Create the full weighted matrix with the matrices on the diagonal using Gaussian weights.

    :param list graphs: the list of graphs.
    :param str edge_data: the name of the scalar data on edges.
    :param float sigma: the variance of the data.
    :return: The weighted adjacency matrix.
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


def randomize_nodes_position(graphs):
    """Randomize the node position inside a graph

    :param list graphs: a list of graph (networkx format)
    :return: the list of new graphs and the new index
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


def get_permutation_matrix_from_matching(matching, sizes, max_node):
    """Create the full permutation matrix from the matching result

    :param matching: the matching result for each graph (nodes number, assignment)
    :param sizes: the list of the size of the different graph
    :param max_node: the maximal label for a node
    :return: the full permutation matrix
    """
    f_size = int(np.sum(sizes))
    res = np.zeros((f_size, max_node))

    idx = 0
    for g in range(len(sizes)):
        for s in range(sizes[g]):
            res[matching[0][idx + matching[1][idx + s]], s] = 1
        idx += sizes[g]

    return res @ res.T
