"""
Random Fourier Features and utility functions

..moduleauthor:: François-Xavier Dupé
"""
import numpy as np
import networkx as nx


def compute_phi(graph, data_name, vectors, offsets):
    """
    Compute the Phi matrix (the vector associated to each edge) using random Fourier features

    :param nx.classes.graph.Graph graph: the graph
    :param str data_name: the name of the data vector
    :param np.ndarray vectors: the random vectors for the new features
    :param np.ndarray offsets: the random offsets
    :return: the Phi matrix compute using RRF
    """
    n = nx.number_of_nodes(graph)
    phi = np.zeros((vectors.shape[1], n, n))
    for u, v, data in graph.edges.data(data_name):
        nvect = np.sqrt(2 / vectors.shape[1]) * np.cos(np.dot(data.T, vectors) + offsets)
        phi[:, u, v] = nvect
        phi[:, v, u] = nvect
    return phi


def create_random_vectors(size, number, sigma):
    """
    Compute the random vectors needed for the RRF
    :param int size: the dimension of features
    :param int number: the number of features
    :param float sigma: the variance of the features
    :return: a tuple with the random vectors and their random offsets
    """
    rng = np.random.default_rng()
    vectors = rng.normal(0.0, sigma, size=(size, number))
    offsets = rng.uniform(0.0, 1.0, size=(number,))
    return vectors, offsets
