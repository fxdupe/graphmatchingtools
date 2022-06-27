"""
Utility functions

..moduleauthor:: François-Xavier Dupé
"""
import numpy as np


def get_dim_data_edges(graph, data_edge):
    """
    Get the dimension of the data on edges
    :param graph: the graph
    :param data_edge: the name of the data vector on edges
    :return: the dimension of the data on edges
    """
    dim_edge = 0
    for u, v, data in graph.edges.data(data_edge):
        if np.isscalar(data):
            dim_edge = 1
        else:
            dim_edge = data.shape[0]
        break
    return dim_edge
