"""This utility module contains some procedures for creating kernels

.. moduleauthor:: François-Xavier Dupé
"""
import numpy as np


def create_gaussian_node_kernel(sigma, node_data):
    """
    Create a Gaussian node kernel
    :param float sigma: the variance hyperparameter (for the Gaussian kernel)
    :param str node_data: the name of the data vector on the nodes
    :return: the node kernel
    """

    def node_kernel(g1, n1, g2, n2):
        """
        Node kernel (on data)
        :param nx.classes.graph.Graph g1: the first graph
        :param int n1: the index of the node inside graph1
        :param nx.classes.graph.Graph g2: the second graph
        :param int n2: the index of the node inside graph2
        :return the Gaussian kernel between the data on each node
        """

        if (not g1.nodes[n1]["is_dummy"]) and (not g2.nodes[n2]["is_dummy"]):

            distance = np.linalg.norm(np.array(g1.nodes[n1][node_data]) - np.array(g2.nodes[n2][node_data]))
            
            return np.exp(-distance**2 / (2.0 * sigma**2.0))

        else:

            return 0.0

    return node_kernel
