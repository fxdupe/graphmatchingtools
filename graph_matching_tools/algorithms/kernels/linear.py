"""This utility module contains some functions for creating linear kernels

.. moduleauthor:: François-Xavier Dupé
"""
import numpy as np


def create_linear_node_kernel(node_data):
    """Create a linear node kernel

    :param str node_data: the name of the data vector on the nodes
    :return: the node kernel
    """

    def node_kernel(g1, n1, g2, n2):
        """Node kernel (on data)

        :param nx.classes.graph.Graph g1: the first graph
        :param int n1: the index of the node inside graph1
        :param nx.classes.graph.Graph g2: the second graph
        :param int n2: the index of the node inside graph2
        :return the Gaussian kernel between the data on each node
        """
        result = np.dot(
            np.array(g1.nodes[n1][node_data]).T, np.array(g2.nodes[n2][node_data])
        )
        return result

    return node_kernel
