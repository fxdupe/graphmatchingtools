"""This utility module contains some functions for creating linear kernels

.. moduleauthor:: François-Xavier Dupé
"""
from typing import Callable

import numpy as np
import networkx as nx


def create_linear_node_kernel(
    node_data: str,
) -> Callable[[nx.Graph, int, nx.Graph, int], float]:
    """Create a linear node kernel.

    :param str node_data: the name of the data vector on the nodes.
    :return: the node kernel.
    :rtype: Callable[[nx.Graph, int, nx.Graph, int], float]
    """

    def node_kernel(g1: nx.Graph, n1: int, g2: nx.Graph, n2: int):
        """Node kernel (on data).

        :param nx.Graph g1: the first graph.
        :param int n1: the index of the node inside graph1.
        :param nx.Graph g2: the second graph.
        :param int n2: the index of the node inside graph2.
        :return the Gaussian kernel between the data on each node.
        :rtype: float
        """
        result = np.dot(
            np.array(g1.nodes[n1][node_data]).T, np.array(g2.nodes[n2][node_data])
        )
        return result

    return node_kernel
