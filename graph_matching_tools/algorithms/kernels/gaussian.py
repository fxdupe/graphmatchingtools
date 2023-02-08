"""This utility module contains some functions for creating Gaussian kernels.

.. moduleauthor:: François-Xavier Dupé
"""
from typing import Callable

import numpy as np
import networkx as nx


def create_gaussian_node_kernel(
    sigma: float, node_data: str
) -> Callable[[nx.Graph, int, nx.Graph, int], float]:
    """Create a Gaussian node kernel.

    :param float sigma: the variance hyperparameter (for the Gaussian kernel).
    :param str node_data: the name of the data vector on the nodes.
    :return: the node kernel.
    :rtype: Callable[[nx.Graph, int, nx.Graph, int], float]
    """

    def node_kernel(g1: nx.Graph, n1: int, g2: nx.Graph, n2: int) -> float:
        """Node kernel (on data).

        :param nx.Graph g1: the first graph.
        :param int n1: the index of the node inside graph1.
        :param nx.Graph g2: the second graph.
        :param int n2: the index of the node inside graph2.
        :return the Gaussian kernel between the data on each node.
        :rtype: float
        """
        distance = np.linalg.norm(
            np.array(g1.nodes[n1][node_data]) - np.array(g2.nodes[n2][node_data])
        )
        return np.exp(-(distance**2) / (2.0 * sigma**2.0))

    return node_kernel
