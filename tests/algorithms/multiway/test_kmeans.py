from unittest import TestCase

import numpy as np
import networkx as nx

import graph_matching_tools.algorithms.multiway.kmeans as kmeans
import graph_matching_tools.algorithms.kernels.gaussian as kern
import graph_matching_tools.algorithms.kernels.utils as utils


class TestKMeans(TestCase):
    def test_get_permutation_with_kmeans(self):
        node_kernel = kern.create_gaussian_node_kernel(0.1, "weight")

        graph1 = nx.Graph()
        graph1.add_node(0, weight=1.0)
        graph1.add_node(1, weight=2.0)
        graph1.add_edge(0, 1, weight=1.0)

        graph2 = nx.Graph()
        graph2.add_node(0, weight=2.0)
        graph2.add_node(1, weight=1.0)
        graph2.add_edge(0, 1, weight=1.0)

        graphs = [graph1, graph2]
        sizes = [2, 2]

        knode = utils.create_full_node_affinity_matrix(graphs, node_kernel)
        res = kmeans.get_permutation_with_kmeans(2, knode)

        truth = np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
            ]
        )
        self.assertEqual(np.linalg.norm(res - truth) < 1e-3, True)
