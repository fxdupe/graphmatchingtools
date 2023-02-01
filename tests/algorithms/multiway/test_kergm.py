import unittest

import numpy as np
import networkx as nx

import graph_matching_tools.algorithms.multiway.kergm as kergm
import graph_matching_tools.algorithms.kernels.gaussian as kern
import graph_matching_tools.algorithms.kernels.utils as ku


class TestDirectMultiwayKerGM(unittest.TestCase):
    def test_multi_pairwise_kergm(self):
        graph1 = nx.Graph()
        graph1.add_node(0, weight=2.0)
        graph1.add_node(1, weight=20.0)
        graph1.add_edge(0, 1, weight=np.array((1.0,)))

        graph2 = nx.Graph()
        graph2.add_node(0, weight=20.0)
        graph2.add_node(1, weight=2.0)
        graph2.add_edge(0, 1, weight=np.array((1.0,)))

        graph3 = nx.Graph()
        graph3.add_node(0, weight=-5.0)
        graph3.add_node(1, weight=30.0)
        graph3.add_node(2, weight=2.0)
        graph3.add_edge(1, 2, weight=np.array((1.0,)))

        graphs = [graph1, graph2, graph3]

        node_kernel = kern.create_gaussian_node_kernel(10.0, "weight")
        knode = ku.create_full_node_affinity_matrix(graphs, node_kernel)

        res = kergm.multi_pairwise_kergm(
            graphs, [2, 2, 3], knode, "weight", 1.0, 2.0, 10, 100, rff=100
        )

        truth = [
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        ]

        self.assertEqual(np.linalg.norm(res - truth) < 1e-3, True)
