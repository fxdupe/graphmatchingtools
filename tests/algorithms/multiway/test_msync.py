from unittest import TestCase

import numpy as np
import networkx as nx

import graph_matching_tools.algorithms.multiway.msync as ms
import graph_matching_tools.algorithms.kernels.gaussian as kern
import graph_matching_tools.algorithms.kernels.utils as utils


class TestMSync(TestCase):

    def test_msync(self):
        node_kernel = kern.create_gaussian_node_kernel(1.0, "weight")

        graph1 = nx.Graph()
        graph1.add_node(0, weight=10.0)
        graph1.add_node(1, weight=20.0)
        graph1.add_edge(0, 1, weight=10.0)

        graph2 = nx.Graph()
        graph2.add_node(0, weight=20.0)
        graph2.add_node(1, weight=10.0)
        graph2.add_edge(0, 1, weight=10.0)

        graphs = [graph1, graph2]
        sizes = [2, 2]

        knode = utils.create_full_node_affinity_matrix(graphs, node_kernel)
        res = ms.msync(knode, sizes, 2)
        res = res @ res.T

        truth = np.array([[1., 0., 0., 1.],
                          [0., 1., 1., 0.],
                          [0., 1., 1., 0.],
                          [1., 0., 0., 1.]])
        self.assertEqual(np.linalg.norm(res - truth) < 1e-3, True)
