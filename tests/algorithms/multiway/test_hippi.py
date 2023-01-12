from unittest import TestCase

import numpy as np
import networkx as nx

import graph_matching_tools.algorithms.multiway.hippi as hippi
import graph_matching_tools.algorithms.kernels.gaussian as kern
import graph_matching_tools.algorithms.kernels.utils as utils


class TestHiPPI(TestCase):

    def test_hippi_multiway_matching(self):
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

        s = np.zeros((4, 4))
        s[0:2, 0:2] = nx.to_numpy_array(graph1, weight=None)
        s[2:4, 2:4] = nx.to_numpy_array(graph2, weight=None)

        knode = utils.create_full_node_affinity_matrix(graphs, node_kernel)
        u = hippi.hippi_multiway_matching(s, sizes, knode, 2, iterations=50)
        res = u @ u.T

        truth = np.array([[1., 0., 0., 1.],
                          [0., 1., 1., 0.],
                          [0., 1., 1., 0.],
                          [1., 0., 0., 1.]])
        self.assertEqual(np.linalg.norm(res - truth) < 1e-3, True)

        init = [[1, 0], [0, 1], [0, 1], [1, 0]]

        u = hippi.hippi_multiway_matching(s, sizes, knode, 2, iterations=1, init=np.array(init))
        self.assertEqual(np.linalg.norm(u @ u.T - truth) < 1e-3, True)
        u = hippi.hippi_multiway_matching(s, sizes, knode, 2, iterations=1)
        self.assertEqual(np.linalg.norm(u @ u.T - truth) < 1e-3, True)
