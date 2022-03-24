from unittest import TestCase

import numpy as np
import networkx as nx

import graph_matching_tools.algorithms.multiway.matcheig as me
import graph_matching_tools.algorithms.kernels.gaussian as kern
import graph_matching_tools.algorithms.kernels.utils as utils


class TestMatchEIG(TestCase):

    def test_matcheig(self):
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
        res = me.matcheig(knode, 2, sizes)

        truth = np.array([[1., 0., 0., 1.],
                          [0., 1., 1., 0.],
                          [0., 1., 1., 0.],
                          [1., 0., 0., 1.]])
        self.assertEqual(np.linalg.norm(res - truth) < 1e-3, True)
