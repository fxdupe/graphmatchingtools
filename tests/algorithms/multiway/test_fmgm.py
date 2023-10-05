import unittest

import numpy as np
import networkx as nx

import graph_matching_tools.algorithms.multiway.fmgm as fmgm
import graph_matching_tools.algorithms.kernels.gaussian as kern


class TestFMGM(unittest.TestCase):
    def test_factorized_multigraph_matching(self):
        node_kernel = kern.create_gaussian_node_kernel(2.0, "weight")

        def edge_kernel(g1, g2, e1, e2):
            w1 = g1.edges[e1[0], e1[1]]["weight"]
            w2 = g2.edges[e2[0], e2[1]]["weight"]
            return w1 * w2

        graph1 = nx.Graph()
        graph1.add_node(0, weight=2.0)
        graph1.add_node(1, weight=20.0)
        graph1.add_edge(0, 1, weight=1.0)

        graph2 = nx.Graph()
        graph2.add_node(0, weight=20.0)
        graph2.add_node(1, weight=2.0)
        graph2.add_edge(0, 1, weight=1.0)

        graph3 = nx.Graph()
        graph3.add_node(0, weight=-5.0)
        graph3.add_node(1, weight=30.0)
        graph3.add_node(2, weight=2.0)
        graph3.add_edge(1, 2, weight=1.0)
        graphs = [graph1, graph2, graph3]

        truth = [
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        ]

        res = fmgm.factorized_multigraph_matching(graphs, 0, node_kernel, edge_kernel)
        self.assertTrue(np.linalg.norm(res - truth) < 1e-3)
