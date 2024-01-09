import unittest

import numpy as np
import networkx as nx

import graph_matching_tools.algorithms.pairwise.grasp as grasp


class TestGRASP(unittest.TestCase):
    def test_grasp(self):
        graph1 = nx.Graph()
        graph1.add_node(0, weight=2.0)
        graph1.add_node(1, weight=20.0)
        graph1.add_node(2, weight=5.0)
        graph1.add_node(3, weight=6.0)
        graph1.add_edge(0, 1, weight=1.0)
        graph1.add_edge(1, 2, weight=1.0)
        graph1.add_edge(2, 0, weight=1.0)

        graph2 = nx.Graph()
        graph2.add_node(0, weight=2.0)
        graph2.add_node(1, weight=20.0)
        graph2.add_node(2, weight=5.0)
        graph2.add_node(3, weight=6.0)
        graph2.add_edge(1, 2, weight=1.0)
        graph2.add_edge(2, 3, weight=1.0)
        graph2.add_edge(3, 1, weight=1.0)

        res = grasp.grasp(graph1, graph2, 2)
        truth = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )
        self.assertTrue(np.linalg.norm(res - truth) < 1e-7)
