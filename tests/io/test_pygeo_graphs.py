from unittest import TestCase

import numpy as np
import networkx as nx

import graph_matching_tools.io.pygeo_graphs as pg


class TestPygeoGraphs(TestCase):
    def test_generate_groundtruth(self):
        res = pg.generate_groundtruth([3, 2], 5, [[0, 1, 2], [0, 1]])
        truth = np.array([[0, 1, 2, 3, 4], [0, 1, 2, 0, 1]])
        self.assertTrue(np.linalg.norm(res - truth) < 1e-3)

    def test_compute_edge_data(self):
        graph = nx.Graph()
        graph.add_node(0, pos=np.array([0, 0]))
        graph.add_node(1, pos=np.array([1, 0]))
        graph.add_node(2, pos=np.array([0, 1]))
        graph.add_node(3, pos=np.array([1, 1]))
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)

        new_graph = pg.compute_edges_data(graph, 1.0)
        self.assertEqual(new_graph.edges[0, 1]["distance"], 1.0)
        self.assertEqual(new_graph.edges[0, 1]["norm_dist"], 1.0)
        self.assertTrue(np.abs(new_graph.edges[0, 1]["weight"] - 0.6065) < 1e-2)
