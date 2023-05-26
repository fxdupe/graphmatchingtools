"""Test of the utility functions
.. moduleauthor:: François-Xavier Dupé
"""

import unittest

import networkx as nx
import numpy as np

import graph_matching_tools.utils.utils as utils


class TestUtils(unittest.TestCase):
    def test_get_dim_data_edges(self):
        g = nx.Graph()
        g.add_node(0, weight=np.array((1.0, 2.0)))
        g.add_node(1, weight=np.array((4.0, 3.0)))
        g.add_node(2, weight=np.array((5.0, 4.0)))
        g.add_edge(0, 1, weight=np.array((5.0, 4.0)))
        g.add_edge(1, 2, weight=np.array((5.0, 4.0)))

        dim = utils.get_dim_data_edges(g, "weight")
        self.assertEqual(dim, 2)

        g = nx.Graph()
        g.add_node(0, weight=np.array((1.0, 2.0)))
        g.add_node(1, weight=np.array((4.0, 3.0)))
        g.add_edge(0, 1, weight=5.0)

        dim = utils.get_dim_data_edges(g, "weight")
        self.assertEqual(dim, 1)

        g = nx.Graph()
        g.add_node(0, weight=np.array((1.0, 2.0)))
        g.add_node(1, weight=np.array((4.0, 3.0)))
        g.add_edge(0, 1)

        dim = utils.get_dim_data_edges(g, "weight")
        self.assertEqual(dim, 0)

    def test_create_full_adjacency_matrix(self):
        g1 = nx.Graph()
        g1.add_node(0, weight=np.array((1.0,)))
        g1.add_node(1, weight=np.array((4.0,)))
        g1.add_node(2, weight=np.array((5.0,)))
        g1.add_edge(0, 1, weight=2.0)
        g1.add_edge(1, 2, weight=3.0)

        g2 = nx.Graph()
        g2.add_node(0, weight=np.array((1.0,)))
        g2.add_node(1, weight=np.array((4.0,)))
        g2.add_edge(0, 1, weight=2.0)

        full_adj = utils.create_full_adjacency_matrix([g1, g2])
        truth = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        self.assertTrue(np.linalg.norm(full_adj - truth) < 1e-3)

    def test_create_full_weight_matrix(self):
        g1 = nx.Graph()
        g1.add_node(0, weight=1.0)
        g1.add_node(1, weight=4.0)
        g1.add_node(2, weight=5.0)
        g1.add_edge(0, 1, weight=2.0)
        g1.add_edge(1, 2, weight=3.0)

        g2 = nx.Graph()
        g2.add_node(0, weight=1.0)
        g2.add_node(1, weight=4.0)
        g2.add_edge(0, 1, weight=2.0)

        full_adj = utils.create_full_weight_matrix([g1, g2], "weight")
        truth = np.array(
            [
                [0.0, 0.13533528, 0.0, 0.0, 0.0],
                [0.13533528, 0.0, 0.011109, 0.0, 0.0],
                [0.0, 0.011109, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.13533528],
                [0.0, 0.0, 0.0, 0.13533528, 0.0],
            ]
        )
        self.assertTrue(np.linalg.norm(full_adj - truth) < 1e-5)

    def test_randomize_nodes_position(self):
        g1 = nx.Graph()
        g1.add_node(0, weight=1.0)
        g1.add_node(1, weight=4.0)
        g1.add_node(2, weight=5.0)
        g1.add_node(3, weight=7.0)
        g1.add_edge(0, 1, weight=2.0)
        g1.add_edge(1, 2, weight=3.0)

        new_g1, new_idx = utils.randomize_nodes_position(
            [
                g1,
            ]
        )
        self.assertTrue(
            g1.nodes[0]["weight"] == new_g1[0].nodes[new_idx[0][0]]["weight"]
        )

    def test_normalized_softperm_matrix(self):
        test_matrix = [
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 5.0],
        ]
        res = utils.normalized_softperm_matrix(np.array(test_matrix), [2, 2], 0.001)
        truth_matrix = [
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 0.5, 0.0, 0.5],
            [0.5, 0.0, 0.0, 0.5],
            [0.0, 0.5, 0.5, 0.0],
        ]
        self.assertTrue(np.linalg.norm(res - np.array(truth_matrix)) < 1e-2)
