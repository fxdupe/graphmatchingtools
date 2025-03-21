from unittest import TestCase

import numpy as np
import networkx as nx

import graph_matching_tools.algorithms.mean.wasserstein_barycenter as mean


class TestWassersteinBarycenter(TestCase):
    def test_get_degree_distributions(self):
        g1 = nx.Graph()
        g1.add_node(0, weight=3.0)
        g1.add_node(1, weight=4.0)
        g1.add_node(2, weight=1.0)
        g1.add_node(3, weight=1.0)
        g1.add_edge(0, 1, weight=3.0)
        g1.add_edge(1, 2, weight=3.0)

        degrees = mean._get_degree_distributions(g1)
        self.assertTrue(
            np.linalg.norm(degrees - np.array([0.25, 0.375, 0.25, 0.125])) < 1e-4
        )

    def test_get_adjacency_matrix_from_costs_with_valuation(self):
        cost = np.array(
            [
                [0.37800054, 0.32182417, 0.23277335, 0.17659698],
                [0.32182417, 0.28437325, 0.19532244, 0.15787152],
                [0.23277335, 0.19532244, 0.14556873, 0.10811781],
                [0.17659698, 0.15787152, 0.10811781, 0.08939235],
            ]
        )
        adj, val = mean.get_adjacency_matrix_from_costs_with_valuation(cost, 0.11)

        truth_adj = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype="f",
        )
        self.assertTrue(np.abs(val - 1.5322) < 1e-2)
        self.assertTrue(np.linalg.norm(adj - truth_adj) < 1e-3)

    def test_fgw_wasserstein_barycenter(self):
        g1 = nx.Graph()
        g1.add_node(0, weight=np.array((3.0,)))
        g1.add_node(1, weight=np.array((4.0,)))
        g1.add_node(2, weight=np.array((1.0,)))
        g1.add_node(3, weight=np.array((0.0,)))
        g1.add_edge(0, 1, weight=3.0)

        g2 = nx.Graph()
        g2.add_node(0, weight=np.array((5.0,)))
        g2.add_node(1, weight=np.array((1.0,)))
        g2.add_node(2, weight=np.array((2.0,)))
        g2.add_node(3, weight=np.array((1.0,)))
        g2.add_edge(0, 1, weight=1.0)
        g2.add_edge(0, 2, weight=4.0)

        g3 = nx.Graph()
        g3.add_node(0, weight=np.array((1.0,)))
        g3.add_node(1, weight=np.array((4.0,)))
        g3.add_node(2, weight=np.array((2.0,)))
        g3.add_node(3, weight=np.array((1.0,)))
        g3.add_edge(0, 1, weight=2.0)
        g3.add_edge(0, 3, weight=2.0)
        g3.add_edge(1, 2, weight=3.0)

        graphs = [g1, g2, g3]
        mean_cost, mean_data = mean.fgw_wasserstein_barycenter(
            graphs,
            0.5,
            10,
            100,
            node_sigma=1.0,
            gamma=0.05,
            inner_iterations_step1=1000,
            inner_iterations_step2=1000,
            random_state=10,
        )

        truth_cost = np.array(
            [
                [0.31935154, 0.85442201, 0.33248795, 0.67673441],
                [0.85442201, 0.10125423, 0.83209174, 0.29453955],
                [0.33248795, 0.83209174, 0.34289364, 0.65310959],
                [0.67673441, 0.29453955, 0.65310959, 0.24662307],
            ]
        )
        truth_data = np.array([3.70289568, 0.87833747, 3.61585852, 1.60290834])

        self.assertEqual(np.linalg.norm(mean_cost - truth_cost) < 1e-3, True)
        self.assertEqual(
            np.linalg.norm(np.squeeze(mean_data) - truth_data) < 1e-3, True
        )
