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
                [1.78395941e-04, 8.57974761e-01, 4.83116751e-01, 9.94789337e-01],
                [8.57974761e-01, 1.96334094e-01, 4.98321343e-01, 3.04248406e-01],
                [4.83116751e-01, 4.98321343e-01, 3.17068082e-01, 7.22273785e-01],
                [9.94789337e-01, 3.04248406e-01, 7.22273785e-01, 6.18636791e-02],
            ]
        )
        truth_data = np.array([4.33311376, 1.60840205, 3.07517121, 0.78331298])

        self.assertEqual(np.linalg.norm(mean_cost - truth_cost) < 1e-3, True)
        self.assertEqual(
            np.linalg.norm(np.squeeze(mean_data) - truth_data) < 1e-3, True
        )
