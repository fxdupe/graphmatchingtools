from unittest import TestCase

import numpy as np
import networkx as nx

import graph_matching_tools.algorithms.multiway.quickmatch as qm
import graph_matching_tools.algorithms.kernels.gaussian as kern
import graph_matching_tools.algorithms.kernels.utils as utils


class TestQuickMatch(TestCase):
    def test_compute_density(self):
        graph1 = nx.Graph()
        graph1.add_node(0, weight=1.0)
        graph1.add_node(1, weight=20.0)
        graph1.add_edge(0, 1, weight=1.0)

        graph2 = nx.Graph()
        graph2.add_node(0, weight=20.0)
        graph2.add_node(1, weight=1.0)
        graph2.add_edge(0, 1, weight=1.0)

        graph3 = nx.Graph()
        graph3.add_node(0, weight=20.0)
        graph3.add_node(1, weight=30.0)
        graph3.add_node(2, weight=1.0)
        graph3.add_edge(1, 2, weight=1.0)

        graphs = [graph1, graph2, graph3]

        densities = qm.compute_density(graphs, [2, 2, 3], "weight", 0.7)
        truth = np.array(
            [
                11.2075169,
                12.49309109,
                12.49309109,
                11.2075169,
                12.49309109,
                8.6125429,
                11.2075169,
            ]
        )
        self.assertEqual(np.linalg.norm(densities - truth) < 1e-1, True)

    def test_compute_parents(self):
        graph1 = nx.Graph()
        graph1.add_node(0, weight=1.0)
        graph1.add_node(1, weight=20.0)
        graph1.add_edge(0, 1, weight=1.0)

        graph2 = nx.Graph()
        graph2.add_node(0, weight=20.0)
        graph2.add_node(1, weight=1.0)
        graph2.add_edge(0, 1, weight=1.0)

        graph3 = nx.Graph()
        graph3.add_node(0, weight=20.0)
        graph3.add_node(1, weight=30.0)
        graph3.add_node(2, weight=1.0)
        graph3.add_edge(1, 2, weight=1.0)

        graphs = [graph1, graph2, graph3]

        densities = qm.compute_density(graphs, [2, 2, 3], "weight", 0.7)
        p, d = qm.compute_parents(graphs, [2, 2, 3], "weight", densities)

        p_truth = np.array([6, 4, 4, 6, 2, 2, 3])
        d_truth = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0])
        self.assertEqual(np.linalg.norm(p - p_truth) < 1e-4, True)
        self.assertEqual(np.linalg.norm(d - d_truth) < 1e-4, True)

    def test_quickmatch(self):
        graph1 = nx.Graph()
        graph1.add_node(0, weight=1.0)
        graph1.add_node(1, weight=20.0)
        graph1.add_edge(0, 1, weight=1.0)

        graph2 = nx.Graph()
        graph2.add_node(0, weight=20.0)
        graph2.add_node(1, weight=1.0)
        graph2.add_edge(0, 1, weight=1.0)

        graph3 = nx.Graph()
        graph3.add_node(0, weight=20.0)
        graph3.add_node(1, weight=30.0)
        graph3.add_node(2, weight=1.0)
        graph3.add_edge(1, 2, weight=1.0)

        graphs = [graph1, graph2, graph3]

        u = qm.quickmatch(graphs, "weight", 0.25, 0.9)
        res = u @ u.T

        truth = np.array(
            [
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            ]
        )
        self.assertEqual(np.linalg.norm(res - truth) < 1e-3, True)
