import unittest

import numpy as np
import networkx as nx

import graph_matching_tools.algorithms.multiway.mixer as mixer
import graph_matching_tools.algorithms.kernels.gaussian as kern
import graph_matching_tools.algorithms.kernels.utils as utils


class TestMixer(unittest.TestCase):
    def test_probability_simplex_projector(self):
        t = np.array([1, 2, 0, 4])
        res = mixer.probability_simplex_projector(t)
        self.assertTrue(np.linalg.norm(res - np.array([0, 0, 0, 1.0])) < 1e-4)

        t = np.array([4, 2, 3, 4])
        res = mixer.probability_simplex_projector(t)
        self.assertTrue(np.linalg.norm(res - np.array([0.5, 0, 0, 0.5])) < 1e-4)

    def test_line_matrix_projector(self):
        t = np.array([[1, 2, 0, 4], [4, 2, 3, 4]])
        res = mixer.line_matrix_projector(t)
        self.assertTrue(np.linalg.norm(res - [[0, 0, 0, 1.0], [0.5, 0, 0, 0.5]]) < 1e-4)

    def test_mixer(self):
        node_kernel = kern.create_gaussian_node_kernel(10.0, "weight")

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

        knode = utils.create_full_node_affinity_matrix(graphs, node_kernel)
        res = mixer.mixer(knode, [2, 2, 3], 0.5, 100)

        truth = [
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        ]

        self.assertTrue(np.linalg.norm(res @ res.T - np.array(truth)) < 1e-3)
