from unittest import TestCase

import numpy as np
import networkx as nx

import graph_matching_tools.algorithms.mean.mm_mean as mean


class TestMMMean(TestCase):
    def test_get_tensor_from_graph(self):
        g = nx.Graph()
        g.add_node(0, weight=4.0)
        g.add_node(1, weight=5.0)
        g.add_node(2, weight=6.0)
        g.add_node(3, weight=7.0)
        g.add_edge(0, 1, weight=4.0)
        g.add_edge(2, 3, weight=5.0)

        tensor = mean.get_tensor_from_graph(g, "weight", "weight")
        self.assertEqual(
            np.linalg.norm(tensor.shape - np.array([1, 4, 4])) < 1e-3, True
        )

        tensor = np.squeeze(tensor)
        truth = np.array(
            [
                [4.0, 4.0, 0.0, 0.0],
                [4.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 6.0, 5.0],
                [0.0, 0.0, 5.0, 7.0],
            ]
        )
        self.assertEqual(np.linalg.norm(tensor - truth) < 1e-3, True)

    def test_tensor_matching(self):
        def node_kernel(x, y):
            return x * y

        t1 = np.array([[10.0, 0.0], [0.0, -2.0]])
        t1 = np.reshape(t1, (1, 2, 2))
        t2 = np.array([[-2.0, 0.0], [0.0, 10.0]])
        t2 = np.reshape(t2, (1, 2, 2))
        r, c = mean.tensor_matching(t1, t2, node_kernel, 0.01, entropy_gamma=1.0)
        truth = np.array([1, 0])
        self.assertEqual(np.linalg.norm(c - truth) < 1e-3, True)

    def test_fast_mean_graph_computation(self):
        def node_kernel(x, y):
            return x * y

        g1 = nx.Graph()
        g1.add_node(0, weight=3.0)
        g1.add_node(1, weight=4.0)
        g1.add_node(2, weight=0.0)
        g1.add_node(3, weight=0.0)
        g1.add_edge(0, 1, weight=3.0)

        g2 = nx.Graph()
        g2.add_node(0, weight=5.0)
        g2.add_node(1, weight=1.0)
        g2.add_node(2, weight=2.0)
        g2.add_node(3, weight=0.0)
        g2.add_edge(0, 1, weight=1.0)
        g2.add_edge(0, 2, weight=4.0)

        g3 = nx.Graph()
        g3.add_node(0, weight=1.0)
        g3.add_node(1, weight=4.0)
        g3.add_node(2, weight=2.0)
        g3.add_node(3, weight=1.0)
        g3.add_edge(0, 1, weight=2.0)
        g3.add_edge(0, 3, weight=2.0)
        g3.add_edge(1, 2, weight=3.0)

        graphs = [g1, g2, g3]
        mean_graph = mean.compute_mean_graph(
            graphs, "weight", node_kernel, "weight", 1.0, entropy_gamma=10.0
        )
        result = np.array(
            [
                [
                    [2.3333, 3.3333, 0.0, 0.0],
                    [3.3333, 4.3333, 1.0, 0.0],
                    [0.0, 1.0, 0.6666, 0.6666],
                    [0.0, 0.0, 0.6666, 0.3333],
                ]
            ]
        )
        self.assertEqual(np.linalg.norm(mean_graph - result) < 1e-3, True)

    def test_get_graph_from_tensor(self):
        tensor = np.array(
            [
                [
                    [4.0, 4.0, 0.0, 0.0],
                    [4.0, 5.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 5.0],
                    [0.0, 0.0, 5.0, 1.0],
                ]
            ]
        )
        g = mean.get_graph_from_tensor(tensor)
        self.assertEqual(nx.number_of_nodes(g), 4)
        self.assertEqual(nx.number_of_edges(g), 2)
        self.assertEqual(g.nodes[0]["data"], 4)
