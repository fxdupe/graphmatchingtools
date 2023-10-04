import unittest

import numpy as np
import networkx as nx

import graph_matching_tools.algorithms.multiway.fmgm as fmgm
import graph_matching_tools.algorithms.kernels.gaussian as kern


class TestFMGM(unittest.TestCase):
    def test_create_pairwise_gradient(self):
        inc1 = np.array([[1, 0], [1, 1], [0, 1]])
        inc2 = np.array([[0, 1], [1, 1], [1, 1]])
        knode = np.array([[0.0, 1.0, 2.0], [0.2, 1.0, 2.0], [0.9, 0.8, 0.2]])
        kedge = np.array([[0.0, 1.0], [0.2, 2.0]])

        grad = fmgm.create_pairwise_gradient(inc1, inc2, knode, kedge)
        res = grad(np.ones((3, 3)))
        truth = np.array([[4.0, 6.0, 8.0], [12.4, 14.4, 16.4], [9.8, 10.0, 8.8]])
        self.assertTrue(np.linalg.norm(res - truth) < 1e-2)

    def test_rrwm(self):
        inc1 = np.array([[1, 0], [1, 1], [0, 1]])
        inc2 = np.array([[0, 1], [1, 1], [1, 1]])
        knode = np.array([[0.0, 1.0, 2.0], [0.2, 1.0, 2.0], [0.9, 0.8, 0.2]])
        kedge = np.array([[0.0, 1.0], [0.2, 2.0]])

        grad = fmgm.create_pairwise_gradient(inc1, inc2, knode, kedge)
        perm = fmgm.rrwm(grad, (3, 3))
        truth = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        self.assertTrue(np.linalg.norm(perm - truth) < 1e-2)

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
