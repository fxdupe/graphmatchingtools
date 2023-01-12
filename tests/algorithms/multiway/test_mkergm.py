import unittest

import numpy as np
import networkx as nx

import graph_matching_tools.algorithms.multiway.mkergm as mkergm
import graph_matching_tools.algorithms.kernels.gaussian as kern
import graph_matching_tools.algorithms.kernels.rff as rff
import graph_matching_tools.utils.utils as utils
import graph_matching_tools.algorithms.kernels.utils as ku


class TestMultiwayKerGM(unittest.TestCase):

    def test_create_fast_gradient(self):
        phi = np.zeros([3, 2, 2])
        phi[:, 0, 1] = np.array([1, 1, 1])
        phi[:, 1, 0] = np.array([1, 1, 1])
        knode = np.identity(2)
        x = np.identity(2)
        gradient = mkergm.create_gradient(phi, knode)
        grad = gradient(x)
        true_res = np.array([[-7, 0], [0, -7]])
        self.assertTrue(np.linalg.norm(grad - true_res) < 1e-5, "Testing gradient computation")

    def test_mkergm(self):
        graph1 = nx.Graph()
        graph1.add_node(0, weight=2.0)
        graph1.add_node(1, weight=20.0)
        graph1.add_edge(0, 1, weight=np.array((1.0, )))

        graph2 = nx.Graph()
        graph2.add_node(0, weight=20.0)
        graph2.add_node(1, weight=2.0)
        graph2.add_edge(0, 1, weight=np.array((1.0, )))

        graph3 = nx.Graph()
        graph3.add_node(0, weight=-5.0)
        graph3.add_node(1, weight=30.0)
        graph3.add_node(2, weight=2.0)
        graph3.add_edge(1, 2, weight=np.array((1.0, )))

        graphs = [graph1, graph2, graph3]

        node_kernel = kern.create_gaussian_node_kernel(10.0, "weight")
        vectors, offsets = rff.create_random_vectors(1, 100, 1.0)
        phi1 = rff.compute_phi(graphs[0], "weight", vectors, offsets)
        phi2 = rff.compute_phi(graphs[1], "weight", vectors, offsets)
        phi3 = rff.compute_phi(graphs[2], "weight", vectors, offsets)
        phi = np.zeros((100, 7, 7))
        phi[:, 0:2, 0:2] = phi1
        phi[:, 2:4, 2:4] = phi2
        phi[:, 4:7, 4:7] = phi3
        sizes = [nx.number_of_nodes(g) for g in graphs]

        knode = ku.create_full_node_affinity_matrix(graphs, node_kernel)
        gradient = mkergm.create_gradient(phi, knode)

        res = mkergm.mkergm(gradient, sizes, 3, iterations=100, init=knode, choice=lambda x: 2)

        truth = [[1., 0., 0., 1., 0., 0., 1.],
                 [0., 1., 1., 0., 0., 1., 0.],
                 [0., 1., 1., 0., 0., 1., 0.],
                 [1., 0., 0., 1., 0., 0., 1.],
                 [0., 0., 0., 0., 1., 0., 0.],
                 [0., 1., 1., 0., 0., 1., 0.],
                 [1., 0., 0., 1., 0., 0., 1.]]

        self.assertEqual(np.linalg.norm(res - truth) < 1e-3, True)

        res = mkergm.mkergm(gradient, sizes, 3, iterations=100, init=knode, choice=lambda x: 2,
                            projection_method="msync")
        self.assertEqual(np.linalg.norm(res - truth) < 1e-3, True)

        res = mkergm.mkergm(gradient, sizes, 3, iterations=100, init=knode, choice=lambda x: 2,
                            projection_method="gpow")
        self.assertEqual(np.linalg.norm(res - truth) < 1e-3, True)

