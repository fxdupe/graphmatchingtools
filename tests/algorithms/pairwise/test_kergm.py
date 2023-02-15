from unittest import TestCase

import numpy as np
import networkx as nx

import graph_matching_tools.algorithms.pairwise.kergm as kergm


class TestKerGM(TestCase):
    def test_create_fast_gradient(self):
        phi1 = np.zeros([3, 2, 2])
        phi1[:, 0, 1] = np.array([1, 1, 1])
        phi1[:, 1, 0] = np.array([1, 1, 1])
        knode = np.identity(2)
        x = np.identity(2)
        gradient = kergm.create_fast_gradient(phi1, phi1, knode)
        grad = gradient(x, 0.1)
        true_res = np.array([[-2.2, 0], [0, -2.2]])
        self.assertTrue(
            np.linalg.norm(grad - true_res) < 1e-5, "Testing gradient computation"
        )

    def test_create_gradient(self):
        graph1 = nx.Graph()
        graph1.add_node(0)
        graph1.add_node(1)
        graph1.add_edge(0, 1)
        knode = np.identity(2)
        x = np.identity(2)

        def kernel(*_):
            return 3

        gradient = kergm.create_gradient(graph1, graph1, kernel, knode)
        grad = gradient(x, 0.1)
        true_res = np.array([[-2.2, 0], [0, -2.2]])
        self.assertTrue(
            np.linalg.norm(grad - true_res) < 1e-5, "Testing gradient computation"
        )

    def test_kergm_method1(self):
        phi1 = np.ones([3, 2, 2]) / 3
        phi2 = np.ones([3, 2, 2]) / 3
        knode = np.array([[0.2, 0.8], [0.8, 0.2]])
        gradient = kergm.create_fast_gradient(phi1, phi2, knode)
        r, c = kergm.kergm_method(
            gradient,
            (2, 2),
            num_alpha=10,
            entropy_gamma=50.0,
            iterations=4,
            inner_iterations=10000,
            inner_tolerance=1e-6,
        )
        self.assertTrue(
            np.linalg.norm(r - np.array([0, 1])) < 1e-5, "Testing matching (1)"
        )

    def test_kergm_method2(self):
        a1 = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype="d")
        a2 = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype="d")
        deg_a1 = np.sum(a1, axis=0)
        deg_a2 = np.sum(a2, axis=0)

        permut = np.array([2, 0, 1])  # np.random.permutation(range(6))

        knode = deg_a1.reshape((3, 1)) @ deg_a2.reshape((1, 3))
        knode /= np.max(knode.flat)

        phi1 = np.zeros((1, 3, 3))
        phi1[0, :, :] = a1
        phi2 = np.zeros((1, 3, 3))
        phi2[0, :, :] = a2
        gradient = kergm.create_fast_gradient(phi1, phi2, knode)

        r, c = kergm.kergm_method(
            gradient,
            (3, 3),
            num_alpha=20,
            entropy_gamma=50.0,
            iterations=1000,
            tolerance=1e-10,
            inner_iterations=10000,
            inner_tolerance=1e-6,
            epsilon=2e-16,
        )
        self.assertTrue(np.linalg.norm(c - permut) < 1e-5, "Testing matching (2)")
