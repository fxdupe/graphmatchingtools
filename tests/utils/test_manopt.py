from unittest import TestCase

import numpy as np

import graph_matching_tools.utils.manopt as manopt


class TestManOpt(TestCase):
    def test_orthogonal_group_optimization(self):
        t = np.array([[1, 0.2, 9], [1.0, 4.0, 0.3], [0.1, 0.1, 2.0]])

        def gradient(x):
            return t

        res = manopt.orthogonal_group_optimization(gradient, 3, epsilon=1e-5)
        self.assertTrue(np.linalg.norm(res @ res.T - np.identity(3)) < 1e-3)

    def test_stiefel_manifold_optimization(self):
        t = np.array([[1, 0.2], [1.0, 4.0], [0.1, 0.1]])

        def gradient(x):
            return t

        res = manopt.stiefel_manifold_optimization(gradient, (3, 2), epsilon=1e-5)
        self.assertTrue(np.linalg.norm(res.T @ res - np.identity(2)) < 1e-3)
