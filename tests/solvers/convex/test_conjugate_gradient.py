from unittest import TestCase

import numpy as np

import graph_matching_tools.solvers.convex.conjugate_gradient as cg


class TestConjugateGradient(TestCase):

    def test_conjugate_gradient(self):
        mat = np.array([[4, 1], [1, 3]])
        b = np.array([1, 2])

        x_res = cg.conjugate_gradient(mat, b.reshape(-1, 1))
        self.assertTrue(np.allclose(np.array([1, 2]).reshape(-1, 1), mat @ x_res))
