from unittest import TestCase

import numpy as np

import graph_matching_tools.algorithms.multiway.boolean_nmf as nmf


class TestBooleanNMF(TestCase):
    def test_boolean_nmf(self):
        perm = np.array(
            [
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            ]
        )

        res = 0
        for i in range(10):
            u = nmf.boolean_nmf(perm, 3, 1)
            res = u @ u.T

            if np.linalg.norm(perm - res) < 1e-3:
                break

        self.assertTrue(np.linalg.norm(perm - res) < 1e-3)
