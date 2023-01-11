from unittest import TestCase

import numpy as np

import graph_matching_tools.algorithms.pairwise.fgw as fgw


class TestFGW(TestCase):

    def test_fgw_direct_matching(self):
        cost_s = np.array([[0., 1., 1.],
                           [1., 0., 0.],
                           [1., 0., 0.]], dtype="d")
        cost_s = 1.0 - cost_s
        cost_t = np.array([[0., 0., 1.],
                           [0., 0., 1.],
                           [1., 1., 0.]], dtype="d")
        cost_t = 1.0 - cost_t
        distances = np.array([[1., 1., 0.],
                              [0., 1., 1.],
                              [1., 0., 1.]], dtype="d")
        mu_s = np.array([2, 1, 1]) / 4.0
        mu_t = np.array([1, 1, 2]) / 4.0

        transport = fgw.fgw_direct_matching(cost_s, cost_t, mu_s, mu_t, distances, 0.5, 100, gamma=0.01)

        match = np.zeros((cost_s.shape[0],)) - 1.0
        for i in range(transport.shape[0]):
            match[i] = np.argmax(transport[i, :])

        permut = np.array([2, 0, 1])
        self.assertTrue(np.linalg.norm(match - permut) < 1e-7, "Matching comparison")
