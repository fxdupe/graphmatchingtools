from unittest import TestCase

import numpy as np

import graph_matching_tools.solvers.ot.sinkhorn as sh


class TestSinkhorn(TestCase):

    def test_sinkhorn_method(self):

        cost = np.array([[0, 1, 0.6], [9, 0, 0.3], [0.5, 0.4, 0]])
        mu_s = np.array([0.333, 0.333, 0.333])
        mu_t = np.array([0.333, 0.333, 0.333])

        res = sh.sinkhorn_method(cost, mu_s, mu_t, 0.1, 1e-6, 1000)

        truth = np.array(
            [
                [3.31644835e-01, 3.97225059e-05, 1.31544263e-03],
                [1.00394996e-40, 3.23238927e-01, 9.76107295e-03],
                [1.35554661e-03, 9.72113905e-03, 3.21923314e-01],
            ]
        )

        self.assertTrue(np.linalg.norm(res - truth) < 1e-3)
