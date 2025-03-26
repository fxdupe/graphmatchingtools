from unittest import TestCase

import numpy as np

import graph_matching_tools.solvers.ot.sns as sns


class TestSNS(TestCase):

    def test_objective_function(self):
        cost = np.array([[0, 1, 0.6], [9, 0, 0.3], [0.5, 0.4, 0]])
        mu_s = np.array([0.333, 0.333, 0.333])
        mu_t = np.array([0.333, 0.333, 0.333])

        z = np.zeros((6, 1))
        res = sns._objective_function(z, cost, mu_s, mu_t, 1000)
        self.assertTrue(abs(res + 0.001103638323) < 1e-4)

    def test_compute_hessian(self):
        p = np.array([[0, 1, 0.6], [9, 0, 0.3], [0.5, 0.4, 0]])
        res = sns._compute_hessian(p, 100)

        truth = np.array(
            [
                [160.0, 0.0, 0.0, 0.0, 100.0, 60.0],
                [0.0, 930.0, 0.0, 900.0, 0.0, 30.0],
                [0.0, 0.0, 90.0, 50.0, 40.0, 0.0],
                [0.0, 900.0, 50.0, 950.0, 0.0, 0.0],
                [100.0, 0.0, 40.0, 0.0, 140.0, 0.0],
                [60.0, 30.0, 0.0, 0.0, 0.0, 90.0],
            ]
        )

        self.assertTrue(np.linalg.norm(res - truth) < 1e-4)

    def test_compute_transport_from_dual(self):
        cost = np.array([[0, 1, 0.6], [9, 0, 0.3], [0.5, 0.4, 0]])
        x = np.reshape(np.array([0.333, 0.333, 0.333]), (-1, 1))
        y = np.reshape(np.array([0.333, 0.333, 0.333]), (-1, 1))

        res = sns._compute_transport_from_dual(
            cost, np.concatenate([x, y], axis=0), 0.1
        )

        truth = np.array(
            [
                [
                    [0.39321451, 0.3557952, 0.37031548],
                    [0.15986909, 0.39321451, 0.38159326],
                    [0.37403721, 0.37779634, 0.39321451],
                ]
            ]
        )

        self.assertTrue(np.linalg.norm(res - truth) < 1e-4)

    def test_sparsify(self):
        matrix = np.array([[10.0, 1, 0.6], [9, 0, 0.3], [0.5, 0.4, 0]])
        truth = np.array([[10.0, 1, 0], [9, 0, 0], [0, 0, 0]])
        res = sns._sparsify(matrix, 3)

        self.assertEqual(res.shape[0], matrix.shape[0])
        self.assertEqual(res.shape[1], matrix.shape[1])
        self.assertTrue(np.linalg.norm(res - truth) < 1e-4)

    def test_linesearch(self):
        cost = np.array([[0, 1, 0.6], [9, 0, 0.3], [0.5, 0.4, 0]])
        mu_s = np.reshape(np.array([0.333, 0.333, 0.333]), (-1, 1))
        mu_t = np.reshape(np.array([0.333, 0.333, 0.333]), (-1, 1))
        z = np.ones((6, 1)) / 6
        direction = np.reshape(np.array([1, 0, 0, 0, 0, 0]), (-1, 1))
        res = sns._linesearch(z, direction, 1.0, cost, mu_s, mu_t, 1.0)
        self.assertTrue(abs(res - 4.440892098500626e-16) < 1e-6)

    def test_sinkhorn_newton_sparse_method(self):

        cost = np.array([[0, 1, 0.6], [9, 0, 0.3], [0.5, 0.4, 0]])
        mu_s = np.array([0.333, 0.333, 0.333])
        mu_t = np.array([0.333, 0.333, 0.333])

        res = sns.sinkhorn_newton_sparse_method(
            cost,
            mu_s.reshape((-1, 1)),
            mu_t.reshape((-1, 1)),
            100.0,
            rho=0.6,
            n1_iterations=100,
            n2_iterations=100,
        )

        truth = np.array(
            [
                [3.33000000e-001, 0.00000000e000, 8.82582052e-262],
                [0.00000000e000, 3.33000000e-001, 1.71435067e-131],
                [2.37248394e-218, 6.37751476e-175, 3.33000000e-001],
            ]
        )

        self.assertTrue(np.linalg.norm(res - truth) < 1e-3)
