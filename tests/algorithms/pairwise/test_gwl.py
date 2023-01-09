from unittest import TestCase

import numpy as np

import graph_matching_tools.algorithms.pairwise.gwl as gwl


class TestGWL(TestCase):

    def test_loss_function(self):
        c_s = np.ones((10, 10))
        c_t = np.ones((10, 10))
        transport = np.ones((10, 10)) * 0.5
        res = gwl._loss_function(c_s, c_t, transport)
        self.assertTrue(np.linalg.norm(res - 0.0) < 1e-4, "Loss function test")

    def test_distance_matrix(self):
        x_s = np.ones((10, 10))
        x_t = np.ones((10, 10))
        res = gwl._distance_matrix(x_s, x_t)
        self.assertTrue(np.linalg.norm(res - 0.0) < 1e-4, "Distance function test")

        x_t[0, 0] = 10
        res = gwl._distance_matrix(x_s, x_t)
        self.assertTrue(np.max(res) > 0.1, "Maximum distance function test")

    def test_gw_proximal_point_solver(self):
        c_s = np.ones((10, 10))
        c_t = np.ones((11, 11))
        mu_s = np.ones((10, ))
        mu_t = np.ones((11, ))
        x_s = np.ones((10, 10))
        x_t = np.ones((11, 10))

        res = gwl._gw_proximal_point_solver(c_s, c_t, mu_s, mu_t, x_s, x_t, 0.1, 0.1, 10, 1)
        self.assertTrue(np.linalg.norm(res - 0.0909) < 1e-2, "Inner S-H loop test")

    def test_update_embeddings_gradient(self):
        c_s = np.ones((10, 10))
        c_t = np.ones((11, 11))
        x_s = np.ones((10, 10))
        x_t = np.ones((11, 10))
        transport = np.ones((10, 11))

        params = dict()
        params["x_s"] = x_s
        params["x_t"] = x_t

        res = gwl._update_embeddings_gradient(params, 1.0, 1.0, c_s, c_t, transport)
        self.assertTrue(np.abs(res - 221.0) < 1.0, "Embedding loss function")

    def test_update_embeddings(self):
        c_s = 1.0 - np.eye(10, 10)
        c_t = 1.0 - np.eye(11, 11)
        transport = 1.0 - np.eye(10, 11)

        x_s, x_t = gwl._update_embeddings(c_s, c_t, transport, 1.0, 1.0, 11, 10, 0.1)
        self.assertEqual(x_s.shape[0], 10, "New source embedding")
        self.assertEqual(x_t.shape[1], 11, "New source embedding")

    def test_gromov_wasserstein_learning(self):
        cost_s = np.array([[0., 1., 1.],
                           [1., 0., 0.],
                           [1., 0., 0.]], dtype="d")
        cost_s = 1.0 - cost_s
        cost_t = np.array([[0., 0., 1.],
                           [0., 0., 1.],
                           [1., 1., 0.]], dtype="d")
        cost_t = 1.0 - cost_t
        mu_s = np.array([2, 1, 1]) / 4.0
        mu_t = np.array([1, 1, 2]) / 4.0

        match = gwl.gromov_wasserstein_learning(cost_s, cost_t, mu_s, mu_t, 10.0, 0.1, 4, 20, 20, 20, 0.001)
        permut = np.array([2, 0, 1])
        print(match)
        self.assertTrue(np.linalg.norm(match - permut) < 1e-7, "Matching comparison")
