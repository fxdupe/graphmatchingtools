from unittest import TestCase

import numpy as np

import graph_matching_tools.algorithms.pairwise.gwl as gwl


class TestGWL(TestCase):

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
        mu_s = np.ones((10, 1)) / 10
        mu_t = np.ones((11, 1)) / 10
        x_s = np.ones((10, 10))
        x_t = np.ones((11, 10))

        res = gwl._gw_proximal_point_solver(
            c_s, c_t, mu_s, mu_t, x_s, x_t, 0.1, 1.0, 10, 1
        )

        self.assertTrue(np.linalg.norm(res - 0.00909091) < 1e-2, "Inner S-H loop test")

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
        rng = np.random.default_rng(seed=1)
        c_s = 1.0 - np.eye(10, 10)
        c_t = 1.0 - np.eye(11, 11)
        transport = 1.0 - np.eye(10, 11)

        x_s, x_t = gwl._update_embeddings(
            c_s, c_t, transport, 1.0, 1.0, 11, 10, 0.1, random_generator=rng
        )
        self.assertEqual(x_s.shape[0], 10, "New source embedding")
        self.assertEqual(x_t.shape[1], 11, "New source embedding")

    def test_gromov_wasserstein_learning(self):
        cost_s = np.array(
            [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype="d"
        )
        cost_s = 1.0 - cost_s
        cost_t = np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype="d"
        )
        cost_t = 1.0 - cost_t
        cost_st = np.array(
            [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]], dtype="d"
        )
        mu_s = np.reshape(np.array([2, 1, 1]) / 4.0, (-1, 1))
        mu_t = np.reshape(np.array([1, 1, 2]) / 4.0, (-1, 1))

        transport = gwl.gromov_wasserstein_learning(
            cost_s,
            cost_t,
            mu_s,
            mu_t,
            10.0,
            0.1,
            4,
            20,
            20,
            20,
            0.001,
            cost_st=cost_st,
            use_cross_cost=True,
            random_seed=1,
        )
        permut = np.array([[0.0, 0.0, 0.5], [0.25, 0.0, 0.0], [0.0, 0.25, 0.0]])
        self.assertTrue(
            np.linalg.norm(transport[0] - permut) < 1e-7, "Matching comparison"
        )
