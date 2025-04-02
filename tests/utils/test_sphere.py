from unittest import TestCase

import numpy as np

import graph_matching_tools.utils.sphere as sphere


class TestSphere(TestCase):

    def test_make_sphere(self):
        res = sphere.make_sphere(0.5)
        self.assertEqual(res[0].shape, (100, 100))
        self.assertEqual(res[1].shape, (100, 100))
        self.assertEqual(res[2].shape, (100, 100))
        self.assertEqual(res[0][0, 0], 0)

    def test_random_coordinate_sampling(self):
        res = sphere.random_coordinate_sampling(10, np.array([10, 20, 30]), 2.0)
        self.assertEqual(res[0].shape, (10,))
        self.assertEqual(res[1].shape, (10,))
        self.assertEqual(res[2].shape, (10,))

    def test_random_sampling(self):
        res = sphere.random_sampling(10, 1.0)
        self.assertEqual(res[0].shape, (10,))
        self.assertEqual(res[1].shape, (10,))
        self.assertEqual(res[2].shape, (10,))
        self.assertTrue(
            (np.linalg.norm(np.array([res[0][0], res[1][0], res[2][0]])) - 1.0) < 1e-3
        )

    def test_regular_sampling(self):
        res = sphere.regular_sampling(10, 1.0)
        self.assertEqual(res[0].shape, (10,))
        self.assertEqual(res[1].shape, (10,))
        self.assertEqual(res[2].shape, (10,))
        self.assertTrue(
            (np.linalg.norm(np.array([res[0][0], res[1][0], res[2][0]])) - 1.0) < 1e-3
        )
