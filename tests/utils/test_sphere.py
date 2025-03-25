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

    def test_sample(self):
        res2 = sphere.sample_sphere(10, np.array([1.0, 1.0, 1.0]), 10)
        self.assertEqual(res2.x.shape, (10,))
        self.assertEqual(res2.kappa, 10)
