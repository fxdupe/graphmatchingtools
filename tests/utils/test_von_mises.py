"""Test of the von Mises Fisher distribution.

.. moduleauthor:: François-Xavier Dupé
"""

import unittest

import numpy as np

import graph_matching_tools.utils.von_mises as von_mises


class TestVonMises(unittest.TestCase):
    def test_sample_von_mises(self):
        res = von_mises.sample_von_mises(np.array([1.0, 1.0]), 10.0, 20)
        self.assertEqual(res.shape, (20, 2))
        self.assertTrue(
            np.linalg.norm(np.mean(res, axis=0) - np.array([1.0, 1.0])) < 1e-1
        )

    def test__sample_weight(self):
        res = von_mises._sample_weight(1.0, 10)
        self.assertIsNotNone(res)
