from unittest import TestCase

import numpy as np

import graph_matching_tools.metrics.matching as matching


class TestMatching(TestCase):

    def test_compute_f1score(self):
        t1 = [[1, 0, 0, 1],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [1, 0, 0, 1]]

        t2 = [[1, 0, 0, 1],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]]

        res = matching.compute_f1score(np.array(t1), np.array(t2))

        self.assertEqual(res[1], 0.5)
        self.assertEqual(res[2], 1.0)
