from unittest import TestCase

import numpy as np
import networkx as nx

import graph_matching_tools.utils.permutations as perm


class TestPermutations(TestCase):
    def test_get_permutation_matrix_from_dictionary(self):
        dic = dict()
        dic["0,0"] = {0: 0, 1: 1}
        dic["0,1"] = {0: 1}
        dic["1,0"] = {1: 0}
        dic["1,1"] = {0: 0, 1: 1}

        res = perm.get_permutation_matrix_from_dictionary(dic, [2, 2])
        truth = [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]]

        self.assertTrue(np.linalg.norm(res - truth) < 1e-3)

    def test_permutation_matrix_from_matching(self):
        sizes = [2, 2]
        matching = np.array([[0, 1], [1, 0]])
        permut = perm.get_permutation_matrix_from_matching(matching, sizes, 3)
        truth = np.array([[1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1]])
        self.assertTrue(np.linalg.norm(permut - truth) < 1e-3)

        matching = np.array([[0, 1], [2, 0]])
        permut = perm.get_permutation_matrix_from_matching(matching, sizes, 3)
        truth = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
        self.assertTrue(np.linalg.norm(permut - truth) < 1e-3)
