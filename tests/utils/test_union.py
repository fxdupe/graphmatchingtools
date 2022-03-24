"""Test of the UNION-FIND algorithm
.. moduleauthor:: François-Xavier Dupé
"""

import unittest
from unittest import TestCase

import numpy as np

import graph_matching_tools.utils.union as uf


class TestUnion(unittest.TestCase):

    def test_union(self):
        parents = uf.create_set(5)
        uf.union(1, 2, parents)
        self.assertEqual(uf.find(1, parents), uf.find(2, parents))
