from unittest import TestCase

import trimesh
import numpy as np

import graph_matching_tools.generators.topology as topology


class Test(TestCase):
    def test_adjacency_matrix(self):
        mesh = trimesh.Trimesh(
            vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], faces=[[0, 1, 2]]
        )
        adj = topology.adjacency_matrix(mesh)
        truth = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        self.assertTrue(np.linalg.norm(truth - adj.todense()) < 1e-3)
