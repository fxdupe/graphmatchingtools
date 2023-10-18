import unittest

import numpy as np
import networkx as nx

import graph_matching_tools.algorithms.multiway.dist_mkergm as mkergm
import graph_matching_tools.algorithms.kernels.gaussian as kern
import graph_matching_tools.algorithms.kernels.rff as rff
import graph_matching_tools.algorithms.kernels.utils as ku


class TestDistributedMultiwayKerGM(unittest.TestCase):
    def test_get_bulk_permutations_from_dict(self):
        dict_perm = dict()
        dict_perm["0,1"] = np.eye(3, 3)
        dict_perm["0,2"] = np.eye(3, 2)
        dict_perm["1,2"] = np.eye(3, 2)

        perm = mkergm.get_bulk_permutations_from_dict(dict_perm, [3, 3, 2])
        truth = [
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        ]

        self.assertEqual(np.linalg.norm(perm - truth) < 1e-3, True)

    def test_stochastic_dist_mkergm(self):
        graph1 = nx.Graph()
        graph1.add_node(0, weight=2.0)
        graph1.add_node(1, weight=20.0)
        graph1.add_edge(0, 1, weight=np.array((1.0,)))

        graph2 = nx.Graph()
        graph2.add_node(0, weight=20.0)
        graph2.add_node(1, weight=2.0)
        graph2.add_edge(0, 1, weight=np.array((1.0,)))

        graph3 = nx.Graph()
        graph3.add_node(0, weight=-5.0)
        graph3.add_node(1, weight=30.0)
        graph3.add_node(2, weight=2.0)
        graph3.add_edge(1, 2, weight=np.array((1.0,)))

        graphs = [graph1, graph2, graph3]

        node_kernel = kern.create_gaussian_node_kernel(10.0, "weight")
        vectors, offsets = rff.create_random_vectors(1, 100, 1.0)
        phi1 = rff.compute_phi(graph1, "weight", vectors, offsets)
        phi2 = rff.compute_phi(graph2, "weight", vectors, offsets)
        phi3 = rff.compute_phi(graph3, "weight", vectors, offsets)

        knodes = dict()
        knodes["0,1"] = ku.compute_knode(graph1, graph2, node_kernel)
        knodes["0,2"] = ku.compute_knode(graph1, graph3, node_kernel)
        knodes["1,2"] = ku.compute_knode(graph2, graph3, node_kernel)

        res = mkergm.stochastic_dist_mkergm(
            graphs, knodes, [phi1, phi2, phi3], 2, 10, 2, 20
        )
        truth_01 = [[0.0, 1.0], [1.0, 0.0]]
        truth_02 = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]

        self.assertEqual(np.linalg.norm(res["0,1"] - truth_01) < 1e-3, True)
        self.assertEqual(np.linalg.norm(res["0,2"] - truth_02) < 1e-3, True)
