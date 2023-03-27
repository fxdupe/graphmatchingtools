from unittest import TestCase

import numpy as np
import networkx as nx

import graph_matching_tools.algorithms.multiway.ga_mgmc as ga_mgmc
import graph_matching_tools.algorithms.kernels.gaussian as kern
import graph_matching_tools.algorithms.kernels.utils as utils


class TestGA_MGMC(TestCase):
    def test_ga_mgmc(self):
        node_kernel = kern.create_gaussian_node_kernel(2.0, "weight")

        graph1 = nx.Graph()
        graph1.add_node(0, weight=2.0)
        graph1.add_node(1, weight=20.0)
        graph1.add_edge(0, 1, weight=1.0)

        graph2 = nx.Graph()
        graph2.add_node(0, weight=20.0)
        graph2.add_node(1, weight=2.0)
        graph2.add_edge(0, 1, weight=1.0)

        graph3 = nx.Graph()
        graph3.add_node(0, weight=-5.0)
        graph3.add_node(1, weight=30.0)
        graph3.add_node(2, weight=2.0)
        graph3.add_edge(1, 2, weight=1.0)
        graphs = [graph1, graph2, graph3]

        knode = utils.create_full_node_affinity_matrix(graphs, node_kernel)
        u = ga_mgmc.ga_mgmc(graphs, knode, 3, "weight", tau=0.1, tau_min=1e-2)
        res = u @ u.T

        truth = [
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        ]

        self.assertEqual(np.linalg.norm(res - truth) < 1e-3, True)

        graph1 = nx.Graph()
        graph1.add_node(0, weight=1.0)
        graph1.add_node(1, weight=2.0)
        graph1.add_edge(0, 1, weight=1.0)

        graph2 = nx.Graph()
        graph2.add_node(0, weight=2.0)
        graph2.add_node(1, weight=1.0)
        graph2.add_edge(0, 1, weight=1.0)

        graphs = [graph1, graph2]

        truth = np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
            ]
        )

        init = [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
        node_kernel = kern.create_gaussian_node_kernel(0.1, "weight")
        knode = utils.create_full_node_affinity_matrix(graphs, node_kernel)
        u = ga_mgmc.ga_mgmc(
            graphs, knode, 2, "weight", init=np.array(init), tau=0.1, tau_min=1e-2
        )
        self.assertEqual(np.linalg.norm(u @ u.T - truth) < 1e-3, True)
        u = ga_mgmc.ga_mgmc(
            graphs,
            knode,
            3,
            "weight",
            normalize_aff=True,
            init=np.array(init),
            tau=0.1,
            tau_min=1e-2,
        )
        self.assertEqual(np.linalg.norm(u @ u.T - truth) < 1e-3, True)
