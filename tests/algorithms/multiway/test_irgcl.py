import unittest

import numpy as np
import networkx as nx

import graph_matching_tools.algorithms.multiway.irgcl as irgcl


class TestIRGCL(unittest.TestCase):

    def test_irgcl(self):
        graph1 = nx.Graph()
        graph1.add_node(0, weight=2.0)
        graph1.add_node(1, weight=20.0)
        graph1.add_edge(0, 1, weight=1.0)

        graph2 = nx.Graph()
        graph2.add_node(0, weight=20.0)
        graph2.add_node(1, weight=2.0)
        graph2.add_edge(0, 1, weight=1.0)

        x_base = np.eye(4)
        x_base[3, 0] = 1
        x_base[0, 3] = 1
        x_base[2, 1] = 1
        x_base[1, 2] = 1

        p_true = np.zeros((4, 2))
        p_true[0, 0] = 1
        p_true[1, 1] = 1
        p_true[2, 1] = 1
        p_true[3, 0] = 1

        p = 1e9
        for i in range(10):
            p = irgcl.irgcl(x_base, irgcl.beta_t, irgcl.alpha_t, irgcl.lambda_t, 2, 2)
            if np.linalg.norm(p - p_true) < 1e-3:
                break

        self.assertTrue(np.linalg.norm(p - p_true) < 1e-3)
