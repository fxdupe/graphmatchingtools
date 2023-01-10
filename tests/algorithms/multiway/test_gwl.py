import unittest

import numpy as np
import networkx as nx

import graph_matching_tools.algorithms.multiway.gwl as gwl
import graph_matching_tools.algorithms.kernels.gaussian as kern
import graph_matching_tools.algorithms.kernels.utils as ku


def get_mu_vector(g):
    """
    Create probability vector from the weights on the node
    :param g: the graph
    :return: the probability vector for each node
    """
    mu = np.ones((nx.number_of_nodes(g), ))
    mu = mu / nx.number_of_nodes(g)
    return mu


class TestDirectMultiwayGWL(unittest.TestCase):

    def test_multi_pairwise_gwl(self):
        graph1 = nx.Graph()
        graph1.add_node(0, weight=2.0)
        graph1.add_node(1, weight=5.0)
        graph1.add_edge(0, 1)

        graph2 = nx.Graph()
        graph2.add_node(0, weight=5.0)
        graph2.add_node(1, weight=2.0)
        graph2.add_edge(0, 1)

        graph3 = nx.Graph()
        graph3.add_node(0, weight=1.0)
        graph3.add_node(1, weight=2.0)
        graph3.add_node(2, weight=5.0)
        graph3.add_edge(1, 2)

        graphs = [graph1, graph2, graph3]

        node_kernel = kern.create_gaussian_node_kernel(1.0, "weight")
        mus = [get_mu_vector(g) for g in graphs]
        costs = [(1.0 - ku.compute_knode(g, g, node_kernel) *
                  (nx.adjacency_matrix(g).todense() + np.eye(nx.number_of_nodes(g)))) for g in graphs]

        cross_costs = dict()
        for i_s in range(len(graphs)):
            for i_t in range(i_s+1, len(graphs)):
                cost_st = 1.0 - ku.compute_knode(graphs[i_s], graphs[i_t], node_kernel)
                cross_costs["{},{}".format(i_s, i_t)] = cost_st

        truth = [[1., 0., 0., 1., 0., 1., 0.],
                 [0., 1., 1., 0., 0., 0., 1.],
                 [0., 1., 1., 0., 0., 0., 1.],
                 [1., 0., 0., 1., 0., 1., 0.],
                 [0., 0., 0., 0., 1., 0., 0.],
                 [1., 0., 0., 1., 0., 1., 0.],
                 [0., 1., 1., 0., 0., 0., 1.]]
        truth = np.array(truth)

        res = gwl.multi_pairwise_gwl(costs, mus, 1.0, 2.0, 5, 20, 20, 2, 0.1, use_cross_cost=True,
                                     cross_costs=cross_costs)
        self.assertEqual(np.linalg.norm(res - truth) < 1e-3, True)
