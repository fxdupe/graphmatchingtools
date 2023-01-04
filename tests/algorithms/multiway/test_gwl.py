import unittest

import numpy as np
import networkx as nx

import graph_matching_tools.algorithms.multiway.gwl as gwl


def get_cost_matrix(g):
    """
    Create a cost matrix from a graph using Gaussian kernel
    :param g: the graph
    :return: the cost matrix linked to the graph
    """
    cost = np.ones((nx.number_of_nodes(g), nx.number_of_nodes(g)))
    for u, v, w in g.edges.data("weight"):
        cost[u, v] = 1.0 / w
        cost[v, u] = cost[u, v]
    return cost


def get_mu_vector(g):
    """
    Create probability vector from the weights on the node
    :param g: the graph
    :return: the probability vector for each node
    """
    mu = np.zeros((nx.number_of_nodes(g), ))
    for n in g:
        mu[n] = g.nodes[n]["weight"]
    mu = mu / np.sum(mu)
    return mu


class TestDirectMultiwayGWL(unittest.TestCase):

    def test_multi_pairwise_gwl(self):
        graph1 = nx.Graph()
        graph1.add_node(0, weight=2.0)
        graph1.add_node(1, weight=20.0)
        graph1.add_edge(0, 1, weight=np.array((10.0, )))

        graph2 = nx.Graph()
        graph2.add_node(0, weight=20.0)
        graph2.add_node(1, weight=2.0)
        graph2.add_edge(0, 1, weight=np.array((10.0, )))

        graph3 = nx.Graph()
        graph3.add_node(0, weight=1.0)
        graph3.add_node(1, weight=4.0)
        graph3.add_node(2, weight=5.0)
        graph3.add_edge(1, 2, weight=np.array((10.0, )))

        graphs = [graph1, graph2, graph3]

        mus = [get_mu_vector(g) for g in graphs]
        costs = [get_cost_matrix(g) for g in graphs]

        truth = [[1., 0., 0., 1., 0., 1., 0.],
                 [0., 1., 1., 0., 0., 0., 1.],
                 [0., 1., 1., 0., 0., 0., 1.],
                 [1., 0., 0., 1., 0., 1., 0.],
                 [0., 0., 0., 0., 1., 0., 0.],
                 [1., 0., 0., 1., 0., 1., 0.],
                 [0., 1., 1., 0., 0., 0., 1.]]
        truth = np.array(truth)

        res = gwl.multi_pairwise_gwl(costs, mus, 10.0, 2.0, 5, 20, 20, 2, 0.1)
        # print(res)
        self.assertEqual(np.linalg.norm(res - truth) < 1e-3, True)
