"""
Utility function for permutation matrix

.. moduleauthor:: François-Xavier Dupé
"""
import random

import numpy as np
import networkx as nx


def get_permutation_matrix_from_dictionary(matching, g_sizes):
    """
    Create the full permutation matrix from the matching result
    :param matching: the matching result for each graph (nodes number, assignment)
    :param g_sizes: the list of the size of the different graph
    :return: the full permutation matrix
    """
    f_size = int(np.sum(g_sizes))
    res = np.zeros((f_size, f_size))

    idx1 = 0
    for i_g1 in range(len(g_sizes)):
        idx2 = 0
        for i_g2 in range(len(g_sizes)):
            match = matching["{},{}".format(i_g1, i_g2)]
            for k in match:
                res[idx1 + int(k), idx2 + match[k]] = 1
            idx2 += g_sizes[i_g2]
        idx1 += g_sizes[i_g1]

    np.fill_diagonal(res, 1)
    return res


def randomize_nodes_position(graphs):
    """
    Randomize the node position inside a graph
    :param list graphs: a list of graph (networkx format)
    :return: the list of new graphs and the new index
    """
    res = []
    new_graphs = []
    for g in graphs:
        nb_nodes = nx.number_of_nodes(g)
        nidx = list(range(nb_nodes))
        random.shuffle(nidx)
        n_g = nx.relabel_nodes(g, dict(zip(g.nodes(), nidx)))
        res.append(nidx)
        new_graphs.append(n_g)
    return new_graphs, res
