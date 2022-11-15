"""This utility module contains some procedures for creating the affinity matrices

.. moduleauthor:: François-Xavier Dupé
"""
import numpy as np
import networkx as nx


def compute_knode(graph1, graph2, kernel):
    """Compute the affinity matrix between the nodes

    :param nx.classes.graph.Graph graph1: the first graph
    :param nx.classes.graph.Graph graph2: the second graph
    :param callable kernel: the kernel between nodes
    :return: the affinity matrix between the nodes
    """
    knode = np.zeros((nx.number_of_nodes(graph1), nx.number_of_nodes(graph2)))
    for n1 in graph1:
        for n2 in graph2:
            knode[n1, n2] = kernel(graph1, n1, graph2, n2)
    return knode


def create_full_node_affinity_matrix(graphs, kernel):
    """Compute the full pairwise matrix from graphs

    :param list graphs: the list of graphs (in networkx format)
    :param callable kernel: the kernel between the node
    :return: a tuple with the full matrix and the sizes of the different graphs
    """
    full_size = 0
    for graph in graphs:
        full_size += nx.number_of_nodes(graph)

    # 1 - Building the full initial permutation matrix
    knode = np.zeros((full_size, full_size))

    # 2 - Compute the pairwise permutations
    index1 = 0
    for idx1 in range(len(graphs)):
        g1 = graphs[idx1]
        index2 = index1
        for idx2 in range(idx1, len(graphs)):
            g2 = graphs[idx2]
            gram = compute_knode(g1, g2, kernel)

            knode[index1:index1 + nx.number_of_nodes(g1), index2:index2 + nx.number_of_nodes(g2)] = gram
            if idx1 != idx2:
                knode[index2:index2 + nx.number_of_nodes(g2), index1:index1 + nx.number_of_nodes(g1)] = gram.T
            index2 += nx.number_of_nodes(g2)

        index1 += nx.number_of_nodes(g1)

    return knode
