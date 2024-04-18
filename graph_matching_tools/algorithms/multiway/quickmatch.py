"""This module contains the quick algorithms for computing an approximation the multigraph matching.

Code from paper,
[1] Tron, R., Zhou, X., Esteves, C., & Daniilidis, K. (2017). Fast multi-image matching via density-based clustering.
In Proceedings of the IEEE international conference on computer vision (pp. 4057-4066).

.. moduleauthor:: François-Xavier Dupé
"""

import numpy as np
import networkx as nx

import graph_matching_tools.utils.union as uf


def _compute_density(
    graphs: list[nx.Graph], sizes: list[int], node_data: str, rho_den: float
) -> np.ndarray:
    """Compute the density vector.

    :param list[nx.Graph] graphs: the list of graphs.
    :param list[int] sizes: the sizes of the different graphs.
    :param str node_data: the name of the data vector.
    :param float rho_den: the density parameter (on the variance of the data).
    :return: the density vector.
    :rtype: np.ndarray
    """
    full_sizes = np.sum(sizes)
    densities = np.zeros((full_sizes,))
    min_distances = np.zeros((full_sizes,))

    index = 0
    for g in graphs:
        for i in g:
            dist = 1e90
            # Get the smallest distance inside the graph
            for j in g:
                if i == j:
                    continue
                new_dist = np.linalg.norm(
                    np.array(g.nodes[i][node_data]) - np.array(g.nodes[j][node_data])
                )
                if new_dist < dist:
                    dist = new_dist
            min_distances[index] = dist
            index += 1

    index = 0
    for g in graphs:
        for i in g:
            # Compute the density for the current node
            density = 0.0
            inner_index = 0
            for graph in graphs:
                for node in graph:
                    dist = min_distances[inner_index]
                    density += np.log(1.0 + dist) * np.exp(
                        -np.linalg.norm(
                            np.array(g.nodes[i][node_data])
                            - np.array(graph.nodes[node][node_data])
                        )
                        ** 2.0
                        / (2.0 * (rho_den * dist) ** 2.0)
                    )
                    inner_index += 1
            densities[index] = density
            index += 1

    return densities


def _compute_parents(
    graphs: list[nx.Graph], sizes: list[int], node_data: str, densities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the parent vector with the distances.

    :param list graphs: the list of graphs.
    :param list sizes: the sizes of the graphs.
    :param str node_data: the name of the node data.
    :param np.ndarray densities: the density vector.
    :return: the parent vector with the associated distances.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    full_size = np.sum(sizes)
    parents = np.zeros((full_size,), dtype="i")
    distances = np.zeros((full_size,))

    index = 0
    for g in graphs:
        for i in g:
            dist = 1e90
            parent = index
            inner_index = 0
            for graph in graphs:
                if graph is g:
                    inner_index += nx.number_of_nodes(g)
                    continue
                for node in graph:
                    new_dist = np.linalg.norm(
                        np.array(g.nodes[i][node_data])
                        - np.array(graph.nodes[node][node_data])
                    )
                    if new_dist <= dist and densities[index] <= densities[inner_index]:
                        parent = inner_index
                        dist = new_dist
                    inner_index += 1
            parents[index] = parent
            distances[index] = dist
            index += 1

    return parents, distances


def quickmatch(
    graphs: list[nx.Graph], node_data: str, rho_den: float, rho_edge: float
) -> np.ndarray:
    """The QuickMatch method for graph matching.

    :param list[nx.Graph] graphs: the list of graphs.
    :param str node_data: the name of the node data vector.
    :param float rho_den: the density parameter.
    :param float rho_edge: the edge merging hyperparameter.
    :return: a node universe for all the graphs.
    :rtype: np.ndarray

    Here an example using NetworkX and some utils:

    .. doctest:

    >>> u = quickmatch.quickmatch(graphs, "weight", 0.25, 0.9)
    >>> u
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 1., 0.],
           [1., 0., 0.],
           [0., 0., 1.],
           [1., 0., 0.],
           [0., 1., 0.]])
    """
    sizes = []
    for g in graphs:
        sizes.append(nx.number_of_nodes(g))
    full_size = np.sum(sizes)

    graph_index = np.zeros((full_size, 2), dtype="i")
    index = 0
    for g in range(len(graphs)):
        for i in range(nx.number_of_nodes(graphs[g])):
            graph_index[index + i, 0] = g
            graph_index[index + i, 1] = i
        index += nx.number_of_nodes(graphs[g])

    densities = _compute_density(graphs, sizes, node_data, rho_den)
    parents, distances = _compute_parents(graphs, sizes, node_data, densities)
    min_distances = np.zeros((full_size,)) + 1e90
    s_index = np.argsort(distances)

    # Compute the clusters
    clusters_uf = uf.create_set(full_size)
    graphs_clusters = np.zeros((len(graphs), full_size)) - 1
    for i in range(full_size):
        x = s_index[i]
        y = parents[x]

        if x == y:
            continue

        # Get the intersection between the clusters (only one instance of each graph by clusters)
        x_root = uf.find(x, clusters_uf)
        y_root = uf.find(y, clusters_uf)

        g_x = graph_index[x, 0]
        g_y = graph_index[y, 0]

        intersect = False
        if graphs_clusters[g_x, y_root] != -1 and graphs_clusters[g_y, x_root] != -1:
            intersect = True

        # Get the min distance
        min_dist = np.min([min_distances[x_root], min_distances[y_root]])
        dist = np.linalg.norm(
            np.array(graphs[g_x].nodes[graph_index[x, 1]][node_data])
            - np.array(graphs[g_y].nodes[graph_index[y, 1]][node_data])
        )

        if not intersect and distances[x] <= rho_edge * min_dist:
            uf.union(x_root, y_root, clusters_uf)
            new_root = uf.find(x, clusters_uf)
            graphs_clusters[g_x, new_root] = 1
            graphs_clusters[g_y, new_root] = 1

            if min_distances[new_root] > dist:
                min_distances[new_root] = dist

    # Do the labelling
    clusters = np.zeros((full_size,), dtype="i")
    labels = np.zeros((full_size,), dtype="i") - 1
    nb_labels = 0
    for i in range(full_size):
        cl = uf.find(i, clusters_uf)
        if labels[cl] == -1:
            labels[cl] = nb_labels
            nb_labels += 1
        clusters[i] = labels[cl]

    u = np.zeros((full_size, nb_labels))
    for i in range(full_size):
        u[i, clusters[i]] = 1

    return u
