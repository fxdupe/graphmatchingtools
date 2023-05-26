"""This module contains the matching algorithm between a set of graphs

This the stochastic extension to multigraph matching of KerGM for graph matching using both edges
and nodes data, from the paper
Zhang, Z., Xiang, Y., Wu, L., Xue, B., & Nehorai, A. (2019). KerGM: Kernelized graph matching. NeurIPS 2019.

This work is genuine and has no previous record as of the 18th of May 2022.

This code only works for graphs of the same size (i.e. number of nodes)

.. moduleauthor:: François-Xavier Dupé
"""
import copy

import numpy as np
import networkx as nx

import graph_matching_tools.algorithms.multiway.mkergm as mkergm


def get_bulk_permutations_from_dict(
    dictionary: dict[str, np.ndarray], size_of_graphs: list[int]
) -> np.ndarray:
    """Build the bulk permutation matrix from the dictionary with permutations.

    :param dict[str, np.ndarray] dictionary: the dictionary with permutation matrix.
    :param list[int] size_of_graphs: the number of nodes of each graph.
    :return: the bulk permutation matrix.
    :rtype: np.ndarray
    """
    res = np.identity(int(np.sum(size_of_graphs)))

    i_size = 0
    for i in range(len(size_of_graphs)):
        j_size = i_size + size_of_graphs[i]
        for j in range(i + 1, len(size_of_graphs)):
            res[
                i_size : i_size + size_of_graphs[i],
                j_size : j_size + size_of_graphs[j],
            ] = dictionary["{},{}".format(i, j)]
            res[
                j_size : j_size + size_of_graphs[j],
                i_size : i_size + size_of_graphs[i],
            ] = dictionary["{},{}".format(i, j)].T
            j_size += size_of_graphs[j]
        i_size += size_of_graphs[i]

    return res


def build_permutation_list(graphs: list[nx.Graph]) -> dict[str, np.ndarray]:
    """Build the permutation matrix list with the correspondence.

    :param list[nx.Graph] graphs: the list of input graphs.
    :return: the list of permutation matrices and the dictionary for indexing.
    :rtype: dict[str, np.ndarray]
    """
    res = {}
    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            res["{},{}".format(i, j)] = np.zeros(
                (nx.number_of_nodes(graphs[i]), nx.number_of_nodes(graphs[j]))
            )
    return res


def compute_gradient(
    x: np.ndarray, knode: np.ndarray, phi: list[np.ndarray]
) -> np.ndarray:
    """Compute the gradient at a given point.

    :param np.ndarray x: the current "permutation" matrix between two graphs.
    :param np.ndarray knode: the node affinity matrix between the two graphs.
    :param list[np.ndarray] phi: the list of edge data tensor between the two graphs.
    :return: the gradient at point x.
    :rtype: np.ndarray
    """
    t1 = np.sum(phi[0] @ x @ phi[1], axis=0)
    grad = knode + 2.0 * t1
    return grad


def stochastic_dist_mkergm(
    graphs: list[nx.Graph],
    nodes_affinities: dict[str, np.ndarray],
    edges_data: list[np.ndarray],
    rank: int,
    batch_size: int,
    batch_nb_graphs: int,
    iterations: int,
    seed: int = 20,
) -> dict[str, np.ndarray]:
    """Stochastic Multi-Graph Matching with difference of convex algorithm.

    :param list[nx.Graph] graphs: list of graphs.
    :param dict[str, np.ndarray] nodes_affinities: dictionary for nodes affinity matrices.
    :param list[np.ndarray] edges_data: list of edge data tensors.
    :param int rank: the dimension of the universe of nodes.
    :param int batch_size: the size of the batch.
    :param int batch_nb_graphs: the number of graphs inside one batch.
    :param int iterations: the number of iterations.
    :param int seed: the seed of the random generator.
    :return: the dictionary of permutations.
    :rtype: dict[str, np.ndarray]
    """
    random_gen = np.random.default_rng(seed=seed)
    set_of_graphs = list(range(len(graphs)))
    permutations = build_permutation_list(graphs)

    # Do the stochastic DCA
    for iteration in range(iterations):
        current_permutations = copy.deepcopy(permutations)
        counter = dict()
        for sampling in range(batch_size):
            index = random_gen.choice(
                set_of_graphs, size=batch_nb_graphs, replace=False, shuffle=True
            )
            index = np.sort(index)  # Simplify the gradient computation
            b_sizes = [nx.number_of_nodes(graphs[i]) for i in index]
            bulk = np.zeros(
                (
                    np.sum(b_sizes),
                    np.sum(b_sizes),
                )
            )

            # Build the next gradient matrix
            i_size = 0
            for i in range(index.shape[0]):
                # Imply diagonal the gradient
                bulk[
                    i_size : i_size + b_sizes[i], i_size : i_size + b_sizes[i]
                ] = np.identity(b_sizes[i])

                j_size = i_size + b_sizes[i]
                for j in range(i + 1, index.shape[0]):
                    # Compute the gradient
                    phi = [edges_data[index[i]], edges_data[index[j]]]
                    knode = nodes_affinities["{},{}".format(index[i], index[j])]
                    permut = current_permutations["{},{}".format(index[i], index[j])]
                    grad = compute_gradient(permut, knode, phi)

                    bulk[
                        i_size : i_size + b_sizes[i],
                        j_size : j_size + b_sizes[j],
                    ] = grad
                    bulk[
                        j_size : j_size + b_sizes[j],
                        i_size : i_size + b_sizes[i],
                    ] = grad.T

                    # Update size (j)
                    j_size += b_sizes[j]

                # Update size (i)
                i_size += b_sizes[i]

            # Apply the projection step
            res = mkergm._rank_projector(bulk, rank, b_sizes, "gpow")

            i_size = 0
            for i in range(index.shape[0]):
                j_size = i_size + b_sizes[i]
                for j in range(i + 1, index.shape[0]):
                    permut = res[
                        i_size : i_size + b_sizes[i],
                        j_size : j_size + b_sizes[j],
                    ]
                    if (index[i], index[j]) in counter:
                        counter[(index[i], index[j])] += 1
                        permutations["{},{}".format(index[i], index[j])] += permut
                    else:
                        counter[(index[i], index[j])] = 1
                        permutations["{},{}".format(index[i], index[j])] = permut

                    j_size += b_sizes[j]
                i_size += b_sizes[i]

        # Reduce step: Get the new permutation matrix with a "majority vote"
        for key in counter:
            (i, j) = key
            permutations["{},{}".format(i, j)] = np.array(
                (permutations["{},{}".format(i, j)] / counter[key]) > 0.5, dtype="f"
            )

    return permutations
