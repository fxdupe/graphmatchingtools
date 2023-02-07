"""Example of using the multiway matching algorithm using sample graphs from INT.

.. moduleauthor:: François-Xavier Dupé
"""
import argparse
import os
import pickle

import numpy as np
import networkx as nx

import graph_matching_tools.algorithms.multiway.hippi as hippi
import graph_matching_tools.algorithms.multiway.matcheig as matcheig
import graph_matching_tools.algorithms.multiway.mkergm as mkergm
import graph_matching_tools.algorithms.multiway.stiefel as stiefel
import graph_matching_tools.utils.utils as utils
import graph_matching_tools.utils.permutations as permutations
import graph_matching_tools.algorithms.kernels.gaussian as gaussian
import graph_matching_tools.algorithms.kernels.utils as kutils
import graph_matching_tools.algorithms.kernels.rff as rff
import graph_matching_tools.metrics.matching as measures


def add_dummy_nodes(graphs, rank):
    """Add dummy nodes to graph to uniform the sizes.

    :param list[nx.Graph] graphs: the list of graphs.
    :param int rank: the rank of the universe of nodes.
    :return: the new list of graph with the new matching index.
    :rtype: np.ndarray
    """
    g_sizes = [nx.number_of_nodes(gr) for gr in graphs]
    max_nodes = np.max(g_sizes)
    new_graphs = []
    new_dummy_index = []

    # Add dummy even in large graphs
    if max_nodes < rank:
        max_nodes = rank

    for idx_g in range(len(g_sizes)):
        dummy_index_node = []

        gr = graphs[idx_g].copy()
        if g_sizes[idx_g] < max_nodes:
            for idn in range(max_nodes - g_sizes[idx_g]):
                # Add dummy nodes
                gr.add_node(g_sizes[idx_g] + idn, coord=(np.ones((3,)) * 1e7))
                dummy_index_node.append(g_sizes[idx_g] + idn)

        new_graphs.append(gr)
        new_dummy_index.append(dummy_index_node)

    return new_graphs, new_dummy_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multiway KerGM matching example.")
    parser.add_argument("directory", help="The directory with the graphs", type=str)
    parser.add_argument(
        "--template",
        help="The file name extension (for filtering)",
        type=str,
        default=".gpickle",
    )
    parser.add_argument(
        "--sigma", help="The sigma parameter (nodes)", type=float, default=200.0
    )
    parser.add_argument(
        "--gamma", help="The gamma parameter (edges)", type=float, default=0.1
    )
    parser.add_argument(
        "--rff",
        help="The number of random Fourier features (edges)",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--rank",
        help="The maximal rank of the results",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--iterations",
        help="The maximal number of iterations",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--ground_truth",
        help="The ground truth (in a numpy file)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--method",
        help="The multi-graph matching method",
        default="new",
        type=str,
        choices=["mkergm", "hippi", "matcheig", "sqad"],
    )
    parser.add_argument(
        "--add_dummy", help="Add dummy nodes", action="store_true", default=False
    )
    args = parser.parse_args()

    all_graphs = []
    with os.scandir(args.directory) as files:
        # noinspection PyTypeChecker
        for file in files:
            if file.name.endswith(args.template):
                print(file.name)
                with open(args.directory + "/" + file.name, "rb") as f:
                    g = pickle.load(f)
                all_graphs.append(g)

    dummy_index = []
    if args.add_dummy:
        all_graphs, dummy_index = add_dummy_nodes(all_graphs, args.rank)

    sizes = [nx.number_of_nodes(g) for g in all_graphs]

    node_kernel = gaussian.create_gaussian_node_kernel(args.sigma, "coord")
    knode = kutils.create_full_node_affinity_matrix(all_graphs, node_kernel)

    if args.method == "hippi":
        a = utils.create_full_adjacency_matrix(all_graphs)
        u_nodes = hippi.hippi_multiway_matching(
            a, sizes, knode, args.rank, iterations=args.iterations
        )
        m_res = u_nodes @ u_nodes.T
    elif args.method == "matcheig":
        m_res = matcheig.matcheig(knode, args.rank, sizes)
    elif args.method == "sqad":
        u_nodes = stiefel.sparse_stiefel_manifold_sync(knode, args.rank, sizes)
        m_res = u_nodes @ u_nodes.T
    else:
        vectors, offsets = rff.create_random_vectors(1, args.rff, args.gamma)
        full_size = knode.shape[0]
        phi = np.zeros((args.rff, full_size, full_size))
        index = 0
        for i in range(len(all_graphs)):
            g_phi = rff.compute_phi(
                all_graphs[i], "geodesic_distance", vectors, offsets
            )
            phi[:, index : index + sizes[i], index : index + sizes[i]] = g_phi
            index += sizes[i]

        knode /= np.max(sizes)

        x_init = knode * 0.0
        gradient = mkergm.create_gradient(phi, knode)
        m_res = mkergm.mkergm(
            gradient,
            sizes,
            args.rank,
            iterations=args.iterations,
            init=x_init,
            projection_method="matcheig",
        )

    if args.ground_truth is not None:
        with open(args.ground_truth, "rb") as f:
            truth = pickle.load(f)
        a_truth = permutations.get_permutation_matrix_from_dictionary(truth, sizes)

        if dummy_index:
            index = 0
            for i_g in range(len(all_graphs)):
                for i_d in dummy_index[i_g]:
                    m_res[index + i_d, :] = 0
                    m_res[:, index + i_d] = 0

        f1_score, precision, recall = measures.compute_f1score(m_res, a_truth)
        print("Precision = {:.3f}".format(precision))
        print("Recall = {:.3f}".format(recall))
        print("F1-Score = {:.3f}".format(f1_score))
