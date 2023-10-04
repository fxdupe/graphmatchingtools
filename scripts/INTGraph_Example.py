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
import graph_matching_tools.algorithms.multiway.matchals as matchals
import graph_matching_tools.algorithms.multiway.mkergm as mkergm
import graph_matching_tools.algorithms.multiway.dist_mkergm as dist_mkergm
import graph_matching_tools.algorithms.multiway.stiefel as stiefel
import graph_matching_tools.algorithms.multiway.irgcl as irgcl
import graph_matching_tools.algorithms.multiway.kergm as multi_kergm
import graph_matching_tools.algorithms.pairwise.kergm as kergm
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
    parser = argparse.ArgumentParser(
        description="Multiway matching for INT graph example."
    )
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
        choices=[
            "dist_mkergm",
            "mkergm",
            "kergm",
            "hippi",
            "matcheig",
            "matchals",
            "gamgm",
            "sqad",
            "irgcl",
        ],
    )
    parser.add_argument(
        "--dist_batch_size",
        help="Size of a batch (number of graph)",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--dist_batch_number",
        help="Number of batches",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--mals_alpha",
        help="The alpha (rank-energy constraint) parameter (MatchALS)",
        default=50.0,
        type=float,
    )
    parser.add_argument(
        "--mals_beta",
        help="The beta (sparsity constraint) parameter (MatchALS)",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--nb_alphas",
        help="The size of the sampling on alpha parameters",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--entropy",
        help="The initial entropy for GA-MGM/KerGM",
        default=2.0,
        type=float,
    )
    parser.add_argument(
        "--add_dummy", help="Add dummy nodes", action="store_true", default=False
    )
    parser.add_argument(
        "--load_res",
        help="Load the result from a given file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--save_res",
        help="Save the result into a given file",
        default=None,
        type=str,
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
    print(sizes)

    m_res = None
    if args.load_res is not None:
        m_res = np.load(args.load_res)

    node_kernel = gaussian.create_gaussian_node_kernel(args.sigma, "coord")
    knode = None
    if args.method != "gamgm":
        knode = kutils.create_full_node_affinity_matrix(all_graphs, node_kernel)

    if args.method == "hippi":
        if m_res is None:
            a = utils.create_full_adjacency_matrix(all_graphs)
        else:
            a = m_res
        u_nodes = hippi.hippi_multiway_matching(
            a, sizes, knode, args.rank, iterations=args.iterations
        )
        m_res = u_nodes @ u_nodes.T
    elif args.method == "matcheig":
        m_res = matcheig.matcheig(knode, args.rank, sizes)
    elif args.method == "matchals":
        m_res = matchals.matchals(
            knode,
            sizes,
            args.rank,
            alpha=args.mals_alpha,
            beta=args.mals_beta,
            iterations=args.iterations,
        )
    elif args.method == "sqad":
        if m_res is None:
            u_nodes = stiefel.sparse_stiefel_manifold_sync(knode, args.rank, sizes)
        else:
            u_nodes = stiefel.sparse_stiefel_manifold_sync(m_res, args.rank, sizes)
        m_res = u_nodes @ u_nodes.T
    elif args.method == "gamgm":
        # Here we assume that the graphs have the same number of nodes
        import pygmtools

        adj_graphs = np.zeros(
            (
                len(sizes),
                nx.number_of_nodes(all_graphs[0]),
                nx.number_of_nodes(all_graphs[0]),
            )
        )
        for i_g in range(len(all_graphs)):
            adj_graphs[i_g, :, :] = nx.to_numpy_array(all_graphs[i_g], weight=None)

        affinities = np.zeros(
            (
                len(sizes),
                len(sizes),
                nx.number_of_nodes(all_graphs[0]),
                nx.number_of_nodes(all_graphs[0]),
            )
        )
        for i_g1 in range(len(all_graphs)):
            for i_g2 in range(i_g1, len(all_graphs)):
                weights = kutils.compute_knode(
                    all_graphs[i_g1], all_graphs[i_g2], node_kernel
                )
                weights = kergm.sinkhorn_method(weights, gamma=3.0, iterations=20)
                affinities[i_g1, i_g2, :, :] = weights
                affinities[i_g2, i_g1, :, :] = np.squeeze(
                    affinities[i_g1, i_g2, :, :]
                ).transpose()

        res = pygmtools.multi_graph_solvers.gamgm(
            adj_graphs,
            affinities,
            n_univ=args.rank,
            sk_init_tau=2.0,
            sk_min_tau=0.1,
            sk_gamma=0.5,
            param_lambda=1.0,
            max_iter=50,
            verbose=True,
        )
        res = pygmtools.utils.MultiMatchingResult.to_numpy(res)
        m_res = np.zeros(
            (
                len(sizes) * nx.number_of_nodes(all_graphs[0]),
                len(sizes) * nx.number_of_nodes(all_graphs[0]),
            )
        )

        size1 = 0
        n_nodes = nx.number_of_nodes(all_graphs[0])
        for i_g1 in range(len(all_graphs)):
            size2 = size1
            for i_g2 in range(i_g1 + 1, len(all_graphs)):
                m_res[size1 : size1 + n_nodes, size2 : size2 + n_nodes] = res[
                    i_g1, i_g2
                ]
                m_res[size2 : size2 + n_nodes, size1 : size1 + n_nodes] = res[
                    i_g2, i_g1
                ]
                size2 += n_nodes
            size1 += n_nodes
        np.fill_diagonal(m_res, 1)
    elif args.method == "kergm":
        m_res = multi_kergm.multi_pairwise_kergm(
            all_graphs,
            sizes,
            knode,
            "geodesic_distance",
            args.gamma,
            args.entropy,
            args.nb_alphas,
            args.iterations,
            rff=args.rff,
            epsilon=1e-9,
        )
    elif args.method == "irgcl":
        if m_res is None:
            m_res = matcheig.matcheig(knode, args.rank, sizes)
        m_res = irgcl.irgcl(
            m_res,
            irgcl.default_beta_t,
            irgcl.default_alpha_t,
            irgcl.default_lambda_t,
            args.rank,
            len(sizes),
        )
        m_res = m_res @ m_res.T
    elif args.method == "dist_mkergm":
        vectors, offsets = rff.create_random_vectors(1, args.rff, args.gamma)
        d_phi = list()
        for i in range(len(all_graphs)):
            g_phi = rff.compute_phi(
                all_graphs[i], "geodesic_distance", vectors, offsets
            )
            d_phi.append(g_phi)

        d_knodes = dict()
        for i_g1 in range(len(all_graphs)):
            for i_g2 in range(i_g1 + 1, len(all_graphs)):
                d_knodes["{},{}".format(i_g1, i_g2)] = (
                    kutils.compute_knode(
                        all_graphs[i_g1], all_graphs[i_g2], node_kernel
                    )
                    * args.rank
                )

        d_perms = dist_mkergm.stochastic_dist_mkergm(
            all_graphs,
            d_knodes,
            d_phi,
            args.rank,
            args.dist_batch_number,
            args.dist_batch_size,
            args.iterations,
        )
        m_res = dist_mkergm.get_bulk_permutations_from_dict(d_perms, sizes)
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

        # knode /= np.max(sizes)
        knode *= args.rank

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

    if args.save_res is not None:
        np.save(args.save_res, m_res)

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
