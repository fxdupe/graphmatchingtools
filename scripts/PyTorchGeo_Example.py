"""Example with the Pytorch Geometric package

.. moduleauthor:: François-Xavier Dupé
"""
import argparse
import sys
import random

import numpy as np
import networkx as nx

import graph_matching_tools.algorithms.kernels.gaussian as gaussian
import graph_matching_tools.algorithms.kernels.utils as ku
import graph_matching_tools.utils.utils as utils
import graph_matching_tools.utils.permutations as perm
import graph_matching_tools.metrics.matching as measures
import graph_matching_tools.algorithms.kernels.rff as rff
import graph_matching_tools.algorithms.multiway.hippi as hippi
import graph_matching_tools.algorithms.multiway.msync as msync
import graph_matching_tools.algorithms.multiway.matcheig as matcheig
import graph_matching_tools.algorithms.multiway.stiefel as stiefel
import graph_matching_tools.algorithms.multiway.quickmatch as quickmatch
import graph_matching_tools.algorithms.multiway.kergm as kergm
import graph_matching_tools.algorithms.multiway.mkergm as mkergm
import graph_matching_tools.algorithms.multiway.dist_mkergm as dist_mkergm
import graph_matching_tools.algorithms.multiway.irgcl as irgcl
import graph_matching_tools.algorithms.multiway.ga_mgmc as ga_mgmc
import graph_matching_tools.algorithms.multiway.matchals as matchals
import graph_matching_tools.algorithms.multiway.boolean_nmf as nmf
import graph_matching_tools.algorithms.multiway.fmgm as fmgm
import graph_matching_tools.algorithms.multiway.mixer as mixer
import graph_matching_tools.io.pygeo_graphs as pyg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--isotropic", help="Build isotropic graphs", action="store_true", default=False
    )
    parser.add_argument(
        "--sigma", help="The sigma parameter (nodes)", type=float, default=200.0
    )
    parser.add_argument(
        "--gamma", help="The gamma parameter (edges)", type=float, default=0.1
    )
    parser.add_argument(
        "--rff",
        help="The number of Random Fourier Features (edges)",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--rank", help="The maximal rank of the results", type=int, default=10
    )
    parser.add_argument(
        "--iterations", help="The maximal number of iterations", type=int, default=100
    )
    parser.add_argument(
        "--tolerance", help="The tolerance for convergence", type=float, default=1e-3
    )
    parser.add_argument(
        "--method",
        help="Select the method",
        type=str,
        choices=[
            "mkergm",
            "hippi",
            "kergm",
            "quickm",
            "mals",
            "msync",
            "matcheig",
            "sqad",
            "matchals",
            "gamgmc",
            "dist_mkergm",
            "fmgm",
            "mixer",
        ],
        default="mkergm",
    )
    parser.add_argument(
        "--database",
        help="The graph database",
        type=str,
        choices=["Willow", "PascalVOC", "PascalPF"],
        default="Willow",
    )
    parser.add_argument(
        "--category", help="The category inside the database", type=str, default="car"
    )
    parser.add_argument(
        "--nb_alphas",
        help="The size of the sampling on alpha parameters",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--repo",
        help="The repository for downloaded data",
        type=str,
        default="/home/fx/Projets/Data/Graphes/pytorch",
    )
    parser.add_argument(
        "--random",
        help="Do a random sampling on the of graphs",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--random_number", help="Size of the random sampling", type=int, default=100
    )
    parser.add_argument(
        "--init_method",
        help="The method for initialization",
        type=str,
        default="uniform",
    )
    parser.add_argument(
        "--mu_init", help="Multiplication on the initial step", type=float, default=0.2
    )
    parser.add_argument(
        "--robust", help="Add robustness step", action="store_true", default=False
    )
    parser.add_argument(
        "--robust_method",
        help="Select the robust projection method",
        type=str,
        choices=["sqad", "irgcl", "nmf"],
        default="irgcl",
    )
    parser.add_argument(
        "--shuffle", help="Shuffle the graphs", action="store_true", default=False
    )
    parser.add_argument(
        "--entropy",
        help="The initial entropy for GA-MGM/KerGM",
        default=2.0,
        type=float,
    )
    parser.add_argument(
        "--gamgm_tau_min",
        help="The minimal value for the entropy for GA-MGM",
        default=1e-2,
        type=float,
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
        "--quickm_dens", help="Density parameter (QuickMatch)", default=0.5, type=float
    )
    parser.add_argument(
        "--quickm_dens_edge",
        help="Edge density parameter (QuickMatch)",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "--add_dummy", help="Add dummy nodes", action="store_true", default=False
    )
    parser.add_argument(
        "--reference_graph",
        help="The number of the reference graph",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--proj_method",
        help="Projection method",
        default="matcheig",
        type=str,
        choices=["matcheig", "msync", "irgcl", "gpow"],
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
    args = parser.parse_args()

    all_graphs = pyg.get_graph_database(
        args.database, args.isotropic, args.category, args.repo + "/" + args.database
    )

    print("Size dataset: {} graphs".format(len(all_graphs)))
    print([nx.number_of_nodes(g) for g in all_graphs])

    if args.random:
        if args.random_number >= len(all_graphs):
            print("Please take a number below the size of the dataset")
            sys.exit(1)
        all_graphs = random.sample(all_graphs, args.random_number)

        print("Size of random dataset: {} graphs".format(len(all_graphs)))
        print([nx.number_of_nodes(g) for g in all_graphs])

    all_graphs = [g for g in all_graphs]

    dummy_index = []
    graph_index = []

    if args.add_dummy:
        dim = 2 if args.database == "PascalPF" else 1024
        all_graphs, graph_index, dummy_index = pyg.add_dummy_nodes(
            all_graphs, args.rank, dimension=dim
        )

    if args.shuffle:
        all_graphs, graph_index = utils.randomize_nodes_position(all_graphs)

    if not args.add_dummy and not args.shuffle:
        graph_index = [list(range(nx.number_of_nodes(g))) for g in all_graphs]

    g_sizes = [nx.number_of_nodes(g) for g in all_graphs]
    full_size = int(np.sum(g_sizes))
    print("Full-size: {} nodes".format(full_size))

    # Compute node affinities
    node_kernel = None
    knode = None

    if (
        args.method != "quickm"
        and args.method != "dist_mkergm"
        and args.method != "fmgm"
    ):
        node_kernel = gaussian.create_gaussian_node_kernel(
            args.sigma, "pos" if args.database == "PascalPF" else "x"
        )
        knode = ku.create_full_node_affinity_matrix(all_graphs, node_kernel)

    if args.method == "hippi":
        a = utils.create_full_weight_matrix(all_graphs, "weight", sigma=args.gamma)
        u_nodes = hippi.hippi_multiway_matching(
            a,
            g_sizes,
            knode,
            args.rank,
            iterations=args.iterations,
            tolerance=args.tolerance,
        )
        m_res = u_nodes @ u_nodes.T
    elif args.method == "gamgmc":
        u_nodes = ga_mgmc.ga_mgmc(
            all_graphs,
            knode,
            args.rank,
            "weight",
            tau=args.entropy,
            tau_min=args.gamgm_tau_min,
            iterations=args.iterations,
        )
        m_res = u_nodes @ u_nodes.T
    elif args.method == "matchals":
        m_res = matchals.matchals(
            knode,
            g_sizes,
            args.rank,
            alpha=args.mals_alpha,
            beta=args.mals_beta,
            iterations=args.iterations,
        )
    elif args.method == "msync":
        u_nodes = msync.msync(knode, g_sizes, args.rank, args.reference_graph)
        m_res = u_nodes @ u_nodes.T
    elif args.method == "matcheig":
        m_res = matcheig.matcheig(knode, args.rank, g_sizes)
    elif args.method == "sqad":
        u_nodes = stiefel.sparse_stiefel_manifold_sync(knode, args.rank, g_sizes)
        m_res = u_nodes @ u_nodes.T
    elif args.method == "mixer":
        u_nodes = mixer.mixer(knode, g_sizes, args.mu_init, args.iterations)
        m_res = u_nodes @ u_nodes.T
    elif args.method == "quickm":
        u_nodes = quickmatch.quickmatch(
            all_graphs,
            "pos" if args.database == "PascalPF" else "x",
            args.quickm_dens,
            args.quickm_dens_edge,
        )
        print("Universe size = {}".format(u_nodes.shape[1]))
        m_res = u_nodes @ u_nodes.T
    elif args.method == "kergm":
        m_res = kergm.multi_pairwise_kergm(
            all_graphs,
            g_sizes,
            knode,
            "weight",
            args.gamma,
            args.entropy,
            args.nb_alphas,
            args.iterations,
            rff=args.rff,
        )
    elif args.method == "dist_mkergm":
        node_kernel = gaussian.create_gaussian_node_kernel(
            args.sigma, "pos" if args.database == "PascalPF" else "x"
        )

        vectors, offsets = rff.create_random_vectors(1, args.rff, args.gamma)
        d_phi = list()
        for i in range(len(all_graphs)):
            g_phi = rff.compute_phi(all_graphs[i], "weight", vectors, offsets)
            d_phi.append(g_phi)

        d_knodes = dict()
        for i_g1 in range(len(all_graphs)):
            for i_g2 in range(i_g1 + 1, len(all_graphs)):
                d_knodes["{},{}".format(i_g1, i_g2)] = (
                    ku.compute_knode(all_graphs[i_g1], all_graphs[i_g2], node_kernel)
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
        m_res = dist_mkergm.get_bulk_permutations_from_dict(d_perms, g_sizes)
    elif args.method == "fmgm":
        node_kernel = gaussian.create_gaussian_node_kernel(
            args.sigma, "pos" if args.database == "PascalPF" else "x"
        )

        def edge_kernel(g1, g2, e1, e2):
            w1 = g1.edges[e1[0], e1[1]]["weight"]
            w2 = g2.edges[e2[0], e2[1]]["weight"]
            return np.exp(-args.gamma * (np.abs(w1 - w2) ** 2.0))

        m_res = fmgm.factorized_multigraph_matching(
            all_graphs,
            args.reference_graph,
            node_kernel,
            edge_kernel,
            iterations=args.iterations,
            tolerance=args.tolerance,
        )
    else:
        # Compute the big phi matrix
        vectors, offsets = rff.create_random_vectors(1, args.rff, args.gamma)
        phi = np.zeros((args.rff, full_size, full_size))
        index = 0
        for i in range(len(all_graphs)):
            g_phi = rff.compute_phi(all_graphs[i], "weight", vectors, offsets)
            phi[:, index : index + g_sizes[i], index : index + g_sizes[i]] = g_phi
            index += g_sizes[i]

        x_init = np.ones(knode.shape) / knode.shape[0]

        # Normalize for the tradeoff between nodes and edges
        # norm_knode = np.median(g_sizes)
        # knode /= norm_knode
        knode *= args.rank

        gradient = mkergm.create_gradient(phi, knode)

        m_res = mkergm.mkergm(
            gradient,
            g_sizes,
            args.rank,
            iterations=args.iterations,
            init=x_init,
            tolerance=args.tolerance,
            projection_method=args.proj_method,
            choice=args.reference_graph,
        )

    # Compare with groundtruth
    truth = pyg.generate_groundtruth(g_sizes, full_size, graph_index)
    a_truth = perm.get_permutation_matrix_from_matching(truth, g_sizes)

    if args.robust:
        if args.robust_method == "irgcl":
            p = irgcl.irgcl(
                m_res,
                irgcl.default_beta_t,
                irgcl.default_alpha_t,
                irgcl.default_lambda_t,
                args.rank,
                len(all_graphs),
                choice=args.reference_graph,
            )
        elif args.robust_method == "nmf":
            p = nmf.boolean_nmf(m_res, args.rank, 1)
        else:
            p = stiefel.sparse_stiefel_manifold_sync(m_res, args.rank, g_sizes)
        m_res = p @ p.T

    # Remove dummy nodes before computing scores
    if dummy_index:
        index = 0
        for i_g in range(len(all_graphs)):
            for i_d in dummy_index[i_g]:
                t_i_d = graph_index[i_g][i_d]
                m_res[index + t_i_d, :] = 0
                m_res[:, index + t_i_d] = 0
                a_truth[:, index + t_i_d] = 0
                a_truth[index + t_i_d, :] = 0
            index += g_sizes[i_g]

    f1_score, precision, recall = measures.compute_f1score(m_res, a_truth)
    print("Precision = {:.3f}".format(precision))
    print("Recall = {:.3f}".format(recall))
    print("F1-Score = {:.3f}".format(f1_score))
