"""
Example of matching with self learning (for improving the features)

.. moduleauthor: François-Xavier Dupé
"""
import argparse

import numpy as np
import networkx as nx

import graph_matching_tools.algorithms.kernels.utils as kutils
import graph_matching_tools.algorithms.kernels.gaussian as gaussian
import graph_matching_tools.algorithms.kernels.rff as rff
import graph_matching_tools.algorithms.multiway.mkergm as mkergm
import graph_matching_tools.algorithms.pairwise.kergm as kergm
import graph_matching_tools.utils.utils as utils
import graph_matching_tools.metrics.matching as measures
import graph_matching_tools.io.pygeo_graphs as pyg
import graph_matching_tools.utils.permutations as perm


def normalized_softperm_matrix(x, sizes, entropy=1.0):
    """

    :param x:
    :param sizes:
    :param entropy:
    :return:
    """

    res = x * 0.0
    i_index = 0  # The global index
    for i_x in range(len(sizes)):
        j_index = i_index
        for i_y in range(i_x, len(sizes)):
            x_ij = x[i_index : i_index + sizes[i_x], j_index : j_index + sizes[i_y]]
            x_ij = kergm.sinkhorn_method(
                x_ij, gamma=entropy
            )  # , mu_s=np.ones((x_ij.shape[0])),
            # mu_t=np.ones((x_ij.shape[1])))
            res[i_index : i_index + sizes[i_x], j_index : j_index + sizes[i_y]] = x_ij
            res[j_index : j_index + sizes[i_y], i_index : i_index + sizes[i_x]] = x_ij.T
            j_index += sizes[i_y]
        i_index += sizes[i_x]

    return res


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
        "--node_factor",
        help="The multiplication factor for node affinity.",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--tolerance", help="The tolerance for convergence", type=float, default=1e-3
    )
    parser.add_argument(
        "--iterations", help="The maximal number of iterations", type=int, default=100
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
        "--shuffle", help="Shuffle the graphs", action="store_true", default=False
    )
    parser.add_argument(
        "--entropy",
        help="The entropy for S-T method",
        default=2.0,
        type=float,
    )
    parser.add_argument(
        "--add_dummy", help="Add dummy nodes", action="store_true", default=False
    )
    parser.add_argument(
        "--proj_method",
        help="Projection method",
        default="matcheig",
        type=str,
        choices=["matcheig", "msync", "irgcl", "gpow"],
    )
    args = parser.parse_args()

    all_graphs = pyg.get_graph_database(
        args.database, args.isotropic, args.category, args.repo + "/" + args.database
    )

    print("Size dataset: {} graphs".format(len(all_graphs)))
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

    node_kernel = gaussian.create_gaussian_node_kernel(
        args.sigma, "pos" if args.database == "PascalPF" else "x"
    )
    knode = kutils.create_full_node_affinity_matrix(all_graphs, node_kernel)

    # Compute the big phi matrix
    vectors, offsets = rff.create_random_vectors(1, args.rff, args.gamma)
    phi = np.zeros((args.rff, full_size, full_size))
    index = 0
    for i in range(len(all_graphs)):
        g_phi = rff.compute_phi(all_graphs[i], "weight", vectors, offsets)
        phi[:, index : index + g_sizes[i], index : index + g_sizes[i]] = g_phi
        index += g_sizes[i]

    x_init = np.ones(knode.shape) / knode.shape[0]
    # x_init = normalized_softperm_matrix(np.ones(knode.shape), g_sizes)

    # Set the tradeoff between nodes and edges
    if args.node_factor is None:
        knode *= args.rank
    else:
        knode *= args.node_factor

    gradient = mkergm.create_gradient(phi, knode)

    m_res = mkergm.mkergm(
        gradient,
        g_sizes,
        args.rank,
        iterations=args.iterations,
        init=x_init,
        tolerance=args.tolerance,
        projection_method=args.proj_method,
    )

    grad = gradient(m_res)

    import matplotlib.pyplot as plt

    plt.imshow(-grad)
    plt.colorbar()
    plt.show()

    # m_res[-grad < 40] = 0.0

    # Compare with groundtruth
    truth = pyg.generate_groundtruth(g_sizes, full_size, graph_index)
    a_truth = perm.get_permutation_matrix_from_matching(truth, g_sizes)

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
