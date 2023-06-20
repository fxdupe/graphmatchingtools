"""
Example of matching with self learning (for improving the features)

.. moduleauthor: François-Xavier Dupé
"""
import argparse

import numpy as np
import networkx as nx
import sklearn.decomposition as skld

import graph_matching_tools.algorithms.kernels.utils as kutils
import graph_matching_tools.algorithms.kernels.gaussian as gaussian
import graph_matching_tools.algorithms.kernels.rff as rff
import graph_matching_tools.algorithms.multiway.mkergm as mkergm
import graph_matching_tools.algorithms.multiway.matcheig as matcheig
import graph_matching_tools.utils.utils as utils
import graph_matching_tools.metrics.matching as measures
import graph_matching_tools.io.pygeo_graphs as pyg
import graph_matching_tools.utils.permutations as perm


def get_node_data_matrix(graph: nx.Graph, data_name: str):
    """Get the node data matrix from a current graph.

    :param nx.Graph graph: a graph.
    :param str data_name: the name of the data vector.
    :return:
    """
    data = np.zeros((nx.number_of_nodes(graph), len(graph.nodes[0][data_name])))
    for node in range(nx.number_of_nodes(graph)):
        data[node, :] = graph.nodes[node][data_name]
    return data


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
        "--iterations", help="The maximal number of iterations", type=int, default=20
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
    parser.add_argument(
        "--load_model", help="The learned model to load", default=None, type=str
    )
    parser.add_argument(
        "--model_name",
        help="The filename for the model weights",
        default=None,
        type=str,
    )
    args = parser.parse_args()

    # Take all the graphs for the learning step
    all_graphs = pyg.get_graph_database(
        args.database, args.isotropic, args.category, args.repo + "/" + args.database
    )

    print("Size dataset: {} graphs".format(len(all_graphs)))
    print([nx.number_of_nodes(g) for g in all_graphs])

    dummy_index = []
    graph_index = []

    data_matrices = [get_node_data_matrix(g, "x") for g in all_graphs]
    data_matrices = np.concatenate(data_matrices, axis=0)

    pca = skld.PCA(n_components=30)
    pca.fit(data_matrices)
    data_reduced = pca.transform(data_matrices)

    new_graphs = []
    for graph in all_graphs:
        new_graph = graph.copy()
        for i_n in range(nx.number_of_nodes(graph)):
            new_graph.nodes[i_n]["x"] = np.squeeze(
                pca.transform(np.array(graph.nodes[i_n]["x"])[np.newaxis, :])
            )
        new_graphs.append(new_graph)

    if args.add_dummy:
        new_graphs, graph_index, dummy_index = pyg.add_dummy_nodes(
            new_graphs, args.rank, dimension=30
        )

    if args.shuffle:
        new_graphs, graph_index = utils.randomize_nodes_position(new_graphs)

    if not args.add_dummy and not args.shuffle:
        graph_index = [list(range(nx.number_of_nodes(g))) for g in all_graphs]

    g_sizes = [nx.number_of_nodes(g) for g in new_graphs]
    full_size = int(np.sum(g_sizes))
    print("Full-size: {} nodes".format(full_size))

    node_kernel = gaussian.create_gaussian_node_kernel(
        args.sigma, "pos" if args.database == "PascalPF" else "x"
    )
    knode = kutils.create_full_node_affinity_matrix(new_graphs, node_kernel)

    # import matplotlib.pyplot as plt
    # plt.imshow(knode)
    # plt.colorbar()
    # plt.show()

    # Compute the big phi matrix
    vectors, offsets = rff.create_random_vectors(1, args.rff, args.gamma)
    phi = np.zeros((args.rff, full_size, full_size))
    index = 0
    for i in range(len(new_graphs)):
        g_phi = rff.compute_phi(new_graphs[i], "weight", vectors, offsets)
        phi[:, index : index + g_sizes[i], index : index + g_sizes[i]] = g_phi
        index += g_sizes[i]

    x_init = np.ones(knode.shape) / knode.shape[0]

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
    # m_res = matcheig.matcheig(knode, args.rank, g_sizes)

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
