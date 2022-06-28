"""Example with the Pytorch Geometric package

.. moduleauthor:: François-Xavier Dupé
"""
import argparse
import sys
import random

import numpy as np
from torch_geometric.datasets import WILLOWObjectClass
from torch_geometric.datasets import PascalVOCKeypoints as PascalVOC
from torch_geometric.datasets import PascalPF
import torch_geometric.transforms as transforms
import torch_geometric.utils as tf_utils
import networkx as nx

import graph_matching_tools.algorithms.kernels.gaussian as gaussian
import graph_matching_tools.algorithms.kernels.utils as ku
import graph_matching_tools.utils.utils as utils
import graph_matching_tools.metrics.matching as measures
import graph_matching_tools.algorithms.kernels.rff as rff
import graph_matching_tools.algorithms.multiway.hippi as hippi
import graph_matching_tools.algorithms.multiway.msync as msync
import graph_matching_tools.algorithms.multiway.matcheig as matcheig
import graph_matching_tools.algorithms.multiway.stiefel as stiefel
import graph_matching_tools.algorithms.multiway.quickmatch as quickmatch
import graph_matching_tools.algorithms.multiway.kergm as kergm
import graph_matching_tools.algorithms.multiway.mkergm as mkergm
import graph_matching_tools.algorithms.multiway.irgcl as irgcl


def add_dummy_nodes(graphs, rank, dim=1024):
    """
    Add dummy nodes to graph to uniform the sizes
    :param list[nx.Graph] graphs: the list of graphs
    :param int rank: the rank of the universe of nodes
    :return: the new list of graph with the new matching index
    """
    sizes = [nx.number_of_nodes(g) for g in graphs]
    max_nodes = np.max(sizes)
    new_graphs = []
    new_index = []
    new_dummy_index = []

    # Add dummy even in large graphs
    if max_nodes < rank:
        max_nodes = rank

    for idx_g in range(len(sizes)):
        match_index_node = list(range(sizes[idx_g]))
        dummy_index_node = []

        g = graphs[idx_g].copy()
        if sizes[idx_g] < max_nodes:
            index_n = 0
            for idn in range(max_nodes - sizes[idx_g]):
                # Add dummy nodes
                g.add_node(sizes[idx_g] + idn, x=(np.zeros((dim, )) + (idn + 1) * 1e8),
                           pos=(-(idn+1)*1e3, -(idn+1)*1e3))
                match_index_node.append(sizes[idx_g] + idn)
                dummy_index_node.append(sizes[idx_g] + idn)

        new_graphs.append(g)
        new_index.append(match_index_node)
        new_dummy_index.append(dummy_index_node)

    return new_graphs, new_index, new_dummy_index


def generate_willow_groundtruth(nb_nodes, nb_graphs):
    """
    Generate groundtruth for the matching
    :param nb_nodes: the number of nodes for each graph
    :param nb_graphs: the number of graphs
    :return: the correspondence map for each node
    """
    row = np.arange(nb_nodes * nb_graphs)
    col = row[:nb_nodes].reshape((1, -1)).repeat(nb_graphs, 0).reshape((-1,))
    return np.stack([row, col], axis=0)


def generate_pascal_groundtruth(graph_sizes, nb_global_nodes, nb_graphs):
    """
    Generate groundtruth for the matching
    :param graph_sizes: the list of the graph sizes
    :param nb_global_nodes: the global number of nodes
    :param nb_graphs: the number of graphs
    :return: the correspondence map for each node
    """
    res = np.zeros((2, nb_global_nodes), dtype="i")
    res[0, :] = np.arange(nb_global_nodes)

    idx = 0
    for size in graph_sizes:
        res[1, idx:idx+size] = np.arange(size)
        idx += size

    return res


def generate_groundtruth(graph_sizes, nb_global_nodes, nb_graphs, indexes):
    """
    Generate groundtruth for the matching
    :param list[int] graph_sizes: the list of the graph sizes
    :param int nb_global_nodes: the global number of nodes
    :param int nb_graphs: the number of graphs
    :param list[list] indexes: the new indexes
    :return: the correspondence map for each node
    """
    res = np.zeros((2, nb_global_nodes), dtype="i")
    res[0, :] = np.arange(nb_global_nodes)

    idx = 0
    for idx_g in range(len(graph_sizes)):
        res[1, idx:idx+graph_sizes[idx_g]] = indexes[idx_g]
        idx += graph_sizes[idx_g]

    return res


def convert_to_networkx(dataset):
    """
    Conversion of the pytorch data to networkx graphs
    :param dataset: the torch geometric dataset
    :return: the converted graphs
    """
    graphs = []
    for idx in range(len(dataset)):
        g = tf_utils.to_networkx(dataset[idx], node_attrs=["pos", "x"], to_undirected=True)
        graphs.append(g)
    return graphs


def get_graph_database(name, isotropic, category, repo):
    """
    Get the Pascal-VOC dataset
    :param str name: the name of the database to load
    :param bool isotropic: get isotropic graphs
    :param str category: the category of images
    :param str repo: the repo for graph (download etc)
    :return: The graphs of keypoint from the image category
    """
    transform = transforms.Compose([
        transforms.Delaunay(),
        transforms.FaceToEdge(),
        transforms.Distance() if isotropic else transforms.Cartesian(),
    ])

    if name == "PascalVOC":
        pre_filter = lambda data: data.pos.size(0) > 0  # noqa
        dataset = PascalVOC(repo,
                            category,
                            train=False,
                            transform=transform,
                            pre_filter=pre_filter)
    elif name == "PascalPF":
        transform = transforms.Compose([
            transforms.Constant(),
            transforms.KNNGraph(k=8),
            transforms.Cartesian(),
        ])
        dataset = PascalPF(repo, category, transform=transform)
    else:
        dataset = WILLOWObjectClass(repo, category=category, transform=transform)

    graphs = convert_to_networkx(dataset)
    for idx in range(len(graphs)):
        graphs[idx] = compute_edges_data(graphs[idx])
    return graphs


def compute_edges_data(graph, mu=10.0, sigma=60.0):
    """
    Compute the distance between the nodes (using Euclidean distance)
    :param graph: the input graph
    :param float mu: the weights scaling factor (default: 1.0)
    :param float sigma: the variance of the keypoint distances
    :return: the new graph with the distance on the edges
    """
    distances = np.zeros((nx.number_of_nodes(graph), )) + 10**9
    for u, v in graph.edges:
        d = np.linalg.norm(np.array(graph.nodes[u]["pos"]) - np.array(graph.nodes[v]["pos"]))
        graph.edges[u, v]["distance"] = d
        if distances[u] > d:
            distances[u] = d
        if distances[v] > d:
            distances[v] = d
    median = np.median(distances)

    for u, v in graph.edges:
        graph.edges[u, v]["norm_dist"] = graph.edges[u, v]["distance"] / median
        graph.edges[u, v]["weight"] = np.exp(-(graph.edges[u, v]["distance"]**2) / (2.0 * median**2 * mu))
        # key_dist = np.linalg.norm(np.array(graph.nodes[u]["x"]) - np.array(graph.nodes[v]["x"]))
        # graph.edges[u, v]["key_weight"] = np.exp(-(key_dist ** 2) / (2.0 * sigma ** 2))

    return graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--isotropic', help="Build isotropic graphs", action="store_true", default=False)
    parser.add_argument("--sigma", help="The sigma parameter (nodes)", type=float, default=200.0)
    parser.add_argument("--gamma", help="The gamma parameter (edges)", type=float, default=0.1)
    parser.add_argument("--rff", help="The number of Random Fourier Features (edges)", type=int, default=100)
    parser.add_argument("--rank", help="The maximal rank of the results", type=int, default=10)
    parser.add_argument("--iterations", help="The maximal number of iterations", type=int, default=100)
    parser.add_argument("--tolerance", help="The tolerance for convergence", type=float, default=1e-3)
    parser.add_argument("--method", help="Select the method", type=str,
                        choices=["mkergm", "hippi", "mgm", "kergm", "quickm", "mals", "msync",
                                 "matcheig", "sqad"], default="mkergm")
    parser.add_argument("--database", help="The graph database", type=str,
                        choices=["Willow", "PascalVOC", "PascalPF"], default="Willow")
    parser.add_argument("--category", help="The category inside the database", type=str, default="car")
    parser.add_argument("--regularized", help="Regularized version", action="store_true", default=False)
    parser.add_argument("--nb_alphas", help="The size of the sampling on alpha parameters", type=int, default=5)
    parser.add_argument("--repo", help="The repository for downloaded data", type=str,
                        default="/home/fx/Projets/Data/Graphes/pytorch")
    parser.add_argument("--random", help="Do a random sampling on the of graphs", action="store_true", default=False)
    parser.add_argument("--random_number", help="Size of the random sampling", type=int, default=100)
    parser.add_argument("--init_method", help="The method for initialization", type=str, default="uniform")
    parser.add_argument("--mu_init", help="Multiplication on the initial step", type=float, default=0.2)
    parser.add_argument("--robust", help="Add robustness step", action="store_true", default=False)
    parser.add_argument("--robust_method", help="Select the robust projection method", type=str,
                        choices=["sqad", "irgcl"], default="irgcl")
    parser.add_argument("--shuffle", help="Shuffle the graphs", action="store_true", default=False)
    parser.add_argument("--entropy", help="The initial entropy for MGM/KerGM", default=2.0, type=float)
    parser.add_argument("--mgm_tau_min", help="The minimal value for the entropy for MGM", default=1e-2, type=float)
    parser.add_argument("--mals_alpha", help="The alpha (rank-energy constraint) parameter (MatchALS)",
                        default=50.0, type=float)
    parser.add_argument("--mals_beta", help="The beta (sparsity constraint) parameter (MatchALS)",
                        default=0.1, type=float)
    parser.add_argument("--quickm_dens", help="Density parameter (QuickMatch)", default=0.5, type=float)
    parser.add_argument("--quickm_dens_edge", help="Edge density parameter (QuickMatch)", default=0.5, type=float)
    parser.add_argument("--add_dummy", help="Add dummy nodes", action="store_true", default=False)
    parser.add_argument("--reference_graph", help="The number of the reference graph", default=0, type=int)
    parser.add_argument("--proj_method", help="Projection method", default="matcheig", type=str,
                        choices=["matcheig", "msync", "irgcl", "gpow"])
    args = parser.parse_args()

    all_graphs = get_graph_database(args.database, args.isotropic,
                                    args.category, args.repo + "/" + args.database)
    print("Size dataset: {} graphs".format(len(all_graphs)))
    print([nx.number_of_nodes(g) for g in all_graphs])

    if args.random:
        if args.random_number >= len(all_graphs):
            print("Please take a number below the size of the dataset")
            sys.exit(1)
        all_graphs = random.sample(all_graphs, args.random_number)

    dummy_index = []
    graph_index = []

    if args.add_dummy:
        dim = 2 if args.database == "PascalPF" else 1024
        all_graphs, graph_index, dummy_index = add_dummy_nodes(all_graphs, args.rank, dim=dim)

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

    if args.method != "quickm":
        node_kernel = gaussian.create_gaussian_node_kernel(args.sigma, "pos" if args.database == "PascalPF" else "x")
        knode = ku.create_full_node_affinity_matrix(all_graphs, node_kernel)

    if args.method == "hippi":
        a = utils.create_weighted_adjacency_matrix(all_graphs, full_size, "weight")
        u_nodes = hippi.hippi_multiway_matching(a, g_sizes, knode, args.rank, iterations=args.iterations,
                                                tolerance=args.tolerance)
        m_res = u_nodes @ u_nodes.T
    elif args.method == "msync":
        u_nodes = msync.msync(knode, g_sizes, args.rank, lambda x: args.reference_graph)
        m_res = u_nodes @ u_nodes.T
    elif args.method == "matcheig":
        m_res = matcheig.matcheig(knode, args.rank, g_sizes)
    elif args.method == "sqad":
        u_nodes = stiefel.sparse_stiefel_manifold_sync(knode, args.rank, g_sizes)
        m_res = u_nodes @ u_nodes.T
    elif args.method == "quickm":
        u_nodes = quickmatch.quickmatch(all_graphs, "pos" if args.database == "PascalPF" else "x",
                                        args.quickm_dens, args.quickm_dens_edge)
        print("Universe size = {}".format(u_nodes.shape[1]))
        m_res = u_nodes @ u_nodes.T
    elif args.method == "kergm":
        m_res = kergm.multi_pairwise_kergm(all_graphs, g_sizes, knode, "weight", args.gamma,
                                           args.entropy, args.nb_alphas,
                                           args.iterations, rff=args.rff)
    else:
        # Compute the big phi matrix
        vectors, offsets = rff.create_random_vectors(1, args.rff, args.gamma)
        phi = np.zeros((args.rff, full_size, full_size))
        index = 0
        for i in range(len(all_graphs)):
            g_phi = rff.compute_phi(all_graphs[i], "weight", vectors, offsets)
            phi[:, index:index + g_sizes[i], index:index + g_sizes[i]] = g_phi
            index += g_sizes[i]

        x_init = np.ones(knode.shape) / knode.shape[0]

        # Normalize for the tradeoff between nodes and edges
        norm_knode = np.median(g_sizes)
        knode /= norm_knode

        gradient = mkergm.create_fast_gradient(phi, knode)

        m_res = mkergm.mkergm(gradient, g_sizes, args.rank, iterations=args.iterations, init=x_init,
                              tolerance=args.tolerance, projection_method=args.proj_method,
                              choice=lambda x: args.reference_graph)

    # Compare with groundtruth
    truth = generate_groundtruth(g_sizes, full_size, len(g_sizes), graph_index)
    a_truth = utils.get_permutation_matrix_from_matching(truth, g_sizes, 50)

    if args.robust:
        if args.robust_method == "irgcl":
            p = irgcl.irgcl(m_res, irgcl.beta_t, irgcl.alpha_t, irgcl.lambda_t, args.rank, len(all_graphs),
                            choice=lambda x: args.reference_graph)
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
