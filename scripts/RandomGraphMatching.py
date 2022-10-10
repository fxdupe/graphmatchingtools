"""
Example of matching with random graphs

.. moduleauthor: François-Xavier Dupé
"""
import argparse
import copy

import numpy as np
import networkx as nx

import graph_matching_tools.algorithms.multiway.mkergm as matching
import graph_matching_tools.algorithms.multiway.matcheig as matcheig
import graph_matching_tools.algorithms.kernels.linear as lin_kern
import graph_matching_tools.utils.permutations as permut
import graph_matching_tools.algorithms.kernels.utils as kutils
import graph_matching_tools.utils.utils as utils
import graph_matching_tools.metrics.matching as metrics
import graph_matching_tools.io.pygeo_graphs as pyg


def create_random_graph_method(size, node_number, edge_proba, shuffle=False,
                               add_noise=False, node_noise_variance=1.0,
                               edge_noise_variance=1.0, node_data_dim=1, edge_data_dim=1,
                               remove_nodes=False, max_node_removed=5):
    """
    Use Erdos-Renyi way of generating graph
    :param size: the number of graphs
    :param node_number: the number of nodes
    :param edge_proba: the proba for edges
    :param shuffle: if True shuffle the nodes
    :param add_noise: if True add noise on data
    :param node_noise_variance: the variance of the noise on node data
    :param edge_noise_variance: the variance of the noise on edge data
    :param node_data_dim: the dimension of the data of nodes
    :param edge_data_dim: the dimension of the data of edges
    :param remove_nodes: True to remove nodes at random (they are transformed into dummy nodes)
    :param max_node_removed: the maximal number of removed nodes
    :return: a random graph
    """
    graphs = []
    graph1 = nx.gnp_random_graph(node_number, edge_proba)
    print(graph1)

    for node in graph1:
        graph1.nodes[node]["weight"] = np.random.uniform(0, 1, size=(node_data_dim,))

    for u, v in graph1.edges:
        graph1.edges[u, v]["weight"] = np.random.uniform(0, 1, size=(edge_data_dim,))

    graphs.append(graph1)
    for idx in range(size - 1):
        graphs.append(copy.deepcopy(graph1))

    if add_noise:
        for graph in graphs:
            for node in graph:
                graph.nodes[node]["weight"] += np.random.randn(node_data_dim) * node_noise_variance
            for u, v in graph.edges:
                graph.edges[u, v]["weight"] += np.random.randn(edge_data_dim) * edge_noise_variance

    if shuffle:
        e_graphs, e_index = permut.randomize_nodes_position(graphs[1:])
        graphs = [graphs[0], ] + e_graphs
        g_index = [list(range(node_number)), ] + e_index
    else:
        g_index = [list(range(node_number)), ]
        for i in range(size - 1):
            g_index.append(list(range(node_number)))

    dummy_index = []
    if remove_nodes:
        for i_g in range(len(graphs)):
            dummies = []
            nb_removed = np.random.randint(0, np.minimum(max_node_removed+1, nx.number_of_nodes(graphs[i_g])))
            idx = np.random.choice(graphs[i_g].nodes, nb_removed, replace=False)
            # Transform the nodes into dummy nodes
            for i_n in idx:
                dummies.append(i_n)
                graphs[i_g].nodes[i_n]["weight"] = np.zeros((node_data_dim, 1)) + 1e9  # Change to impossible values
                neigs = list(nx.neighbors(graphs[i_g], i_n))
                for neig in neigs:
                    graphs[i_g].remove_edge(i_n, neig)
            dummy_index.append(dummies)

    return graphs, g_index, dummy_index


def run_graph_generation(args):
    """
    Run everything
    :param args: the arguments from the command line
    :return: the f1-score
    """
    all_graphs, all_index, dummy_index = create_random_graph_method(args.number_of_graphs, args.node_number, 0.05,
                                                                    shuffle=True,
                                                                    add_noise=args.add_noise,
                                                                    node_noise_variance=args.node_noise,
                                                                    node_data_dim=args.data_dimension,
                                                                    edge_noise_variance=args.edge_noise,
                                                                    edge_data_dim=args.data_dimension,
                                                                    remove_nodes=args.remove_nodes,
                                                                    max_node_removed=args.max_node_removed
                                                                    )

    g_sizes = [nx.number_of_nodes(g) for g in all_graphs]
    full_size = int(np.sum(g_sizes))

    phi = np.zeros((args.data_dimension, full_size, full_size))
    index = 0
    for i in range(len(all_graphs)):
        g_phi = np.zeros((args.data_dimension, args.node_number, args.node_number))
        for u, v, data in all_graphs[i].edges.data("weight"):
            g_phi[:, u, v] = data
            g_phi[:, v, u] = data
        phi[:, index:index + g_sizes[i], index:index + g_sizes[i]] = g_phi
        index += g_sizes[i]

    node_kernel = lin_kern.create_linear_node_kernel("weight")
    knode = kutils.create_full_node_affinity_matrix(all_graphs, node_kernel) / args.node_number

    truth = pyg.generate_groundtruth(g_sizes, full_size, len(g_sizes), all_index)
    a_truth = utils.get_permutation_matrix_from_matching(truth, g_sizes, 5000)

    x_init = knode * 0.0
    gradient = matching.create_gradient(phi, knode)
    m_res = matching.mkergm(gradient, g_sizes, args.rank,
                            iterations=args.iterations, init=x_init,
                            projection_method=args.proj_method)

    m_res_eig = matcheig.matcheig(knode, args.rank, g_sizes)

    if dummy_index:
        index = 0
        for i_g in range(len(all_graphs)):
            for i_d in dummy_index[i_g]:
                t_i_d = all_index[i_g][i_d]
                m_res[index + t_i_d, :] = 0
                m_res[:, index + t_i_d] = 0
                m_res_eig[index + t_i_d, :] = 0
                m_res_eig[:, index + t_i_d] = 0
                a_truth[:, index + t_i_d] = 0
                a_truth[index + t_i_d, :] = 0
            index += g_sizes[i_g]

    f1_score_new, _, _ = metrics.compute_f1score(m_res, a_truth)
    f1_score_eig, _, _ = metrics.compute_f1score(m_res_eig, a_truth)

    return f1_score_new, f1_score_eig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multiway random graph testing.')
    parser.add_argument("--rank", help="The maximal rank of the results", type=int, default=20)
    parser.add_argument("--iterations", help="The maximal number of iterations", type=int, default=100)
    parser.add_argument("--number_of_runs", help="The number of runs", type=int, default=20)
    parser.add_argument("--number_of_graphs", help="The number of graphs", type=int, default=10)
    parser.add_argument("--node_number", help="The number of nodes", type=int, default=20)
    parser.add_argument("--data_dimension", help="The dimension of the attributes", type=int, default=10)
    parser.add_argument("--remove_nodes", help="Remove node at random", action="store_true", default=False)
    parser.add_argument("--max_node_removed", help="The maximal number of node removed", type=int, default=5)
    parser.add_argument("--add_noise", help="Add noise to the data on nodes and edges", action="store_true",
                        default=False)
    parser.add_argument("--node_noise", help="The variance of the noise on the node data", type=float, default=1.0)
    parser.add_argument("--edge_noise", help="The variance of the noise on the node data", type=float, default=1.0)
    parser.add_argument("--method", help="The method used for solving the optimization problem",
                        type=str, default="new")
    parser.add_argument("--proj_method", help="The projection method onto the permutation set", type=str,
                        default="matcheig")
    p_args = parser.parse_args()

    scores = []
    scores_eig = []
    for iteration in range(p_args.number_of_runs):
        score, score_eig = run_graph_generation(p_args)
        scores.append(score)
        scores_eig.append(score_eig)

    print("F1-Score (Ours) = {:.3f}".format(np.mean(scores)))
    print("F1-Score (MathEIG) = {:.3f}".format(np.mean(scores_eig)))
