"""This module contains code to generate graph family.

.. moduleauthor:: Marius Thorre, Rohit Yadav
"""

import networkx as nx

import graph_matching_tools.generators.noisy_graph as ng


def generation_graph_family(
    nb_sample_graphs: int,
    ref_graph: nx.Graph,
    kappa_noise_node: float = 1.0,
    outlier_mu: float = 12.0,
    outlier_sigma: float = 4.0,
    edge_delete_percent: float = 0.1,
    suppress_nodes: bool = True,
    add_outliers: bool = True,
) -> list[nx.graph]:
    """Generate noisy graphs from a reference graph.

    :param int nb_sample_graphs: the number of graphs to generate.
    :param nx.Graph ref_graph: the reference graph.
    :param float kappa_noise_node: the amount of noise to add on the nodes.
    :param float outlier_mu: the mean number of outliers.
    :param float outlier_sigma: the standard deviation of the outliers.
    :param float edge_delete_percent: the percent of removed edges for the edges.
    :param bool suppress_nodes: if True some nodes are suppressed.
    :param bool add_outliers: if True outlier nodes are added.
    :return: a list of graphs.
    :rtype: list[nx.Graph]
    """
    # Generate the reference graph
    reference_graph = ref_graph

    # Initialise the list of noisy_graphs
    list_noisy_graphs = []

    graph_index = 0
    while graph_index < nb_sample_graphs:

        noisy_graph = ng.noisy_graph_generation(
            reference_graph,
            kappa_noise_node,
            outlier_mu=outlier_mu,
            outlier_sigma=outlier_sigma,
            suppress_nodes=suppress_nodes,
            add_outliers=add_outliers,
            edge_delete_percent=edge_delete_percent,
        )

        if nx.is_connected(noisy_graph):
            list_noisy_graphs.append(noisy_graph)
            graph_index += 1

    return list_noisy_graphs
