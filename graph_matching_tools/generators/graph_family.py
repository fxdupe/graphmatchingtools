"""This module contains code to generate graph family.

.. moduleauthor:: Marius Thorre, Rohit Yadav
"""

import networkx as nx

import graph_matching_tools.generators.noisy_graph as ng


def generation_graph_family(
    nb_sample_graphs: int,
    nb_vertices: int,
    ref_graph: nx.Graph,
    noise_node: float = 1,
    noise_edge: float = 1,
) -> list[nx.graph]:
    """Generate n noisy graphs from a reference graph.

    :param int nb_sample_graphs: the number of graphs to generate.
    :param int nb_vertices: the number of vertices in the reference graph.
    :param nx.Graph ref_graph: the reference graph.
    :param float noise_node: the amount of noise to add on the nodes.
    :param float noise_edge:the amount of noise to add on the edges.
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
            reference_graph, nb_vertices, noise_node, noise_edge
        )

        if nx.is_connected(noisy_graph):
            list_noisy_graphs.append(noisy_graph)
            graph_index += 1

    return list_noisy_graphs
