"""Example of using graph generation tools.

.. moduleauthor:: François-Xavier Dupé
"""

import networkx as nx
import matplotlib.pyplot as plt

import graph_matching_tools.generators.graph_family as graph_family
import graph_matching_tools.generators.reference_graph as reference_graph


def graph_generations():
    graph_reference = reference_graph.generate_reference_graph(100, 1.0)
    list_noisy_graph = graph_family.generation_graph_family(
        10, 100, graph_reference, 1.0, 10.0
    )
    return graph_reference, list_noisy_graph


if __name__ == "__main__":
    ref_graph, noisy_graphs = graph_generations()
    plt.figure(1)
    nx.draw(ref_graph, with_labels=True)
    plt.figure(2)
    nx.draw(noisy_graphs[7], with_labels=True)
    plt.show()

    for node, node_data in noisy_graphs[1].nodes.items():
        print(node_data["coord"])
        print(node_data["label"])
        print(node_data["is_outlier"])

    for edge, edge_data in noisy_graphs[1].edges.items():
        print(edge_data["geodesic_distance"])
        print(edge_data["id"])
