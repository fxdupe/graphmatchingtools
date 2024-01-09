"""
Some plotting methods for illustration purpose.

.. moduleauthor:: François-Xavier Dupé
"""
import matplotlib.pyplot as plt
import networkx as nx


def draw_letter(graph: nx.Graph) -> None:
    """Draw a letter in blue.

    :param nx.Graph graph: the graph representing a letter
    """

    plt.title("Letter")
    for edge in graph.edges:
        x1 = graph.nodes[edge[0]]["x"][0]
        x2 = graph.nodes[edge[1]]["x"][0]
        y1 = graph.nodes[edge[0]]["x"][1]
        y2 = graph.nodes[edge[1]]["x"][1]
        plt.plot((x1, x2), (y1, y2), "bo-")
    plt.show()
