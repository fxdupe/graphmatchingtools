"""
Set of functions for Pytorch-Geometrics datasets

.. moduleauthor:: François-Xavier Dupé
"""
import numpy as np
import networkx as nx
from torch_geometric.datasets import WILLOWObjectClass
from torch_geometric.datasets import PascalVOCKeypoints as PascalVOC
from torch_geometric.datasets import PascalPF
import torch_geometric.transforms as transforms
import torch_geometric.utils as tf_utils


def generate_groundtruth(
    graph_sizes: list[int],
    nb_global_nodes: int,
    indexes: list[list[int]],
) -> np.ndarray:
    """Generate groundtruth for the matching.

    :param list[int] graph_sizes: the list of the graph sizes.
    :param int nb_global_nodes: the global number of nodes.
    :param list[list[int]] indexes: the new indexes.
    :return: the correspondence map for each node.
    :rtype: np.ndarray
    """
    res = np.zeros((2, nb_global_nodes), dtype="i")
    res[0, :] = np.arange(nb_global_nodes)

    idx = 0
    for idx_g in range(len(graph_sizes)):
        res[1, idx : idx + graph_sizes[idx_g]] = indexes[idx_g]
        idx += graph_sizes[idx_g]

    return res


def _convert_to_networkx(dataset) -> list[nx.Graph]:  # pragma: no cover
    """Conversion of the pytorch data to networkx graphs.

    :param dataset: the torch geometric dataset.
    :return: the converted graphs.
    :rtype: list[nx.Graph]
    """
    graphs = []
    for idx in range(len(dataset)):
        g = tf_utils.to_networkx(
            dataset[idx], node_attrs=["pos", "x"], to_undirected=True
        )
        graphs.append(g)
    return graphs


def get_graph_database(
    name: str, isotropic: bool, category: str, repo: str
) -> list[nx.Graph]:  # pragma: no cover
    """Get a given graph dataset

    :param str name: the name of the database to load (PascalVOC, PascalPF or Willow [default]).
    :param bool isotropic: get isotropic graphs.
    :param str category: the category of images.
    :param str repo: the repo for graph (download etc.).
    :return: The graphs of keypoint from the image category.
    :rtype: list[nx.Graph]
    """
    transform = transforms.Compose(
        [
            transforms.Delaunay(),
            transforms.FaceToEdge(),
            transforms.Distance() if isotropic else transforms.Cartesian(),
        ]
    )

    if name == "PascalVOC":
        pre_filter = lambda data: data.pos.size(0) > 0  # noqa
        dataset = PascalVOC(
            repo, category, train=False, transform=transform, pre_filter=pre_filter
        )
    elif name == "PascalPF":
        transform = transforms.Compose(
            [
                transforms.Constant(),
                transforms.KNNGraph(k=8),
                transforms.Cartesian(),
            ]
        )
        dataset = PascalPF(repo, category, transform=transform)
    else:
        dataset = WILLOWObjectClass(repo, category=category, transform=transform)

    graphs = _convert_to_networkx(dataset)
    for idx in range(len(graphs)):
        graphs[idx] = compute_edges_data(graphs[idx])
    return graphs


def compute_edges_data(graph: nx.Graph, mu: float = 10.0) -> nx.Graph:
    """Compute the distance between the nodes (using Euclidean distance)

    :param nx.Graph graph: the input graph.
    :param float mu: the weights scaling factor (default: 1.0).
    :return: the new graph with the distance on the edges.
    :rtype: nx.Graph
    """
    distances = np.zeros((nx.number_of_nodes(graph),)) + 10**9
    for u, v in graph.edges:
        d = np.linalg.norm(
            np.array(graph.nodes[u]["pos"]) - np.array(graph.nodes[v]["pos"])
        )
        graph.edges[u, v]["distance"] = d
        if distances[u] > d:
            distances[u] = d
        if distances[v] > d:
            distances[v] = d
    median = np.median(distances)

    for u, v in graph.edges:
        graph.edges[u, v]["norm_dist"] = graph.edges[u, v]["distance"] / median
        graph.edges[u, v]["weight"] = np.exp(
            -(graph.edges[u, v]["distance"] ** 2) / (2.0 * median**2 * mu)
        )

    return graph
