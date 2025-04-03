"""This module contains code to generate noisy graph

.. moduleauthor:: Marius Thorre, Rohit Yadav, François-Xavier Dupé
"""

import random

from scipy.stats import betabinom
import networkx as nx
import numpy as np
import trimesh

import graph_matching_tools.generators.topology as topology
import graph_matching_tools.generators.reference_graph as reference_graph
import graph_matching_tools.utils.sphere as sphere


def compute_hull_from_vertices(vertices: np.ndarray) -> trimesh.Trimesh:
    """Compute faces from vertices using trimesh convex hull.

    :param np.ndarray vertices: a set of vertices.
    :return: the convex hull from the faces.
    :rtype: trimesh.Trimesh
    """
    return trimesh.Trimesh(vertices=vertices, process=False).convex_hull


def edge_len_threshold(graph: nx.Graph, percent: float) -> list:
    """Get percentage of edges at random.

    :param nx.Graph graph: the input graph.
    :param float percent: the percentage of new edges.
    :return: the set of selected edges.
    :rtype: list
    """
    return random.sample(list(graph.edges), round(len(graph.edges) * percent))


def compute_beta(alpha: float, n: int, mean: float) -> float:
    """Get the beta parameter of the beta-binomial distribution for a given mean and alpha.

    :param float alpha: the alpha parameter of the beta-binomial distribution.
    :param int n: the size of the support.
    :param float mean: the mean of the beta-binomial distribution.
    :return: the corresponding beta value.
    :rtype: float
    """
    return (1 - mean / n) / (mean / n) * alpha


def compute_alpha(n: int, mean: float, variance: float) -> float:
    """Get the alpha parameter of the beta-binomial distribution for a given mean and variance.

    :param int n: the size of the support.
    :param float mean: the expected mean.
    :param float variance: the expected variance.
    :return: the corresponding alpha value.
    :rtype: float
    """
    ratio = (1 - mean / n) / (mean / n)
    alpha = ((1 + ratio) ** 2 * variance - n**2 * ratio) / (
        n * ratio * (1 + ratio) - variance * (1 + ratio) ** 3
    )
    return alpha


def generate_outliers_numbers(
    nb_vertices: int = 25, mu: float = 10, sigma: float = 4.0
) -> tuple[int, int]:
    """Sample nb_outliers and nb_supress from a Normal distance following the std of real data.

    :param int nb_vertices: the number of vertices (default: 25).
    :param float mu: the mean of the distribution (default: 10).
    :param float sigma: the standard deviation of the distribution (default: 4.0).
    :return: Tuple which contains nb outliers and nb supress
    :rtype: tuple[int, int]
    """
    alpha = compute_alpha(
        nb_vertices, mu, sigma**2
    )  # corresponding alpha with respect to given mu and sigma
    beta = compute_beta(alpha, nb_vertices, mu)  # corresponding beta

    nb_suppress = betabinom.rvs(nb_vertices, alpha, beta, size=1)[0]
    nb_outliers = betabinom.rvs(nb_vertices, alpha, beta, size=1)[0]

    return int(nb_outliers), int(nb_suppress)


def von_mises_sampling(original_graph: nx.Graph, kappa_noise_nodes: float) -> dict:
    """Perturbed the coordinates of a given graph.

    :param nx.Graph original_graph: the input unperturbed graph.
    :param float kappa_noise_nodes: the variance of the noise.
    :return: a dictionary with the noisy attributes for each node.
    :rtype: dict
    """
    noisy_coord = {}
    for index in range(original_graph.number_of_nodes()):
        # Sampling from Von Mises - Fisher distribution
        original_coord = original_graph.nodes[index]["coord"]
        mean_original = original_coord / np.linalg.norm(
            original_coord
        )  # convert to unit vector
        noisy_coordinate = sphere.random_coordinate_sampling(
            1, mu=mean_original, kappa=kappa_noise_nodes
        )

        # rescale to original size
        noisy_coordinate = list(noisy_coordinate)
        for dim in range(len(noisy_coordinate)):
            noisy_coordinate[dim] = np.squeeze(noisy_coordinate[dim]) * np.linalg.norm(
                original_coord
            )

        noisy_coord[index] = {
            "coord": np.array(noisy_coordinate),
            "label": index + 1,
            "is_outlier": False,
        }
    return noisy_coord


def noisy_graph_generation(
    original_graph: nx.Graph,
    kappa_noise_nodes: float = 1.0,
    radius: float = 1.0,
    label_outlier: int = -1,
    edge_delete_percent: float = 0.1,
    outlier_mu: float = 12.0,
    outlier_sigma: float = 4.0,
    suppress_nodes: bool = True,
    add_outliers: bool = True,
) -> nx.Graph:
    """Generate a noisy version of a reference graph.

    :param nx.Graph original_graph: the reference graph.
    :param float kappa_noise_nodes: the variance of the noise on the attributes of the nodes.
    :param float radius: the size the sphere used for the sampling.
    :param int label_outlier: the label of the outliers.
    :param float edge_delete_percent: the percent of removed edges for the edges.
    :param float outlier_mu: the mean number of outliers and suppressed nodes.
    :param float outlier_sigma: the standard deviation for the outlier generation.
    :param bool suppress_nodes: if True some nodes are suppressed.
    :param bool add_outliers: if True outlier nodes are added.
    :return: the noisy graph.
    :rtype: nx.Graph
    """
    noisy_coord_nodes = von_mises_sampling(original_graph, kappa_noise_nodes)
    nb_outliers, nb_suppress = generate_outliers_numbers(
        original_graph.number_of_nodes(), mu=outlier_mu, sigma=outlier_sigma
    )

    if suppress_nodes and nb_suppress > 0:
        suppress_list = random.sample(list(noisy_coord_nodes.keys()), nb_suppress)
        for i in suppress_list:
            del noisy_coord_nodes[i]

    if add_outliers and nb_outliers > 0:
        x_out, y_out, z_out = sphere.random_sampling(
            vertex_number=nb_outliers, radius=radius
        )
        max_key = original_graph.number_of_nodes() + 1
        for outlier in range(nb_outliers):
            noisy_coord_nodes[max_key + outlier] = {
                "coord": np.array([x_out[outlier], y_out[outlier], z_out[outlier]]),
                "label": label_outlier,
                "is_outlier": True,
            }

    sorted_nodes = sorted(
        noisy_coord_nodes.items(),
        key=lambda element: (element[1]["label"] < 0, element[1]["label"]),
    )
    noisy_nodes = {}
    for item in range(len(sorted_nodes)):
        noisy_nodes[item] = sorted_nodes[item][1]

    all_coord = np.array([node["coord"] for node in noisy_nodes.values()])
    compute_noisy_edges = compute_hull_from_vertices(
        all_coord
    )  # take all perturbed coord and comp conv hull.
    adj_matrix = topology.adjacency_matrix(
        compute_noisy_edges
    )  # compute the new adjacency mat.

    noisy_graph = nx.from_numpy_array(adj_matrix.todense())
    nx.set_node_attributes(noisy_graph, noisy_nodes)

    edge_to_remove = edge_len_threshold(noisy_graph, edge_delete_percent)
    noisy_graph.remove_edges_from(edge_to_remove)
    noisy_graph.remove_edges_from(nx.selfloop_edges(noisy_graph))
    noisy_graph = reference_graph.compute_edges_attributes(noisy_graph, radius)

    return noisy_graph
