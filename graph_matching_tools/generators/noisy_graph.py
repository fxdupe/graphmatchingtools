"""This module contains code to generate noisy graph

.. moduleauthor:: Marius Thorre, Rohit Yadav, François-Xavier Dupé
"""

import random

from scipy.stats import betabinom
import networkx as nx
import numpy as np
import trimesh

import graph_matching_tools.generators.topology as topology
import graph_matching_tools.utils.sphere as sphere


def sphere_random_sampling(vertex_number: int = 100, radius: float = 1.0) -> np.ndarray:
    """Generate a sphere with random sampling.

    :param int vertex_number: the number of vertices.
    :param float radius: the radius of the sphere.
    :return: a sphere coordinate array.
    :rtype: np.ndarray
    """
    coords = np.zeros((vertex_number, 3))
    for i in range(vertex_number):
        m = np.random.normal(size=(3, 3))
        q, r = np.linalg.qr(m)
        coords[i, :] = q[:, 0].transpose() * np.sign(r[0, 0])
    coords = radius * coords
    return coords


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

    print(alpha, beta)
    nb_supress = betabinom.rvs(nb_vertices, alpha, beta, size=1)[0]
    nb_outliers = betabinom.rvs(nb_vertices, alpha, beta, size=1)[0]

    return int(nb_outliers), int(nb_supress)


def von_mises_sampling(
    nb_vertices: int, original_graph: nx.Graph, sigma_noise_nodes: float
) -> dict:
    """Perturbed the coordinates of a given graph.

    :param nb_vertices: the number of vertices.
    :param nx.Graph original_graph: the input unperturbed graph.
    :param float sigma_noise_nodes: the variance of the noise.
    :return: a dictionary with the noisy attributes for each node.
    :rtype: dict
    """
    noisy_coord = {}
    for index in range(nb_vertices):
        # Sampling from Von Mises - Fisher distribution
        original_coord = original_graph.nodes[index]["coord"]
        mean_original = original_coord / np.linalg.norm(
            original_coord
        )  # convert to unit vector
        noisy_coordinate = sphere.sample_sphere(
            1, mu=mean_original, kappa=sigma_noise_nodes
        ).sample[0]

        noisy_coordinate = noisy_coordinate * np.linalg.norm(
            original_coord
        )  # rescale to original size
        noisy_coord[index] = {
            "coord": noisy_coordinate,
            "label": index + 1,
            "is_outlier": False,
        }
    return noisy_coord


def noisy_graph_generation(
    original_graph: nx.Graph,
    nb_vertices: int,
    sigma_noise_nodes: float = 1.0,
    sigma_noise_edges: float = 1.0,
    radius: int = 100,
    label_outlier: int = 0,
    edge_delete_percent: float = 0.1,
) -> nx.Graph:
    """Generate a noisy version of a reference graph.

    :param nx.Graph original_graph: the reference graph.
    :param int nb_vertices: the number of vertices of the reference graph.
    :param float sigma_noise_nodes: the variance of the noise on the attributes of the nodes.
    :param float sigma_noise_edges: the variance of the noise on the attributes of the edges.
    :param int radius: the size the sphere used for the sampling.
    :param int label_outlier: the label of the outliers.
    :param float edge_delete_percent: the percent of removed edges for the edges.
    :return: the noisy graph.
    :rtype: nx.Graph
    """
    sample_nodes = von_mises_sampling(nb_vertices, original_graph, sigma_noise_nodes)
    nb_outliers, nb_supress = generate_outliers_numbers(nb_vertices)

    outliers = sphere_random_sampling(vertex_number=nb_outliers, radius=radius)
    for outlier in outliers:
        random_key = random.choice(list(sample_nodes.items()))[0]
        sample_nodes[random_key] = {
            "coord": outlier,
            "is_outlier": True,
            "label": label_outlier,
        }

    sample_nodes = dict(
        sorted(
            sample_nodes.items(),
            key=lambda item: (item[1]["label"] >= 0, item[1]["label"]),
        )
    )

    all_coord = np.array([node["coord"] for node in sample_nodes.values()])
    compute_noisy_edges = compute_hull_from_vertices(
        all_coord
    )  # take all perturbed coord and comp conv hull.
    adj_matrix = topology.adjacency_matrix(
        compute_noisy_edges
    )  # compute the new adjacency mat.

    noisy_graph = nx.from_numpy_array(adj_matrix.todense())
    nx.set_node_attributes(noisy_graph, sample_nodes)
    nx.set_edge_attributes(noisy_graph, 1.0, name="weight")

    edge_to_remove = edge_len_threshold(noisy_graph, edge_delete_percent)
    noisy_graph.remove_edges_from(edge_to_remove)
    noisy_graph.remove_edges_from(nx.selfloop_edges(noisy_graph))

    return noisy_graph
