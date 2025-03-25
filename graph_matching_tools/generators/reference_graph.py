"""This module contains code to generate reference graph.

.. moduleauthor:: Marius Thorre, Sylvain Takerkart, François-Xavier Dupé
"""

import numpy as np
import networkx as nx

import graph_matching_tools.generators.noisy_graph as ng
import graph_matching_tools.generators.topology as topology


def get_geodesic_distance_sphere(
    coord_a: np.ndarray, coord_b: np.ndarray, radius: float
) -> float:
    """Compute the geodesic distance of two 3D vectors on a sphere

    :param np.ndarray coord_a: first vector.
    :param np.ndarray coord_b: second vector.
    :param float radius: radius of the sphere.
    :return: geodesic distance.
    :rtype: float
    """
    return radius * np.arccos(
        np.clip(np.dot(coord_a, coord_b) / np.power(radius, 2), -1, 1)
    )


def fibonacci(
    nb_point: int, radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate random points on the sphere.

    :param int nb_point: the number of points to generate.
    :param float radius: the radius of the sphere.
    :return: the generated points.
    :rtype: tuple[np.ndarray]
    """
    inc = np.pi * (3 - np.sqrt(5))
    off = 2.0 / nb_point
    k = np.arange(0, nb_point)
    y = k * off - 1.0 + 0.5 * off
    r = np.sqrt(1 - y * y)
    phi = k * inc
    x = np.cos(phi) * r
    z = np.sin(phi) * r
    return x * radius, y * radius, z * radius


def generate_reference_graph(nb_vertices: int, radius: float) -> nx.Graph:
    """Generate a reference graph.

    :param int nb_vertices: the number of vertices.
    :param float radius: the radius of the sphere.
    :return: the generated reference graph.
    :rtype: nx.Graph
    """
    x, y, z = fibonacci(nb_point=nb_vertices, radius=radius)
    sphere_random_sampling = np.vstack((x, y, z)).T
    sphere_random_sampling = ng.compute_hull_from_vertices(
        sphere_random_sampling
    )  # Computing convex hull (adding edges)

    adja = topology.adjacency_matrix(sphere_random_sampling)
    graph = nx.from_numpy_array(adja.todense())

    node_attribute_dict = {}
    for node, label in enumerate(graph.nodes()):
        # we set the label of nodes in the same order as in graph
        node_attribute_dict[node] = {
            "coord": np.array(sphere_random_sampling.vertices[node]),
            "label": label,
        }
    # add the node attributes to the graph
    nx.set_node_attributes(graph, node_attribute_dict)
    #
    # # We add a default weight on each edge of 1
    nx.set_edge_attributes(graph, 1.0, name="weight")

    # We add a geodesic distance between the two ends of an edge
    edge_attribute_dict = {}
    id_counter = 0  # useful for affinity matrix calculation
    for edge in graph.edges:
        # We calculate the geodesic distance
        end_a = graph.nodes()[edge[0]]["coord"]
        end_b = graph.nodes()[edge[1]]["coord"]
        geodesic_dist = get_geodesic_distance_sphere(end_a, end_b, radius)

        # add the information in the dictionary
        edge_attribute_dict[edge] = {
            "geodesic_distance": geodesic_dist,
            "id": id_counter,
        }
        id_counter += 1

    # add the edge attributes to the graph
    nx.set_edge_attributes(graph, edge_attribute_dict)
    return graph
