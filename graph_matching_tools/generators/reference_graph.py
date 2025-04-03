"""This module contains code to generate reference graph.

.. moduleauthor:: Marius Thorre, Sylvain Takerkart, François-Xavier Dupé
"""

import numpy as np
import networkx as nx

import graph_matching_tools.generators.noisy_graph as ng
import graph_matching_tools.generators.topology as topology
import graph_matching_tools.utils.sphere as sphere


def compute_edges_attributes(graph: nx.Graph, radius: float) -> nx.Graph:
    """Compute the edges attributes for a graph

    :param nx.Graph graph: the input graph.
    :param float radius: the radius of the sphere.
    :return: the graph.
    :rtype: nx.Graph
    """
    # We add a geodesic distance between the two ends of an edge
    edge_attribute_dict = dict()
    id_counter = 0  # useful for affinity matrix calculation

    for edge in graph.edges():
        # We calculate the geodesic distance
        end_a = graph.nodes()[edge[0]]["coord"]
        end_b = graph.nodes()[edge[1]]["coord"]
        geodesic_dist = get_geodesic_distance_sphere(end_a, end_b, radius)

        # add the information in the dictionary
        edge_attribute_dict[edge] = {
            "geodesic_distance": geodesic_dist,
            "id": id_counter,
            "weight": 1.0,
        }

        id_counter += 1

    # add the edge attributes to the graph
    nx.set_edge_attributes(graph, edge_attribute_dict)

    return graph


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


def generate_reference_graph(nb_vertices: int, radius: float) -> nx.Graph:
    """Generate a reference graph.

    :param int nb_vertices: the number of vertices.
    :param float radius: the radius of the sphere.
    :return: the generated reference graph.
    :rtype: nx.Graph
    """
    x, y, z = sphere.random_sampling(vertex_number=nb_vertices, radius=radius)
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
            "is_outlier": False,
        }
    # add the node and edge attributes to the graph
    nx.set_node_attributes(graph, node_attribute_dict)
    graph = compute_edges_attributes(graph, radius)

    return graph


def generated_max_geodesic_reference_graph(
    nb_vertices: int, radius: float, nb_iterations: int = 1000
) -> nx.Graph:
    """Generate a reference graph with the biggest minimal geodesic distance.

    :param int nb_vertices: the number of vertices.
    :param float radius: the radius of the sphere.
    :param int nb_iterations: the number of iterations (1000 by default).
    :return: the generated reference graph.
    :rtype: nx.Graph
    """
    max_ref_graph = generate_reference_graph(nb_vertices=nb_vertices, radius=radius)
    edge_graph_geo = [
        z["geodesic_distance"] for x, y, z in list(max_ref_graph.edges.data())
    ]
    min_geo_ref_graph = min(edge_graph_geo)

    for iter in range(nb_iterations - 1):
        new_graph = generate_reference_graph(nb_vertices=nb_vertices, radius=radius)
        edge_graph_geo = [
            z["geodesic_distance"] for x, y, z in list(new_graph.edges.data())
        ]
        min_geo_new_graph = min(edge_graph_geo)
        if min_geo_new_graph > min_geo_ref_graph:
            min_geo_ref_graph = min_geo_new_graph
            max_ref_graph = new_graph

    return max_ref_graph
