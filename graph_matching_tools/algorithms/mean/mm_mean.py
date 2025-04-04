"""This module contains the algorithm for computing the mean graph between a set of graph
Implementation of the MM method proposed by Brijnesh J. Jain

.. moduleauthor:: François-Xavier Dupé
"""

from typing import Callable, Optional

import numpy as np
import networkx as nx

import graph_matching_tools.utils.utils as utils
import graph_matching_tools.algorithms.pairwise.kergm as kergm
import graph_matching_tools.algorithms.kernels.rff as rff


def get_graph_from_tensor(tensor: np.ndarray, tolerance: float = 1e-5) -> nx.Graph:
    """Create a graph (networkx) from its tensor representation. All the vectors are named "data".

    :param np.ndarray tensor: the tensor representation of the graph.
    :param float tolerance: the tolerance to detect existing edges (through a norm value).
    :return: the corresponding graph.
    :rtype: nx.Graph
    """
    g = nx.Graph()
    for i in range(tensor.shape[1]):
        g.add_node(i, data=np.squeeze(tensor[:, i, i]))
    for i in range(tensor.shape[1]):
        for j in range(i + 1, tensor.shape[1]):
            vector = np.squeeze(tensor[:, i, j])
            if np.linalg.norm(vector) > tolerance:
                g.add_edge(i, j, data=vector)
    return g


def get_tensor_from_graph(
    graph: nx.Graph, data_node: str, data_edge: str
) -> np.ndarray:
    """Compute a tensor representation of the graph.

    :param nx.Graph graph: the input graph.
    :param str data_node: the name of the data vector on nodes.
    :param str data_edge: the name of the data vector on edges.
    :return: the tensor.
    :rtype: np.ndarray
    """
    # 1 - First we need the maximal dimension of the data of edges and nodes
    if np.isscalar(graph.nodes[0][data_node]):
        dim_node = 1
    else:
        dim_node = graph.nodes[0][data_node].shape[0]
    dim_edge = utils.get_dim_data_edges(graph, data_edge)
    dim = dim_node
    if dim_edge > dim_node:
        dim = dim_edge
    # 2 - Build the tensor
    tensor = np.zeros((dim, nx.number_of_nodes(graph), nx.number_of_nodes(graph)))
    # 2.1 - The nodes
    for i in range(tensor.shape[1]):
        tensor[0:dim_node, i, i] = graph.nodes[i][data_node]
    # 2.2 - The edges
    for u, v, data in graph.edges.data(data_edge):
        tensor[0:dim_edge, u, v] = data
        tensor[0:dim_edge, v, u] = data

    return tensor


def tensor_matching(
    t1: np.ndarray,
    t2: np.ndarray,
    node_kernel: Callable[[np.ndarray, np.ndarray], float],
    edge_gamma: float,
    rff_dim: int = 200,
    num_alpha: int = 20,
    entropy_gamma: float = 0.2,
    random_seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Match two graphs using their tensor representation.

    :param np.ndarray t1: the first graph.
    :param np.ndarray t2: the second graph.
    :param Callable[[np.ndarray, np.ndarray], int] node_kernel: the kernel on the node vectors (must take two vectors).
    :param float edge_gamma: the hyperparameter of the kernel on the vectors on edges.
    :param int rff_dim: the dimension of the random Fourier features.
    :param int num_alpha: the number of step in the regularization path of KerGM.
    :param float entropy_gamma: the regularization of assigment in KerGM.
    :param Optional[int] random_seed: the seed for the random generator.
    :return: the matching between the nodes of the two graphs.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    # Remove node vectors to have the phi matrices
    phi1 = np.zeros((rff_dim, t1.shape[1], t1.shape[2]))
    phi2 = np.zeros((rff_dim, t2.shape[1], t2.shape[2]))

    # Compute the RFF
    vectors, offsets = rff.create_random_vectors(
        t1.shape[0], rff_dim, edge_gamma, random_seed=random_seed
    )
    for i in range(phi1.shape[1]):
        for j in range(i + 1, phi1.shape[2]):
            phi1[:, i, j] = np.sqrt(2 / vectors.shape[1]) * np.cos(
                t1[:, i, j] @ vectors + offsets
            )
            phi1[:, j, i] = phi1[:, i, j]

    for i in range(phi2.shape[1]):
        for j in range(i + 1, phi2.shape[2]):
            phi2[:, i, j] = np.sqrt(2 / vectors.shape[1]) * np.cos(
                t2[:, i, j] @ vectors + offsets
            )
            phi2[:, j, i] = phi2[:, i, j]

    # Compute knode
    knode = np.zeros((t1.shape[1], t2.shape[1]))
    for i in range(knode.shape[0]):
        for j in range(knode.shape[1]):
            knode[i, j] = np.squeeze(node_kernel(t1[:, i, i], t2[:, j, j]))

    # Get the gradient
    gradient = kergm.create_fast_gradient(phi1, phi2, knode)

    # Match
    r, c = kergm.kergm_method(
        gradient, knode.shape, num_alpha=num_alpha, entropy_gamma=entropy_gamma
    )

    return r, c


def compute_mean_graph(
    graphs: list[nx.Graph],
    node_data: str,
    node_kernel: Callable[[np.ndarray, np.ndarray], float],
    edge_data: str,
    edge_gamma: float,
    reference_graph: int = 0,
    rff_dim: int = 200,
    num_alpha: int = 20,
    entropy_gamma: float = 0.2,
    iterations: int = 10,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Compute the mean graph from a set of graphs.

    :param list[nx.Graph] graphs: the list of graphs.
    :param str node_data: the name of the data vector.
    :param Callable[[np.ndarray, np.ndarray], float] node_kernel: the node kernel.
    :param str edge_data: the name of the edge vector.
    :param float edge_gamma: the hyperparameter of the kernel on the vectors on edges.
    :param int reference_graph: the reference graph to begin the mean computation.
    :param int rff_dim: the dimension of the random Fourier features.
    :param int num_alpha: the number of step in the regularization path of KerGM.
    :param float entropy_gamma: the regularization of assigment in KerGM.
    :param int iterations: the number of iterations for computing the mean graph.
    :param Optional[int] random_seed: the seed for the random generator.
    :return: the mean graph in tensor format.
    :rtype: np.ndarray

    Here an example using NetworkX and some utils:

    .. doctest:

    >>> g1 = nx.Graph()
    >>> g1.add_node(0, weight=np.array((3.0,)))
    >>> g1.add_node(1, weight=np.array((4.0,)))
    >>> g1.add_node(2, weight=np.array((1.0,)))
    >>> g1.add_node(3, weight=np.array((1.0,)))
    >>> g1.add_edge(0, 1, weight=3.0)
    >>> g2 = nx.Graph()
    >>> g2.add_node(0, weight=np.array((5.0,)))
    >>> g2.add_node(1, weight=np.array((1.0,)))
    >>> g2.add_node(2, weight=np.array((2.0,)))
    >>> g2.add_node(3, weight=np.array((1.0,)))
    >>> g2.add_edge(0, 1, weight=1.0)
    >>> g2.add_edge(0, 2, weight=4.0)
    >>> g3 = nx.Graph()
    >>> g3.add_node(0, weight=np.array((1.0,)))
    >>> g3.add_node(1, weight=np.array((4.0,)))
    >>> g3.add_node(2, weight=np.array((2.0,)))
    >>> g3.add_node(3, weight=np.array((1.0,)))
    >>> g3.add_edge(0, 1, weight=2.0)
    >>> g3.add_edge(0, 3, weight=2.0)
    >>> g3.add_edge(1, 2, weight=3.0)
    >>> graphs = [g1, g2, g3]
    >>> def node_kernel(x, y):
    ...    return x * y
    >>> mean_graph = mm_mean.compute_mean_graph(graphs, "weight", node_kernel, "weight",
    ...     1.0, entropy_gamma=10.0, random_seed=1)
    >>> mean_graph
    array([[[2.33333333, 3.33333333, 0.        , 0.        ],
            [3.33333333, 4.33333333, 1.        , 0.        ],
            [0.        , 1.        , 1.        , 0.66666667],
            [0.        , 0.        , 0.66666667, 1.        ]]])
    """
    # 0 - Compute the tensor representation of each graph
    tensors = []
    for g in graphs:
        tensor = get_tensor_from_graph(g, node_data, edge_data)
        tensors.append(tensor)

    mean_graph = tensors[reference_graph]  # Initialization
    for iteration in range(iterations):
        # 1 - Get the matching between the current mean and the set of graphs
        r_perms = []
        c_perms = []
        for tensor in tensors:
            r, c = tensor_matching(
                tensor,
                mean_graph,
                node_kernel,
                edge_gamma,
                rff_dim=rff_dim,
                num_alpha=num_alpha,
                entropy_gamma=entropy_gamma,
                random_seed=random_seed,
            )
            r_perms.append(r)
            c_perms.append(c)

        # 2 - Update the mean graph
        new_mean = np.zeros(mean_graph.shape)
        for i in range(len(r_perms)):
            p = np.eye(mean_graph.shape[1], dtype=int)[c_perms[i]]
            new_mean += p.T @ tensors[i] @ p
        mean_graph = new_mean / len(tensors)

    return mean_graph
