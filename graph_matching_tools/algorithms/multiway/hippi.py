"""
HiPPI algorithm as described in ICCV 2019 paper.

.. moduleauthor:: François-Xavier Dupé
"""
import numpy as np

import graph_matching_tools.algorithms.multiway.utils as utils


def hippi_multiway_matching(
    s, sizes, knode, u_dim, iterations=100, tolerance=1e-6, init=None
):
    """HiPPI method for multi-graph matching based on a power method

    :param np.ndarray s: the bulk matrix with the adjacency matrices on the diagonal
    :param list sizes: the number of nodes of the different graphs (in order)
    :param np.ndarray knode: the node affinity matrix
    :param int u_dim: the dimension of the universe of nodes
    :param int iterations: the maximal number of iterations
    :param float tolerance: the tolerance for convergence
    :param np.ndarray init: the initialization, random if None
    :return: the universe of node projection for all the nodes
    :rtype: np.ndarray

    Here an example using NetworkX and some utils:

    .. doctest:

    >>> import numpy as np  # Load the different modules
    >>> import networkx as nx
    >>> import graph_matching_tools.algorithms.kernels.gaussian as kern
    >>> import graph_matching_tools.algorithms.kernels.utils as utils
    >>> import graph_matching_tools.algorithms.multiway.hippi as hippi
    >>> node_kernel = kern.create_gaussian_node_kernel(0.1, "weight")  # Load a Gaussian kernel
    >>> graph1 = nx.Graph()  # The first graph
    >>> graph1.add_node(0, weight=1.0)
    >>> graph1.add_node(1, weight=2.0)
    >>> graph1.add_edge(0, 1, weight=1.0)
    >>> graph2 = nx.Graph()  # The second graph
    >>> graph2.add_node(0, weight=2.0)
    >>> graph2.add_node(1, weight=1.0)
    >>> graph2.add_edge(0, 1, weight=1.0)
    >>> graphs = [graph1, graph2]
    >>> sizes = [2, 2]
    >>> s = np.zeros((4, 4))
    >>> s[0:2, 0:2] = nx.to_numpy_array(graph1, weight=None)
    >>> s[2:4, 2:4] = nx.to_numpy_array(graph2, weight=None)
    >>> knode = utils.create_full_node_affinity_matrix(graphs, node_kernel)  # Get the affinity matrix between nodes
    >>> u = hippi.hippi_multiway_matching(s, sizes, knode, 2, iterations=50)  # Compute the universe of nodes
    >>> u @ u.T  # Get the permutation matrix
    array([[1., 0., 0., 1.],
           [0., 1., 1., 0.],
           [0., 1., 1., 0.],
           [1., 0., 0., 1.]])
    """
    if init is None:
        u = np.ones((s.shape[0], u_dim)) / u_dim + 1e-3 * np.random.randn(
            s.shape[0], u_dim
        )
    else:
        u = init

    w = knode.T @ s @ knode
    vi = w @ u @ u.T @ w @ u
    fu = np.trace(u.T @ vi)

    for i in range(iterations):
        u = utils.u_projector(vi, sizes)
        vi = w @ u @ u.T @ w @ u
        n_fu = np.trace(u.T @ vi)
        if np.abs(n_fu - fu) < tolerance:
            break
        fu = n_fu

    return u
