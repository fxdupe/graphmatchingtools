"""
Direct multigraph version of KerGM

.. moduleauthor:: François-Xavier Dupé
"""

import numpy as np
import networkx as nx

import graph_matching_tools.algorithms.kernels.rff as rffo
import graph_matching_tools.algorithms.pairwise.kergm as kergm


def multi_pairwise_kergm(
    graphs: list[nx.Graph],
    sizes: list[int],
    knode: np.ndarray,
    name_data_edge: str,
    gamma: float,
    entropy: float,
    nb_alphas: int,
    iterations: int,
    rff: int = 100,
    epsilon: float = 1e-16,
) -> np.ndarray:
    """Direct pairwise matching on a set of graphs using KerGM.

    :param list[nx.Graph] graphs: the list of graph to match.
    :param list sizes: the sizes of the different graph.
    :param np.ndarray knode: the full node affinity matrix.
    :param str name_data_edge: the name of the data vector on edges.
    :param float gamma: the hyperparameter for edge kernel.
    :param float entropy: the entropy parameter for the Sinkhorn.
    :param int nb_alphas: the number of alpha values.
    :param int iterations: the maximal number of iterations for each alpha.
    :param int rff: the size of the random Fourier features.
    :param float epsilon: an additional term to avoid division by 0.
    :return: the full permutation matrix.
    :rtype: np.ndarray

    Here an example using NetworkX and some utils:

    .. doctest:

    >>> node_kernel = kern.create_gaussian_node_kernel(10.0, "weight")
    >>> knode = utils.create_full_node_affinity_matrix(graphs, node_kernel)
    >>> res = kergm.multi_pairwise_kergm(graphs, [2, 2, 3], knode, "weight", 1.0, 2.0, 10, 100, rff=100)
    >>> res
    array([[1., 0., 0., 1., 0., 1., 0.],
           [0., 1., 1., 0., 0., 0., 1.],
           [0., 1., 1., 0., 0., 0., 1.],
           [1., 0., 0., 1., 0., 1., 0.],
           [0., 0., 0., 0., 1., 0., 0.],
           [1., 0., 0., 1., 0., 1., 0.],
           [0., 1., 1., 0., 0., 0., 1.]])
    """
    vectors, offsets = rffo.create_random_vectors(1, rff, gamma)

    full_size = int(np.sum(sizes))
    res = np.eye(full_size, full_size)

    index1 = 0
    for i_g1 in range(len(graphs)):
        phi1 = rffo.compute_phi(graphs[i_g1], name_data_edge, vectors, offsets)

        index2 = index1 + sizes[i_g1]
        for i_g2 in range(i_g1 + 1, len(graphs)):
            phi2 = rffo.compute_phi(graphs[i_g2], name_data_edge, vectors, offsets)
            gradient = kergm.create_fast_gradient(
                phi1,
                phi2,
                knode[index1 : index1 + sizes[i_g1], index2 : index2 + sizes[i_g2]]
                / sizes[i_g1],
            )
            r, c = kergm.kergm_method(
                gradient,
                (sizes[i_g1], sizes[i_g2]),
                entropy_gamma=entropy,
                iterations=iterations,
                num_alpha=nb_alphas,
                epsilon=epsilon,
            )

            m_res = np.zeros((sizes[i_g1], sizes[i_g2]))
            for i in range(len(r)):
                m_res[r[i], c[i]] = 1

            res[index1 : index1 + sizes[i_g1], index2 : index2 + sizes[i_g2]] = m_res
            res[index2 : index2 + sizes[i_g2], index1 : index1 + sizes[i_g1]] = m_res.T

            index2 += sizes[i_g2]
        index1 += sizes[i_g1]

    return res
