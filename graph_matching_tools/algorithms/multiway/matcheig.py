"""
MatchEIG algorithm as described in ICCV 2017 paper.

.. moduleauthor:: François-Xavier Dupé
"""
import numpy as np
import scipy.optimize as sco


def matcheig(x: np.ndarray, rank: int, sizes: list[int]) -> np.ndarray:
    """Spectral way of building the permutation matrix with a given rank.

    :param np.ndarray x: the input affinity matrix.
    :param int rank: the dimension of the universe of nodes.
    :param list[int] sizes: the size of the different graphs.
    :return: the bulk permutation matrix.
    :rtype: np.ndarray

    Here an example using NetworkX and some utils:

    .. doctest:

    >>> node_kernel = kern.create_gaussian_node_kernel(0.1, "weight")  # Load a Gaussian kernel
    >>> graphs = [graph1, graph2]
    >>> sizes = [2, 2]
    >>> s = np.zeros((4, 4))
    >>> s[0:2, 0:2] = nx.to_numpy_array(graph1, weight=None)
    >>> s[2:4, 2:4] = nx.to_numpy_array(graph2, weight=None)
    >>> knode = utils.create_full_node_affinity_matrix(graphs, node_kernel)  # Get the affinity matrix between nodes
    >>> perm = matcheig.matcheig(knode, 2, sizes)  # Compute the universe of nodes
    >>> perm  # Show the permutation matrix
    array([[1., 0., 0., 1.],
           [0., 1., 1., 0.],
           [0., 1., 1., 0.],
           [1., 0., 0., 1.]])
    """
    u, s, _ = np.linalg.svd(x)
    u = np.real(u[:, 0:rank]) * np.sqrt(np.real(s[0:rank]))

    n = int(np.sum(sizes))
    res = np.eye(n)

    index1 = 0
    for i1_g in range(len(sizes)):
        index2 = 0
        for i2_g in range(len(sizes)):
            if i1_g != i2_g:
                corr = (
                    u[index1 : index1 + sizes[i1_g], :]
                    @ u[index2 : index2 + sizes[i2_g], :].T
                )
                r, c = sco.linear_sum_assignment(-corr)
                for i in range(r.shape[0]):
                    res[index1 + r[i], index2 + c[i]] = 1
            index2 += sizes[i2_g]
        index1 += sizes[i1_g]

    return res
