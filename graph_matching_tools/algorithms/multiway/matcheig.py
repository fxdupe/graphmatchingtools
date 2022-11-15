"""
MatchEIG algorithm as described in ICCV 2017 paper

.. moduleauthor:: François-Xavier Dupé
"""
import numpy as np
import scipy.optimize as sco


def matcheig(x, rank, sizes):
    """Spectral way of building the permutation matrix with a given rank

    :param np.ndarray x: the input affinity matrix
    :param int rank: the dimension of the universe of nodes
    :param list[int] sizes: the size of the different graphs
    :return: the bulk permutation matrix
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
                corr = u[index1:index1 + sizes[i1_g], :] @ u[index2:index2 + sizes[i2_g], :].T
                r, c = sco.linear_sum_assignment(-corr)
                for i in range(r.shape[0]):
                    res[index1 + r[i], index2 + c[i]] = 1
            index2 += sizes[i2_g]
        index1 += sizes[i1_g]

    return res
