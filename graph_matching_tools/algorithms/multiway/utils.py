"""
Utility functions for the multiway matching
"""
import numpy as np
import scipy.optimize as sco


def permutation_projector(v, sizes, choice):
    """Projections over a set of permutation for each graph (with a reference graph)

    :param np.ndarray v: the current approximation matrix
    :param list[int] sizes: the sizes of the different graphs
    :param callable choice: the choosing function for the reference graph
    :return: the projected version of v
    """
    u = np.zeros(v.shape)
    index = 0

    # Take one graph as reference
    i_max = choice(range(len(sizes)))
    i_begin = int(np.sum(sizes[0:i_max]))
    ref = v[i_begin : i_begin + sizes[i_max], :]

    for size in sizes:
        vi = v[index : index + size, :] @ ref.T
        r, c = sco.linear_sum_assignment(-vi)
        for i in range(r.shape[0]):
            u[r[i] + index, c[i]] = 1
        index += size
    return u


def u_projector(v, sizes):
    """Projections over a set of permutation for each graph (without any reference graph)

    :param v: the current approximation matrix
    :param sizes: the sizes of the different graphs
    :return: the projected version of v
    """
    u = np.zeros(v.shape)
    index = 0
    for size in sizes:
        vi = v[index : index + size, :]
        r, c = sco.linear_sum_assignment(-vi)
        for i in range(r.shape[0]):
            u[r[i] + index, c[i]] = 1
        index += size
    return u
