"""
Utility functions for the multiway matching
"""
import numpy as np
import scipy.optimize as sco


def u_projector(v, sizes):
    """
    Projections over a set of permutation (for each graph)

    :param v: the current approximation matrix
    :param sizes: the sizes of the different graphs
    :return: the projected version of v
    """
    u = np.zeros(v.shape)
    index = 0
    for size in sizes:
        vi = v[index:index+size, :]
        r, c = sco.linear_sum_assignment(-vi)
        for i in range(r.shape[0]):
            u[r[i]+index, c[i]] = 1
        index += size
    return u
