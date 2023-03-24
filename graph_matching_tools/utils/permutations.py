"""
Utility function for permutation matrix

.. moduleauthor:: François-Xavier Dupé
"""
import numpy as np


def get_permutation_matrix_from_dictionary(
    matching: dict[str, dict[int, int]], g_sizes: list[int]
) -> np.ndarray:
    """Create the full permutation matrix from the matching result

    :param dict[str, dict[int, int]] matching: the matching result for each graph (nodes number, assignment)
    :param list[int] g_sizes: the list of the size of the different graph
    :return: the full permutation matrix
    :rtype: np.ndarray
    """
    f_size = int(np.sum(g_sizes))
    res = np.zeros((f_size, f_size))

    idx1 = 0
    for i_g1 in range(len(g_sizes)):
        idx2 = 0
        for i_g2 in range(len(g_sizes)):
            match = matching["{},{}".format(i_g1, i_g2)]
            for k in match:
                res[idx1 + int(k), idx2 + match[k]] = 1
            idx2 += g_sizes[i_g2]
        idx1 += g_sizes[i_g1]

    np.fill_diagonal(res, 1)
    return res


def get_permutation_matrix_from_matching(
    matching: np.ndarray, g_sizes: list[int]
) -> np.ndarray:
    """Create the full permutation matrix from the matching result.

    :param np.ndarray matching: the matching result for each graph (nodes number, assignment).
    :param list[int] g_sizes: the sizes of the different graphs (in order).
    :return: the full permutation matrix.
    :rtype: np.ndarray
    """
    max_node = np.max(matching[1, :])
    f_size = int(np.sum(g_sizes))
    res = np.zeros((f_size, max_node + 1))

    idx = 0
    for g in range(len(g_sizes)):
        for s in range(g_sizes[g]):
            res[matching[0, idx + matching[1, idx + s]], s] = 1
        idx += g_sizes[g]

    return res @ res.T
