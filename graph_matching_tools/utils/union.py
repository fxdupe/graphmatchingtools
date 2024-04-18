"""This utility module contains the Union-Find algorithm

    parents is an (nb_element, 2) matrix, 0 contains the parent index and 1 the rank

.. moduleauthor:: François-Xavier Dupé
"""

import numpy as np


def create_set(numbers_of_elements: int) -> np.ndarray:
    """Create a default parents matrix for the union-find method.

    :param numbers_of_elements: the number of elements.
    :return: the default parent matrix.
    :rtype: np.ndarray
    """
    res = np.zeros((numbers_of_elements, 2), dtype="i")
    res[:, 0] = np.array(range(numbers_of_elements))
    return res


def find(x: int, parents: np.ndarray) -> int:
    """The FIND part of the method.

    :param int x: the researched element.
    :param np.ndarray parents: the parents table.
    :return: the label of x.
    :rtype: int
    """
    if parents[x, 0] == x:
        return x

    parents[x, 0] = find(parents[x, 0], parents)
    return parents[x, 0]


def union(x: int, y: int, parents: np.ndarray) -> None:
    """The UNION part of the method.

    :param int x: the index of the first element.
    :param int y: the index of the second element.
    :param np.ndarray parents: the parents table.
    """
    x_root = find(x, parents)
    y_root = find(y, parents)

    if parents[x_root, 1] > parents[y_root, 1]:
        parents[y_root, 0] = x_root
    elif parents[x_root, 1] < parents[y_root, 1]:
        parents[x_root, 0] = y_root
    elif x_root != y_root:
        parents[y_root, 0] = x_root
        parents[x_root, 1] += 1
