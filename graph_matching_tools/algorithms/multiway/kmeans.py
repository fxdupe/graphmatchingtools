"""
KMean way of doing multigraph matching by clustering the nodes

.. moduleauthor:: Rohit Yadav
"""
import numpy as np
from sklearn.cluster import KMeans


def _create_perm_from_labels(labels: list[int]) -> np.ndarray:
    """Create permutation matrix using labels from KMeans.

    :param list[int] labels: the labels of the cluster.
    :return: the permutation matrix.
    :rtype: np.ndarray
    """
    u = np.zeros((len(labels), len(set(labels))))
    for node, label in zip(range(u.shape[0]), labels):
        u[node, label] = 1
    return u @ u.T


def _get_labels_from_k_means(k: int, data: np.ndarray) -> list[int]:
    """Get labels from the KMeans algorithm.

    :param int k: the number of clusters.
    :param np.ndarray data: the data to cluster.
    :return: the labels for each cluster.
    :rtype: list[int]
    """
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(data)
    return kmeans.labels_


def get_permutation_with_kmeans(k: int, data: np.ndarray) -> np.ndarray:
    """Apply KMeans to get permutation matrix.

    :param int k: the number of clusters.
    :param np.ndarray data: the data array.
    :return: the permutation matrix.
    :rtype: np.ndarray

    Here an example using NetworkX and some utils:

    .. doctest:

    >>> node_kernel = kern.create_gaussian_node_kernel(1.0, "weight")
    >>> knode = utils.create_full_node_affinity_matrix(graphs, node_kernel)
    >>> res = kmeans.get_permutation_with_kmeans(2, knode)
    >>> res
    array([[1., 0., 0., 1., 1., 1., 0.],
           [0., 1., 1., 0., 0., 0., 1.],
           [0., 1., 1., 0., 0., 0., 1.],
           [1., 0., 0., 1., 1., 1., 0.],
           [1., 0., 0., 1., 1., 1., 0.],
           [1., 0., 0., 1., 1., 1., 0.],
           [0., 1., 1., 0., 0., 0., 1.]])
    """
    kmeans_labels = _get_labels_from_k_means(k, data)
    perm = _create_perm_from_labels(kmeans_labels)
    return perm
