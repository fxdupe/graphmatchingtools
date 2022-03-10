"""
KMean way of doing multigraph matching by clustering the nodes

.. moduleauthor:: Rohit Yadav
"""
import numpy as np
from sklearn.cluster import KMeans


def create_perm_from_labels(labels):
    """
    Create permutation matrix using labels from KMeans
    :param labels: the labels of the cluster
    :return: the permutation matrix
    """
    u = np.zeros((len(labels), len(set(labels))))
    for node, label in zip(range(u.shape[0]), labels):
        u[node, label] = 1
    return u @ u.T


def get_labels_from_k_means(k, data):
    """
    Get labels from the KMeans algorithm
    :param k: the number of clusters
    :param data: the data to cluster
    :return: the labels for each cluster
    """
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    return kmeans.labels_


def get_permutation_with_kmeans(k, data):
    """
    Apply KMeans to get permutation matrix
    :param k: the number of clusters
    :param data: the data array
    :return: the permutation matrix
    """
    kmeans_labels = get_labels_from_k_means(k, data)
    perm = create_perm_from_labels(kmeans_labels)
    return perm
