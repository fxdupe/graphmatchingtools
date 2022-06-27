"""
Random Fourier Feature

..moduleauthor:: François-Xavier Dupé
"""
import numpy as np


def create_random_vectors(size, number, sigma):
    """
    Compute the random vectors needed for the RRF
    :param int size: the dimension of features
    :param int number: the number of features
    :param float sigma: the variance of the features
    :return: a tuple with the random vectors and their random offsets
    """
    rng = np.random.default_rng()
    vectors = rng.normal(0.0, sigma, size=(size, number))
    offsets = rng.uniform(0.0, 1.0, size=(number,))
    return vectors, offsets
