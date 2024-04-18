"""
Score for matching comparison

.. moduleauthor:: François-Xavier Dupé
"""

import numpy as np


def compute_f1score(
    result: np.ndarray, truth: np.ndarray
) -> tuple[float, float, float]:
    """Compute the F1-score from permutation matrix.

    :param np.ndarray result: the produced bulk permutation matrix.
    :param np.ndarray truth: the ground truth permutation matrix.
    :return: a tuple with the f1-score, the precision and the recall.
    :rtype: tuple[float, float, float]
    """
    truth -= np.diag(np.diag(truth))
    result -= np.diag(np.diag(result))
    score = np.trace(truth @ result)
    precision = score / np.sum(result)
    recall = score / np.sum(truth)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score, precision, recall
