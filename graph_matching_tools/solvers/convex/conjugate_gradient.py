"""
This code is an implementation of the Conjugate Gradient Method.

See https://en.wikipedia.org/wiki/Conjugate_gradient_method for more information

.. moduleauthor:: François-Xavier Dupé
"""

import numpy as np


def conjugate_gradient(
    mat: np.ndarray, b: np.ndarray, iterations: int = 100, tolerance: float = 1e-3
) -> np.ndarray:
    """
    Conjugate gradient method for solving Ax=b equations.

    :param np.ndarray mat: the coefficient matrix (aka A).
    :param np.ndarray b: the target vector.
    :param int iterations: the maximum number of iterations.
    :param float tolerance: the tolerance for convergence.
    :return: the estimated solution.
    :rtype: np.ndarray
    """
    x_k = np.zeros((mat.shape[0], 1))
    r_k = b - mat @ x_k
    p_k = r_k

    if np.linalg.norm(r_k) < tolerance:
        return x_k

    for iteration in range(iterations):
        alpha_k = (r_k.T @ r_k) / (p_k.T @ mat @ p_k)
        x_kpp = x_k + alpha_k * p_k
        r_kpp = r_k - alpha_k * mat @ p_k

        if np.linalg.norm(r_kpp) < tolerance:
            break

        beta_k = (r_kpp.T @ r_kpp) / (r_k.T @ r_k)
        p_kpp = r_kpp + beta_k * p_k

        x_k = x_kpp
        p_k = p_kpp
        r_k = r_kpp

    return x_kpp
