"""This module contains an optimized version of the Sinkhorn-Knopp algorithm for optimal transport.

This is the implementation of Tang, X., Shavlovsky, M., Rahmanian, H., Tardini, E., Thekumparampil, K. K., Xiao, T.,
 & Ying, L. (2023, October). Accelerating Sinkhorn algorithm with sparse Newton iterations.
  In The Twelfth International Conference on Learning Representations.

.. moduleauthor: François-Xavier Dupé
"""

import numpy as np


def sinkhorn_newton_sparse_method(
    cost: np.ndarray,
    mu_s: np.ndarray,
    mu_t: np.ndarray,
    eta: float,
    n1_iterations: int = 100,
    n2_iterations: int = 100,
    rho: float = 1.0,
    x_init: np.ndarray = None,
    y_init: np.ndarray = None,
) -> np.ndarray:
    """Sinkhorn with a sparse Newton steps (best for some hard constrained optimal transport

    :param cost:
    :param rho:
    :param mu_s:
    :param mu_t:
    :param eta:
    :param n1_iterations:
    :param n2_iterations:
    :param x_init:
    :param y_init:
    :return:
    """
    x = x_init
    y = y_init
    if x is None:
        x = 0 * mu_s
    if y is None:
        y = 0 * mu_t

    # The classical steps
    p = cost * 0
    for i in range(n1_iterations):
        p = np.exp(
            eta
            * (
                -cost
                + x @ np.ones((1, cost.shape[0]))
                + np.ones((cost.shape[1], 1)) @ y.T
            )
            - 1.0
        )
        x = x + (np.log(mu_s) - np.log(p @ np.ones((p.shape[1], 1)))) / eta
        p = np.exp(
            eta
            * (
                -cost
                + x @ np.ones((1, cost.shape[0]))
                + np.ones((cost.shape[1], 1)) @ y.T
            )
            - 1.0
        )
        y = y + (np.log(mu_t) - np.log(p.T @ np.ones((p.shape[0], 1)))) / eta

    # The Newton steps
    hessian = np.zeros((cost.shape[0] + cost.shape[1], cost.shape[0] + cost.shape[1]))
    for i in range(n2_iterations):
        # Hessian is the M matrix in the paper
        hessian[0 : cost.shape[0], 0 : cost.shape[0]] = np.diag(
            p @ np.ones((p.shape[1], 1))
        )
        hessian[cost.shape[0] :, cost.shape[0] :] = np.diag(
            p.T @ np.ones((p.shape[0], 1))
        )
        hessian[0 : cost.shape[0], cost.shape[0] :] = p
        hessian[cost.shape[0] :, 0 : cost.shape[0]] = p.T
        idx = np.abs(hessian) < rho
        hessian[idx] = 0

        # TODO: continue writing the algorithm

    return p
