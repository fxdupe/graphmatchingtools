"""This module contains the classical Sinkhorn-Knopp algorithm for optimal transport.

One version of this method can be found in Cuturi, M. (2013). Sinkhorn distances: Lightspeed
 computation of optimal transport. Advances in neural information processing systems, 26.

.. moduleauthor:: François-Xavier Dupé
"""

from typing import Optional

import numpy as np


def sinkhorn_method(
    x: np.ndarray,
    mu_s: Optional[np.ndarray] = None,
    mu_t: Optional[np.ndarray] = None,
    gamma: float = 1.0,
    tolerance: float = 1e-6,
    iterations: int = 10000,
) -> np.ndarray:
    """Sinkhorn-Knopp algorithm as proposed by M. Cuturi.

    :param np.ndarray x: the input affinity matrix.
    :param float gamma: the weight of the entropy term.
    :param np.ndarray mu_s: the initial probability for source distribution (uniform by default).
    :param np.ndarray mu_t: the initial probability for target distribution (uniform by default).
    :param float tolerance: the tolerance for convergence (default: 1e-6).
    :param int iterations: the maximum number of iterations (default: 10000).
    :return: the approximate optimal transport from one side to another.
    :rtype: np.ndarray
    """
    u = np.ones(x.shape[0])
    v = np.ones(x.shape[1])
    if mu_s is None:
        mu_s = np.ones(x.shape[0]) / x.shape[0]
    if mu_t is None:
        mu_t = np.ones(x.shape[1]) / x.shape[1]
    c = np.exp(-x / gamma)
    for iteration in range(iterations):
        v = mu_t / (c.T @ u)
        unew = mu_s / (c @ v)

        u_norm = np.linalg.norm(u)
        if u_norm > 1e-3:
            error = np.linalg.norm(unew - u) / u_norm
        else:
            error = np.inf
        u = unew
        if error < tolerance:
            break
    res = np.diag(u.flat) @ c @ np.diag(v.flat)
    return res
