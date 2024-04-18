"""
MatchALS multigrpah method

.. moduleauthor:: François-Xavier Dupé
"""

from typing import Optional

import numpy as np

import scipy.linalg as scl


def _diagonal_projection(
    diag: np.ndarray, sum_target: float, tolerance: float = 1e-6
) -> np.ndarray:
    """Project a vector to the closed positive vector with a given sum.

    :param np.ndarray diag: the vector to project.
    :param float sum_target: the targeted sum.
    :param float tolerance: the tolerance for convergence.
    :return: the projection.
    :rtype: np.ndarray
    """
    x = diag
    for i in range(1000):
        tmp = x - (np.sum(x) - sum_target) / diag.shape[0]
        xnew = x + np.maximum(2 * tmp - x, 0) - tmp
        if np.linalg.norm(xnew - x) / np.linalg.norm(x) < tolerance:
            break
        x = xnew
    return x


def _apply_constraints(
    x: np.ndarray, sizes: list[int], p_select: float = 1.0
) -> np.ndarray:
    """Apply the constraints for having true assignment.

    :param np.ndarray x: the current estimate.
    :param list[int] sizes: the different size of the graphs.
    :param float p_select: the percentage of the element on diagonals.
    :return: the projection of x onto the constraint set.
    :rtype: np.ndarray
    """
    index = 0
    # Constraint on diagonals
    if p_select >= 1.0:
        for size in sizes:
            x[index : index + size, index : index + size] = np.identity(size)
            index += size
    else:
        diag = np.diag(x)
        new_diag = _diagonal_projection(diag, p_select * x.shape[0])
        np.fill_diagonal(x, new_diag)
    # Constraint on values
    idx = x < 0
    x[idx] = 0
    idx = x > 1
    x[idx] = 1
    return x


def matchals(
    s: np.ndarray,
    sizes: list[int],
    k: int,
    alpha: float = 50.0,
    beta: float = 0.01,
    p_select: float = 1.0,
    iterations: int = 1000,
    tolerance: float = 5e-4,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """MatchALS algorithm for multiple graph matching (python version).

    This version follows the Matlab code in https://github.com/zju-3dv/multiway/blob/master/utils/mmatch_CVX_ALS.m

    :param np.ndarray s: the pairwise affinity matrix.
    :param list[int] sizes: the sizes of the different graphs (node numbers).
    :param int k: the rank parameter.
    :param float alpha: the regularisation parameter.
    :param float beta: sparsity constraint parameter.
    :param float p_select: percentage of non-null element on the diagonal.
    :param int iterations: the number of iterations.
    :param float tolerance: the tolerance for convergence.
    :param Optional[int] random_seed: the seed for the random generator.
    :return: the full assignment matrix.
    :rtype: np.ndarray


    Here an example using NetworkX and some utils:

    .. doctest:

    >>> node_kernel = kern.create_gaussian_node_kernel(2.0, "weight")
    >>> knode = utils.create_full_node_affinity_matrix(graphs, node_kernel)
    >>> res = matchals.matchals(knode, [2, 2, 3], 3, alpha=200.0, random_seed=11)
    >>> res
    array([[1., 0., 0., 1., 0., 1., 0.],
           [0., 1., 1., 0., 0., 0., 1.],
           [0., 1., 1., 0., 0., 0., 1.],
           [1., 0., 0., 1., 0., 1., 0.],
           [0., 0., 0., 0., 1., 0., 0.],
           [1., 0., 0., 1., 0., 1., 0.],
           [0., 1., 1., 0., 0., 0., 1.]])
    """
    rng = np.random.default_rng(seed=random_seed)
    mu = 64.0
    w = s.copy()
    w -= np.diag(np.diag(w))  # Removing the diagonal (see Matlab code)
    w = (w + w.T) / 2
    x = w.copy()
    z = w.copy()
    y = np.zeros(x.shape)
    a = rng.uniform(0.0, 1.0, size=(w.shape[0], k))
    for iteration in range(iterations):
        x0 = x
        x = z - (y - w + beta) / mu
        b = scl.solve(a.T @ a + alpha / mu * np.identity(k), a.T @ x).T
        a = scl.solve(b.T @ b + alpha / mu * np.identity(k), b.T @ x.T).T
        x = a @ b.T
        z = x + y / mu
        z = _apply_constraints(z, sizes, p_select=p_select)
        y = y + mu * (x - z)

        pres = np.linalg.norm(x - z) / w.shape[0]
        dres = mu * np.linalg.norm(x0 - x) / w.shape[0]
        if pres < tolerance and dres < tolerance:
            break
        if pres > 10 * dres:
            mu = 2 * mu
        elif dres > 10 * pres:
            mu = mu / 2

    x = (x + x.T) / 2
    x = np.array(x > 0.5, dtype="f8")  # Thresholding as in available code
    return x
