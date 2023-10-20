"""
This code is directly from the paper
"Fast and accurate optimization on the orthogonal manifold without retraction" by Ablin and Peyré (AISTATS, 2022).

.. moduleauthor:: François-Xavier Dupé
"""
from typing import Callable

import numpy as np


def _skew(x: np.ndarray) -> np.ndarray:
    """Compute the skew of a matrix.

    :param np.ndarray x: the input matrix.
    :return: the skew of the matrix.
    :rtype: np.ndarray
    """
    return 0.5 * (x - x.T)


def _phi(x: np.ndarray, gradient: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Compute the relative gradient.

    :param np.ndarray x: the current matrix.
    :param Callable[[np.ndarray], np.ndarray] gradient: the classical gradient function.
    :return: the relative gradient at current point.
    :rtype: np.ndarray
    """
    return _skew(gradient(x) @ x.T)


def _go_landing(
    x: np.ndarray, lbd: float, gradient: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Compute the landing function (for orthogonal group).

    :param np.ndarray x: the current matrix.
    :param float lbd: the landing hyperparameter.
    :param gradient: the classical gradient function.
    :return: the landing point.
    :rtype: np.ndarray
    """
    res = _phi(x, gradient) @ x + lbd * (x @ x.T - np.identity(x.shape[0])) @ x
    return res


def _sm_landing(
    x: np.ndarray, lbd: float, gradient: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Compute the landing function (for Stiefel manifold).

    :param np.ndarray x: the current matrix.
    :param float lbd: the landing hyperparameter.
    :param gradient: the classical gradient function.
    :return: the landing point.
    :rtype: np.ndarray
    """
    res = _phi(x, gradient) @ x + lbd * ((x.T @ x - np.identity(x.shape[1])) @ x.T).T
    return res


def _go_compute_eta(
    x: np.ndarray,
    gradient: Callable[[np.ndarray], np.ndarray],
    lbd: float,
    epsilon: float,
) -> float:
    """Compute the step for the orthogonal group optimization.

    :param np.ndarray x: the current matrix.
    :param Callable[[np.ndarray], np.ndarray] gradient: the classical gradient.
    :param float lbd: the landing hyperparameter.
    :param float epsilon: the tolerance for the step.
    :return: a inner step.
    :rtype: float
    """
    a = np.linalg.norm(_phi(x, gradient))
    d = 0.25 * np.linalg.norm(x @ x.T - np.identity(x.shape[0]))
    alpha = 2.0 * lbd * d - 2.0 * a * d - 2.0 * lbd * d**2.0
    beta = a**2.0 + lbd**2.0 * d**3.0 + 2.0 * lbd * a * d**2.0 + a**2.0 * d
    eta = (np.sqrt(alpha**2.0 + 4.0 * beta * (epsilon - d)) + alpha) / (2.0 * beta)
    return eta


def _sm_compute_eta(
    x: np.ndarray,
    gradient: Callable[[np.ndarray], np.ndarray],
    lbd: float,
    epsilon: float,
) -> float:
    """Compute the step for the Stiefel manifold optimization.

    :param np.ndarray x: the current matrix.
    :param Callable[[np.ndarray], np.ndarray] gradient: the classical gradient.
    :param float lbd: the landing hyperparameter.
    :param float epsilon: the tolerance for the step.
    :return: a inner step.
    :rtype: float
    """
    a = np.linalg.norm(_phi(x, gradient))
    d = 0.25 * np.linalg.norm(x.T @ x - np.identity(x.shape[1]))
    alpha = 2.0 * lbd * d - 2.0 * a * d - 2.0 * lbd * d**2.0
    beta = a**2.0 + lbd**2.0 * d**3.0 + 2.0 * lbd * a * d**2.0 + a**2.0 * d
    eta = (np.sqrt(alpha**2.0 + 4.0 * beta * (epsilon - d)) + alpha) / (2.0 * beta)
    return eta


def orthogonal_group_optimization(
    gradient: Callable[[np.ndarray], np.ndarray],
    size: int,
    step: float = 0.1,
    lbd: float = 1.0,
    epsilon: float = 0.5,
    iterations: int = 100,
    tolerance: float = 1e-20,
) -> np.ndarray:
    """Optimization over the orthogonal group.

    :param Callable[[np.ndarray], np.ndarray] gradient: the gradient of the function to optimize.
    :param int size: the order of the space.
    :param float step: the initial descent step.
    :param float lbd: the landing hyperparameter.
    :param float epsilon: tolerance over the step.
    :param int iterations: the maximal number of iterations.
    :param float tolerance: the tolerance for convergence.
    :return: the solution.
    :rtype: np.ndarray
    """
    x = np.identity(size)
    for it in range(iterations):
        eta = _go_compute_eta(x, gradient, lbd, epsilon)
        eta = np.min([step, eta])
        x_new = x - eta * _go_landing(x, lbd, gradient)
        if np.linalg.norm(x - x_new) < tolerance:
            break
        x = x_new

    return x


def stiefel_manifold_optimization(
    gradient: Callable[[np.ndarray], np.ndarray],
    size: tuple[int, int],
    step: float = 0.1,
    lbd: float = 1.0,
    epsilon: float = 0.5,
    iterations: int = 100,
    tolerance: float = 1e-20,
) -> np.ndarray:
    """Optimization over the Stiefel manifold.

    :param Callable[[np.ndarray], np.ndarray] gradient: the gradient of the function to optimize.
    :param tuple[int, int] size: the order of the space.
    :param float step: the initial descent step.
    :param float lbd: the landing hyperparameter.
    :param float epsilon: tolerance over the step.
    :param int iterations: the maximal number of iterations.
    :param float tolerance: the tolerance for convergence.
    :return: the solution.
    :rtype: np.ndarray
    """
    x = np.eye(size[0], size[1])
    for it in range(iterations):
        eta = _sm_compute_eta(x, gradient, lbd, epsilon)
        eta = np.min([step, eta])
        x_new = x - eta * _sm_landing(x, lbd, gradient)
        if np.linalg.norm(x - x_new) < tolerance:
            break
        x = x_new

    return x
