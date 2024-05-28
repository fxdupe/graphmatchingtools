"""This module contains an optimized version of the Sinkhorn-Knopp algorithm for optimal transport.

This is the implementation of Tang, X., Shavlovsky, M., Rahmanian, H., Tardini, E., Thekumparampil, K. K., Xiao, T.,
 & Ying, L. (2023, October). Accelerating Sinkhorn algorithm with sparse Newton iterations.
  In The Twelfth International Conference on Learning Representations.

.. moduleauthor: François-Xavier Dupé
"""

import numpy as np
import scipy.optimize as sco


def _objective_function(
    z: np.ndarray, cost: np.ndarray, mu_s: np.ndarray, mu_t: np.ndarray, eta: float
) -> float:
    """The optimal transport objective function with Lyapunov potential.

    :param np.ndarray z: the current dual variable.
    :param np.ndarray cost: the cost matrix.
    :param np.ndarray mu_s: the source distribution.
    :param np.ndarray mu_t: the target distribution.
    :param float eta: the regularization parameter.
    :return: the value at z.
    :rtype: float
    """
    tmp = -cost + z[0 : cost.shape[0], 0] + z[cost.shape[0] :, 0].T
    p1 = -np.exp(eta * tmp - 1) / eta
    p2 = z.T @ np.concatenate([mu_s, mu_t], axis=0)
    return p1.sum() + p2


def _compute_hessian(p: np.ndarray, eta: float) -> np.ndarray:
    """Compute the Hessian matrix from current transport.

    :param np.ndarray p: the current transport.
    :param float eta: the regularization parameter.
    :return: the Hessian matrix.
    :rtype: np.ndarray
    """
    hessian = np.zeros((p.shape[0] + p.shape[1], p.shape[0] + p.shape[1]))
    hessian[0 : p.shape[0], 0 : p.shape[0]] = np.diag(
        (p @ np.ones((p.shape[1], 1))).flat
    )
    hessian[p.shape[0] :, p.shape[0] :] = np.diag((p.T @ np.ones((p.shape[0], 1))).flat)
    hessian[0 : p.shape[0], p.shape[0] :] = p
    hessian[p.shape[0] :, 0 : p.shape[0]] = p.T
    hessian *= eta
    return hessian


def _compute_transport_from_dual(
    cost: np.ndarray, z: np.ndarray, eta: float
) -> np.ndarray:
    """Compute the transport matrix from dual variable.

    :param np.ndarray cost: the cost matrix.
    :param np.ndarray z: the dual variable.
    :param float eta: the regularization parameter.
    :return: the transport matrix.
    :rtype: np.ndarray
    """
    (x, y) = (z[0 : cost.shape[0], :], z[cost.shape[0] :, :])
    transport = np.exp(
        eta
        * (-cost + x @ np.ones((1, cost.shape[0])) + np.ones((cost.shape[1], 1)) @ y.T)
        - 1
    )
    return transport


def _sparsify(tensor: np.ndarray, nb_keep: int) -> np.ndarray:
    """Sparsify a given tensor.

    :param np.ndarray tensor: the input tensor.
    :param int nb_keep: the number of elements to keep.
    :return: the sparsified tensor.
    :rtype: np.ndarray
    """
    tensor_array = tensor.reshape((-1,))
    idx = np.argsort(np.abs(tensor_array))
    result = tensor_array
    result[idx[0 : result.shape[0] - nb_keep]] = 0.0
    return result.reshape(tensor.shape)


def _linesearch(
    z: np.ndarray,
    direction: np.ndarray,
    alpha: float,
    cost: np.ndarray,
    mu_s: np.ndarray,
    mu_t: np.ndarray,
    eta: float,
) -> float:
    """Linesearch for finding good descent step.

    :param np.ndarray z: the current dual variable.
    :param np.ndarray direction: the current direction of descent.
    :param float alpha: the initial step value.
    :param np.ndarray cost: the cost matrix.
    :param np.ndarray mu_s: the source distribution.
    :param np.ndarray mu_t: the target distribution.
    :param float eta: the regularization parameter.
    :return: the "best" step.
    :rtype: float
    """
    new_z = z + alpha * direction
    current_val = _objective_function(z, cost, mu_s, mu_t, eta)
    new_val = _objective_function(new_z, cost, mu_s, mu_t, eta)
    while new_val < current_val:
        alpha = alpha / 2.0
        new_z = z + alpha * direction
        new_val = _objective_function(new_z, cost, mu_s, mu_t, eta)

    return alpha


def sinkhorn_newton_sparse_method(
    cost: np.ndarray,
    mu_s: np.ndarray,
    mu_t: np.ndarray,
    eta: float,
    n1_iterations: int = 100,
    n2_iterations: int = 100,
    rho: float = 0.5,
    x_init: np.ndarray = None,
    y_init: np.ndarray = None,
    alpha: float = 1.0,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Sinkhorn with a sparse Newton steps (best for some hard constrained optimal transport).

    :param np.ndarray cost: the cost matrix.
    :param np.ndarray mu_s: the source distribution.
    :param np.ndarray mu_t: the target distribution.
    :param float eta: the regularization parameter.
    :param int n1_iterations: the number of Sinkhorn iterations (100 by default).
    :param int n2_iterations: the number of Newton iterations.
    :param float rho: the percentage of kept values in Hessian matrix (0.5 by default).
    :param np.ndarray x_init: the initial first dual variable (0 by default).
    :param np.ndarray y_init: the initial second dual variable (0 by default).
    :param float alpha: the initial step for the line search.
    :param float tolerance: the convergence tolerance.
    :return: the transport matrix.
    :rtype: np.ndarray
    """
    x = x_init
    y = y_init
    if x is None:
        x = 0 * mu_s
    if y is None:
        y = 0 * mu_t

    # Compute the number of elements to keep
    if rho > 1.0 or rho < 0.0:
        rho = 0.5
    nb_keep = int((cost.shape[0] + cost.shape[1]) * rho)

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
        new_y = y + (np.log(mu_t) - np.log(p.T @ np.ones((p.shape[0], 1)))) / eta

        if np.linalg.norm(new_y - y) / np.linalg.norm(new_y) < tolerance:
            break
        y = new_y

    # The Newton steps
    z = np.concatenate([x, y], axis=0)
    for i in range(n2_iterations):
        # Get current transport
        p = _compute_transport_from_dual(cost, z, eta)

        # Hessian is the M matrix in the paper
        hessian = _compute_hessian(p, eta)

        # Sparsification of the matrix
        hessian = _sparsify(hessian, nb_keep)

        # Compute the gradient
        gradient = np.zeros((p.shape[0] + p.shape[1], 1))
        gradient[0 : p.shape[0], :] = mu_s - p @ np.ones((p.shape[1], 1))
        gradient[p.shape[0] :, :] = mu_t - p.T @ np.ones((p.shape[0], 1))

        # Get the direction
        result = sco.lsq_linear(hessian, (-gradient).flat)
        direction = result["x"]
        direction = np.reshape(direction, (-1, 1))

        # Linesearch (concave function)
        alpha = _linesearch(z, direction, alpha * 2.0, cost, mu_s, mu_t, eta)

        # Update
        new_z = z + alpha * direction
        if np.linalg.norm(new_z - z) / np.linalg.norm(new_z) < tolerance:
            break
        z = new_z

    # Build transport from dual variables
    transport = _compute_transport_from_dual(cost, z, eta)
    return transport
