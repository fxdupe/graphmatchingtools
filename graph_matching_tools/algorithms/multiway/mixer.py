"""
This code is directly from the paper
Lusk, P. C., Fathian, K., & How, J. P. (2023). MIXER: Multiattribute, Multiway Fusion of Uncertain Pairwise Affinities.
IEEE Robotics and Automation Letters, 8(5), 2462-2469.

.. moduleauthor:: François-Xavier Dupé
"""
import numpy as np


def probability_simplex_projector(x: np.ndarray) -> np.ndarray:
    """Projector onto the probability simplex (each line of a matrix)

    :param np.ndarray x: the matrix to project.
    :return: the projection.
    :rtype: np.ndarray
    """
    tmp1 = np.sort(x, axis=1)[:, ::-1]
    sum_t = np.cumsum(tmp1, axis=1)
    tmp2 = tmp1 + (1 - sum_t) / np.arange(1, x.shape[1] + 1)
    rho = x.shape[1] - np.argmax(tmp2[:, ::-1] > 0, axis=1)
    lbd = (1 - sum_t[np.arange(0, x.shape[0]), rho - 1]) / rho
    res = np.maximum(x + lbd.reshape(-1, 1), 0)
    return res


def _objective_function(
    u: np.ndarray, s: np.ndarray, po: np.ndarray, pd: np.ndarray, d: float
):
    """The MIXER objective function.

    :param np.ndarray u: the universe of node matrix.
    :param np.ndarray s: the affinity matrix.
    :param np.ndarray po: the orthogonality penalty matrix.
    :param np.ndarray pd: the distinctivness penalty matrix.
    :param float d: the strength of the penalty.
    :return: the value at the current point.
    :rtype: float
    """
    t1 = np.trace(u.T @ u @ (1 - 2.0 * s))
    t2 = np.trace((u.T @ u) @ po) + np.trace((u @ u.T) @ pd)
    return t1 + d * t2


def mixer(
    knode: np.ndarray, sizes: list[int], step: float, iterations: int
) -> np.ndarray:
    """MIXER method for attributes alignment

    :param np.ndarray knode: the attributes/nodes affinity matrix.
    :param list[int] sizes: the size of the different elements.
    :param float step: the initial step of the gradient descent.
    :param int iterations: the number of iterations for convergence.
    :return: the permutation matrix (in the universe of nodes).
    :rtype: np.ndarray

    Here an example using NetworkX and some utils:

    .. doctest:

    >>> node_kernel = kern.create_gaussian_node_kernel(0.1, "weight")
    >>> knode = utils.create_full_node_affinity_matrix(graphs, node_kernel)
    >>> res = mixer.mixer(knode, [2, 2, 3], 0.1, 100)
    >>> res @ res.T
    array([[1., 0., 0., 1., 0., 1., 0.],
           [0., 1., 1., 0., 0., 0., 1.],
           [0., 1., 1., 0., 0., 0., 1.],
           [1., 0., 0., 1., 0., 1., 0.],
           [0., 0., 0., 0., 1., 0., 0.],
           [1., 0., 0., 1., 0., 1., 0.],
           [0., 1., 1., 0., 0., 0., 1.]])
    """
    aff = 1.0 - 2.0 * knode
    _, eigv = np.linalg.eigh(aff)
    u = probability_simplex_projector(eigv.T)

    po = np.ones(knode.shape) - np.identity(knode.shape[0])
    pd = np.zeros(knode.shape)
    cum_idx = 0
    for size in sizes:
        pd[cum_idx : cum_idx + size, cum_idx : cum_idx + size] = 1
        pd[cum_idx : cum_idx + size, cum_idx : cum_idx + size] -= np.identity(size)
        cum_idx += size

    # div = u @ po + pd @ u
    # not_null_idx = (div.flat > 0) & (u.flat > 0)
    # d = np.median(-aff.flat[not_null_idx] / div.flat[not_null_idx])
    # if d < 0:
    #     d *= -1
    d = 0.01

    while d < knode.shape[0] + 1:
        current_value = _objective_function(u, knode, po, pd, d)

        for ite in range(iterations):
            grad = 2.0 * aff @ u + 2 * d * (
                u @ (po + np.random.uniform(0, 0.1, size=po.shape))
                + (pd + np.random.uniform(0, 0.1, size=pd.shape)) @ u
            )

            step *= 2.0
            u_new = probability_simplex_projector(u - step * grad)
            new_value = _objective_function(u_new, knode, po, pd, d)
            while new_value > current_value:
                step /= 2
                u_new = probability_simplex_projector(u - step * grad)
                new_value = _objective_function(u_new, knode, po, pd, d)
                if step < 1e-6:
                    break

            # Detect convergence
            if np.abs(current_value - new_value) < 1e-12:
                break

            current_value = new_value
            u = u_new

        d *= 2.0

        # Check orthogonality and distinctiveness
        if np.trace((u.T @ u) @ po) < 1e-12 and np.trace((u @ u.T) @ pd) < 1e-12:
            break

    return u
