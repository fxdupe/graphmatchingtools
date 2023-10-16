"""
This code is directly from the paper
Lusk, P. C., Fathian, K., & How, J. P. (2023). MIXER: Multiattribute, Multiway Fusion of Uncertain Pairwise Affinities.
IEEE Robotics and Automation Letters, 8(5), 2462-2469.

.. moduleauthor:: François-Xavier Dupé
"""
import numpy as np


def probability_simplex_projector(x: np.ndarray) -> np.ndarray:
    """Projector onto the probability simplex (vector version)

    :param np.ndarray x: the vector to project.
    :return: the projection.
    :rtype: np.ndarray
    """
    u = np.sort(x)[::-1]

    rho = 0
    cumul = 0.0
    optim = 0.0
    for i in range(u.shape[0]):
        cumul += u[i]
        tmp = u[i] + 1.0 / (i + 1) * (1.0 - cumul)
        if tmp > 0:
            optim = cumul
            rho = i

    lbd = 1.0 / (rho + 1) * (1 - optim)
    res = np.maximum(x + lbd, 0)
    return res


def line_matrix_projector(x: np.ndarray) -> np.ndarray:
    """Project each line of a matrix onto the probability simplex

    :param np.ndarray x: the input matrix.
    :return: the projection.
    :rtype: np.ndarray
    """
    res = np.zeros(x.shape)
    for i_line in range(x.shape[0]):
        res[i_line, :] = probability_simplex_projector(x[i_line, :])
    return res


def mixer(
    knode: np.ndarray, sizes: list[int], step: float, iterations: int
) -> np.ndarray:
    """MIXER method for attributes alignment

    :param np.ndarray knode: the attributes/nodes affinity matrix.
    :param list[int] sizes: the size of the different elements.
    :param float step: the step of the gradient descent.
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
    u = line_matrix_projector(eigv.T)

    po = np.ones(knode.shape) - np.identity(knode.shape[0])
    pd = np.zeros(knode.shape)
    cum_idx = 0
    for size in sizes:
        pd[cum_idx : cum_idx + size, cum_idx : cum_idx + size] = 1
        pd[cum_idx : cum_idx + size, cum_idx : cum_idx + size] -= np.identity(size)
        cum_idx += size

    div = u @ po + pd @ u
    not_null_idx = (div.flat > 0) & (u.flat > 0)
    d = np.median(-aff.flat[not_null_idx] / div.flat[not_null_idx])
    if d < 0:
        d *= -1

    while d < knode.shape[0] + 1:
        for ite in range(iterations):
            grad = 2.0 * aff @ u + 2 * d * (
                u @ (po + np.random.uniform(0, 0.1, size=po.shape))
                + (pd + np.random.uniform(0, 0.1, size=pd.shape)) @ u
            )
            u = line_matrix_projector(u - step * grad)
        d *= 2.0

        # Check orthogonality and distinctiveness
        if np.trace((u.T @ u).T @ po) < 1e-6 and np.trace((u @ u.T).T @ pd) < 1e-6:
            break

    return u
