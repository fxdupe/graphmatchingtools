"""
This code is from the paper
"Symmetric Sparse Boolean Matrix Factorization and Applications" (ITCS 2022) by
Chen, S., Song, Z., Tao, R., & Zhang, R.

.. moduleauthor:: François-Xavier Dupé
"""
import numpy as np


def _factorial(n: int) -> int:
    """Compute n!

    :param int n: the input number.
    :return: n!
    """
    res = 1
    for idx in range(1, n + 1):
        res = res * idx
    return res


def _get_tensor_value(
    x: np.ndarray, position: tuple[int, int, int], rank: int, sparsity: int
) -> float:
    """Get the value of the tensor at a given position

    :param np.ndarray x: the input boolean square matrix.
    :param tuple[int, int, int] position: the position of the value.
    :param int rank: the rank of the factor matrix.
    :param int sparsity: the sparsity of the lines.
    :return: T[a, b, c].
    :rtype: float
    """
    # Get mu_abc
    count = 0
    a, b, c = position
    for idx in range(x.shape[0]):
        if x[a, idx] == 0 and x[b, idx] == 0 and x[c, idx] == 0:
            count += 1

    mu_abc = count / x.shape[0]

    # Compute t_abc
    value = 1e99
    t_abc = 0
    for i_t in range(rank - sparsity + 1):
        mu_t = (
            _factorial(rank - i_t)
            * _factorial(rank - sparsity)
            / (_factorial(rank) * _factorial(rank - i_t - sparsity))
        )
        distance = np.abs(mu_t - mu_abc)
        if distance < value:
            t_abc = i_t
            value = distance

    return t_abc


def _compute_intermediate_matrix(
    x: np.ndarray, v1: np.ndarray, v2: np.array, rank: int, sparsity: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the intermediate matrices Ma and Mb

    :param np.ndarray x: the input boolean square matrix.
    :param np.ndarray v1: a random vector on the unit sphere.
    :param np.ndarray v2: a random vector on the unit sphere.
    :param int rank: the rank of the factor matrix.
    :param int sparsity: the sparsity of the lines of the factor matrix.
    :return: a tuple with Ma and Mb.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    m1 = np.zeros(x.shape)
    m2 = np.zeros(x.shape)

    for i_a in range(x.shape[0]):
        for i_b in range(x.shape[0]):
            t_ab = _get_tensor_value(x, (i_a, i_b, i_b), rank, sparsity)
            for i_c in range(x.shape[0]):
                t_ac = _get_tensor_value(x, (i_a, i_c, i_c), rank, sparsity)
                t_bc = _get_tensor_value(x, (i_b, i_c, i_c), rank, sparsity)
                t_abc = _get_tensor_value(x, (i_a, i_b, i_c), rank, sparsity)

                tensor_abc = t_abc - t_ab - t_ac - t_bc + 3 * sparsity
                m1[i_a, i_b] += tensor_abc * v1[i_c]
                m2[i_a, i_b] += tensor_abc * v2[i_c]

    return m1, m2


def boolean_nmf(x: np.ndarray, rank: int, sparsity: int) -> np.ndarray:
    """Compute the boolean NMF for a given rank and sparsity.

    :param np.ndarray x: the input boolean square matrix.
    :param int rank: the rank of the factor matrix.
    :param int sparsity: the sparsity of the lines of the factor matrix.
    :return: the factor matrix.
    :rtype: np.ndarray

    Here an example:

    .. doctest:

    >>> perm = np.array(
    ...         [
    ...             [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    ...             [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    ...             [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    ...             [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    ...             [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ...             [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    ...             [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    ...          ]
    ...     )
    >>> u = nmf.boolean_nmf(perm, 3, 1)
    """
    # The two random vectors on the unit sphere needed for the approximation
    v1 = np.random.randn(x.shape[0])
    v2 = np.random.randn(x.shape[0])
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    # We now compute the two matrices M1 and M2
    m1, m2 = _compute_intermediate_matrix(x, v1, v2, rank, sparsity)

    # Compute the eigenvectors
    sigma, w = np.linalg.eig(m1 @ np.linalg.pinv(m2))
    s_idx = np.argsort(np.abs(sigma))
    w_r = w[:, s_idx[x.shape[0] - rank : x.shape[0]]]

    # Get the "closest" boolean vectors
    res = np.zeros(w_r.shape)
    for line in range(x.shape[0]):
        i_max = np.argmax(np.squeeze(np.abs(w_r[line, :])))
        res[line, i_max] = 1

    return res
