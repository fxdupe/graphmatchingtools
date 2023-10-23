"""
This code is directly from the paper
"GRASP: Scalable Graph Alignment by Spectral Corresponding Functions" by Hermanns et al (ACM Trans KDD, 2023).

.. moduleauthor:: François-Xavier Dupé
"""
from typing import Optional, Callable

import numpy as np
import networkx as nx
import scipy.optimize as sco

import graph_matching_tools.utils.manopt as manopt


def _create_gradient(
    phi: np.ndarray,
    psi: np.ndarray,
    f: np.ndarray,
    g: np.ndarray,
    lbd2: np.ndarray,
    mu: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create the gradient for the base alignment problem.

    :param np.ndarray phi: the eigenvectors of the source graph.
    :param np.ndarray psi: the eigenvectors of the target graph.
    :param np.ndarray f: the corresponding functions of the source graph.
    :param np.ndarray g: the corresponding functions of the target graph.
    :param np.ndarray lbd2: the eigenvalues of the target graph.
    :param float mu: the equilibrium between matching and data.
    :return: the corresponding gradient function.
    :rtype: Callable[[np.ndarray], np.ndarray]
    """
    a = np.ones((lbd2.shape[0], lbd2.shape[1])) - np.identity(lbd2.shape[0])

    def gradient(m) -> np.ndarray:
        """The gradient at a given point.

        :param np.ndarray m: the current permutation matrix.
        :return: the gradient at the given point.
        :rtype: np.ndarray
        """
        p1 = lbd2.T @ m @ a + lbd2 @ m @ a.T
        p2 = -2.0 * psi.T @ g @ f.T @ phi + 2.0 * psi.T @ g @ g.T @ psi @ m
        return p1 + mu * p2

    return gradient


def grasp(
    source: nx.Graph,
    target: nx.Graph,
    k: int,
    mu: float = 1.0,
    time_steps: Optional[list[float]] = None,
) -> np.ndarray:
    """GRASP: scalable alignment by spectral corresponding functions

    :param nx.Graph source: the source graph.
    :param nx.Graph target: the target graph.
    :param int k: the number of eigenvalues (dimension reduction).
    :param float mu: the equilibrium between matching and data.
    :param Optional[list[float]] time_steps: a list of time steps.
    :return: the permutation matrix.
    :rtype: np.ndarray

    Here an example using NetworkX and some utils:

    .. doctest:

    >>> graph4 = nx.Graph()
    >>> graph4.add_node(0, weight=2.0)
    >>> graph4.add_node(1, weight=20.0)
    >>> graph4.add_node(2, weight=5.0)
    >>> graph4.add_node(3, weight=6.0)
    >>> graph4.add_edge(0, 1, weight=1.0)
    >>> graph4.add_edge(1, 2, weight=1.0)
    >>> graph4.add_edge(2, 0, weight=1.0)
    >>> graph5 = nx.Graph()
    >>> graph5.add_node(0, weight=2.0)
    >>> graph5.add_node(1, weight=20.0)
    >>> graph5.add_node(2, weight=5.0)
    >>> graph5.add_node(3, weight=6.0)
    >>> graph5.add_edge(1, 2, weight=1.0)
    >>> graph5.add_edge(2, 3, weight=1.0)
    >>> graph5.add_edge(3, 1, weight=1.0)
    >>> res = grasp.grasp(graph4, graph5, 2)
    >>> res
    array([[0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.],
           [1., 0., 0., 0.]])
    """
    if time_steps is None:
        time_steps = np.linspace(0.1, 50)
    else:
        time_steps = np.array(time_steps)

    # Step 1: eigendecomposition of normalized Laplacian
    l1 = nx.normalized_laplacian_matrix(source).toarray()
    l2 = nx.normalized_laplacian_matrix(target).toarray()

    lbd1, phi = np.linalg.eigh(l1)
    lbd2, psi = np.linalg.eigh(l2)

    # Step 2: compute the corresponding functions
    f = np.zeros((l1.shape[0], len(time_steps)))
    g = np.zeros((l2.shape[0], len(time_steps)))

    for t_i in range(time_steps.shape[0]):
        fi = np.zeros((l1.shape[0],))
        gi = np.zeros((l2.shape[0],))

        for i in range(k):
            if lbd1[-(1 + i)] > 1e-5:
                fi += (
                    np.exp(-time_steps[t_i] / lbd1[-(1 + i)])
                    * phi[:, -(1 + i)]
                    * phi[:, -(1 + i)]
                )
            if lbd2[-(1 + i)] > 1e-5:
                gi += (
                    np.exp(-time_steps[t_i] / lbd2[-(1 + i)])
                    * psi[:, -(1 + i)]
                    * psi[:, -(1 + i)]
                )

        f[:, t_i] = fi
        g[:, t_i] = gi

    # Step 3: base alignment
    phi_bar = phi[:, : phi.shape[0] - k - 1 : -1]
    psi_bar = psi[:, : psi.shape[0] - k - 1 : -1]
    lbd2_bar = np.diag(lbd2[: lbd2.shape[0] - k - 1 : -1])

    grad_func = _create_gradient(phi_bar, psi_bar, f, g, lbd2_bar, mu)
    m = manopt.orthogonal_group_optimization(grad_func, k, epsilon=1e-4)

    psi_hat = psi_bar @ m

    # Step 4: calculate mapping matrix
    left_part = np.zeros((len(time_steps) * k, k))
    for i in range(len(time_steps)):
        left_part[i * k : (i + 1) * k, :] = np.diag(g[:, i] @ psi_hat)

    right_part = (phi_bar.T @ f).reshape(
        -1,
    )
    c_mapping = sco.lsq_linear(left_part, right_part)

    # Step 5: matching by linear assignment
    cost = phi_bar @ np.diag(c_mapping["x"]) @ psi_hat.T
    r, c = sco.linear_sum_assignment(-cost)

    res = np.zeros(cost.shape)
    for i in range(r.shape[0]):
        res[r[i], c[i]] = 1

    return res
