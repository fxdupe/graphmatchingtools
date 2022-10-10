"""
Multigraph version of KerGM

As published in "Kernelized multi-graph matching", ACML 2022

.. moduleauthor:: François-Xavier Dupé
"""
import numpy as np

import graph_matching_tools.algorithms.multiway.matcheig as matcheig
import graph_matching_tools.algorithms.multiway.msync as msync
import graph_matching_tools.algorithms.multiway.irgcl as irgcl


def _iterative_rank_projection(x, rank, sizes, iterations=20, tolerance=1e-3):
    """
    Generalized power method for permutation matrix estimation
    :param np.ndarray x: the input bulk permutation matrix
    :param int rank: the dimension of the universe of nodes
    :param list[int] sizes: the size of the different graphs
    :param int iterations: the maximal number of iterations
    :param float tolerance: the tolerance for convergence
    :return: the projected permutation matrix
    """
    res = matcheig.matcheig(x, rank, sizes)
    for i in range(iterations):
        n_res = matcheig.matcheig(x @ res, rank, sizes)
        if np.linalg.norm(res - n_res) < tolerance:
            break
        res = n_res

    return res


def create_gradient(phi, knode):
    """
    Compute the gradient for the minimization
    :param np.ndarray phi: the data matrix for the edges of the graphs (stack on the first index)
    :param np.ndarray knode: the current node universe projection
    :return: the corresponding gradient
    """

    def gradient(x):
        """
        Compute the gradient at a given point

        :param np.ndarray x: the current bulk "permutation" matrix
        :return: the gradient at point x
        """
        t1 = np.sum(phi @ x @ phi, axis=0)
        grad = -knode - 2.0 * t1
        return grad

    return gradient


def _rank_projector(x, rank, sizes, method, choice=None):
    """
    Generalized rank projector method
    :param np.ndarray x: the matrix to project
    :param int rank: the size of the universe of nodes
    :param list[int] sizes: the size of the graphs
    :param str method: the projection method
    :param callable choice: the function for choosing the reference graph (when needed)
    :return: the projection
    """

    if method == "matcheig":
        res = matcheig.matcheig(x, rank, sizes)
    elif method == "msync":
        res = msync.msync(x, sizes, rank, choice)
        res = res @ res.T
    elif method == "irgcl":
        res = matcheig.matcheig(x, rank, sizes)
        res = irgcl.irgcl(res, irgcl.beta_t, irgcl.alpha_t, irgcl.lambda_t, rank, len(sizes), choice=choice)
        res = res @ res.T
    elif method == "gpow":
        res = _iterative_rank_projection(x, rank, sizes)
    else:
        res = None

    return res


def mkergm(gradient, sizes, u_dim, init, iterations=100, tolerance=1e-2,
           projection_method="matcheig", choice=None):
    """
    Multi-graph matching extension of KerGM
    :param callable gradient: the gradient function
    :param list sizes: the number of nodes of the different graphs (in order)
    :param int u_dim: the dimension of the universe of nodes
    :param int iterations: the maximal number of iterations
    :param float tolerance: the tolerance for convergence
    :param np.ndarray init: an initialization
    :param str projection_method: select the projection method for the rank constraints
    :param callable choice: a choosing function for the reference graph
    :return: the bulk permutation matrix
    """
    x = init
    x_old = init
    for i in range(iterations):
        g = -gradient(x)
        x_new = _rank_projector(g, u_dim, sizes, projection_method, choice)
        if np.linalg.norm(x - x_new) < tolerance:
            break
        # Loop detection
        if np.linalg.norm(x_new - x_old) < tolerance:
            break
        x_old = x
        x = x_new

    return x
