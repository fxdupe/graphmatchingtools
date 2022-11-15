"""
This code is directly from the papers
"Robust Multi-Object Matching via Iterative Reweighting of the Graph Connection Laplacian" (NeurIPS 2020)

.. moduleauthor:: François-Xavier Dupé
"""
import numpy as np

import graph_matching_tools.algorithms.multiway.stiefel as stiefel


def beta_t(t):
    """Example of beta(t)

    :param int t: the iteration number
    :return: beta at t
    """
    return np.minimum(2 ** t, 40)


def alpha_t(t):
    """Example of alpha(t)

    :param int t: the iteration number
    :return: alpha at t
    """
    return np.minimum(1.2 ** (t-1), 40)


def lambda_t(t):
    """Example of lambda(t)

    :param int t: the iteration number
    :return: lambda at t
    """
    return t / (t+1)


def block_scalar_product(x, y, node_number, graph_number):
    """Block scalar product (one scalar by graph)

    :param np.ndarray x: the left matrix
    :param np.ndarray y: the right matrix
    :param int node_number: the number of nodes
    :param int graph_number: the number of graphs
    :return: the matrix with the scalar product for each block
    """
    res = np.zeros((graph_number, graph_number))
    index1 = 0
    for g1 in range(graph_number):
        index2 = 0
        for g2 in range(graph_number):
            res[g1, g2] = np.trace(x[index1:index1 + node_number, index2:index2 + node_number].T @
                                   y[index1:index1 + node_number, index2:index2 + node_number])
            index2 += node_number
        index1 += node_number

    return res


def cemp(x, adj, t0, beta, node_number, graph_number):
    """CEMP algorithm

    :param np.ndarray x: the initial bulk permutation matrix
    :param np.ndarray adj: the adjency matrix between the graph
    :param int t0: the number of iterations
    :param callable beta: the increasing beta(t)
    :param int node_number: the number of nodes
    :param int graph_number: the number of graphs
    :return: the initial weight matrix
    """
    one_m = np.ones((node_number, node_number))
    w_init = adj
    a_init = None
    for t in range(t0):
        s_init = np.kron(w_init, one_m) * x
        tmp = 1/node_number * (s_init @ s_init) / (np.kron(w_init @ w_init, one_m) + 1e-3)  # Avoid division by 0
        a_init = block_scalar_product(tmp, x, node_number, graph_number)
        w_init = np.exp(beta(t) * a_init)

    return a_init


def irgcl(x, beta, alpha, lbd, node_number, graph_number, t0=5, t_max=100, choice=None):
    """The Iteratively Reweighted Graph Connection Laplacian (IRGCL)

    :param np.ndarray x: the estimated bulk permutation matrix
    :param callable beta: the increasing beta(t) for initialization
    :param callable alpha: the increasing alpha(t)
    :param callable lbd: the increasing lambda(t) for the equilibrium at update step
    :param int node_number: the number of node inside one graph (also the rank)
    :param int graph_number: the number of graphs
    :param int t0: number of iterations for initialization
    :param int t_max: maximal number of iterations
    :param callable choice: the choosing function for the reference graph
    :return:
    """
    if choice is None:
        rng = np.random.default_rng()  # Use default random generator
        choice = rng.choice

    one_m = np.ones((node_number, node_number))
    adj = np.ones((graph_number, graph_number)) - np.eye(graph_number)
    a_t = cemp(x, adj, 100, beta, node_number, graph_number)
    w_t = a_t
    # p_t = mwk.u_rank_projector(np.kron(w_t, one_m) * x, [node_number, ] * graph_number, node_number, choice)
    p_t = stiefel.sparse_stiefel_manifold_sync(np.kron(w_t, one_m) * x, node_number, [node_number, ] * graph_number)
    for t in range(t_max):
        x_t = p_t @ p_t.T
        a1_t = block_scalar_product(x_t, x, node_number, graph_number) / graph_number
        w1_t = np.exp(alpha(t) * a1_t)
        s_t = np.kron(w1_t, one_m) * x
        a2_t = block_scalar_product((s_t @ s_t) / (np.kron(w1_t @ w1_t, one_m)), x, node_number, graph_number)
        a_t = (1 - lbd(t)) * a1_t + lbd(t) * a2_t
        w_t = a_t
        # p_tt = mwk.u_rank_projector(np.kron(w_t, one_m) * x, [node_number, ] * graph_number, node_number, choice)
        p_tt = stiefel.sparse_stiefel_manifold_sync(np.kron(w_t, one_m) * x, node_number,
                                                    [node_number, ] * graph_number)
        if np.linalg.norm(p_tt - p_t) < 1e-3:
            break
        pt = p_tt

    return p_t
