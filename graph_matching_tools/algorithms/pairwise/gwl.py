"""This module contains the matching algorithm between a pair of graph

Implementation of Gromov-Wasserstein Learning method for graph matching, from the paper
Xu, H., Luo, D., Zha, H., & Duke, L. C. (2019, May). Gromov-wasserstein learning for graph matching and node embedding.
 In International conference on machine learning (pp. 6932-6941). PMLR.

.. moduleauthor:: François-Xavier Dupé
"""
import numpy as np
import jax
import jax.numpy as jnp


def _loss_function(cost_s, cost_t, transport):
    """Loss function (here L2-norm).

    :param cost_s: the cost matrix for the "source" graph.
    :param cost_t: the cost matrix for the "target" graph.
    :param transport: the transport matrix between the two graphs.
    :return: the loss function for each element.
    """
    res = np.zeros((cost_s.shape[0], cost_t.shape[0]))
    for j_s in range(cost_s.shape[0]):
        for j_t in range(cost_t.shape[0]):
            c_j_s = jnp.squeeze(cost_s[:, j_s])
            c_j_t = jnp.squeeze(cost_t[:, j_t])
            res[j_s, j_t] = jnp.sum(((c_j_s[:, jnp.newaxis] - c_j_t[jnp.newaxis, :]) ** 2) * transport)
    return res


def _distance_matrix(x_s, x_t):
    """Distance matrix from node embeddings (normalized version).

    :param x_s: the embeddings from the source matrix.
    :param x_t: the embeddings from the target matrix.
    :return: the normalized distance matrix.
    """
    # n_s = np.linalg.norm(x_s, axis=1)
    n_s = jnp.sqrt(jnp.sum(x_s ** 2.0, axis=1))
    # n_t = np.linalg.norm(x_t, axis=1)
    n_t = jnp.sqrt(jnp.sum(x_t ** 2.0, axis=1))
    dist = 1.0 - jnp.diag(1.0/n_s) @ x_s @ x_t.T @ jnp.diag(1.0/n_t)
    return dist


def _gw_proximal_point_solver(cost_s, cost_t, mu_s, mu_t, x_s, x_t, alpha, gamma, outer_iterations, inner_iterations):
    """Proximal point method for Gromov-Wasserstein discrepancy.

    :param cost_s: the cost matrix for the "source" graph.
    :param cost_t: the cost matrix for the "target" graph.
    :param mu_s: the probabilities for source nodes.
    :param mu_t: the probabilities for target nodes.
    :param x_s: the embeddings of the source nodes.
    :param x_t: the embeddings of the target nodes.
    :param alpha: the equilibrium between general loss and embeddings.
    :param gamma: the descent step.
    :param outer_iterations: the number of iterations for the outer (global) descent.
    :param inner_iterations: the number of iterations of Sinkhorn-Knopp (inner).
    :return: the update transport.
    """
    t_m = mu_s.reshape(-1, 1) @ mu_t.reshape(1, -1)
    a = mu_s
    b = mu_t
    for n in range(outer_iterations):
        cmn = _loss_function(cost_s, cost_t, t_m) + alpha * _distance_matrix(x_s, x_t) + gamma
        g = jnp.exp(-cmn / gamma) * t_m
        for j in range(inner_iterations):
            b = mu_t / (g.T @ a)
            a = mu_s / (g @ b)
        t_m = jnp.diag(a) @ g @ jnp.diag(b)
    return t_m


def _update_embeddings_gradient(params):
    """Embedding Loss value computation for JAX system with dictionary.
    All parameters are given through a dictionary with the following elements as keys,

    - x_s: the current embedding for the source graph.
    - x_t: the current embedding for the target graph.
    :param params: the dictionary with all the parameter.
    :return: the loss value for given embeddings.
    """
    dist_st = _distance_matrix(params["x_s"], params["x_t"])
    dist_ss = _distance_matrix(params["x_s"], params["x_s"])
    dist_tt = _distance_matrix(params["x_t"], params["x_t"])

    r_s = jnp.sum((dist_ss - params["cost_s"]) ** 2)
    r_t = jnp.sum((dist_tt - params["cost_t"]) ** 2)
    res = params["alpha"] * jnp.trace(dist_st.T @ params["transport"]) + params["beta"] * (r_s + r_t)
    return res


def _update_embeddings(cost_s, cost_t, transport, alpha, beta, node_dim, iterations, descent_step,
                       starting_embeddings=None):
    """Gradient descent for embedding update.

    :param cost_s: the cost matrix of the source graph.
    :param cost_t: the cost matrix of the target graph.
    :param transport: the transport between the two graphs.
    :param alpha: the equilibrium for the transport regularization.
    :param beta: the equilibrium for the embeddings.
    :param node_dim: the size of the embedding space.
    :param iterations: the number of iterations.
    :param descent_step: the descent step.
    :param starting_embeddings: a tuple with the starting embeddings (random if None)
    :return: the new embeddings for each graph.
    """
    gradient = jax.grad(_update_embeddings_gradient)
    # Random initialization
    if starting_embeddings is None:
        x_s = np.random.randn(cost_s.shape[0], node_dim)
        x_t = np.random.randn(cost_t.shape[0], node_dim)
    else:
        x_s = starting_embeddings[0]
        x_t = starting_embeddings[1]

    # Gradient descent (with jax magic)
    params = dict()
    params["alpha"] = alpha
    params["beta"] = beta
    params["cost_s"] = cost_s
    params["cost_t"] = cost_t
    params["transport"] = transport
    for iteration in range(iterations):
        params["x_s"] = x_s
        params["x_t"] = x_t
        grad = gradient(params)
        x_s = x_s - descent_step * grad["x_s"]
        x_t = x_t - descent_step * grad["x_t"]

    return x_s, x_t


def gromov_wasserstein_learning(cost_s, cost_t, mu_s, mu_t, beta, gamma, node_dim, outer_iterations,
                                inner_iterations, embed_iterations, embed_step):
    """Gromov-Wasserstein Learning method for graph matching.

    :param cost_s: the cost matrix for the "source" graph.
    :param cost_t: the cost matrix for the "target" graph.
    :param mu_s: the probabilities for source nodes.
    :param mu_t: the probabilities for target nodes.
    :param beta: embedding regularization equilibrium.
    :param gamma: regularization equilibrium.
    :param node_dim: the dimension of the node space.
    :param outer_iterations: the number of outer iterations.
    :param inner_iterations: the number of inner iterations.
    :param embed_iterations: the number of iterations for the embedding update.
    :param embed_step: the descent step of the embedding update.
    :return: a matching between the two graphs.
    """
    # Learning steps
    x_s = np.random.randn(cost_s.shape[0], node_dim)
    x_t = np.random.randn(cost_t.shape[0], node_dim)
    t_m = None
    for m in range(outer_iterations):
        alpha_m = m / outer_iterations
        t_m = _gw_proximal_point_solver(cost_s, cost_t, mu_s, mu_t, x_s, x_t, alpha_m, gamma, inner_iterations, 1)
        x_s, x_t = _update_embeddings(cost_s, cost_t, t_m, alpha_m, beta, node_dim, embed_iterations, embed_step,
                                      starting_embeddings=(x_s, x_t))

    # Matching steps
    matchs = np.zeros((cost_s.shape[0], )) - 1.0
    for i in range(t_m.shape[0]):
        matchs[i] = np.argmax(t_m[i, :])

    return matchs
