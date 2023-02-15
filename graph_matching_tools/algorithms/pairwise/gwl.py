"""This module contains the matching algorithm between a pair of graph

Implementation of Gromov-Wasserstein Learning method for graph matching, from the paper
Xu, H., Luo, D., Zha, H., & Carin, L. (2019, May). Gromov-wasserstein learning for graph matching and node embedding.
In International conference on machine learning (pp. 6932-6941). PMLR.

TODO: code need to be improved with a better gradient computation

.. moduleauthor:: François-Xavier Dupé
"""
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp


def _loss_function(
    cost_s: np.ndarray, cost_t: np.ndarray, transport: np.ndarray
) -> np.ndarray:
    """Loss function (here L2-norm).

    :param np.ndarray cost_s: the cost matrix for the "source" graph.
    :param np.ndarray cost_t: the cost matrix for the "target" graph.
    :param np.ndarray transport: the transport matrix between the two graphs.
    :return: the loss function for each element.
    :rtype: np.ndarray
    """
    res = np.zeros((cost_s.shape[0], cost_t.shape[0]))
    for j_s in range(cost_s.shape[0]):
        for j_t in range(cost_t.shape[0]):
            c_j_s = jnp.squeeze(cost_s[:, j_s])
            c_j_t = jnp.squeeze(cost_t[:, j_t])
            res[j_s, j_t] = jnp.sum(
                ((c_j_s[:, jnp.newaxis] - c_j_t[jnp.newaxis, :]) ** 2) * transport
            )
    return res


def _distance_matrix(x_s: np.ndarray, x_t: np.ndarray) -> jax.Array:
    """Distance matrix from node embeddings (normalized version).

    :param np.ndarray x_s: the embeddings from the source matrix.
    :param np.ndarray x_t: the embeddings from the target matrix.
    :return: the normalized distance matrix.
    :rtype: jax.Array
    """
    # n_s = np.linalg.norm(x_s, axis=1)
    n_s = jnp.sqrt(jnp.sum(x_s**2.0, axis=1))
    # n_t = np.linalg.norm(x_t, axis=1)
    n_t = jnp.sqrt(jnp.sum(x_t**2.0, axis=1))
    dist = 1.0 - jnp.diag(1.0 / n_s) @ x_s @ x_t.T @ jnp.diag(1.0 / n_t)
    return dist


def _gw_proximal_point_solver(
    cost_s: np.ndarray,
    cost_t: np.ndarray,
    mu_s: np.ndarray,
    mu_t: np.ndarray,
    x_s: np.ndarray,
    x_t: np.ndarray,
    alpha: float,
    gamma: float,
    outer_iterations: int,
    inner_iterations: int,
):
    """Proximal point method for Gromov-Wasserstein discrepancy.

    :param np.ndarray cost_s: the cost matrix for the "source" graph.
    :param np.ndarray cost_t: the cost matrix for the "target" graph.
    :param np.ndarray mu_s: the probabilities for source nodes.
    :param np.ndarray mu_t: the probabilities for target nodes.
    :param np.ndarray x_s: the embeddings of the source nodes.
    :param np.ndarray x_t: the embeddings of the target nodes.
    :param float alpha: the equilibrium between general loss and embeddings.
    :param float gamma: the descent step.
    :param int outer_iterations: the number of iterations for the outer (global) descent.
    :param int inner_iterations: the number of iterations of Sinkhorn-Knopp (inner).
    :return: the update transport.
    :rtype: np.ndarray
    """
    t_m = mu_s.reshape(-1, 1) @ mu_t.reshape(1, -1)
    a = mu_s
    b = mu_t
    for n in range(outer_iterations):
        cmn = (
            _loss_function(cost_s, cost_t, t_m)
            + alpha * _distance_matrix(x_s, x_t)
            + gamma
        )
        g = jnp.exp(-cmn / gamma) * t_m
        for j in range(inner_iterations):
            b = mu_t / (g.T @ a)
            a = mu_s / (g @ b)
        t_m = jnp.diag(a) @ g @ jnp.diag(b)
    return t_m


def _update_embeddings_gradient(
    params: dict[str, np.ndarray],
    alpha: float,
    beta: float,
    cost_s: np.ndarray,
    cost_t: np.ndarray,
    transport: np.ndarray,
    use_cross_cost: bool = False,
    cost_st: Optional[np.ndarray] = None,
) -> jax.Array:
    """Embedding Loss value computation for JAX system with dictionary.
    Embedding parameters are given through a dictionary with the following elements as keys,

    - "x_s": the current embedding for the source graph.
    - "x_t": the current embedding for the target graph.
    :param dict[str, np.ndarray] params: the dictionary with all the parameter.
    :param float alpha: equilibrium for the transport regularization.
    :param float beta: the equilibrium for the embeddings.
    :param np.ndarray cost_s: the cost matrix of the source graph.
    :param np.ndarray cost_t: the cost matrix of the target graph.
    :param np.ndarray transport: the current transport matrix.
    :param bool use_cross_cost: toggle to use the cross cost matrix if available.
    :param Optional[np.ndarray] cost_st: the cost matrix of the between the two graphs (|s| x |t|).
    :return: the loss value for given embeddings.
    :rtype: np.ndarray
    """
    dist_st = _distance_matrix(params["x_s"], params["x_t"])
    dist_ss = _distance_matrix(params["x_s"], params["x_s"])
    dist_tt = _distance_matrix(params["x_t"], params["x_t"])

    r_s = jnp.sum((dist_ss - cost_s) ** 2)
    r_t = jnp.sum((dist_tt - cost_t) ** 2)
    res = alpha * jnp.trace(dist_st.T @ transport) + beta * (r_s + r_t)

    if use_cross_cost:
        r_st = jnp.sum((dist_st - cost_st) ** 2)
        res += beta * r_st

    return res


def _update_embeddings(
    cost_s: np.ndarray,
    cost_t: np.ndarray,
    transport: np.ndarray,
    alpha: float,
    beta: float,
    node_dim: int,
    iterations: int,
    descent_step: float,
    random_generator: np.random.Generator,
    starting_embeddings: Optional[tuple[np.ndarray, np.ndarray]] = None,
    cost_st: Optional[np.ndarray] = None,
    use_cross_cost: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient descent for embedding update.

    :param np.ndarray cost_s: the cost matrix of the source graph.
    :param np.ndarray cost_t: the cost matrix of the target graph.
    :param np.ndarray transport: the transport between the two graphs.
    :param float alpha: the equilibrium for the transport regularization.
    :param float beta: the equilibrium for the embeddings.
    :param int node_dim: the size of the embedding space.
    :param int iterations: the number of iterations.
    :param float descent_step: the descent step.
    :param np.random.Generator random_generator: the random generator.
    :param Optional[tuple[np.ndarray, np.ndarray]] starting_embeddings: a tuple with the starting embeddings (random if None).
    :param Optional[np.ndarray] cost_st: the cost matrix of the between the two graphs (|s| x |t|).
    :param bool use_cross_cost: toggle to use the cross cost matrix if available.
    :return: the new embeddings for each graph.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    gradient = jax.grad(
        _update_embeddings_gradient, argnums=0
    )  # Only applied on the params
    # Random initialization
    if starting_embeddings is None:
        x_s = random_generator.uniform(0, 1.0, size=(cost_s.shape[0], node_dim))
        x_t = random_generator.uniform(0, 1.0, size=(cost_t.shape[0], node_dim))
    else:
        x_s = starting_embeddings[0]
        x_t = starting_embeddings[1]

    # Gradient descent (with jax magic)
    params = dict()
    for iteration in range(iterations):
        params["x_s"] = x_s
        params["x_t"] = x_t
        grad = gradient(
            params, alpha, beta, cost_s, cost_t, transport, use_cross_cost, cost_st
        )
        x_s = x_s - descent_step * grad["x_s"]
        x_t = x_t - descent_step * grad["x_t"]

    return x_s, x_t


def gromov_wasserstein_learning(
    cost_s: np.ndarray,
    cost_t: np.ndarray,
    mu_s: np.ndarray,
    mu_t: np.ndarray,
    beta: float,
    gamma: float,
    node_dim: int,
    outer_iterations: int,
    inner_iterations: int,
    embed_iterations: int,
    embed_step: float,
    cost_st: Optional[np.ndarray] = None,
    use_cross_cost: bool = False,
    random_seed: Optional[int] = None,
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """Gromov-Wasserstein Learning method for graph matching.

    :param np.ndarray cost_s: the cost matrix for the "source" graph.
    :param np.ndarray cost_t: the cost matrix for the "target" graph.
    :param np.ndarray mu_s: the probabilities for source nodes.
    :param np.ndarray mu_t: the probabilities for target nodes.
    :param float beta: embedding regularization equilibrium.
    :param float gamma: regularization equilibrium.
    :param int node_dim: the dimension of the node space.
    :param int outer_iterations: the number of outer iterations.
    :param int inner_iterations: the number of inner iterations.
    :param int embed_iterations: the number of iterations for the embedding update.
    :param float embed_step: the descent step of the embedding update.
    :param Optional[np.ndarray] cost_st: the cost matrix of the between the two graphs of size :math:`|source|\\times|target|`.
    :param bool use_cross_cost: toggle to use the cross cost matrix if available.
    :param Optional[int] random_seed: the seed for the random generator.
    :return: the probability matching matrix between the two graphs.
    :rtype: [np.ndarray, np.ndarray, np.ndarray]

    Note: the output is a probability matrix, thus high values lead to good match.

    Here an example using NetworkX and some utils:

    .. doctest:

    >>> node_kernel = kern.create_gaussian_node_kernel(1.0, "weight")
    >>> cost_s = 1.0 - utils.create_full_node_affinity_matrix([graph1, ], node_kernel)
    >>> cost_t = 1.0 - utils.create_full_node_affinity_matrix([graph2, ], node_kernel)
    >>> mu_s = np.ones((nx.number_of_nodes(graph1), )) / nx.number_of_nodes(graph1)
    >>> mu_t = np.ones((nx.number_of_nodes(graph2), )) / nx.number_of_nodes(graph2)
    >>> distance = 1.0 - utils.compute_knode(graph1, graph2, node_kernel)
    >>> t_m, x_s, x_t = gwl_pairwise.gromov_wasserstein_learning(cost_s, cost_t, mu_s, mu_t, 1.0, 1.0,
    ...     3, 20, 20, 20, 0.01, cost_st=distance, use_cross_cost=True, random_seed=1)
    >>> t_m
    array([[6.4347525e-17, 4.9999997e-01],
           [5.0000000e-01, 6.4452484e-17]], dtype=float32)
    """
    # Learning steps
    rng = np.random.default_rng(seed=random_seed)
    x_s = rng.uniform(0, 1.0, size=(cost_s.shape[0], node_dim))
    x_t = rng.uniform(0, 1.0, size=(cost_t.shape[0], node_dim))
    t_m = None
    for m in range(outer_iterations):
        alpha_m = m / outer_iterations
        t_m = _gw_proximal_point_solver(
            cost_s, cost_t, mu_s, mu_t, x_s, x_t, alpha_m, gamma, inner_iterations, 1
        )
        x_s, x_t = _update_embeddings(
            cost_s,
            cost_t,
            t_m,
            alpha_m,
            beta,
            node_dim,
            embed_iterations,
            embed_step,
            random_generator=rng,
            starting_embeddings=(x_s, x_t),
            cost_st=cost_st,
            use_cross_cost=use_cross_cost,
        )

    return np.array(t_m), np.array(x_s), np.array(x_t)
