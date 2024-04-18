"""
Multigraph matching method proposed in
Wang, R., Yan, J., & Yang, X. (2020). Graduated Assignment for Joint Multi-Graph Matching and Clustering
with Application to Unsupervised Graph Matching Network Learning.
Advances in Neural Information Processing Systems, 33.

.. moduleauthor:: François-Xavier Dupé
"""

from typing import Optional

import numpy as np
import networkx as nx
import scipy.optimize as sco

# import ot

import graph_matching_tools.algorithms.pairwise.kergm as kergm
import graph_matching_tools.utils.utils as utils


def ga_mgmc(
    graphs: list[nx.Graph],
    aff_node: np.ndarray,
    u_dim: int,
    edge_data: str,
    mu: float = 0.5,
    tau_node: float = 10.0,
    tau: float = 10.0,
    tau_min: float = 1e-3,
    gamma: float = 0.8,
    sigma: float = 1.0,
    iterations: int = 200,
    init: Optional[np.ndarray] = None,
    normalize_aff: bool = False,
    inner_iterations: int = 10,
) -> np.ndarray:
    """Graduated assignment multi-graph matching (GA-MGM).

    :param list[nx.Graph] graphs: the list of graphs.
    :param np.ndarray aff_node: the affinity matrix between all the nodes.
    :param int u_dim: the dimension of the universe of nodes.
    :param str edge_data: the name of the data on edges.
    :param float mu: the equilibrium between the node affinities and the matching score.
    :param float tau_node: the entropy for the first node assignment using Sinkhorn (applied on parts of knode).
    :param float tau: the initial entropy for Sinkhorn.
    :param float tau_min: the minimal value for tau.
    :param float gamma: the descent factor.
    :param float sigma: the variance of the value on edge.
    :param int iterations: the maximal number of iterations.
    :param np.ndarray init: an initialization (optional).
    :param bool normalize_aff: True to normalize the affinity node matrix.
    :param int inner_iterations: The number of iterations for Sinkhorn.
    :return: the list the node projections (graph by graph).
    :rtype: np.ndarray

    Here an example using NetworkX and some utils:

    .. doctest:

    >>> node_kernel = kern.create_gaussian_node_kernel(2.0, "weight")
    >>> knode = utils.create_full_node_affinity_matrix(graphs, node_kernel)
    >>> u = ga_mgmc.ga_mgmc(graphs, knode, 3, "weight", tau=0.1, tau_min=1e-2)
    >>> u @ u.T
    array([[1., 0., 0., 1., 0., 1., 0.],
           [0., 1., 1., 0., 0., 0., 1.],
           [0., 1., 1., 0., 0., 0., 1.],
           [1., 0., 0., 1., 0., 1., 0.],
           [0., 0., 0., 0., 1., 0., 0.],
           [1., 0., 0., 1., 0., 1., 0.],
           [0., 1., 1., 0., 0., 0., 1.]])

    """
    rng = np.random.default_rng()
    # 0 - Initialization of the different matrices and lists
    a = utils.create_full_weight_matrix(graphs, edge_data, sigma=sigma)
    sizes = [nx.number_of_nodes(g) for g in graphs]

    if init is None:
        u = 1.0 / u_dim + rng.uniform(0, 1, size=(a.shape[0], u_dim))
    else:
        u = init
    v = np.zeros(u.shape)

    # 1 - Get the weights
    knode = aff_node
    if normalize_aff:
        knode = np.zeros(aff_node.shape)
        index_i = 0
        for i in range(len(sizes)):
            index_j = index_i + sizes[i]
            for j in range(i + 1, len(sizes)):
                weights = aff_node[
                    index_i : index_i + sizes[i], index_j : index_j + sizes[j]
                ]
                # res = ot.bregman.sinkhorn_log(
                #     np.ones(weights.shape[0]),
                #     np.ones(weights.shape[1]),
                #     weights,
                #     reg=tau_node,
                #     numItermax=inner_iterations,
                # )
                res = kergm.sinkhorn_method(
                    weights, gamma=tau_node, iterations=inner_iterations
                )
                knode[index_i : index_i + sizes[i], index_j : index_j + sizes[j]] = res
                knode[index_j : index_j + sizes[j], index_i : index_i + sizes[i]] = (
                    res.T
                )
                index_j += sizes[j]
            index_i += sizes[i]
        knode += np.identity(knode.shape[0])

    # 2 - The optimization loop
    final_stage = False
    while not final_stage:
        for iteration in range(iterations):
            # Update step
            v = mu * a @ u @ u.T @ a @ u + knode @ u

            # Projection step
            index = 0
            for i in range(len(sizes)):
                vi = v[index : index + sizes[i], :]
                # u[index : index + sizes[i], :] = ot.bregman.sinkhorn_log(
                #     np.ones(vi.shape[0]),
                #     np.ones(vi.shape[1]),
                #     vi,
                #     reg=tau,
                #     numItermax=inner_iterations,
                # )
                u[index : index + sizes[i], :] = kergm.sinkhorn_method(
                    vi, gamma=tau, iterations=inner_iterations
                )
                index += sizes[i]

        # Managing tau
        if not final_stage and tau >= tau_min:
            tau *= gamma
        elif not final_stage and tau < tau_min:
            final_stage = True
        else:
            break

    # Final step
    for iteration in range(iterations):
        v = mu * a @ u @ u.T @ a @ u + knode @ u
        u *= 0.0
        index = 0
        for i in range(len(sizes)):
            vi = v[index : index + sizes[i], :]
            r, c = sco.linear_sum_assignment(-vi)
            for j in range(r.shape[0]):
                u[index + r[j], c[j]] = 1
            index += sizes[i]

    return u
