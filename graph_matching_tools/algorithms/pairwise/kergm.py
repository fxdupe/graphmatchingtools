"""This module contains the matching algorithm between a pair of graph.

Implementation of KerGM for graph matching using both edges and nodes data, from the paper
Zhang, Z., Xiang, Y., Wu, L., Xue, B., & Nehorai, A. (2019). KerGM: Kernelized graph matching. NeurIPS 2019.

.. moduleauthor:: François-Xavier Dupé
"""
import numpy as np
import scipy.optimize as sco
import networkx as nx


def _compute_axb(g1, h1, g2, h2, k):
    """Compute the AXB (see paper) formula.

    :param np.ndarray g1: the head matrix of the first graph.
    :param np.ndarray h1: the tail matrix of the first graph.
    :param np.ndarray g2: the head matrix of the second graph.
    :param np.ndarray h2: the tail matrix of the second graph.
    :param np.ndarray k: the kernel matrix between the nodes of the first graph (lines) against the second's (columns).
    :return: the AXB product.
    """
    return h1 @ (g1.T @ g2 * k) @ h2.T + h1 @ (g1.T @ h2 * k) @ g2.T +\
        g1 @ (h1.T @ g2 * k) @ h2.T + g1 @ (h1.T @ h1 * k) @ g2.T


def _compute_edge_kernel(graph1, graph2, kernel):
    """Inner function for computing part of the Gram matrix between edges.

    :param nx.classes.graph.Graph graph1: the first graph.
    :param nx.classes.graph.Graph graph2: the second graph.
    :param callable kernel: the kernel function between edges.
    :return: the affinity/Gram matrix between the edges.
    """
    k12 = np.zeros((nx.number_of_edges(graph1), nx.number_of_edges(graph2)))
    for ite1 in range(nx.number_of_edges(graph1)):
        for ite2 in range(nx.number_of_edges(graph2)):
            if ite2 < ite1:
                continue
            k12[ite1, ite2] = kernel(graph1, graph2, ite1, ite2)
            k12[ite2, ite1] = k12[ite1, ite2]

    return k12


def create_gradient(graph1, graph2, kernel, knode):
    """Compute the gradient for the Frank-Wolfe minimization (exact version).
    The *kernel* function must take four arguments: the two graphs followed by the index of their respective edges.
    For example: kernel(g1, g2, e1, e2) with g1, g2 are the two graphs, e1 is the index of one edge in g1 and e2 an
    edge of g2.

    :param nx.classes.graph.Graph graph1: the first graph.
    :param nx.classes.graph.Graph graph2: the second graph.
    :param callable kernel: the kernel function between two edges.
    :param np.ndarray knode: the node affinity matrix.
    :return: the corresponding gradient.
    """
    g1 = np.zeros((nx.number_of_nodes(graph1), nx.number_of_edges(graph1)))
    g2 = np.zeros((nx.number_of_nodes(graph2), nx.number_of_edges(graph2)))
    h1 = np.zeros(g1.shape)
    h2 = np.zeros(g2.shape)

    # Get head and tail matrices
    counter = 0
    for e in graph1.edges:
        g1[e[0], counter] = 1
        h1[e[1], counter] = 1
        counter += 1

    counter = 0
    for e in graph2.edges:
        g2[e[0], counter] = 1
        h2[e[1], counter] = 1
        counter += 1

    # Compute the kernel matrices
    k11 = _compute_edge_kernel(graph1, graph1, kernel)
    k22 = _compute_edge_kernel(graph2, graph2, kernel)
    k12 = _compute_edge_kernel(graph1, graph2, kernel)

    sphi1 = _compute_axb(g1, h1, g1, h1, k11)
    sphi2 = _compute_axb(g2, h2, g2, h2, k22)

    def gradient(x, alpha=0.0):
        """Compute the gradient at a given point.

        :param np.ndarray x: the current *permutation* matrix.
        :param float alpha: the regularization hyperparameter.
        :return: the gradient at point x.
        """
        temp = h1 @ (g1.T @ x @ g2 * k12) @ h2.T + h1 @ (g1.T @ x @ h2 * k12) @ g2.T +\
            g1 @ (h1.T @ x @ g2 * k12) @ h2.T + g1 @ (h1.T @ x @ h1 * k12) @ g2.T
        grad = (1 - 2 * alpha) * (sphi1 @ x + x @ sphi2) - 2 * temp - knode
        return grad

    return gradient


def create_fast_gradient(phi1, phi2, knode):
    """Compute the gradient for the Frank-Wolfe minimization (fast version).

    :param np.ndarray phi1: the data matrix for the edges of the first graph (stack on the first index).
    :param np.ndarray phi2: the data matrix for the edges of the second graph (stack on the first index).
    :param np.ndarray knode: the node affinity matrix.
    :return: the corresponding gradient.
    """
    sphi1 = np.sum(phi1 @ phi1, axis=0)
    sphi2 = np.sum(phi2 @ phi2, axis=0)

    def gradient(x, alpha=0.0):
        """Compute the gradient at a given point.

        :param np.ndarray x: the current *permutation* matrix.
        :param float alpha: the regularization hyperparameter.
        :return: the gradient at point x.
        """
        sphi12 = np.sum((phi1 @ x) @ phi2, axis=0)
        grad = (1 - 2*alpha) * (sphi1 @ x + x @ sphi2) - 2 * sphi12 - knode
        return grad
    return gradient


def q_value(x, y, grad):
    """Compute the value of the function Q (for convergence testing).

    :param np.ndarray x: the current permutation matrix.
    :param np.ndarray y: the new permutation matrix.
    :param np.ndarray grad: the gradient at y-x.
    :return: the value of Q(x, y).
    """
    qval = 0.5 * np.sum(grad * (y - x))
    return qval


def gap_value(x, x_grad, y, gamma, epsilon=3e-16):
    """Compute the value of the gap (for convergence testing).

    :param np.ndarray x: the current permutation matrix.
    :param np.ndarray x_grad: the gradient at x.
    :param np.ndarray y: the new permutation matrix.
    :param float gamma: the weighted of the entropy term.
    :param float epsilon: the float precision (to avoid overflow).
    :return: the value of the gap between x and y.
    """
    x_part = np.sum(x_grad * x) + gamma * np.sum(x * np.log(x + epsilon))
    y_part = np.sum(x_grad * y) + gamma * np.sum(y * np.log(y + epsilon))
    return x_part - y_part


def sinkhorn_method(x, mu_s=None, mu_t=None, gamma=1.0, tolerance=1e-6, iterations=10000):
    """Sinkhorn-Knopp algorithm as proposed by M. Cuturi.

    :param np.ndarray x: the input affinity matrix.
    :param float gamma: the weight of the entropy term.
    :param np.ndarray mu_s: the initial probability for source distribution (uniform by default).
    :param np.ndarray mu_t: the initial probability for target distribution (uniform by default).
    :param float tolerance: the tolerance for convergence (default: 1e-6).
    :param int iterations: the maximum number of iterations (default: 10000).
    :return: the approximate optimal transport from one side to another.
    """
    u = np.ones(x.shape[0])
    v = np.ones(x.shape[1])
    if mu_s is None:
        mu_s = np.ones(x.shape[0]) / x.shape[0]
    if mu_t is None:
        mu_t = np.ones(x.shape[1]) / x.shape[1]
    c = np.exp(- x / gamma)
    for iteration in range(iterations):
        v = mu_t / (c.T @ u)
        unew = mu_s / (c @ v)

        u_norm = np.linalg.norm(u)
        if u_norm > 1e-3:
            error = np.linalg.norm(unew - u) / u_norm
        else:
            error = np.inf
        u = unew
        if error < tolerance:
            break
    res = np.diag(u.flat) @ c @ np.diag(v.flat)
    return res


def kergm_fw_method(gradient, init, alpha, entropy_gamma=0.005, iterations=1000, tolerance=1e-8,
                    inner_iterations=10000, inner_tolerance=1e-6, epsilon=3e-6):
    """The Frank-Wolfe method to solve the assignment problem.

    :param callable gradient: the gradient function
    :param np.ndarray init: the assignment initialization
    :param float alpha: the equilibrium between convex and concave terms
    :param float entropy_gamma: the weight of the entropy term
    :param int iterations: the maximal number of iterations (for global convergence)
    :param float tolerance: the tolerance for the global convergence
    :param int inner_iterations: the maximal number of iterations (for inner convergence)
    :param float inner_tolerance: the tolerance for the inner problem
    :param float epsilon: the float precision
    :return: an affinity assignment matrix
    """
    xt = init
    # nodes = np.ones(init.shape[0]) / init.shape[0]

    for iteration in range(iterations):
        grad = gradient(xt, alpha)
        yt = sinkhorn_method(grad, gamma=entropy_gamma, tolerance=inner_tolerance, iterations=inner_iterations)
        gt = gap_value(xt, grad, yt, entropy_gamma, epsilon=epsilon)
        if np.abs(gt) < epsilon:
            break
        qt = q_value(xt, yt, gradient(yt - xt, alpha))
        st = 1
        if qt > 0:
            st = np.min([gt/(2*qt), 1])
        xtt = xt + st * (yt - xt)
        error = np.linalg.norm(xtt - xt) / np.linalg.norm(xt)
        xt = xtt
        if error < tolerance:
            break

    return xt


def kergm_method(gradient, number_of_nodes, num_alpha=10, entropy_gamma=0.005, iterations=100, tolerance=1e-8,
                 inner_iterations=10000, inner_tolerance=1e-6, epsilon=3e-16) -> tuple[np.ndarray, np.ndarray]:
    """
    Graph assignment method KerGM.

    :param Callable[[np.ndarray], np.ndarray] gradient: the gradient function (build using one of the two methods)
    :param int | tuple[int] number_of_nodes: the number of nodes of both graphs (tuple or scalar)
    :param int num_alpha: the number of values for alpha
    :param float entropy_gamma: the weight of the entropy term
    :param int iterations: the maximal number of iterations (for global convergence)
    :param float tolerance: the tolerance for the global convergence
    :param int inner_iterations: the maximal number of iterations (for inner convergence)
    :param float inner_tolerance: the tolerance for the inner problem
    :param float epsilon: the float precision
    :return: a tuple with the row and column assignment
    """
    alphas = np.linspace(0, 1, num_alpha)
    if np.isscalar(number_of_nodes):
        x_alpha = np.ones((number_of_nodes, number_of_nodes)) / number_of_nodes +\
                  1e-3 * np.random.randn(number_of_nodes, number_of_nodes)  # Initialization
    else:
        x_alpha = np.ones(number_of_nodes) / number_of_nodes[0] +\
                  1e-3 * np.random.randn(number_of_nodes[0], number_of_nodes[1])

    for alpha in alphas:
        x_alpha = kergm_fw_method(gradient, x_alpha, alpha, entropy_gamma, iterations, tolerance,
                                  inner_iterations, inner_tolerance, epsilon)
    r_ind, c_ind = sco.linear_sum_assignment(-x_alpha)

    return r_ind, c_ind
