"""Example of using some barycenter computations.

    Most code is from Titouan Vayer <titouan.vayer@irisa.fr>

.. moduleauthor:: François-Xavier Dupé
"""

import argparse

import numpy as np
import matplotlib.pylab as pl
import ot
import math
import networkx as nx
from scipy.sparse.csgraph import shortest_path
import matplotlib.colors as mcol
import matplotlib.pyplot as plt
from matplotlib import cm

# import graph_matching_tools.solvers.ot.sns as sns
# import graph_matching_tools.algorithms.mean.wasserstein_barycenter as bary
import graph_matching_tools.algorithms.pairwise.fgw as fgw


def find_thresh(C, inf=0.5, sup=3, step=10):
    """Trick to find the adequate thresholds from where value of the C matrix are considered
    close enough to say that nodes are connected.
    The threshold is found by a linesearch between values "inf" and "sup" with "step" thresholds tested.
    The optimal threshold is the one which minimizes the reconstruction error between
    the shortest_path matrix coming from the thresholded adjacency matrix and the original matrix.

    :param ndarray C: The structure matrix to threshold.
    :param float inf: The beginning of the linesearch.
    :param float sup: The end of the linesearch.
    :param integer step: Number of thresholds tested.
    :return: a tuple with the search value and the distance
    :rtype: tuple
    """
    dist = []
    search = np.linspace(inf, sup, step)
    for thresh in search:
        Cprime = sp_to_adjacency(C, 0, thresh)
        SC = shortest_path(Cprime, method="D")
        SC[SC == float("inf")] = 100
        dist.append(np.linalg.norm(SC - C))
    return search[np.argmin(dist)], dist


def sp_to_adjacency(C, threshinf=0.2, threshsup=1.8):
    """Thresholds the structure matrix in order to compute an adjacency matrix.
    All values between threshinf and threshsup are considered representing connected nodes and set to 1. Else are set to 0

    :param np.ndarray C: The structure matrix to threshold.
    :param float threshinf: The minimum value of distance from which the new value is set to 1
    :param float threshsup: The maximum value of distance from which the new value is set to 1
    :return: The threshold matrix. Each element is in {0,1}
    :rtype: np.ndarray
    """
    H = np.zeros_like(C)
    np.fill_diagonal(H, np.diagonal(C))
    C = C - H
    C = np.minimum(np.maximum(C, threshinf), threshsup)
    C[C == threshsup] = 0
    C[C != 0] = 1

    return C


def build_noisy_circular_graph(
    N=20, mu=0, sigma=0.3, with_noise=False, structure_noise=False, p=None
):
    """Create a noisy circular graph"""
    g = nx.Graph()
    g.add_nodes_from(list(range(N)))
    for i in range(N):
        noise = float(np.random.normal(mu, sigma, 1))
        if with_noise:
            g.add_node(i, attr_name=math.sin((2 * i * math.pi / N)) + noise)
        else:
            g.add_node(i, attr_name=math.sin(2 * i * math.pi / N))
        g.add_edge(i, i + 1)
        if structure_noise:
            randomint = np.random.randint(0, p)
            if randomint == 0:
                if i <= N - 3:
                    g.add_edge(i, i + 2)
                if i == N - 2:
                    g.add_edge(i, 0)
                if i == N - 1:
                    g.add_edge(i, 1)
    g.add_edge(N, 0)
    noise = float(np.random.normal(mu, sigma, 1))
    if with_noise:
        g.add_node(N, attr_name=math.sin((2 * N * math.pi / N)) + noise)
    else:
        g.add_node(N, attr_name=math.sin(2 * N * math.pi / N))
    return g


def graph_colors(nx_graph, vmin=0, vmax=7):
    cnorm = mcol.Normalize(vmin=vmin, vmax=vmax)
    cpick = cm.ScalarMappable(norm=cnorm, cmap="viridis")
    cpick.set_array([])
    val_map = {}
    for k, v in nx.get_node_attributes(nx_graph, "attr_name").items():
        val_map[k] = cpick.to_rgba(v)
    colors = []
    for node in nx_graph.nodes():
        colors.append(val_map[node])
    return colors


def plot_barycenter_fgw():
    np.random.seed(30)
    X0 = []
    for k in range(9):
        X0.append(
            build_noisy_circular_graph(
                np.random.randint(15, 25), with_noise=True, structure_noise=True, p=3
            )
        )

    plt.figure(figsize=(8, 10))
    for i in range(len(X0)):
        plt.subplot(3, 3, i + 1)
        g = X0[i]
        pos = nx.kamada_kawai_layout(g)
        nx.draw(
            g,
            pos=pos,
            node_color=graph_colors(g, vmin=-1, vmax=1),
            with_labels=False,
            node_size=100,
        )
    plt.suptitle("Dataset of noisy graphs. Color indicates the label", fontsize=20)
    plt.show()

    # Cs = [shortest_path(nx.adjacency_matrix(x).todense()) for x in X0]
    # ps = [np.ones(len(x.nodes())) / len(x.nodes()) for x in X0]
    # Ys = [
    #     np.array([v for (k, v) in nx.get_node_attributes(x, "attr_name").items()]).reshape(
    #         -1, 1
    #     )
    #     for x in X0
    # ]
    # lambdas = np.array([np.ones(len(Ys)) / len(Ys)]).ravel()
    # sizebary = 15  # we choose a barycenter with 15 nodes

    # TODO: improve current code


def plot_fgw():

    # We create two 1D random measures
    n = 20  # number of points in the first distribution
    n2 = 30  # number of points in the second distribution
    sig = 1  # std of first distribution
    sig2 = 0.1  # std of second distribution

    np.random.seed(0)

    phi = np.arange(n)[:, None]
    xs = phi + sig * np.random.randn(n, 1)
    ys = np.vstack(
        (np.ones((n // 2, 1)), 0 * np.ones((n // 2, 1)))
    ) + sig2 * np.random.randn(n, 1)

    phi2 = np.arange(n2)[:, None]
    xt = phi2 + sig * np.random.randn(n2, 1)
    yt = np.vstack(
        (np.ones((n2 // 2, 1)), 0 * np.ones((n2 // 2, 1)))
    ) + sig2 * np.random.randn(n2, 1)
    yt = yt[::-1, :]

    p = ot.unif(n)
    q = ot.unif(n2)

    # plot the distributions

    pl.figure(1, (7, 7))

    pl.subplot(2, 1, 1)

    pl.scatter(ys, xs, c=phi, s=70)
    pl.ylabel("Feature value a", fontsize=20)
    pl.title("$\\mu=\\sum_i \\delta_{x_i,a_i}$", fontsize=25, y=1)
    pl.xticks(())
    pl.yticks(())
    pl.subplot(2, 1, 2)
    pl.scatter(yt, xt, c=phi2, s=70)
    pl.xlabel("coordinates x/y", fontsize=25)
    pl.ylabel("Feature value b", fontsize=20)
    pl.title("$\\nu=\\sum_j \\delta_{y_j,b_j}$", fontsize=25, y=1)
    pl.yticks(())
    pl.tight_layout()
    pl.show()

    # Structure matrices and across-features distance matrix
    C1 = ot.dist(xs)
    C2 = ot.dist(xt)
    M = ot.dist(ys, yt)
    Got = ot.emd([], [], M)

    cmap = "Reds"

    pl.figure(2, (5, 5))
    fs = 15
    l_x = [0, 5, 10, 15]
    l_y = [0, 5, 10, 15, 20, 25]
    gs = pl.GridSpec(5, 5)

    ax1 = pl.subplot(gs[3:, :2])

    pl.imshow(C1, cmap=cmap, interpolation="nearest")
    pl.title("$C_1$", fontsize=fs)
    pl.xlabel("$k$", fontsize=fs)
    pl.ylabel("$i$", fontsize=fs)
    pl.xticks(l_x)
    pl.yticks(l_x)

    ax2 = pl.subplot(gs[:3, 2:])

    pl.imshow(C2, cmap=cmap, interpolation="nearest")
    pl.title("$C_2$", fontsize=fs)
    pl.ylabel("$l$", fontsize=fs)
    pl.xticks(())
    pl.yticks(l_y)
    ax2.set_aspect("auto")

    ax3 = pl.subplot(gs[3:, 2:], sharex=ax2, sharey=ax1)
    pl.imshow(M, cmap=cmap, interpolation="nearest")
    pl.yticks(l_x)
    pl.xticks(l_y)
    pl.ylabel("$i$", fontsize=fs)
    pl.title("$M_{AB}$", fontsize=fs)
    pl.xlabel("$j$", fontsize=fs)
    pl.tight_layout()
    ax3.set_aspect("auto")
    pl.show()

    # Computing FGW
    alpha = 1e-3
    Gwg = fgw.fgw_direct_matching(
        C1,
        C2,
        p,
        q,
        M,
        alpha,
        10,
        gamma=0.4,
        rho=0.4,
        inner_iterations_step1=1000,
        inner_iterations_step2=100,
    )

    # visu OT matrix
    cmap = "Blues"
    fs = 15
    pl.figure(3, (13, 5))
    pl.clf()
    pl.subplot(1, 2, 1)
    pl.imshow(Got, cmap=cmap, interpolation="nearest")
    pl.ylabel("$i$", fontsize=fs)
    pl.xticks(())

    pl.title("Wasserstein ($M$ only)")

    pl.subplot(1, 2, 2)
    pl.imshow(Gwg, cmap=cmap, interpolation="nearest")
    pl.title("FGW  ($M+C_1,C_2$)")

    pl.xlabel("$j$", fontsize=fs)
    pl.ylabel("$i$", fontsize=fs)

    pl.tight_layout()
    pl.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Barycenter computation examples.")

    plot_fgw()
