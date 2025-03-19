"""Example of using some optimal transport methods.

.. moduleauthor:: François-Xavier Dupé
"""

import argparse

import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot

import graph_matching_tools.solvers.ot.sns as sns


def plot_ot_2d_sample():
    """
    Example from ttps://pythonot.github.io/auto_examples/plot_OT_2D_samples.html
    """
    n = 50  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    mu_t = np.array([4, 4])
    cov_t = np.array([[1, -0.8], [-0.8, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n, mu_s, cov_s)
    xt = ot.datasets.make_2D_samples_gauss(n, mu_t, cov_t)

    a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples

    # loss matrix
    M = ot.dist(xs, xt)

    pl.figure(1)
    pl.plot(xs[:, 0], xs[:, 1], "+b", label="Source samples")
    pl.plot(xt[:, 0], xt[:, 1], "xr", label="Target samples")
    pl.legend(loc=0)
    pl.title("Source and target distributions")

    pl.figure(2)
    pl.imshow(M, interpolation="nearest")
    pl.title("Cost matrix M")
    pl.show()

    # reg term
    lambd = 10.0
    Gs = sns.sinkhorn_newton_sparse_method(
        M, a.reshape((-1, 1)), b.reshape((-1, 1)), eta=lambd, rho=0.6
    )

    pl.figure(3)
    pl.imshow(Gs, interpolation="nearest")
    pl.title("OT matrix sinkhorn")

    pl.figure(4)
    ot.plot.plot2D_samples_mat(xs, xt, Gs, color=[0.5, 0.5, 1])
    pl.plot(xs[:, 0], xs[:, 1], "+b", label="Source samples")
    pl.plot(xt[:, 0], xt[:, 1], "xr", label="Target samples")
    pl.legend(loc=0)
    pl.title("OT matrix Sinkhorn with samples")

    pl.show()


def plot_optim_otreg():
    """
    Example from https://pythonot.github.io/auto_examples/plot_optim_OTreg.html
    """
    n = 100  # nb bins

    # bin positions
    x = np.arange(n, dtype=np.float64)

    # Gaussian distributions
    a = ot.datasets.make_1D_gauss(n, m=20, s=5)  # m= mean, s= std
    b = ot.datasets.make_1D_gauss(n, m=60, s=10)

    # loss matrix
    M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)))
    M /= M.max()

    lambd = 5000
    Gs = sns.sinkhorn_newton_sparse_method(
        M, a.reshape((-1, 1)), b.reshape((-1, 1)), eta=lambd, rho=0.2
    )

    pl.figure(5, figsize=(5, 5))
    ot.plot.plot1D_mat(a, b, Gs, "OT matrix Entrop. reg")
    pl.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimal Transport example.")

    plot_ot_2d_sample()
    plot_optim_otreg()
