"""Generate multivariate von Mises Fisher samples.
This solution originally appears here:
http://stats.stackexchange.com/questions/156729/sampling-from-von-mises-fisher-distribution-in-python

Also see:
Sampling from vMF on :math:`S^2`:
https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
http://www.stat.pitt.edu/sungkyu/software/randvonMisesFisher3.pdf

This code was taken from the following project: https://github.com/clara-labs/spherecluster

.. moduleauthor:: Rohit Yadav
"""

import numpy as np


def sample_von_mises(mu: np.ndarray, kappa: float, num_samples: int) -> np.ndarray:
    r"""Generate N-dimensional samples from von Mises Fisher
    distribution around center :math:`\mu \in R^N` with concentration :math:`\kappa`.

    :param np.ndarray mu: the mean of the distribution.
    :param float kappa: parameter of vMF distribution.
    :param int num_samples: number of samples to generate.
    :return: N-dimensional samples.
    :rtype: np.ndarray
    """
    dim = mu.shape[0]
    result = np.zeros((num_samples, dim))
    for nn in range(num_samples):
        # sample offset from center (on sphere) with spread kappa
        w = _sample_weight(kappa, dim)

        # sample a point v on the unit sphere that's orthogonal to mu
        v = _sample_orthonormal(mu)

        # compute new point
        result[nn, :] = v * np.sqrt(1.0 - w**2) + w * mu

    return result


def _sample_weight(kappa: float, dim: int) -> np.ndarray:
    """Rejection sampling scheme for sampling distance from center on surface of the sphere.

    :param float kappa: parameter of vMF distribution.
    :param int dim: dimensionality of samples.
    :return: N-dimensional samples.
    :rtype: np.ndarray
    """
    dim = dim - 1  # since S^{n-1}
    b = dim / (np.sqrt(4.0 * kappa**2 + dim**2) + 2 * kappa)
    x = (1.0 - b) / (1.0 + b)
    c = kappa * x + dim * np.log(1 - x**2)

    while True:
        z = np.random.beta(dim / 2.0, dim / 2.0)
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        u = np.random.uniform(low=0, high=1)
        if kappa * w + dim * np.log(1.0 - x * w) - c >= np.log(u):
            return w


def _sample_orthonormal(mu: np.ndarray) -> np.ndarray:
    r"""Sample point on sphere orthogonal to :math:`\mu`.

    :param np.ndarray mu: the mean of the distribution.
    :return: N-dimensional point.
    :rtype: np.ndarray
    """
    v = np.random.randn(mu.shape[0])
    proj_mu_v = mu * np.dot(mu, v) / np.linalg.norm(mu)
    orthto = v - proj_mu_v
    return orthto / np.linalg.norm(orthto)
