"""This module contains tools to plotting and sampling points on a sphere tools.

..moduleauthor:: Marius Thorre, Rohit Yadav
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from graph_matching_tools.utils.von_mises import sample_von_mises


class vMFSample:
    """
    Class for 3D coordinates from the sample_von_mises function
    and kappa value.
    """

    def __init__(self, sample, kappa):
        """
        Store 3D coordinates from the sample_von_mises function
        and kappa value.

        :param np.ndarray sample: one sample.
        :param float kappa: parameter of vMF distribution.
        """
        tp = np.transpose(sample)
        self.x = tp[0]
        self.y = tp[1]
        self.z = tp[2]
        self.kappa = kappa
        self.sample = sample


def make_sphere(radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get mesh for a sphere of a given radius.
    :param float radius: the radius of the sphere.
    :return: tuple which contains the sphere coordinate
    :rtype: tuple
    """
    phi, theta = np.mgrid[0.0 : np.pi : 100j, 0.0 : 2.0 * np.pi : 100j]
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return x, y, z


def draw_sphere(ax, radius: float) -> None:  # pragma: no cover
    """Draw an empty sphere
    :param ax: axis where to plot sphere
    :param float radius: value of sphere radius
    """
    x, y, z = make_sphere(radius - 0.01)  # subtract a little so points show on sphere
    ax.plot_surface(
        x,
        y,
        z,
        rstride=1,
        cstride=1,
        color=sns.xkcd_rgb["light grey"],
        alpha=0.5,
        linewidth=0,
    )


def plot(data, radius: float = 1.0) -> None:  # pragma: no cover
    """Plot a sphere with samples superposed, if supplied.

    :param float radius: radius of the base sphere
    :param tuple data: list of sample objects
    """
    sns.set_style("dark")
    ax = plt.figure().add_subplot(projection="3d")

    draw_sphere(ax=ax, radius=radius)
    N = len(data)

    colors = [sns.color_palette("GnBu_d", N)[i] for i in reversed(range(N))]
    data_check = data[0]

    if type(data_check) is vMFSample:
        i = 0
        ax.scatter(
            data.x,
            data.y,
            data.z,
            s=50,
            alpha=0.7,
            label="$\\kappa = $" + str(data.kappa),
            color=colors[i],
        )
    else:
        print("Error: data type not recognised")

    ax.set_axis_off()
    ax.legend(bbox_to_anchor=[0.65, 0.75])


def sample_sphere(
    nb_sample: int,
    mu: np.ndarray,
    kappa: float,
):
    """Sample points on a spherical surface.
    :param int nb_sample: number of samples.
    :param Optional[np.ndarray] mu: parameter of vMF distribution.
    :param Optional[float] kappa: parameter of vMF distribution.
    """
    s = sample_von_mises(mu, kappa, nb_sample)
    vMFSample(s, kappa)
    return vMFSample(s, kappa)
