"""This module contains tools to plotting and sampling points on a sphere tools.

..moduleauthor:: Marius Thorre, Rohit Yadav
"""

from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from graph_matching_tools.utils.von_mises import sample_von_mises


class Sphere:
    """
    A 3D sphere
    """

    def __init__(self):
        self.sphere = []
        self.samples = []
        self.x = None
        self.y = None
        self.z = None
        self.distribution = ["uniform", "vMF"]

    def make_sphere(self, radius: float) -> tuple:
        """
        Get mesh for a sphere of a given radius.
        :param float radius: the radius of the sphere.
        :return: tuple which contains the sphere coordinate
        :rtype: tuple
        """
        pi = np.pi
        cos = np.cos
        sin = np.sin
        phi, theta = np.mgrid[0.0:pi:100j, 0.0 : 2.0 * pi : 100j]
        self.x = radius * sin(phi) * cos(theta)
        self.y = radius * sin(phi) * sin(theta)
        self.z = radius * cos(phi)

        self.sphere = (self.x, self.y, self.z)
        return self.sphere

    def draw_sphere(self, ax, radius: float) -> None:
        """Draw an empty sphere
        :param ax: axis where to plot sphere
        :param float radius: value of sphere radius
        """
        x, y, z = self.make_sphere(
            radius - 0.01
        )  # subtract a little so points show on sphere
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

    def plot(self, radius: float = 1.0, data: tuple = None) -> None:
        """Plot a sphere with samples superposed, if supplied.

        :param float radius: radius of the base sphere
        :param tuple data: list of sample objects
        """
        sns.set_style("dark")
        ax = plt.figure().add_subplot(projection="3d")

        self.draw_sphere(ax=ax, radius=radius)

        if data is None:
            is_list = True
            data = self.samples
            N = len(data)
        else:
            is_list = False
            N = 1

        colors = [sns.color_palette("GnBu_d", N)[i] for i in reversed(range(N))]

        data_check = data[0] if is_list else data

        if type(data_check) is vMFSample:
            i = 0
            if is_list:
                for d in data:
                    ax.scatter(
                        d.x,
                        d.y,
                        d.z,
                        s=50,
                        alpha=0.7,
                        label="$\\kappa = $" + str(d.kappa),
                        color=colors[i],
                    )
                    i += 1
            else:
                ax.scatter(
                    data.x,
                    data.y,
                    data.z,
                    s=50,
                    alpha=0.7,
                    label="$\\kappa = $" + str(data.kappa),
                    color=colors[i],
                )

        elif type(data_check) is Coord3D:
            i = 0
            if is_list:
                for d in data:
                    ax.scatter(
                        d.x,
                        d.y,
                        d.z,
                        s=50,
                        alpha=0.7,
                        label="uniform samples",
                        color=sns.color_palette("GnBu_d", N),
                    )
                    i += 1
            else:
                ax.scatter(
                    data.x,
                    data.y,
                    data.z,
                    s=50,
                    alpha=0.7,
                    label="uniform samples",
                    color=sns.color_palette("GnBu_d", N),
                )

        else:
            print("Error: data type not recognised")

        ax.set_axis_off()
        ax.legend(bbox_to_anchor=[0.65, 0.75])

    def sample(
        self,
        nb_sample: int,
        radius: float = 1.0,
        distribution: str = "uniform",
        mu: Optional[np.ndarray] = None,
        kappa: Optional[float] = None,
    ):
        """Sample points on a spherical surface.
        :param int nb_sample: number of samples.
        :param float radius: the sphere radius.
        :param str distribution: type of distribution (uniform by default).
        :param Optional[np.ndarray] mu: parameter of vMF distribution.
        :param Optional[float] kappa: parameter of vMF distribution.
        """
        if distribution == "uniform":
            u = np.random.uniform(0, 1, nb_sample)
            v = np.random.uniform(0, 1, nb_sample)

            theta = 2 * np.pi * u
            phi = np.arccos(2 * v - 1)

            # convert to cartesian
            x = radius * np.cos(theta) * np.sin(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(phi)
            self.samples = (x, y, z)

        elif distribution == "vMF":
            try:
                s = sample_von_mises(mu, kappa, nb_sample)
            except Exception:
                print("Error: mu and kappa must be defined when sampling from vMF")
                return
            self.samples = vMFSample(s, kappa)

        else:
            print(
                "Error: sampling distribution not recognised (try 'uniform' or 'vMF')"
            )

        return self.samples


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


class Coord3D:
    """
    Class for 3D coordinates.
    """

    def __init__(self, x, y, z):
        """
        Store 3D coordinates.

        :param float x: the x coordinate.
        :param float y: the y coordinate.
        :param float z: the z coordinate.
        """
        self.x = x
        self.y = y
        self.z = z
