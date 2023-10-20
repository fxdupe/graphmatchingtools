"""
This code is directly from the paper
"GRASP: Scalable Graph Alignment by Spectral Corresponding Functions" by Hermanns et al (ACM Trans KDD, 2023).

.. moduleauthor:: François-Xavier Dupé
"""
from typing import Optional

import numpy as np
import networkx as nx


def grasp(
    source: nx.Graph, target: nx.Graph, k: int, time_steps: Optional[list[float]] = None
) -> np.ndarray:
    if time_steps is None:
        time_steps = np.linspace(0.1, 50)
    else:
        time_steps = np.array(time_steps)

    # Step 1: eigendecomposition of normalized Laplacian
    l1 = nx.normalized_laplacian_matrix(source)
    l2 = nx.normalized_laplacian_matrix(target)

    lbd1, phi = np.linalg.eigh(l1)
    lbd2, psi = np.linalg.eigh(l2)

    # Step 2: compute the corresponding functions
    f = np.zeros((l1.shape[0], k))
    g = np.zeros((l2.shape[0], k))

    for t_i in range(time_steps.shape[0]):
        fi = np.zeros((l1.shape[0], 1))
        gi = np.zeros((l2.shape[0], 1))

        for i in range(k):
            fi += (
                np.exp(-time_steps[t_i] / lbd1[-(1 + t_i)])
                * phi[:, -(1 + t_i)]
                * phi[:, -(1 + t_i)]
            )
            gi += (
                np.exp(-time_steps[t_i] / lbd2[-(1 + t_i)])
                * psi[:, -(1 + t_i)]
                * psi[:, -(1 + t_i)]
            )

        f[:, t_i] = fi
        g[:, t_i] = gi

    # Step 3: base alignment
