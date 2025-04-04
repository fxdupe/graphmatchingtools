"""Code for graph topology.

Extract from slam code.

.. moduleauthor:: François-Xavier Dupé
"""

import numpy as np
import scipy
import trimesh


def adjacency_matrix(mesh: trimesh.Trimesh) -> scipy.sparse.coo_matrix:
    """Compute the adjacency matrix of the graph corresponding to the input mesh.
    See https://en.wikipedia.org/wiki/Adjacency_matrix

    :param nx.Graph mesh: the input mesh.
    :return: nb_vertex X bn_vertex sparse matrix.
    :rtype: scipy.sparse.coo_matrix
    """
    adja = trimesh.graph.edges_to_coo(
        mesh.edges, data=np.ones(mesh.edges.shape[0], dtype=np.int64)
    )
    adja = adja + adja.transpose()
    adja[adja > 0] = 1
    return adja
