Tutorial
========

In this tutorial we explain how to use our toolbox on the Willow dataset. We add the imports when
we need new functions.

1 - Load the Willow dataset
---------------------------

Here we use `pytorch-geometric <https://pytorch-geometric.readthedocs.io>`_ to handle the data.
We propose a module to download and build the graphs (in NetworkX format).
The following lines will load the *car* category from Willow using isotropic graph,

>>> import numpy as np
>>> import networkx as nx
>>> import graph_matching_tools.io.pygeo_graphs as pyg
>>> graphs = pyg.get_graph_database("Willow", True, "car", "/tmp/willow")

Notice that in this example, we default the repository for the graphs to */tmp/willow*. The graphs
will have attributes on nodes (named *x*) and edges (named *weight*).


2 - Building th node affinity matrix
------------------------------------

Now, we must build the bulk node affinity matrix. First, we need to build a kernel for example
a Gaussian kernel,

>>> import graph_matching_tools.algorithms.kernels.gaussian as gaussian
>>> node_kernel = gaussian.create_gaussian_node_kernel(70.0, "x")

The arguments are, :math:`\sigma^2`, the variance of the Gaussian kernel and the name of the data
vector for the nodes. Then we can build the bulk node affinity matrix,

>>> import graph_matching_tools.algorithms.kernels.utils as ku
>>> knode = ku.create_full_node_affinity_matrix(graphs, node_kernel)

3 - Building the edge tensors
-----------------------------

The next step is the construction of the edge tensors. We begin by building the Randon Fourier Features needed to
approximate a Gaussian kernel while keeping a linear scalar product.

>>> import graph_matching_tools.algorithms.kernels.rff as rff
>>> vectors, offsets = rff.create_random_vectors(1, 100, 0.01)

The parameters are the following.

1. The dimension of the input vector (1 here as it is only a weight).
2. The dimension of thr Fourier features (the larger the better the approximation is).
3. The hyperparameter, :math:`\gamma` of the Gaussian kernel as a radial basis function with
    :math:`\gamma\propto\frac{1}{\sigma^2}`.

Then we can build the tensors for each graph.

>>> sizes = [nx.number_of_nodes(g) for g in graphs]  # The size the graphs
>>> full_size = sum(sizes)  # The full size the permutation matrix
>>> phi = np.zeros((100, full_size, full_size))
>>> index = 0
>>> for i in range(len(graphs)):  # Compute the phi for each graph
...     g_phi = rff.compute_phi(graphs[i], "weight", vectors, offsets)
...     phi[:, index : index + sizes[i], index : index + sizes[i]] = g_phi
...     index += sizes[i]


4 - Compute the multiple graph matching
---------------------------------------

First in order to equilibrate the nodes versus the edge, we normalize the node affinity matrix by dividing it
by the median size of the graphs,

>>> norm_knode = np.median(sizes)
>>> knode /= norm_knode

Once both the node affinity matrix and edge tensors built, we can create the gradient linked to the objective
function to optimize.

>>> import graph_matching_tools.algorithms.multiway.mkergm as mkergm
>>> gradient = mkergm.create_gradient(phi, knode)

Then we can apply the multigraph matching algorithm using a uniform initialization.

>>> x_init = np.ones(knode.shape) / knode.shape[0]  # Uniform initialization
>>> m_res = mkergm.mkergm(
...     gradient,
...     sizes,
...     10,  # The dimension of the universe of nodes (i.e. the rank)
...     iterations=20,
...     init=x_init,
...     tolerance=1e-3,
...     projection_method="matcheig",  # This is default projector
... )

5 - Evaluate the results
------------------------

To evaluate the results, we first need to generate the ground truth (this is specific for Willow).

>>> import graph_matching_tools.utils.permutations as perm
>>> graph_index = [list(range(nx.number_of_nodes(g))) for g in graphs]
>>> match_truth = pyg.generate_groundtruth(sizes, full_size, graph_index)
>>> truth = perm.get_permutation_matrix_from_matching(match_truth, sizes)

Then we can compute the precision, recall and F1-score.

>>> import graph_matching_tools.metrics.matching as measures
>>> f1_score, precision, recall = measures.compute_f1score(m_res, truth)

You should expect a F1-score around 0.89 (with some fluctuation due to the randomness of the Fourier Features).
