Tutorial
========

In this tutorial we explain how to use our toolbox on the Willow dataset.

1 - Load the Willow dataset
---------------------------

Here we use `pytorch-geometric <https://pytorch-geometric.readthedocs.io>`_ to handle the data.
We propose a module to download and build the graphs (in NetworkX format).
The following lines will load the *car* category from Willow using isotropic graph,

>>> import numpy as np
>>> import graph_matching_tools.io.pygeo_graphs as pyg
>>> graphs = pyg.get_graph_database("Willow", True, "car", "/tmp/willow")

Notice that in this example, we default the repository for the graphs to */tmp/willow*. The graphs
will have attributes on nodes (named *x*) and edges (named *weight*).


2 - Building th node affinity matrix
------------------------------------

Now, we must build the bulk node affinity matrix. First, we need to build a kernel for example
a Gaussian kernel,

>>> import graph_matching_tools.algorithms.kernels.gaussian as gaussian
>>> node_kernel = gaussian.create_gaussian_node_kernel(2.0, "x")

The arguments are, :math:`\sigma^2`, the variance of the Gaussian kernel and the name of the data
vector for the nodes. Then we can build the bulk node affinity matrix,

>>> import graph_matching_tools.algorithms.kernels.utils as ku
>>> knode = ku.create_full_node_affinity_matrix(graphs, node_kernel)

3 - Building the edge tensors
-----------------------------

The next step is the construction of the edge tensors. We begin by building the Randon Fourier Features needed to
approximate a Gaussian kernel while keeping a linear scalar product.

>>> import graph_matching_tools.algorithms.kernels.rff as rff
>>> vectors, offsets = rff.create_random_vectors(1, 100, 0.1)

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

5 - Evaluate the results
------------------------
