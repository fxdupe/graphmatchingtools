Tutorial
========

In this tutorial we explain how to use our toolbox on the Willow dataset.

1 - Load the Willow dataset
---------------------------

Here we use `pytorch-geometric <https://pytorch-geometric.readthedocs.io>`_ to handle the data.
We propose a module to download and build the graphs (in NetworkX format).
The following lines will load the *car* category from Willow using isotropic graph,

>>> import graph_matching_tools.io.pygeo_graphs as pyg
>>> graphs = pyg.get_graph_database("Willow", True, "car", "/tmp/willow")

Notice that in this example, we default the repository for the graphs to */tmp/willow*.


2 - Building th node affinity matrix
------------------------------------

Now, we must build the bulk node affinity matrix. First, we need to build a kernel for example
a Gaussian kernel,

>>> import graph_matching_tools.algorithms.kernels.gaussian as gaussian
>>> node_kernel = gaussian.create_gaussian_node_kernel(args.sigma, "x")


3 - Building the edge tensors
-----------------------------

4 - Compute the multiple graph matching
---------------------------------------

5 - Evaluate the results
------------------------
