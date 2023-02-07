.. GraphMatchingTools documentation master file, created by
   sphinx-quickstart on Tue Jan 24 17:21:48 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GraphMatchingTools' documentation!
==============================================

This is the documentation of GraphMatchingTools a set of method dedicated to graph matching.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorial
   autoapi/index
   authors
   changelog

.. _install-guide:

Installation Guide
==================

* Step 1: Create and activate a virtual environment
* Step 2: pip install it


Current available methods
=========================

Pairwise method
---------------

- [KerGM](https://papers.nips.cc/paper/2019/hash/cd63a3eec3319fd9c84c942a08316e00-Abstract.html)
- [GWL](https://proceedings.mlr.press/v97/xu19b.html)
- [FGW](https://proceedings.mlr.press/v97/titouan19a.html)

Multiway methods (state-of-art)
-------------------------------

- [HiPPI](https://openaccess.thecvf.com/content_ICCV_2019/html/Bernard_HiPPI_Higher-Order_Projected_Power_Iterations_for_Scalable_Multi-Matching_ICCV_2019_paper.html)
- KMeans (as a naive way of doing the multimatch)
- [MSync](https://papers.nips.cc/paper/2013/hash/3df1d4b96d8976ff5986393e8767f5b2-Abstract.html)
- [MatchEIG](https://openaccess.thecvf.com/content_iccv_2017/html/Maset_Practical_and_Efficient_ICCV_2017_paper.html)
- [QuickMatch](https://openaccess.thecvf.com/content_iccv_2017/html/Tron_Fast_Multi-Image_Matching_ICCV_2017_paper.html)
- [Sparse Quadratic Optimisation over the Stiefel Manifold with Application to Permutation Synchronisation](https://openreview.net/forum?id=sl_0rQmHxQk)
- [IRGCL](https://papers.nips.cc/paper/2020/hash/ae06fbdc519bddaa88aa1b24bace4500-Abstract.html)

Multiway methods (homemade)
---------------------------

- Kernelized multi-graph matching

Mean graph methods
------------------

- [MMM method based on pairwise matching](https://www.sciencedirect.com/science/article/abs/pii/S003132031630139X)
- [Wasserstein barycenter using FGW](https://proceedings.mlr.press/v97/titouan19a.html)



About the examples
==================

All the examples in the documentation are done with the following imports,

.. doctest::

   >>> import numpy as np
   >>> import networkx as nx
   >>> import graph_matching_tools.algorithms.kernels.gaussian as kern
   >>> import graph_matching_tools.algorithms.kernels.utils as utils
   >>> import graph_matching_tools.algorithms.multiway.gwl as gwl
   >>> import graph_matching_tools.algorithms.mean.wasserstein_barycenter as wb
   >>> import graph_matching_tools.algorithms.multiway.matcheig as matcheig
   >>> import graph_matching_tools.algorithms.multiway.hippi as hippi

We use three graphs for illustration purpose. These graphs are build using Networkx using
the following code:

.. doctest::

   >>> graph1 = nx.Graph()
   >>> graph1.add_node(0, weight=2.0)
   >>> graph1.add_node(1, weight=5.0)
   >>> graph1.add_edge(0, 1)
   >>> graph2 = nx.Graph()
   >>> graph2.add_node(0, weight=5.0)
   >>> graph2.add_node(1, weight=2.0)
   >>> graph2.add_edge(0, 1)
   >>> graph3 = nx.Graph()
   >>> graph3.add_node(0, weight=3.0)
   >>> graph3.add_node(1, weight=2.0)
   >>> graph3.add_node(2, weight=5.0)
   >>> graph3.add_edge(1, 2)
   >>> # The list of graphs
   >>> graphs = [graph1, graph2, graph3]


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
