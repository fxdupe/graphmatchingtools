Graph Generation
================

The toolbox proposes one method for generating sucal pits graphs.

Sucal pits graph
----------------

We recommend to read `this paper <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0293886>`_
in order to understand the influence of the different parameters.

First we need to generate a reference graph. For example, to generate a graph with 25 nodes (using a sphere of
radius 1.0),

>>> import graph_matching_tools.generators.reference_graph as reference_graph
>>> ref_graph = reference_graph.generate_reference_graph(25, 1.0)
>>> ref_graph.number_of_nodes()
25

From the reference, we can now a noisy version (here with :math:`\kappa = 200`),

>>> import graph_matching_tools.generators.noisy_graph as noisy_graph
>>> noisy_version = noisy_graph.noisy_graph_generation(ref_graph, kappa_noise_node=200)

By default, we use the parameters from the paper.

It is possible to directly generate a set of graphs. For example to generate 10 graphs with :math:`\kappa = 200`,

>>> import graph_matching_tools.generators.graph_family as graph_family
>>> graphs_list = graph_family.generation_graph_family(10, ref_graph, kappa_noise_node=200)
>>> len(graphs_list)
10
