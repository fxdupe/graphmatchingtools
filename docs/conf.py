# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "GraphMatchingTools"
copyright = "2023, François-Xavier Dupé, Rohit Yadav"
author = "François-Xavier Dupé, Rohit Yadav"
release = "0.7.0"
show_authors = True

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "autoapi.extension",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autoapi_dirs = ["../graph_matching_tools"]
autoapi_type = "python"
autoapi_template_dir = "_templates/autoapi"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_keep_files = True
autodoc_typehints = "signature"

# Doctest configuration to unify the graphs
doctest_global_setup = """
import numpy as np  # Load the different modules
import networkx as nx
import graph_matching_tools.algorithms.kernels.gaussian as kern
import graph_matching_tools.algorithms.kernels.utils as utils
import graph_matching_tools.algorithms.multiway.gwl as gwl
import graph_matching_tools.algorithms.mean.wasserstein_barycenter as wb
import graph_matching_tools.algorithms.multiway.matcheig as matcheig
import graph_matching_tools.algorithms.multiway.hippi as hippi


graph1 = nx.Graph()
graph1.add_node(0, weight=2.0)
graph1.add_node(1, weight=5.0)
graph1.add_edge(0, 1)
graph2 = nx.Graph()
graph2.add_node(0, weight=5.0)
graph2.add_node(1, weight=2.0)
graph2.add_edge(0, 1)
graph3 = nx.Graph()
graph3.add_node(0, weight=3.0)
graph3.add_node(1, weight=2.0)
graph3.add_node(2, weight=5.0)
graph3.add_edge(1, 2)

graphs = [graph1, graph2, graph3]
"""


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = "GTM {}".format(release)
html_theme = "furo"
html_static_path = ["_static"]
