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
copyright = (
    "2023-2025, François-Xavier Dupé, Rohit Yadav, Guillaume Auzias, Sylvain Takerkart"
)
author = "François-Xavier Dupé, Rohit Yadav, Guillaume Auzias, Sylvain Takerkart"
release = "0.10.0"
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
import graph_matching_tools.algorithms.multiway.fgw as fgw
import graph_matching_tools.algorithms.mean.wasserstein_barycenter as wb
import graph_matching_tools.algorithms.kernels.rff as rff
import graph_matching_tools.algorithms.multiway.matcheig as matcheig
import graph_matching_tools.algorithms.multiway.msync as msync
import graph_matching_tools.algorithms.multiway.hippi as hippi
import graph_matching_tools.algorithms.multiway.irgcl as irgcl
import graph_matching_tools.algorithms.multiway.kergm as kergm
import graph_matching_tools.algorithms.multiway.kmeans as kmeans
import graph_matching_tools.algorithms.multiway.mkergm as mkergm
import graph_matching_tools.algorithms.multiway.quickmatch as quickmatch
import graph_matching_tools.algorithms.multiway.stiefel as stiefel
import graph_matching_tools.algorithms.multiway.ga_mgmc as ga_mgmc
import graph_matching_tools.algorithms.multiway.matchals as matchals
import graph_matching_tools.algorithms.multiway.boolean_nmf as nmf
import graph_matching_tools.algorithms.mean.wasserstein_barycenter as fgw_barycenter
import graph_matching_tools.algorithms.mean.mm_mean as mm_mean
import graph_matching_tools.algorithms.pairwise.fgw as fgw_pairwise
import graph_matching_tools.algorithms.pairwise.gwl as gwl_pairwise
import graph_matching_tools.algorithms.pairwise.rrwm as rrwm
import graph_matching_tools.algorithms.pairwise.kergm as kergm_pairwise
import graph_matching_tools.algorithms.pairwise.grasp as grasp
import graph_matching_tools.algorithms.multiway.fmgm as fmgm
import graph_matching_tools.algorithms.multiway.mixer as mixer


graph1 = nx.Graph()
graph1.add_node(0, weight=np.array([2.0, ]))
graph1.add_node(1, weight=np.array([5.0, ]))
graph1.add_edge(0, 1, weight=np.array([1.0, ]))
graph2 = nx.Graph()
graph2.add_node(0, weight=np.array([5.0, ]))
graph2.add_node(1, weight=np.array([2.0, ]))
graph2.add_edge(0, 1, weight=np.array([1.0, ]))
graph3 = nx.Graph()
graph3.add_node(0, weight=np.array([3.0, ]))
graph3.add_node(1, weight=np.array([2.0, ]))
graph3.add_node(2, weight=np.array([5.0, ]))
graph3.add_edge(1, 2, weight=np.array([1.0, ]))

graphs = [graph1, graph2, graph3]
"""


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = "GTM {}".format(release)
html_theme = "furo"
html_static_path = ["_static"]
