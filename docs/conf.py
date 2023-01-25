# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GraphMatchingTools'
copyright = '2023, François-Xavier Dupé, Rohit Yadav'
author = 'François-Xavier Dupé, Rohit Yadav'
release = '0.7.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.coverage',
              'autoapi.extension']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autoapi_dirs = ['../graph_matching_tools']
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

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = "GTM {}".format(release)
html_theme = 'furo'
html_static_path = ['_static']
