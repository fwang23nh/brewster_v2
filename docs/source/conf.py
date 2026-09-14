# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath("../.."))



project = 'Brewster v2'
copyright = '2026, Fei & Ben'
author = 'Fei & Ben'
release = '2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",    # for Google/NumPy style docstrings
    "sphinx.ext.viewcode",    # show source links
    "sphinx.ext.mathjax",
]
extensions.append("nbsphinx")

autodoc_mock_imports = [
    "forwardmodel",
    "ciamod",
    "mpi4py",
    "pymultinest",
]



templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_css_files = ['custom.css']
html_title = "Brewster v2 documentation"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
}

nbsphinx_execute = "never"
nbsphinx_allow_errors = True
