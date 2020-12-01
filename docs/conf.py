# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
from recommonmark.transform import AutoStructify
from recommonmark.parser import CommonMarkParser


# -- Project information -----------------------------------------------------

project = "ONR Detonation Tube Documentation"
copyright = "2020, Mick Carter"
author = "Mick Carter"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "recommonmark",
    "nbsphinx",
]
autosummary_generate = True
napoleon_google_docstring = False

source_parsers = {
    ".md": CommonMarkParser
}
source_suffix = [".rst", ".md"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static", "pages/tube_operation/images"]


# noinspection PyUnusedLocal
def convert(my_path, prev=""):
    split = os.path.split(my_path)
    if len(split[1]):
        prev = convert(split[0], split[1])
        return os.path.join(prev, split[1])
    else:
        return split[1]


def setup(app):
    """
    https://recommonmark.readthedocs.io/en/latest/
    """
    app.add_config_value(
        "recommonmark_config",
        {
            "url_resolver": lambda url: convert(url),
            "enable_auto_toc_tree": True,
            "auto_toc_max_depth": 4,
        },
        True
    )
    app.add_transform(AutoStructify)
