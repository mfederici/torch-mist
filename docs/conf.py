# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------

project = "torch_mist"
copyright = "2023, Marco Federici"
author = "Marco Federici"

# -- General configuration ---------------------------------------------------

extensions = [
    "nbsphinx",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
autoapi_dirs = ["../src"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_logo = "_static/logo.png"
# Change the color of search and top bar
html_theme_options = {
    "style_nav_header_background": "#1a1a1a",
    "logo_only": True,
}

html_favicon = "_static/favicon.ico"
