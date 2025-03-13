import os
import sys

sys.path.insert(0, os.path.abspath("."))

import matplotlib
from utils import linkcode_resolve

from hidimstat import __version__

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "HiDimStat"
copyright = "2025, The hidimstat developers"
author = "The hidimstat developers"
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.bibtex",
    "sphinx.ext.mathjax",
    "sphinx_gallery.gen_gallery",
    "numpydoc",
    "sphinx.ext.linkcode",  # use the function linkcode_resolve for the definition of the link
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]
html_logo = "_static/logo.png"
html_theme = "pydata_sphinx_theme"
html_title = "HiDimStat"
html_context = {
    "display_github": True,  # Integrate GitHub
    "github_repo": "hidimstat",
}

plot_formats = []

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# sphinxcontrib-bibtex
bibtex_bibfiles = ["./references.bib"]
bibtex_style = "unsrt"
bibtex_reference_style = "author_year"
bibtex_footbibliography_header = ""


# -- Options for autodoc / autosummary ----------------------------------------
# generate autosummary even if no references
autosummary_generate = True

# -- Options for Numpydoc ----------------------------------------------------
# https://numpydoc.readthedocs.io/en/latest/install.html
numpydoc_show_class_members = False

autodoc_default_options = {
    "inherited-members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/mind-inria/hidimstat/",
            "icon": "fa-brands fa-github",
        },
    ],
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "header_links_before_dropdown": 4,
}


# -- Options for gallery ----------------------------------------------------
# Generate the plots for the gallery
matplotlib.use("agg")
sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["auto_examples"],
    "plot_gallery": "True",
    "thumbnail_size": (160, 112),
    "min_reported_time": 1.0,
    "abort_on_example_error": False,
    "image_scrapers": ("matplotlib",),
    "show_memory": True,
    "doc_module": "hidimstat",
    "backreferences_dir": "generated",
    "reference_url": {
        # The module we locally document (so, hidimstat) uses None
        "hidimstat": None,
        # We don't specify the other modules as we use the intershpinx ext.
        # See https://sphinx-gallery.github.io/stable/configuration.html#link-to-documentation  # noqa
    },
}

git_root_url = "https://github.com/mind-inria/hidimstat"


# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/devdocs", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "numba": ("https://numba.readthedocs.io/en/stable/", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
    "pyvista": ("https://docs.pyvista.org", None),
}
