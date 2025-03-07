import inspect
import os
import sys

from sphinx.ext.autosummary import Autosummary

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../src"))


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None

    try:
        module_name = info["module"]
        fullname = info["fullname"]
        submod = sys.modules.get(module_name)
        if submod is None:
            return None

        obj = submod
        for part in fullname.split("."):
            try:
                obj = getattr(obj, part)
            except AttributeError:
                return None

        fn = inspect.getsourcefile(obj)
        source, lineno = inspect.getsourcelines(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(__file__))

        # Adjust if project inside src folder
        if fn.startswith("src"):
            fn = fn[len("src/") :]

        return f"https://github.com/mind-inria/hidimstat/blob/main/src/{fn}#L{lineno}-L{lineno + len(source) - 1}"
    except Exception:
        return None


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "HiDimStat"
copyright = "2025, The hidimstat developers"
author = "The hidimstat developers"
release = "0.2.0b"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.bibtex",
    "sphinx.ext.mathjax",
    "sphinx_gallery.gen_gallery",
    "numpydoc",
    "sphinx.ext.linkcode",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]


html_logo = "_static/logo.png"
html_theme = "pydata_sphinx_theme"
html_title = "HiDimStat"

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

# Generate the plots for the gallery
plot_gallery = "True"
examples_dirs = ["../examples"]
gallery_dirs = ["auto_examples"]
scrapers = ("matplotlib",)


autosummary_generate = True
numpydoc_show_class_members = False

autodoc_default_options = {
    "inherited-members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}
autodoc_mock_imports = [
    "sklearn",
]

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
    # "navbar_links": [("API", "api/index")],  # Ensure the right path
}


sphinx_gallery_conf = {
    "doc_module": "groupmne",
    "reference_url": dict(groupmne=None),
    "examples_dirs": examples_dirs,
    "gallery_dirs": gallery_dirs,
    "plot_gallery": "True",
    "thumbnail_size": (160, 112),
    "min_reported_time": 1.0,
    "backreferences_dir": os.path.join("generated"),
    "abort_on_example_error": False,
    "image_scrapers": scrapers,
    "show_memory": True,
}

git_root_url = "https://github.com/mind-inria/hidimstat"

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_repo": "hidimstat",
}
