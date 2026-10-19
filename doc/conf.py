# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0,os.path.abspath(os.path.join("..")))
sys.setrecursionlimit(1500)

import pyvista
from pyvista.plotting.utilities.sphinx_gallery import DynamicScraper
pyvista.BUILDING_GALLERY = True
pyvista.set_plot_theme("document")
pyvista.set_error_output_file("errors.txt")


os.environ["PYVISTA_BUILDING_GALLERY"] = "true"


project = 'SHAnC'
copyright = '2026, re-gr, mate-pr'
author = 're-gr, mate-pr'
release = '0.1'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration



extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx.ext.mathjax",
    "pyvista.ext.plot_directive",
    "pyvista.ext.viewer_directive",
    "sphinx_design",
    "sphinxcontrib.asciinema",
    ]
napoleon_custom_sections = [('Returns', 'params_style'), ("Attributes","params_style")]

templates_path = ['_templates']
exclude_patterns = []



html_theme = 'pydata_sphinx_theme'


napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_ivar = True
html_show_sourcelink = False
html_show_copyright = False
html_theme_options = {
  # ...
  "secondary_sidebar_items": ["edit-this-page", "sourcelink"],
  # ...
}



html_static_path = ['_static']

html_css_files = [
    'css/custom.css',
]
