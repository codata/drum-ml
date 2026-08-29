"""Sphinx documentation configuration for DRUM-ML."""

import os
import sys
from pathlib import Path

# Add src to sys.path for autodoc
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

# Project information
project = "DRUM-ML"
copyright = "2026, CODATA DRUM Working Group (Lead: Pascal Heus)"
author = "Pascal Heus (Lead), CODATA DRUM Working Group"
release = "0.1.0"
version = "0.1.0"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Source suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# HTML output theme
html_theme = "furo"
html_title = "DRUM-ML Documentation"
html_static_path = ["_static"]

# Theme options
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
suppress_warnings = ["sphinx_autodoc_typehints.forward_reference"]
