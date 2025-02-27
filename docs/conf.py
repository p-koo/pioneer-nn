# docs/conf.py

import os
import sys
# Add the project root to sys.path if needed:
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'pioneer-nn'
author = 'Your Name'
release = '0.1.0'  # The full version, including alpha/beta/rc tags

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',    # Automatically generate documentation from docstrings
    'sphinx.ext.napoleon',   # Support for NumPy and Google style docstrings
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = 'alabaster'
html_static_path = ['_static']
