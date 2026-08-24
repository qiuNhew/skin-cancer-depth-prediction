"""
Breslow Depth Prediction Package

A deep learning framework for predicting melanoma Breslow depth from histopathology images.
"""

__version__ = "0.1.0"
__author__ = "Breslow Depth Prediction Team"

from . import data
from . import models
from . import training
from . import visualization
from .config import Config, load_config, save_config, get_default_config
