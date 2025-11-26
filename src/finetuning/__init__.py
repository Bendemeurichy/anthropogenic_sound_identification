"""
Finetuning module for airplane YAMNet model.

This module provides functionality to finetune the YAMNet model for airplane
sound detection using transfer learning.
"""

from .config import FinetuneConfig
from .model import create_classifier
from .trainer import train

__all__ = ["FinetuneConfig", "create_classifier", "train"]
