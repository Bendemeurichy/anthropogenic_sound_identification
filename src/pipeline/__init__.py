"""
Pipeline modules for combining separation models with activity filters.

This package provides modular pipelines that integrate TUSS separation
with various preprocessing and optimization strategies like mask recycling
and scene splitting.
"""

from .separation_pipeline import SeparationPipeline

__all__ = ["SeparationPipeline"]
