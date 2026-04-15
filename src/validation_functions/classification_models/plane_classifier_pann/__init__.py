"""
PANN-based plane classifier using CNN14 pretrained on AudioSet.

This package provides a complete pipeline for fine-tuning PANN (Pretrained Audio
Neural Networks) on binary plane detection tasks.

Main components:
    - model: CNN14 architecture and PlaneClassifierPANN
    - model_loader: Load pretrained and fine-tuned models
    - dataset: PyTorch Dataset and DataLoader for PANN
    - train: Two-phase training pipeline
    - inference: Inference wrapper for predictions
    - config: Configuration dataclasses

Example usage:
    # Training
    >>> from main import main
    >>> main()
    
    # Inference
    >>> from inference import PlaneClassifierInference
    >>> classifier = PlaneClassifierInference("checkpoints/final_model.pth")
    >>> result = classifier.predict_file("audio.wav")
    >>> print(result['prediction'], result['confidence'])
"""

from .config import TrainingConfig, ModelConfig
from .model import Cnn14, PlaneClassifierPANN
from .model_loader import (
    load_pretrained_cnn14,
    create_plane_classifier,
    load_trained_model,
)
from .inference import PlaneClassifierInference

__all__ = [
    'TrainingConfig',
    'ModelConfig',
    'Cnn14',
    'PlaneClassifierPANN',
    'load_pretrained_cnn14',
    'create_plane_classifier',
    'load_trained_model',
    'PlaneClassifierInference',
]

__version__ = '1.0.0'
