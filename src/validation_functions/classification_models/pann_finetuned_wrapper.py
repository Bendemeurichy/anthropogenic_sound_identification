"""
Wrapper for fine-tuned PANN plane classifier.

This wrapper adapts the PlaneClassifierPANN (fine-tuned CNN14 model) to conform
to the unified AudioClassifier interface used in test_pipeline.py.
"""

from pathlib import Path
from typing import Tuple, Optional

import torch
import numpy as np


class PANNFinetunedWrapper:
    """Wrapper for fine-tuned PANN plane classifier conforming to AudioClassifier protocol.
    
    This classifier uses a CNN14 backbone pre-trained on AudioSet and then fine-tuned
    on a binary plane detection task.
    
    Example:
        >>> wrapper = PANNFinetunedWrapper(
        ...     checkpoint_path="checkpoints/final_model.pth",
        ...     device="cuda"
        ... )
        >>> waveform = torch.randn(320000)  # 10 seconds at 32kHz
        >>> prediction, confidence = wrapper(waveform)
        >>> print(f"Plane detected: {prediction}, Confidence: {confidence:.4f}")
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config=None,  # TrainingConfig, but avoid import
        device: str = "cuda",
        threshold: float = 0.5,
    ):
        """Initialize the fine-tuned PANN classifier.
        
        Args:
            checkpoint_path: Path to the fine-tuned model checkpoint (.pth file)
            config: TrainingConfig (uses defaults if None)
            device: Device for inference ("cuda", "cuda:0", "cpu")
            threshold: Classification threshold (default: 0.5)
        """
        # Import here to avoid circular dependencies
        from validation_functions.classification_models.plane_classifier_pann.model_loader import load_trained_model
        from validation_functions.classification_models.plane_classifier_pann.config import ModelConfig
        
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.threshold = threshold
        
        # Load the model
        self.model = load_trained_model(
            checkpoint_path=checkpoint_path,
            config=ModelConfig() if config is None else None,
            training_config=config,
            device=device,
        )
        self.model.eval()
        
        # PANN works at 32kHz with 10-second segments
        self._sample_rate = 32000
        self._segment_length = 10.0
        self._segment_samples = int(self._sample_rate * self._segment_length)
        
        print(f"Loaded fine-tuned PANN classifier from {checkpoint_path}")
        print(f"  Sample rate: {self._sample_rate} Hz")
        print(f"  Segment length: {self._segment_length} s ({self._segment_samples} samples)")
        print(f"  Threshold: {threshold}")
    
    @property
    def sample_rate(self) -> int:
        """Expected sample rate for input waveforms (32kHz for PANN)."""
        return self._sample_rate
    
    def __call__(self, waveform: torch.Tensor) -> Tuple[int, float]:
        """Classify an audio waveform.
        
        Args:
            waveform: Audio tensor (1D) at 32kHz.
                     Should be in range [-1, 1] (normalized audio).
                     Expected length: 320000 samples (10 seconds at 32kHz).
        
        Returns:
            Tuple of (prediction, confidence) where:
            - prediction: Binary class label (0 or 1)
                         1 = plane detected
                         0 = no plane
            - confidence: Sigmoid probability in range [0.0, 1.0]
        """
        # Ensure waveform is 1D
        if waveform.ndim > 1:
            waveform = waveform.squeeze()
        
        # Pad or truncate to expected length
        if waveform.shape[0] < self._segment_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self._segment_samples - waveform.shape[0])
            )
        else:
            waveform = waveform[: self._segment_samples]
        
        # Move to device and add batch dimension
        waveform = waveform.to(self.device).unsqueeze(0)  # (1, samples)
        
        # Run model inference
        self.model.eval()
        with torch.no_grad():
            logit = self.model(waveform)  # (1, 1)
            confidence = torch.sigmoid(logit).item()
        
        # Apply threshold
        prediction = 1 if confidence >= self.threshold else 0
        
        return prediction, confidence
