"""Wrapper for PlaneClassifier to match the unified classifier interface.

The PlaneClassifier uses a TensorFlow model with YAMNet backbone for airplane detection.
This wrapper adapts it to the common AudioClassifier protocol.
"""

from typing import Tuple, Optional
import torch
import numpy as np

from .plane_clasifier.inference import PlaneClassifierInference
from .plane_clasifier.config import TrainingConfig


class PlaneClassifierWrapper:
    """Wrapper for PlaneClassifierInference that follows the AudioClassifier protocol.
    
    This class wraps the original PlaneClassifierInference to provide a consistent
    interface with other classifiers. It converts the dict-based output to a simple
    (prediction, confidence) tuple and handles tensor/numpy conversions.
    
    Attributes:
        sample_rate: Expected sample rate (from config, typically 16000 Hz)
        classifier: Underlying PlaneClassifierInference instance
        threshold: Classification threshold for binary decision
    """
    
    def __init__(
        self,
        weights_path: str,
        config: Optional[TrainingConfig] = None,
        threshold: float = 0.5,
        device: str = "cpu"  # Not used by TensorFlow but kept for API consistency
    ):
        """Initialize the PlaneClassifier wrapper.
        
        Args:
            weights_path: Path to the .weights.h5 file containing trained model weights
            config: TrainingConfig instance with model/audio settings. If None, uses
                   default config with sample_rate=16000 and audio_duration=5.0
            threshold: Classification threshold. Predictions with confidence >= threshold
                      are classified as positive (plane detected = 1)
            device: Device string (kept for API consistency but ignored since TensorFlow
                   handles device placement automatically)
        """
        self.config = config if config is not None else TrainingConfig()
        self.classifier = PlaneClassifierInference(weights_path, self.config)
        self.threshold = threshold
        self._sample_rate = self.config.sample_rate
    
    @property
    def sample_rate(self) -> int:
        """Expected sample rate in Hz for input waveforms.
        
        Returns:
            Sample rate from the TrainingConfig (typically 16000 Hz for YAMNet)
        """
        return self._sample_rate
    
    def __call__(self, waveform: torch.Tensor) -> Tuple[int, float]:
        """Classify an audio waveform for airplane presence.
        
        Args:
            waveform: Audio tensor (1D) at self.sample_rate Hz.
                     Should be normalized to [-1, 1] range.
                     Length should match config.audio_duration or will be
                     truncated/padded automatically by the underlying model.
        
        Returns:
            A tuple of (prediction, confidence) where:
            - prediction: 1 if airplane detected, 0 otherwise
            - confidence: Sigmoid probability in range [0.0, 1.0]
        
        Example:
            >>> wrapper = PlaneClassifierWrapper("model.weights.h5")
            >>> waveform = torch.randn(16000 * 5)  # 5 seconds at 16kHz
            >>> pred, conf = wrapper(waveform)
            >>> print(f"Airplane: {pred}, Confidence: {conf:.4f}")
        """
        # Convert torch tensor to numpy if needed
        if isinstance(waveform, torch.Tensor):
            wav_np = waveform.detach().cpu().numpy()
        else:
            wav_np = np.asarray(waveform)
        
        # Run inference using the underlying classifier
        result = self.classifier.predict_waveform(wav_np, threshold=self.threshold)
        
        # Convert string prediction ("plane"/"no_plane") to binary (1/0)
        pred = 1 if result["prediction"] == "plane" else 0
        conf = float(result["confidence"])
        
        return pred, conf
    
    def predict_batch(self, waveforms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Classify a batch of audio waveforms for airplane presence.
        
        This method leverages TensorFlow's native batch processing for efficient inference.
        
        Args:
            waveforms: Audio tensor of shape (B, T) at self.sample_rate Hz.
                      Should be normalized to [-1, 1] range.
        
        Returns:
            A tuple of (predictions, confidences) where:
            - predictions: Binary tensor of shape (B,) with values 0 or 1
            - confidences: Confidence tensor of shape (B,) in range [0.0, 1.0]
        """
        import tensorflow as tf
        
        # Convert torch tensor to numpy
        if isinstance(waveforms, torch.Tensor):
            wavs_np = waveforms.detach().cpu().numpy()
        else:
            wavs_np = np.asarray(waveforms)
        
        # Ensure shape is (B, T) - batch dimension should already be there
        if len(wavs_np.shape) == 1:
            wavs_np = np.expand_dims(wavs_np, 0)
        
        # Run batch inference using TensorFlow model (automatically handles batches)
        logits = self.classifier.model(wavs_np, training=False)
        probabilities = tf.sigmoid(logits).numpy().squeeze(-1)  # Shape: (B,)
        
        # Convert to predictions
        predictions = (probabilities >= self.threshold).astype(np.int64)
        
        return torch.tensor(predictions, dtype=torch.long), torch.tensor(probabilities, dtype=torch.float32)
