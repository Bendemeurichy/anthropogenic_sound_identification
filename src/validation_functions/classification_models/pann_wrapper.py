"""Wrapper for PANN (Pretrained Audio Neural Networks) AudioTagging classifier.

PANN is a PyTorch-based AudioSet classifier that can detect various audio events
including aircraft sounds. This wrapper adapts it to the unified classifier interface.
"""

from typing import List, Tuple
import torch

from .pann_inference import (
    load_pann_model,
    run_pann_inference,
    PANN_SAMPLE_RATE,
    DEFAULT_PANN_POSITIVE_LABELS,
)


class PANNClassifierWrapper:
    """Wrapper for PANN AudioTagging that follows the AudioClassifier protocol.
    
    PANN is trained on AudioSet (527 classes) and uses sigmoid activations for
    multi-label classification. This wrapper aggregates predictions across
    specified positive labels to produce a binary decision.
    
    Attributes:
        sample_rate: PANN's expected sample rate (32000 Hz)
        device: PyTorch device for model inference
        model: Loaded PANN AudioTagging model
        positive_labels: AudioSet labels treated as positive class
        threshold: Classification threshold for binary decision
    """
    
    def __init__(
        self,
        device: str = "cpu",
        positive_labels: List[str] = None,
        threshold: float = 0.5,
    ):
        """Initialize the PANN classifier wrapper.
        
        Args:
            device: PyTorch device string (e.g., "cpu", "cuda", "cuda:0", "cuda:1")
            positive_labels: List of AudioSet label names to treat as positive class.
                           If None, uses default airplane-related labels:
                           ["Fixed-wing aircraft, airplane", "Aircraft", "Jet aircraft",
                            "Propeller, airscrew", "Turboprop, small aircraft"]
                           Unrecognized labels are ignored with a warning.
            threshold: Classification threshold. Predictions with max confidence across
                      positive_labels >= threshold are classified as positive (1).
        """
        self.device = device
        self.positive_labels = positive_labels if positive_labels is not None else DEFAULT_PANN_POSITIVE_LABELS
        self.threshold = threshold
        
        # Load the PANN CNN14 AudioTagging model
        self.model = load_pann_model(device=device)
        self._sample_rate = PANN_SAMPLE_RATE
    
    @property
    def sample_rate(self) -> int:
        """Expected sample rate in Hz for input waveforms.
        
        PANN was trained on 32kHz audio and expects this rate.
        The underlying run_pann_inference will resample if needed,
        but for consistency callers should provide 32kHz audio.
        
        Returns:
            32000 (PANN's native sample rate)
        """
        return self._sample_rate
    
    def __call__(self, waveform: torch.Tensor) -> Tuple[int, float]:
        """Classify an audio waveform using PANN.
        
        The waveform is expected to be at PANN_SAMPLE_RATE (32kHz).
        PANN computes sigmoid scores for all 527 AudioSet classes,
        then this wrapper takes the maximum score across positive_labels
        as the confidence and applies the threshold for binary classification.
        
        Args:
            waveform: Audio tensor (1D) at 32000 Hz.
                     Should be normalized to [-1, 1] range.
        
        Returns:
            A tuple of (prediction, confidence) where:
            - prediction: 1 if max confidence >= threshold, 0 otherwise
            - confidence: Maximum sigmoid score across positive_labels in [0.0, 1.0]
        
        Example:
            >>> wrapper = PANNClassifierWrapper(device="cuda")
            >>> waveform = torch.randn(32000 * 3)  # 3 seconds at 32kHz
            >>> pred, conf = wrapper(waveform)
            >>> print(f"Aircraft: {pred}, Confidence: {conf:.4f}")
        """
        # run_pann_inference handles resampling internally if needed,
        # but we assume waveform is already at PANN_SAMPLE_RATE for consistency
        return run_pann_inference(
            self.model,
            waveform,
            self._sample_rate,
            self.positive_labels,
            threshold=self.threshold,
        )
