"""Wrapper for AST (Audio Spectrogram Transformer) classifier.

AST is a HuggingFace transformer model fine-tuned on AudioSet for audio event detection.
This wrapper adapts it to the unified classifier interface.
"""

from typing import List, Tuple
import torch

from .ast_inference import (
    load_ast_model,
    run_ast_inference,
    AST_SAMPLE_RATE,
    DEFAULT_AST_POSITIVE_LABELS,
)


class ASTClassifierWrapper:
    """Wrapper for AST that follows the AudioClassifier protocol.
    
    AST is a transformer-based AudioSet classifier that can detect various audio events.
    Like PANN, it uses sigmoid activations for multi-label classification. This wrapper
    aggregates predictions across specified positive labels to produce a binary decision.
    
    Attributes:
        sample_rate: AST's expected sample rate (16000 Hz)
        device: PyTorch device for model inference
        extractor: HuggingFace feature extractor
        model: Loaded AST AudioClassification model
        positive_labels: AudioSet labels treated as positive class
        threshold: Classification threshold for binary decision
    """
    
    def __init__(
        self,
        device: str = "cpu",
        positive_labels: List[str] = None,
        threshold: float = 0.5,
    ):
        """Initialize the AST classifier wrapper.
        
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
        self.positive_labels = positive_labels if positive_labels is not None else DEFAULT_AST_POSITIVE_LABELS
        self.threshold = threshold
        
        # Load the AST feature extractor and model
        self.extractor, self.model = load_ast_model(device=device)
        self._sample_rate = AST_SAMPLE_RATE
    
    @property
    def sample_rate(self) -> int:
        """Expected sample rate in Hz for input waveforms.
        
        AST was fine-tuned on 16kHz audio and expects this rate.
        The underlying run_ast_inference will resample if needed,
        but for consistency callers should provide 16kHz audio.
        
        Returns:
            16000 (AST's native sample rate)
        """
        return self._sample_rate
    
    def __call__(self, waveform: torch.Tensor) -> Tuple[int, float]:
        """Classify an audio waveform using AST.
        
        The waveform is expected to be at AST_SAMPLE_RATE (16kHz).
        AST computes sigmoid scores for AudioSet classes, then this wrapper
        takes the maximum score across positive_labels as the confidence
        and applies the threshold for binary classification.
        
        Args:
            waveform: Audio tensor (1D) at 16000 Hz.
                     Should be normalized to [-1, 1] range.
        
        Returns:
            A tuple of (prediction, confidence) where:
            - prediction: 1 if max confidence >= threshold, 0 otherwise
            - confidence: Maximum sigmoid score across positive_labels in [0.0, 1.0]
        
        Example:
            >>> wrapper = ASTClassifierWrapper(device="cuda")
            >>> waveform = torch.randn(16000 * 3)  # 3 seconds at 16kHz
            >>> pred, conf = wrapper(waveform)
            >>> print(f"Aircraft: {pred}, Confidence: {conf:.4f}")
        """
        # run_ast_inference handles resampling internally if needed,
        # but we assume waveform is already at AST_SAMPLE_RATE for consistency
        return run_ast_inference(
            self.extractor,
            self.model,
            waveform,
            self._sample_rate,
            self.positive_labels,
            device=self.device,
            threshold=self.threshold,
        )
