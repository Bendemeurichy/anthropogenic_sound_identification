"""
Unified classifier interface for audio classification models.

This module provides a common interface for all audio classifiers used in the validation
pipeline. Classifiers follow a simple callable protocol: given an audio waveform tensor,
they return a binary prediction (0/1) and a confidence score (0.0-1.0).

Supported classifiers:
- "plane": PlaneClassifier (TensorFlow/YAMNet-based CNN for airplane detection)
- "pann": PANN AudioTagging (PyTorch-based AudioSet classifier)
- "pann_finetuned": Fine-tuned PANN CNN14 (PyTorch-based, trained specifically for plane detection)
- "ast": Audio Spectrogram Transformer (HuggingFace transformer for AudioSet)
- "birdnet": BirdNET (acoustic bird species classifier)

Example usage:
    >>> from Classifier import create_classifier
    >>> 
    >>> # Create a plane classifier
    >>> classifier = create_classifier(
    ...     "plane",
    ...     weights_path="path/to/weights.h5",
    ...     device="cuda"
    ... )
    >>> 
    >>> # Load audio and classify
    >>> waveform = load_audio("audio.wav")  # torch.Tensor at classifier.sample_rate
    >>> prediction, confidence = classifier(waveform)
    >>> print(f"Prediction: {prediction}, Confidence: {confidence:.4f}")
    
    >>> # Or use a different classifier - same interface!
    >>> birdnet = create_classifier("birdnet", device="cuda")
    >>> prediction, confidence = birdnet(waveform)
"""

from typing import Tuple, Protocol, Optional, List, runtime_checkable
import torch


@runtime_checkable
class AudioClassifier(Protocol):
    """Protocol defining the unified classifier interface.
    
    All classifiers must implement:
    - __call__(waveform) -> (prediction, confidence)
    - sample_rate property
    
    This allows classifiers to be used interchangeably in the validation pipeline.
    """
    
    def __call__(self, waveform: torch.Tensor) -> Tuple[int, float]:
        """Classify an audio waveform.
        
        Args:
            waveform: Audio tensor (1D) at self.sample_rate Hz.
                     The tensor should be in the range [-1, 1] (normalized audio).
        
        Returns:
            A tuple of (prediction, confidence) where:
            - prediction: Binary class label (0 or 1)
                         1 = positive class (plane/bird detected)
                         0 = negative class (no plane/bird)
            - confidence: Confidence score in range [0.0, 1.0]
        """
        ...
    
    @property
    def sample_rate(self) -> int:
        """Expected sample rate in Hz for input waveforms.
        
        Callers should resample audio to this rate before calling the classifier.
        """
        ...


def create_classifier(
    classifier_type: str,
    device: str = "cpu",
    threshold: float = 0.5,
    **kwargs
) -> AudioClassifier:
    """Factory function to create audio classifiers.
    
    Creates and returns a configured classifier that follows the AudioClassifier protocol.
    All classifiers have a consistent interface: they accept a waveform tensor and return
    (prediction, confidence).
    
    Args:
        classifier_type: Type of classifier to create. One of:
            - "plane": Airplane detection using custom CNN (TensorFlow/YAMNet)
            - "pann": PANN AudioTagging for AudioSet classification
            - "pann_finetuned": Fine-tuned PANN CNN14 for airplane detection
            - "ast": Audio Spectrogram Transformer for AudioSet
            - "birdnet": BirdNET for bird species detection
        device: Device for model inference. Examples: "cpu", "cuda", "cuda:0", "cuda:1"
        threshold: Classification threshold in range [0.0, 1.0]. Predictions with
                  confidence >= threshold are classified as positive (1).
        **kwargs: Classifier-specific arguments (see below)
    
    Classifier-specific kwargs:
    
        Plane classifier:
            weights_path (str, required): Path to .weights.h5 file
            config (TrainingConfig, optional): Configuration object with sample_rate,
                                              audio_duration, etc.
        
        PANN classifier:
            positive_labels (List[str], optional): AudioSet labels for positive class.
                Default: ["Fixed-wing aircraft, airplane", "Aircraft", "Jet aircraft",
                         "Propeller, airscrew", "Turboprop, small aircraft"]
        
        PANN fine-tuned classifier:
            checkpoint_path (str, required): Path to fine-tuned model checkpoint (.pth file)
            config (TrainingConfig, optional): Training configuration
        
        AST classifier:
            positive_labels (List[str], optional): AudioSet labels for positive class.
                Default: Same as PANN
        
        BirdNET classifier:
            model_version (str, optional): BirdNET version, default "2.4"
            backend (str, optional): "tf" (TensorFlow) or "tflite", default "tf"
            target_species (List[str], optional): List of bird species to detect.
                If None, detects any bird species.
            detect_any_bird (bool, optional): If True (default), return positive for
                any bird species. If False, only return positive for target_species.
    
    Returns:
        An AudioClassifier instance that can be called with a waveform tensor.
    
    Raises:
        ValueError: If classifier_type is not recognized
        ImportError: If required dependencies for the classifier are not installed
    
    Examples:
        >>> # Create plane classifier
        >>> plane_clf = create_classifier(
        ...     "plane",
        ...     weights_path="models/plane.weights.h5",
        ...     device="cuda:0"
        ... )
        >>> 
        >>> # Create PANN with custom labels
        >>> pann_clf = create_classifier(
        ...     "pann",
        ...     device="cuda:1",
        ...     positive_labels=["Aircraft", "Jet aircraft"],
        ...     threshold=0.7
        ... )
        >>> 
        >>> # Create BirdNET for any bird detection
        >>> bird_clf = create_classifier(
        ...     "birdnet",
        ...     device="cpu",
        ...     detect_any_bird=True
        ... )
        >>> 
        >>> # All have the same interface
        >>> for clf in [plane_clf, pann_clf, bird_clf]:
        ...     pred, conf = clf(waveform)
        ...     print(f"Sample rate: {clf.sample_rate} Hz")
        ...     print(f"Prediction: {pred}, Confidence: {conf:.4f}")
    """
    classifier_type = classifier_type.lower().strip()
    
    if classifier_type == "plane":
        from .plane_wrapper import PlaneClassifierWrapper
        
        weights_path = kwargs.pop("weights_path", None)
        if weights_path is None:
            raise ValueError("'weights_path' is required for plane classifier")
        
        config = kwargs.pop("config", None)
        return PlaneClassifierWrapper(
            weights_path=weights_path,
            config=config,
            threshold=threshold,
            device=device,
            **kwargs
        )
    
    elif classifier_type == "pann":
        from .pann_wrapper import PANNClassifierWrapper
        
        return PANNClassifierWrapper(
            device=device,
            threshold=threshold,
            **kwargs
        )
    
    elif classifier_type == "pann_finetuned":
        from .pann_finetuned_wrapper import PANNFinetunedWrapper
        
        checkpoint_path = kwargs.pop("checkpoint_path", None)
        if checkpoint_path is None:
            raise ValueError("'checkpoint_path' is required for pann_finetuned classifier")
        
        config = kwargs.pop("config", None)
        return PANNFinetunedWrapper(
            checkpoint_path=checkpoint_path,
            config=config,
            device=device,
            threshold=threshold,
            **kwargs
        )
    
    elif classifier_type == "ast":
        from .ast_wrapper import ASTClassifierWrapper
        
        return ASTClassifierWrapper(
            device=device,
            threshold=threshold,
            **kwargs
        )
    
    elif classifier_type == "birdnet":
        from .birdnet_wrapper import BirdNETClassifierWrapper
        
        return BirdNETClassifierWrapper(
            device=device,
            threshold=threshold,
            **kwargs
        )
    
    else:
        raise ValueError(
            f"Unknown classifier_type: {classifier_type!r}. "
            f"Must be one of: 'plane', 'pann', 'pann_finetuned', 'ast', 'birdnet'"
        )