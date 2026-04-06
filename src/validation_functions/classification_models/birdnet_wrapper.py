"""Wrapper for BirdNET bird species acoustic classifier.

BirdNET is a deep learning model for identifying bird species by their sounds.
This wrapper adapts it to the unified classifier interface for binary bird detection.
"""

from typing import List, Tuple, Optional
import sys
import torch
import numpy as np

# Optional import - only loaded when BirdNET classifier is used
try:
    import birdnet
    BIRDNET_AVAILABLE = True
except ImportError:
    BIRDNET_AVAILABLE = False
    birdnet = None


class BirdNETClassifierWrapper:
    """Wrapper for BirdNET acoustic classifier that follows the AudioClassifier protocol.
    
    BirdNET can identify 6,522 bird species from audio. This wrapper provides binary
    detection: either detect ANY bird species or detect specific target species.
    
    The model returns predictions for 3-second chunks by default, so input waveforms
    should ideally be at least 3 seconds long.
    
    Attributes:
        sample_rate: Expected sample rate (48000 Hz for BirdNET 2.4)
        device: Device for model inference
        model: Loaded BirdNET acoustic model
        threshold: Classification threshold for binary decision
        target_species: Optional list of specific species to detect
        detect_any_bird: If True, detect any bird; if False, only target_species
    """
    
    def __init__(
        self,
        device: str = "cpu",
        model_version: str = "2.4",
        backend: str = "tf",  # or "tflite"
        target_species: Optional[List[str]] = None,
        threshold: float = 0.5,
        detect_any_bird: bool = True,
    ):
        """Initialize the BirdNET classifier wrapper.
        
        Args:
            device: Device for inference. Note that BirdNET's device handling may
                   differ from PyTorch - "cpu" is safest for compatibility.
            model_version: BirdNET model version, default "2.4" (supports 6,522 species)
            backend: Model backend - "tf" (TensorFlow ProtoBuf) or "tflite" (TensorFlow Lite).
                    TensorFlow backend supports GPU, TFLite is CPU-only.
            target_species: List of bird species names to detect (BirdNET format).
                          Example: ["Turdus merula_Common Blackbird", "Parus major_Great Tit"]
                          If None and detect_any_bird=True, detects all 6,522 species.
            threshold: Classification threshold. Predictions with confidence >= threshold
                      are classified as positive (bird detected = 1).
            detect_any_bird: If True (default), return positive (1) when ANY bird species
                           is detected above threshold. If False, only return positive
                           when one of the target_species is detected above threshold.
        
        Raises:
            ImportError: If birdnet package is not installed
        """
        if not BIRDNET_AVAILABLE:
            raise ImportError(
                "BirdNET is not installed. Install with:\n"
                "  pip install birdnet\n"
                "For GPU support (Linux only):\n"
                "  pip install birdnet[and-cuda]"
            )
        
        self.device = device
        self.threshold = threshold
        self.target_species = target_species
        self.detect_any_bird = detect_any_bird
        self.model_version = model_version
        self.backend = backend
        
        # Load BirdNET acoustic model
        print(f"Loading BirdNET {model_version} model (backend={backend})...", file=sys.stderr)
        self.model = birdnet.load("acoustic", model_version, backend)
        print(f"BirdNET model loaded successfully.", file=sys.stderr)
        
        # BirdNET v2.4 expects 48kHz audio
        # Note: The actual sample rate may vary by model version
        self._sample_rate = 48000
    
    @property
    def sample_rate(self) -> int:
        """Expected sample rate in Hz for input waveforms.
        
        BirdNET v2.4 models expect 48kHz audio. Older versions may use different rates.
        The model handles resampling internally if needed.
        
        Returns:
            48000 (BirdNET v2.4's native sample rate)
        """
        return self._sample_rate
    
    def __call__(self, waveform: torch.Tensor) -> Tuple[int, float]:
        """Classify an audio waveform for bird presence.
        
        BirdNET analyzes the audio and returns species predictions with confidence scores.
        This wrapper aggregates those predictions into a binary decision:
        - If detect_any_bird=True: positive if ANY species detected above threshold
        - If detect_any_bird=False: positive only if target_species detected above threshold
        
        Args:
            waveform: Audio tensor (1D) at 48000 Hz (or will be resampled internally).
                     Should be normalized to [-1, 1] range.
                     Recommended length: at least 3 seconds (144000 samples at 48kHz).
        
        Returns:
            A tuple of (prediction, confidence) where:
            - prediction: 1 if bird detected above threshold, 0 otherwise
            - confidence: Maximum confidence score across detected species in [0.0, 1.0]
        
        Example:
            >>> wrapper = BirdNETClassifierWrapper(detect_any_bird=True)
            >>> waveform = torch.randn(48000 * 3)  # 3 seconds at 48kHz
            >>> pred, conf = wrapper(waveform)
            >>> print(f"Bird detected: {pred}, Confidence: {conf:.4f}")
            
            >>> # Detect specific species only
            >>> wrapper2 = BirdNETClassifierWrapper(
            ...     target_species=["Turdus merula_Common Blackbird"],
            ...     detect_any_bird=False
            ... )
            >>> pred, conf = wrapper2(waveform)
        """
        # Convert torch tensor to numpy if needed
        if isinstance(waveform, torch.Tensor):
            wav_np = waveform.detach().cpu().numpy()
        else:
            wav_np = np.asarray(waveform)
        
        # Ensure correct dtype
        if wav_np.dtype != np.float32:
            wav_np = wav_np.astype(np.float32)
        
        # Run BirdNET prediction
        # The model.predict() method expects audio as numpy array
        # and returns a DataFrame with columns: ['start_time', 'end_time', 'species_name', 'confidence']
        try:
            if self.detect_any_bird and self.target_species is None:
                # Detect all species
                predictions = self.model.predict(wav_np)
            else:
                # Detect specific species or filter by target_species
                predictions = self.model.predict(
                    wav_np,
                    custom_species_list=self.target_species
                )
            
            # Get maximum confidence across all predictions
            if predictions is not None and len(predictions) > 0:
                # predictions is a pandas DataFrame or similar structure
                if hasattr(predictions, 'confidence'):
                    # It's a DataFrame
                    confidences = predictions['confidence'].values
                    max_conf = float(np.max(confidences)) if len(confidences) > 0 else 0.0
                elif hasattr(predictions, '__getitem__'):
                    # Try to access as dict-like or array-like
                    try:
                        confidences = predictions['confidence']
                        max_conf = float(np.max(confidences)) if len(confidences) > 0 else 0.0
                    except (KeyError, TypeError):
                        # Fallback: assume it's an array of confidence values
                        max_conf = float(np.max(predictions)) if len(predictions) > 0 else 0.0
                else:
                    max_conf = 0.0
            else:
                max_conf = 0.0
            
            # Apply threshold for binary classification
            pred = 1 if max_conf >= self.threshold else 0
            return pred, max_conf
            
        except Exception as e:
            # If prediction fails, log warning and return negative prediction
            print(
                f"[BirdNET] Warning: Prediction failed: {e}. Returning negative prediction.",
                file=sys.stderr
            )
            return 0, 0.0
