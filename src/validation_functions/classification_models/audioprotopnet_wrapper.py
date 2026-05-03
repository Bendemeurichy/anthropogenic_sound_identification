import torch
from transformers import AutoFeatureExtractor, AutoModelForSequenceClassification
from typing import Tuple
import os

class AudioProtoPNetClassifierWrapper:
    """Wrapper for AudioProtoPNet-20-BirdSet-XCL model from HuggingFace."""

    def __init__(self, model_id="DBD-research-group/AudioProtoPNet-20-BirdSet-XCL", device="cpu", threshold=0.5, **kwargs):
        self.device = device
        self.threshold = threshold
        self.model_id = model_id
        
        print(f"Loading AudioProtoPNet model from {self.model_id}...")
        
        # Windows compatibility: trust_remote_code causes SIGALRM error on Windows
        # Always trust code for HuggingFace models to avoid timeout issues
        # Set env var to disable symlink warnings
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        
        # Load with trust_remote_code=True and local_files_only fallback for faster loading
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                local_files_only=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                local_files_only=True
            ).to(self.device)
        except Exception as e:
            # Fallback to downloading if not cached
            print(f"  Model not cached locally (error: {e}), downloading from HuggingFace (this may take a while)...")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                trust_remote_code=True
            ).to(self.device)
        
        self.model.eval()
        
        self._sample_rate = self.feature_extractor.sampling_rate
        print(f"  AudioProtoPNet loaded successfully (sample_rate={self._sample_rate} Hz)")

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @torch.inference_mode()
    def __call__(self, waveform: torch.Tensor) -> Tuple[int, float]:
        """Single waveform inference."""
        preds, confs = self.predict_batch(waveform.unsqueeze(0))
        return int(preds[0].item()), float(confs[0].item())

    @torch.inference_mode()
    def predict_batch(self, waveforms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch inference for better performance.
        
        Args:
            waveforms: (B, T) batch of waveforms
            
        Returns:
            Tuple of (predictions, confidences) both shape (B,)
        """
        # Move to CPU and convert to numpy for feature extraction
        waveforms_np = waveforms.cpu().numpy()
        
        # Feature extraction (handles batching internally)
        inputs = self.feature_extractor(
            waveforms_np,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Model inference
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # Take max probability across all bird classes
        probs = torch.sigmoid(logits)
        max_probs, _ = torch.max(probs, dim=1)
        
        preds = (max_probs >= self.threshold).long()
        return preds, max_probs
