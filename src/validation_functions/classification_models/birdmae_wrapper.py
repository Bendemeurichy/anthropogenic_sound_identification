import torch
import transformers
from transformers import AutoFeatureExtractor, AutoModelForSequenceClassification
from typing import Tuple
import os

def apply_monkeypatches():
    # Patch 1: torch.linspace crash on meta tensors
    if not hasattr(torch, '_original_linspace'):
        torch._original_linspace = torch.linspace
        def patched_linspace(*args, **kwargs):
            if 'device' in kwargs and str(kwargs['device']) == 'meta':
                kwargs['device'] = 'cpu'
                return torch._original_linspace(*args, **kwargs).to('meta')
            return torch._original_linspace(*args, **kwargs)
        torch.linspace = patched_linspace

    # Patch 2: all_tied_weights_keys missing
    if not hasattr(transformers.modeling_utils.PreTrainedModel, 'all_tied_weights_keys'):
        transformers.modeling_utils.PreTrainedModel.all_tied_weights_keys = property(lambda self: {})

class BirdMaeClassifierWrapper:
    """Wrapper for BirdMAE-XCL model from HuggingFace."""

    def __init__(self, model_id="DBD-research-group/BirdMAE-XCL", device="cpu", threshold=0.5, **kwargs):
        self.device = device
        self.threshold = threshold
        self.model_id = model_id
        
        apply_monkeypatches()
        
        print(f"Loading Bird-MAE model from {self.model_id}...")
        
        # Windows compatibility and performance: disable symlink warnings, try local first
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        
        # Try loading from cache first for faster startup
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
        except Exception:
            # Fallback to downloading if not cached
            print("  Model not cached locally, downloading from HuggingFace (this may take a while)...")
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
        print(f"  Bird-MAE loaded successfully (sample_rate={self._sample_rate} Hz)")

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
        
        # Feature extraction
        inputs = self.feature_extractor(waveforms_np)
        
        # Handle different return types from custom feature extractor
        if isinstance(inputs, dict):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
        else:
            # Convert to tensor if needed and move to device
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.tensor(inputs).to(self.device)
            else:
                inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            
        logits = outputs.logits
        
        # Take max probability across all bird classes
        probs = torch.sigmoid(logits)
        max_probs, _ = torch.max(probs, dim=1)
        
        preds = (max_probs >= self.threshold).long()
        return preds, max_probs
