import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from typing import Tuple

class AudioProtoPNetClassifierWrapper:
    """Wrapper for AudioProtoPNet-20-BirdSet-XCL model from HuggingFace."""

    def __init__(self, model_id="DBD-research-group/AudioProtoPNet-20-BirdSet-XCL", device="cpu", threshold=0.5, **kwargs):
        self.device = device
        self.threshold = threshold
        self.model_id = model_id
        
        print(f"Loading AudioProtoPNet model from {self.model_id}...")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
        self.model = AutoModelForAudioClassification.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        
        self._sample_rate = self.feature_extractor.sampling_rate

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @torch.inference_mode()
    def __call__(self, waveform: torch.Tensor) -> Tuple[int, float]:
        preds, confs = self.predict_batch(waveform.unsqueeze(0))
        return int(preds[0].item()), float(confs[0].item())

    @torch.inference_mode()
    def predict_batch(self, waveforms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # waveforms: (B, T)
        waveforms = waveforms.cpu().numpy()
        inputs = self.feature_extractor(waveforms, sampling_rate=self.sample_rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        logits = outputs.logits
        # We take the max probability across all bird classes (if any class is > threshold)
        probs = torch.sigmoid(logits)
        max_probs, _ = torch.max(probs, dim=1)
        
        preds = (max_probs >= self.threshold).long()
        return preds, max_probs
