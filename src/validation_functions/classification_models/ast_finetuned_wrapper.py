"""
Wrapper for fine-tuned AST plane classifier.

This wrapper adapts the PlaneClassifierAST to conform
to the unified AudioClassifier interface used in test_pipeline.py.
"""

from typing import Tuple, Optional

import torch

class ASTFinetunedWrapper:
    """Wrapper for fine-tuned AST plane classifier conforming to AudioClassifier protocol."""
    
    def __init__(
        self,
        checkpoint_path: str,
        config=None,  # TrainingConfig
        device: str = "cuda",
        threshold: float = 0.5,
    ):
        """Initialize the fine-tuned AST classifier."""
        from validation_functions.classification_models.plane_classifier_ast.model_loader import load_trained_model
        from validation_functions.classification_models.plane_classifier_ast.config import ModelConfig
        
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.threshold = threshold
        
        self.model = load_trained_model(
            checkpoint_path=checkpoint_path,
            config=ModelConfig() if config is None else None,
            training_config=config,
            device=device,
        )
        self.model.eval()
        
        from transformers import ASTFeatureExtractor
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        
        # AST works at 16kHz
        self._sample_rate = self.feature_extractor.sampling_rate
        self._segment_length = 10.0
        self._segment_samples = int(self._sample_rate * self._segment_length)
        
        print(f"Loaded fine-tuned AST classifier from {checkpoint_path}")
        print(f"  Sample rate: {self._sample_rate} Hz")
        print(f"  Segment length: {self._segment_length} s ({self._segment_samples} samples)")
        print(f"  Threshold: {threshold}")
    
    @property
    def sample_rate(self) -> int:
        return self._sample_rate
    
    @torch.inference_mode()
    def __call__(self, waveform: torch.Tensor) -> Tuple[int, float]:
        preds, confs = self.predict_batch(waveform.unsqueeze(0))
        return int(preds[0].item()), float(confs[0].item())

    @torch.inference_mode()
    def predict_batch(self, waveforms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Classify a batch of audio waveforms."""
        # Pad or truncate to expected length
        if waveforms.shape[1] < self._segment_samples:
            waveforms = torch.nn.functional.pad(
                waveforms, (0, self._segment_samples - waveforms.shape[1])
            )
        else:
            waveforms = waveforms[:, :self._segment_samples]
        
        # Extract features using ASTFeatureExtractor
        inputs = self.feature_extractor(
            waveforms.cpu().numpy(), 
            sampling_rate=self._sample_rate, 
            return_tensors="pt"
        )
        input_values = inputs.input_values.to(self.device)
        
        self.model.eval()
        logits = self.model(input_values)  # (B, 1)
        confidences = torch.sigmoid(logits).squeeze(-1)  # (B,)
        
        predictions = (confidences >= self.threshold).long()
        
        return predictions, confidences
