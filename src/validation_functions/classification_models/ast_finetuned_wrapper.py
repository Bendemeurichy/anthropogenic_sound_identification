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
        checkpoint_path: Optional[str] = None,
        config=None,  # TrainingConfig
        device: str = "cuda",
        threshold: float = 0.5,
    ):
        """Initialize the AST classifier.

        Args:
            checkpoint_path: Path to a fine-tuned model checkpoint (.pth file).
                             If None, loads the pretrained AST backbone only
                             (no task-specific finetuning).
            config: TrainingConfig (uses defaults if None)
            device: Device for inference ("cuda", "cuda:0", "cpu")
            threshold: Classification threshold (default: 0.5)
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.threshold = threshold
        self._use_pretrained_fallback = False
        self._ast_for_embedding = None
        
        if checkpoint_path is not None:
            from validation_functions.classification_models.plane_classifier_ast.model_loader import (
                load_trained_model,
            )
            from validation_functions.classification_models.plane_classifier_ast.config import ModelConfig
            
            self.model = load_trained_model(
                checkpoint_path=checkpoint_path,
                config=ModelConfig() if config is None else None,
                training_config=config,
                device=device,
            )
        else:
            # Without a checkpoint, use the pretrained AST model directly.
            # The pretrained model outputs 527 AudioSet classes, including:
            #   - Index 335: Aircraft
            #   - Index 336: Aircraft engine
            #   - Index 340: Fixed-wing aircraft, airplane
            # We'll extract all airplane-related labels for binary classification.
            from transformers import ASTForAudioClassification
            self.model = ASTForAudioClassification.from_pretrained(
                "MIT/ast-finetuned-audioset-10-10-0.4593",
                num_labels=527
            ).to(device)
            
            # Indices for all airplane-related labels in AudioSet
            self._airplane_label_indices = [335, 336, 340]  # Aircraft, Aircraft engine, Fixed-wing aircraft
            self._use_pretrained_fallback = True
        
        self.model.eval()
        
        from transformers import ASTFeatureExtractor
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        
        # AST works at 16kHz
        self._sample_rate = self.feature_extractor.sampling_rate
        self._segment_length = 10.0
        self._segment_samples = int(self._sample_rate * self._segment_length)
        
        if checkpoint_path is not None:
            print(f"Loaded fine-tuned AST classifier from {checkpoint_path}")
        else:
            print("Loaded pretrained AST classifier (no finetuning)")
        print(f"  Sample rate: {self._sample_rate} Hz")
        print(f"  Segment length: {self._segment_length} s ({self._segment_samples} samples)")
        print(f"  Threshold: {threshold}")
    
    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def segment_samples(self) -> int:
        """Number of samples expected per segment at self.sample_rate."""
        return self._segment_samples
    
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
        
        if self._use_pretrained_fallback:
            # Use the pretrained AST model's airplane labels (indices 335, 336, 340)
            # The model outputs 527 AudioSet class logits
            outputs = self.model(input_values)
            all_logits = outputs.logits  # (B, 527)
            # Extract and aggregate all airplane-related label logits
            airplane_logits = torch.index_select(
                all_logits, 1, 
                torch.tensor(self._airplane_label_indices, device=self.device)
            )  # (B, 3)
            # Take the maximum logit across all airplane labels as the final logit
            logits = torch.max(airplane_logits, dim=1, keepdim=True)[0]  # (B, 1)
        else:
            # Use the full fine-tuned PlaneClassifierAST
            logits = self.model(input_values)  # (B, 1)
        
        confidences = torch.sigmoid(logits).squeeze(-1)  # (B,)
        
        predictions = (confidences >= self.threshold).long()
        
        return predictions, confidences
