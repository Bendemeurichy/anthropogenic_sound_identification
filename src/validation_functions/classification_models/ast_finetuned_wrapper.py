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
        from validation_functions.classification_models.plane_classifier_ast.model_loader import (
            load_trained_model,
            create_plane_classifier,
        )
        from validation_functions.classification_models.plane_classifier_ast.config import ModelConfig
        
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.threshold = threshold
        self._use_pretrained_fallback = False
        self._ast_for_embedding = None
        
        if checkpoint_path is not None:
            self.model = load_trained_model(
                checkpoint_path=checkpoint_path,
                config=ModelConfig() if config is None else None,
                training_config=config,
                device=device,
            )
        else:
            # Without a checkpoint, use a simpler approach:
            # Extract AST embeddings and use a lightweight classifier
            from validation_functions.classification_models.plane_classifier_ast.model_loader import (
                load_pretrained_ast,
            )
            self._ast_for_embedding = load_pretrained_ast(
                config=ModelConfig() if config is None else None,
                device=device,
            )
            
            # Create a simple classifier head for plane detection
            # Using a lightweight 2-layer MLP on the AST embedding (768 dims)
            import torch.nn as nn
            self.model = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1)
            ).to(device)
            
            # Initialize with better weights
            for module in self.model.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            
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
            # Use AST embedding + simple classifier
            ast_outputs = self._ast_for_embedding(input_values)
            embedding = ast_outputs.pooler_output
            logits = self.model(embedding)
        else:
            # Use the full fine-tuned PlaneClassifierAST
            logits = self.model(input_values)  # (B, 1)
        
        confidences = torch.sigmoid(logits).squeeze(-1)  # (B,)
        
        predictions = (confidences >= self.threshold).long()
        
        return predictions, confidences
