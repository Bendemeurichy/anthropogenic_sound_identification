import torch
import torch.nn as nn
from transformers import ASTModel, ASTConfig, ASTFeatureExtractor
from typing import Optional

from config import ModelConfig

class PlaneClassifierAST(nn.Module):
    """
    Binary classifier for plane/no-plane audio using AST (Audio Spectrogram Transformer) backbone.
    
    Architecture:
        Input: Raw waveform (batch_size, samples)
        ↓
        AST backbone → 768-dim embedding (cls token)
        ↓
        Dense(512) + ReLU + BatchNorm + Dropout(0.3)
        ↓
        Dense(256) + ReLU + BatchNorm + Dropout(0.2)
        ↓
        Dense(128) + ReLU + BatchNorm + Dropout(0.1)
        ↓
        Dense(1) → Binary logit
    """
    
    def __init__(
        self,
        ast_model: ASTModel,
        config: Optional[ModelConfig] = None,
        fine_tune: bool = False
    ):
        super(PlaneClassifierAST, self).__init__()
        
        if config is None:
            config = ModelConfig()
            
        self.config = config
        self.ast = ast_model
        self._fine_tune = fine_tune
        
        self.set_fine_tune(fine_tune)
        
        layers = []
        prev_units = config.embedding_dim
        
        for i, (units, dropout) in enumerate(zip(config.hidden_units, config.dropout_rates)):
            layers.append(nn.Linear(prev_units, units))
            if config.activation == 'relu':
                layers.append(nn.ReLU())
            elif config.activation == 'swish':
                layers.append(nn.SiLU())
            else:
                raise ValueError(f"Unsupported activation: {config.activation}")
            
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(units))
                
            layers.append(nn.Dropout(dropout))
            prev_units = units
            
        layers.append(nn.Linear(prev_units, 1))
        self.classifier = nn.Sequential(*layers)
        
        self._init_classifier_weights()

    def _init_classifier_weights(self):
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.)
            elif isinstance(module, nn.BatchNorm1d):
                module.bias.data.fill_(0.)
                module.weight.data.fill_(1.)

    def set_fine_tune(self, fine_tune: bool):
        self._fine_tune = fine_tune
        for param in self.ast.parameters():
            param.requires_grad = fine_tune

    @property
    def fine_tune(self) -> bool:
        return self._fine_tune
        
    @fine_tune.setter
    def fine_tune(self, value: bool):
        self.set_fine_tune(value)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            input_values: Processed spectrograms from ASTFeatureExtractor
                          shape (batch_size, max_length, num_mel_bins)
        """
        # ast output contains pooler_output (cls token)
        outputs = self.ast(input_values)
        embedding = outputs.pooler_output
        
        logits = self.classifier(embedding)
        return logits

    def get_embedding(self, input_values: torch.Tensor) -> torch.Tensor:
        outputs = self.ast(input_values)
        return outputs.pooler_output
