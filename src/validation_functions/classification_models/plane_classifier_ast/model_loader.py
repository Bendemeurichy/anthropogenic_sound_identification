"""
Utility to load pretrained and trained AST models.
"""

import torch
from pathlib import Path
from typing import Optional
from transformers import ASTModel

from model import PlaneClassifierAST
from config import ModelConfig, TrainingConfig


def load_pretrained_ast(
    config: Optional[ModelConfig] = None,
    device: str = 'cuda'
) -> ASTModel:
    if config is None:
        config = ModelConfig()
        
    print(f"Loading pretrained AST model: {config.ast_model_name}")
    model = ASTModel.from_pretrained(config.ast_model_name)
    model = model.to(device)
    
    print("Pretrained AST loaded successfully")
    return model


def create_plane_classifier(
    config: Optional[ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    fine_tune: bool = False,
    device: str = 'cuda'
) -> PlaneClassifierAST:
    if config is None:
        config = ModelConfig()
        
        if training_config is not None:
            config.hidden_units = training_config.hidden_units
            config.dropout_rates = [
                training_config.dropout_rate_1,
                training_config.dropout_rate_2,
                training_config.dropout_rate_3,
            ]
            
    ast_model = load_pretrained_ast(config, device)
    model = PlaneClassifierAST(ast_model, config, fine_tune=fine_tune)
    model = model.to(device)
    
    return model


def load_trained_model(
    checkpoint_path: str,
    config: Optional[ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    device: str = 'cuda'
) -> PlaneClassifierAST:
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    model = create_plane_classifier(
        config=config,
        training_config=training_config,
        fine_tune=False,
        device=device
    )
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    print("Model loaded successfully")
    
    return model


def save_checkpoint(
    model: PlaneClassifierAST,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    save_path: str,
    val_metrics: Optional[dict] = None
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'fine_tune': model.fine_tune,
    }
    
    if val_metrics is not None:
        checkpoint['val_metrics'] = val_metrics
        
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")
