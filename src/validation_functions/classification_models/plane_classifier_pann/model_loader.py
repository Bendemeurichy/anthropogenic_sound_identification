"""
Utility to load pretrained and trained PANN models.

Provides functions to:
- Download pretrained CNN14 weights from Zenodo
- Load pretrained CNN14 model
- Load fine-tuned PlaneClassifierPANN from checkpoint
"""

import torch
import requests
from pathlib import Path
from tqdm import tqdm
from typing import Optional
import os

from model import Cnn14, PlaneClassifierPANN
from config import ModelConfig, TrainingConfig


def download_pretrained_weights(
    url: str,
    save_path: str,
    force_download: bool = False
) -> str:
    """
    Download pretrained PANN weights from URL.
    
    Args:
        url: URL to download from (Zenodo link)
        save_path: Path to save the weights file
        force_download: If True, download even if file exists
        
    Returns:
        Path to downloaded weights file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_path.exists() and not force_download:
        print(f"Pretrained weights already exist at {save_path}")
        return str(save_path)
    
    print(f"Downloading pretrained weights from {url}")
    print(f"Saving to {save_path}")
    
    # Stream download with progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with open(save_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"Downloaded pretrained weights to {save_path}")
    return str(save_path)


def load_pretrained_cnn14(
    config: Optional[ModelConfig] = None,
    weights_path: Optional[str] = None,
    device: str = 'cuda'
) -> Cnn14:
    """
    Load pretrained CNN14 model.
    
    Args:
        config: ModelConfig (uses defaults if None)
        weights_path: Path to pretrained weights .pth file.
                     If None, downloads from config.pretrained_weights_url
        device: Device to load model on ('cuda' or 'cpu')
        
    Returns:
        CNN14 model with pretrained weights loaded
        
    Example:
        >>> cnn14 = load_pretrained_cnn14()
        >>> embedding = cnn14(waveform, return_embedding=True)
    """
    if config is None:
        config = ModelConfig()
    
    # Create CNN14 model
    cnn14 = Cnn14(
        sample_rate=config.pann_sample_rate,
        window_size=config.pann_window_size,
        hop_size=config.pann_hop_size,
        mel_bins=config.pann_mel_bins,
        fmin=config.pann_fmin,
        fmax=config.pann_fmax,
        classes_num=527  # AudioSet classes
    )
    
    # Get weights path
    if weights_path is None:
        if config.pretrained_weights_path is not None:
            weights_path = config.pretrained_weights_path
        else:
            # Download from Zenodo
            cache_dir = Path.home() / '.cache' / 'pann'
            weights_path = download_pretrained_weights(
                config.pretrained_weights_url,
                cache_dir / 'Cnn14_mAP=0.431.pth'
            )
    
    # Load pretrained weights
    print(f"Loading pretrained weights from {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    cnn14.load_state_dict(state_dict, strict=False)
    cnn14 = cnn14.to(device)
    
    print("Pretrained CNN14 loaded successfully")
    return cnn14


def create_plane_classifier(
    config: Optional[ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    pretrained_weights_path: Optional[str] = None,
    fine_tune: bool = False,
    device: str = 'cuda'
) -> PlaneClassifierPANN:
    """
    Create PlaneClassifierPANN model with pretrained CNN14 backbone.
    
    Args:
        config: ModelConfig for architecture
        training_config: TrainingConfig (alternative to config)
        pretrained_weights_path: Path to pretrained CNN14 weights
        fine_tune: Whether CNN14 backbone should be trainable
        device: Device to load model on
        
    Returns:
        PlaneClassifierPANN model ready for training or inference
        
    Example:
        >>> # For training
        >>> model = create_plane_classifier(fine_tune=False, device='cuda')
        >>> 
        >>> # For fine-tuning
        >>> model = create_plane_classifier(fine_tune=True, device='cuda')
    """
    if config is None:
        config = ModelConfig()
        
        # If training_config provided, use its parameters
        if training_config is not None:
            config.hidden_units = training_config.hidden_units
            config.dropout_rates = [
                training_config.dropout_rate_1,
                training_config.dropout_rate_2,
                training_config.dropout_rate_3,
            ]
    
    # Load pretrained CNN14
    cnn14 = load_pretrained_cnn14(config, pretrained_weights_path, device)
    
    # Create classifier
    model = PlaneClassifierPANN(cnn14, config, fine_tune=fine_tune)
    model = model.to(device)
    
    return model


def load_trained_model(
    checkpoint_path: str,
    config: Optional[ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    device: str = 'cuda'
) -> PlaneClassifierPANN:
    """
    Load a fine-tuned PlaneClassifierPANN model from checkpoint.
    
    Args:
        checkpoint_path: Path to saved checkpoint (.pth file)
        config: ModelConfig (must match training architecture)
        training_config: TrainingConfig (alternative to config)
        device: Device to load model on
        
    Returns:
        Loaded PlaneClassifierPANN model
        
    Example:
        >>> model = load_trained_model(
        ...     "checkpoints/best_model_phase2.pth",
        ...     device='cuda'
        ... )
        >>> logits = model(waveform)
        >>> probs = torch.sigmoid(logits)
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create model architecture (will load pretrained CNN14)
    model = create_plane_classifier(
        config=config,
        training_config=training_config,
        fine_tune=False,  # Will be set from checkpoint
        device=device
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Assume checkpoint is just the state dict
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully")
    
    return model


def save_checkpoint(
    model: PlaneClassifierPANN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    save_path: str,
    val_metrics: Optional[dict] = None
):
    """
    Save model checkpoint.
    
    Args:
        model: PlaneClassifierPANN model
        optimizer: Optimizer
        epoch: Current epoch number
        save_path: Path to save checkpoint
        val_metrics: Optional validation metrics dict
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'fine_tune': model.fine_tune,
    }
    
    if val_metrics is not None:
        checkpoint['val_metrics'] = val_metrics
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


if __name__ == "__main__":
    # Test loading pretrained model
    import sys
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("\n" + "="*70)
    print("Testing pretrained CNN14 loading")
    print("="*70)
    
    cnn14 = load_pretrained_cnn14(device=device)
    print(f"CNN14 loaded. Total parameters: {sum(p.numel() for p in cnn14.parameters()):,}")
    
    print("\n" + "="*70)
    print("Testing PlaneClassifierPANN creation")
    print("="*70)
    
    model = create_plane_classifier(device=device)
    print(f"Model created. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    print("\n" + "="*70)
    print("Testing forward pass")
    print("="*70)
    
    batch_size = 2
    audio_length = 32000 * 10  # 10 seconds at 32kHz
    dummy_input = torch.randn(batch_size, audio_length).to(device)
    
    with torch.no_grad():
        logits = model(dummy_input)
        probs = torch.sigmoid(logits)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output probabilities: {probs.squeeze().cpu().numpy()}")
    
    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)
