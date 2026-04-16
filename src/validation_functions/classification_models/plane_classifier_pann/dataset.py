"""
PyTorch Dataset and DataLoader for PANN plane classifier.

This module provides a thin wrapper around the shared AudioClassificationDataset
with PANN-specific configuration (32kHz sample rate, 10-second duration).
"""

import pandas as pd
from torch.utils.data import DataLoader
from typing import Tuple

from common.audio_dataset import AudioClassificationDataset
from config import TrainingConfig
from data_config import DataLoaderConfig


class PANNDataset(AudioClassificationDataset):
    """
    PANN-specific dataset configuration.
    
    Preconfigured with:
    - Sample rate: 32kHz (PANN native sample rate)
    - Duration: 10 seconds (PANN works well with longer clips)
    - Augmentation: Time stretch, noise, gain
    
    Args:
        df: DataFrame with audio metadata
        config: TrainingConfig or DataLoaderConfig
        augment: Whether to apply augmentations (training only)
    
    Example:
        >>> train_dataset = PANNDataset(train_df, config, augment=True)
        >>> waveform, label = train_dataset[0]
        >>> print(waveform.shape)  # (320000,) = 10s at 32kHz
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        config,  # TrainingConfig or DataLoaderConfig
        augment: bool = False
    ):
        # Extract configuration parameters
        if hasattr(config, 'aug_time_stretch_prob'):
            # TrainingConfig
            aug_config = {
                'time_stretch_prob': config.aug_time_stretch_prob if config.use_augmentation else 0.0,
                'time_stretch_range': config.aug_time_stretch_range,
                'noise_prob': config.aug_noise_prob if config.use_augmentation else 0.0,
                'noise_stddev': config.aug_noise_stddev,
                'gain_prob': config.aug_gain_prob if config.use_augmentation else 0.0,
                'gain_range': config.aug_gain_range,
            } if augment else None
        else:
            # DataLoaderConfig
            aug_config = {
                'time_stretch_prob': config.aug_time_stretch_prob if config.use_augmentation else 0.0,
                'time_stretch_range': config.aug_time_stretch_range,
                'noise_prob': config.aug_noise_prob if config.use_augmentation else 0.0,
                'noise_stddev': config.aug_noise_stddev,
                'gain_prob': config.aug_gain_prob if config.use_augmentation else 0.0,
                'gain_range': config.aug_gain_range,
            } if augment else None
        
        super().__init__(
            df=df,
            target_sample_rate=config.sample_rate,  # 32000 Hz
            audio_duration=config.audio_duration,  # 10 seconds
            augment=augment,
            augmentation_config=aug_config,
            filename_col=config.filename_column,
            start_time_col=config.start_time_column,
            end_time_col=config.end_time_column,
            label_col=config.label_column,
            split_long=config.split_long,
            min_clip_length=getattr(config, 'min_clip_length', 0.5),
        )


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: TrainingConfig
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders for PANN training.
    
    Args:
        train_df: Training data DataFrame
        val_df: Validation data DataFrame
        test_df: Test data DataFrame
        config: TrainingConfig instance
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        
    Example:
        >>> train_loader, val_loader, test_loader = create_dataloaders(
        ...     train_df, val_df, test_df, config
        ... )
        >>> for waveforms, labels in train_loader:
        ...     logits = model(waveforms)
        ...     loss = criterion(logits, labels)
    """
    # Create datasets
    train_dataset = PANNDataset(train_df, config, augment=True)
    val_dataset = PANNDataset(val_df, config, augment=False)
    test_dataset = PANNDataset(test_df, config, augment=False)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False  # Keep all samples
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_dataset)} samples ({len(val_loader)} batches)")
    print(f"  Test:  {len(test_dataset)} samples ({len(test_loader)} batches)")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset creation
    import torch
    from config import TrainingConfig
    
    # Create dummy data
    dummy_df = pd.DataFrame({
        'filename': ['test.wav'] * 10,
        'start_time': [0.0] * 10,
        'end_time': [10.0] * 10,
        'label': [0, 1] * 5,
        'split': ['train'] * 10,
    })
    
    config = TrainingConfig()
    
    print("Testing PANNDataset creation...")
    dataset = PANNDataset(dummy_df, config, augment=True)
    print(f"Dataset size: {len(dataset)}")
    print(f"Target sample rate: {dataset.target_sample_rate} Hz")
    print(f"Target duration: {dataset.audio_duration} seconds")
    print(f"Target samples: {dataset.target_samples}")
    
    print("\nTesting DataLoader creation...")
    train_df, val_df, test_df = dummy_df.copy(), dummy_df.copy(), dummy_df.copy()
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, config
    )
    
    print("\nAll tests passed!")
