"""Main entry point for PANN plane classifier training"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import torch
import numpy as np

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from .config import TrainingConfig
from .dataset import create_dataloaders
from .model_loader import create_plane_classifier
from .train import train_plane_classifier

from src.common.audio_validation import validate_dataset_files
from src.label_loading.metadata_loader import (
    load_metadata_datasets,
    split_seperation_classification,
)
from src.label_loading.sampler import get_coi, sample_non_coi


def main():
    """Main training function for PANN plane classifier"""
    
    parser = argparse.ArgumentParser(description="Train PANN plane classifier")
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config file (optional, uses defaults if not provided)'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to train on (cuda or cpu)'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig()
    config.checkpoint_dir = args.checkpoint_dir
    config.device = args.device
    
    # Set random seeds for reproducibility
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)
    
    print("="*70)
    print("PANN PLANE CLASSIFIER TRAINING")
    print("="*70)
    print(f"Device: {config.device}")
    print(f"Sample rate: {config.sample_rate} Hz")
    print(f"Audio duration: {config.audio_duration} seconds")
    print(f"Batch size: {config.batch_size}")
    print(f"Random seed: {config.random_seed}")
    
    # 1. Load all dataset metadata
    print("\n" + "="*70)
    print("LOADING DATASET METADATA")
    print("="*70)
    
    project_root = Path(__file__).parent.parent.parent.parent.parent
    datasets_path = str(project_root / "data")
    audio_base_path = str(project_root.parent / "datasets")
    
    all_metadata = load_metadata_datasets(datasets_path, audio_base_path)
    _, classification_metadata = split_seperation_classification(all_metadata)
    
    print(f"Loaded {len(all_metadata)} total samples from all datasets")
    print(f"Datasets included: {all_metadata['dataset'].unique()}")
    print(f"Classification samples: {len(classification_metadata)}")
    
    # 2. Define target classes (plane-related sounds)
    target_classes = [
        "airplane",
        "Aircraft",
        "Fixed-wing aircraft, airplane",
        "Aircraft engine",
        "Fixed-wing_aircraft_and_airplane",
        "Helicopter",
        "helicopter",
        "Propeller, airscrew",
        "Propeller airscrew",
        "Turboprop, small aircraft",
        "Jet aircraft",
    ]
    
    print(f"\nTarget classes: {target_classes}")
    
    # 3. Sample data to get balanced dataset
    print("\n" + "="*70)
    print("SAMPLING DATA")
    print("="*70)
    
    coi_df = get_coi(classification_metadata, target_classes)
    sampled_df = sample_non_coi(
        classification_metadata,
        coi_df,
        coi_ratio=0.25,  # Aim for 25% plane sounds (can adjust)
    )
    
    print(f"Class-of-interest samples: {len(coi_df)}")
    print(f"Total sampled: {len(sampled_df)}")
    
    # 4. Create binary labels
    sampled_df["binary_label"] = sampled_df["label"].apply(
        lambda x: (
            1
            if (isinstance(x, list) and any(label in target_classes for label in x))
            or (isinstance(x, str) and x in target_classes)
            else 0
        )
    )
    
    # 5. Check for missing files
    print("\n" + "="*70)
    print("CHECKING FOR MISSING AUDIO FILES")
    print("="*70)
    
    sampled_df["file_exists"] = sampled_df["filename"].apply(lambda f: Path(f).exists())
    missing_mask = ~sampled_df["file_exists"]
    
    if missing_mask.any():
        missing_df = sampled_df[missing_mask].copy()
        print(f"⚠️  Found {len(missing_df)} missing files out of {len(sampled_df)} total samples")
        
        # Filter to only existing files
        sampled_df = sampled_df[sampled_df["file_exists"]].copy()
        print(f"Continuing with {len(sampled_df)} samples that exist")
    else:
        print("✅ All audio files present!")
    
    sampled_df = sampled_df.drop(columns=["file_exists"])
    
    # 6. Use existing splits or create new ones
    if sampled_df["split"].notna().all():
        print("\n" + "="*70)
        print("USING EXISTING DATASET SPLITS")
        print("="*70)
        
        train_df = sampled_df[sampled_df["split"] == "train"].copy()
        val_df = sampled_df[sampled_df["split"] == "val"].copy()
        test_df = sampled_df[sampled_df["split"] == "test"].copy()
    else:
        print("\n" + "="*70)
        print("CREATING TRAIN/VAL/TEST SPLITS")
        print("="*70)
        
        from sklearn.model_selection import train_test_split
        
        # 70% train, 15% val, 15% test
        train_df, temp_df = train_test_split(
            sampled_df,
            test_size=0.3,
            stratify=sampled_df["binary_label"],
            random_state=config.random_seed,
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            stratify=temp_df["binary_label"],
            random_state=config.random_seed,
        )
        
        train_df["split"] = "train"
        val_df["split"] = "val"
        test_df["split"] = "test"
    
    # 7. Update label column to use binary labels
    for df in [train_df, val_df, test_df]:
        df["label"] = df["binary_label"]
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    print(f"\nClass distribution:")
    print(f"  Train - Plane: {train_df['label'].sum()}, Non-plane: {(train_df['label'] == 0).sum()}")
    print(f"  Val   - Plane: {val_df['label'].sum()}, Non-plane: {(val_df['label'] == 0).sum()}")
    print(f"  Test  - Plane: {test_df['label'].sum()}, Non-plane: {(test_df['label'] == 0).sum()}")
    
    print(f"\nDatasets in train split:")
    print(train_df["dataset"].value_counts())
    
    # 8. Validate audio files
    print("\n" + "="*70)
    print("VALIDATING AUDIO FILES")
    print("="*70)
    
    print("\nValidating training files...")
    train_df = validate_dataset_files(train_df, config.filename_column, verbose=True)
    
    print("\nValidating validation files...")
    val_df = validate_dataset_files(val_df, config.filename_column, verbose=True)
    
    print("\nValidating test files...")
    test_df = validate_dataset_files(test_df, config.filename_column, verbose=True)
    
    # 9. Create DataLoaders
    print("\n" + "="*70)
    print("CREATING DATALOADERS")
    print("="*70)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, config
    )
    
    # 10. Create model
    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)
    
    model = create_plane_classifier(
        training_config=config,
        fine_tune=False,  # Will be set to True in phase 2
        device=config.device
    )
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters (phase 1): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 11. Train model
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    model, hist1, hist2, test_results = train_plane_classifier(
        train_loader, val_loader, test_loader, model, config, config.device
    )
    
    # 12. Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Model checkpoints saved to: {config.checkpoint_dir}")
    print(f"Logs saved to: {config.log_dir}")
    
    print("\nPhase 1 best validation PR-AUC:", max(hist1['val_pr_auc']))
    print("Phase 2 best validation PR-AUC:", max(hist2['val_pr_auc']))
    
    print("\nFinal test results:")
    for metric, value in test_results.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nTo use the trained model:")
    print("  from model_loader import load_trained_model")
    print(f"  model = load_trained_model('{config.checkpoint_dir}/final_model.pth')")
    
    return model


if __name__ == "__main__":
    main()
