"""Main entry point for training"""

from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from .train import train_plane_classifier
from .config import TrainingConfig
from label_loading.sampler import get_coi, sample_non_coi
from label_loading.metadata_loader import (
    load_metadata_datasets,
    split_seperation_classification,
)


def main():
    """Main training function"""

    # 1. Load all dataset metadata
    print("Loading dataset metadata...")
    # Use absolute path from project root
    project_root = Path(__file__).parent.parent.parent.parent
    datasets_path = str(project_root / "data" / "metadata")
    audio_base_path = str(project_root / "datasets")

    # Load metadata with full file paths
    all_metadata = load_metadata_datasets(datasets_path, audio_base_path)

    _, classification_metadata = split_seperation_classification(all_metadata)

    print(f"Loaded {len(all_metadata)} total samples from all datasets")
    print(f"Datasets included: {all_metadata['dataset'].unique()}")
    print(f"\nColumns: {list(all_metadata.columns)}")

    # 2. Define your target class (plane-related sounds)
    # You'll need to check what plane-related labels exist in your datasets
    # Common airplane labels in AudioSet: "Aircraft", "Airplane", "Fixed-wing aircraft"
    # In ESC-50: "airplane"
    target_classes = [
        "airplane",
        "Aircraft",
        "Fixed-wing aircraft, airplane",
        "Aircraft engine",
        "Fixed-wing_aircraft_and_airplane",
    ]

    print(f"\nTarget classes: {target_classes}")

    # 3. Sample data to get balanced dataset
    print("\nSampling data with class-of-interest ratio...")
    coi_df = get_coi(classification_metadata, target_classes)
    sampled_df = sample_non_coi(
        classification_metadata,
        coi_df,
        coi_ratio=0.25,  # Aim for 50% plane sounds
    )

    # 4. Create binary labels: 1 for plane, 0 for non-plane
    sampled_df["binary_label"] = sampled_df["label"].apply(
        lambda x: (
            1
            if (isinstance(x, list) and any(label in target_classes for label in x))
            or (isinstance(x, str) and x in target_classes)
            else 0
        )
    )

    # 5. Use existing splits if available, otherwise create new ones
    if sampled_df["split"].notna().all():
        print("\nUsing existing dataset splits...")
        train_df = sampled_df[sampled_df["split"] == "train"].copy()
        val_df = sampled_df[sampled_df["split"] == "val"].copy()
        test_df = sampled_df[sampled_df["split"] == "test"].copy()

    else:
        print("\nCreating train/val/test splits...")
        from sklearn.model_selection import train_test_split

        train_df, temp_df = train_test_split(
            sampled_df,
            test_size=0.3,
            stratify=sampled_df["binary_label"],
            random_state=42,
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df["binary_label"], random_state=42
        )

        train_df["split"] = "train"
        val_df["split"] = "val"
        test_df["split"] = "test"

    # 6. Update label column to use binary labels
    for df in [train_df, val_df, test_df]:
        df["label"] = df["binary_label"]

    print("\nDataset splits:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")

    print("\nClass distribution:")
    print(
        f"  Train - Plane: {train_df['label'].sum()}, Non-plane: {(train_df['label'] == 0).sum()}"
    )
    print(
        f"  Val   - Plane: {val_df['label'].sum()}, Non-plane: {(val_df['label'] == 0).sum()}"
    )
    print(
        f"  Test  - Plane: {test_df['label'].sum()}, Non-plane: {(test_df['label'] == 0).sum()}"
    )

    print("\nDatasets in train split:")
    print(train_df["dataset"].value_counts())

    # 7. Create training configuration
    config = TrainingConfig(
        filename_column="filename",
        start_time_column="start_time",
        end_time_column="end_time",
        label_column="label",
        split_column="split",
        sample_rate=16000,
        audio_duration=5.0,
        split_long=True,  # Split long annotations into multiple clips
        min_clip_length=0.5,
        batch_size=32,
        shuffle_buffer=10000,
        phase1_epochs=30,
        phase2_epochs=20,
        phase1_lr=1e-3,
        phase2_lr=1e-5,
        checkpoint_dir="./checkpoints",
        log_dir="./logs",
        # Augmentation settings (only applied during training)
        use_augmentation=True,
        aug_time_stretch_prob=0.5,
        aug_time_stretch_range=(0.9, 1.1),
        aug_noise_prob=0.3,
        aug_noise_stddev=0.002,
        aug_gain_prob=0.5,
        aug_gain_range=(0.8, 1.2),
    )

    # 8. Train model using the lazy loader
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    model, hist1, hist2, test_results = train_plane_classifier(
        train_df, val_df, test_df, config=config
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Model saved to: {config.checkpoint_dir}/best_model_phase2.keras")
    print(f"Logs saved to: {config.log_dir}")
    print("\nTest Results:")
    print(f"  Loss:     {test_results[0]:.4f}")
    print(f"  Accuracy: {test_results[1]:.4f}")
    print("\nTo view training progress:")
    print(f"  tensorboard --logdir={config.log_dir}")

    return model


if __name__ == "__main__":
    main()
