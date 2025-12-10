"""Main entry point for training"""

from pathlib import Path
import sys
import pandas as pd
import argparse

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from train import train_plane_classifier
from config import TrainingConfig
from label_loading.sampler import get_coi, sample_non_coi
from label_loading.metadata_loader import (
    load_metadata_datasets,
    split_seperation_classification,
)
from common.audioset_downloader import download_missing_files_from_df


def main(optimize_hyperparams=False, n_trials=20):
    """Main training function

    Args:
        optimize_hyperparams: Whether to run Optuna hyperparameter optimization
        n_trials: Number of Optuna trials if optimization is enabled
    """

    # 1. Load all dataset metadata
    print("Loading dataset metadata...")
    # Use absolute path from project root
    project_root = Path(__file__).parent.parent.parent.parent
    datasets_path = str(project_root / "data")
    audio_base_path = str(project_root.parent / "datasets")

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

    # 5. Check for missing files and download if needed
    print("\nChecking for missing audio files...")

    # Check which files are missing
    sampled_df["file_exists"] = sampled_df["filename"].apply(lambda f: Path(f).exists())
    missing_mask = ~sampled_df["file_exists"]

    if missing_mask.any():
        missing_df = sampled_df[missing_mask].copy()
        print(
            f"\n⚠️  Found {len(missing_df)} missing files out of {len(sampled_df)} total samples"
        )

        # Filter for AudioSet only - drop all other datasets
        audioset_missing = missing_df[missing_df["dataset"] == "audioset"].copy()
        other_missing = missing_df[missing_df["dataset"] != "audioset"]

        if len(other_missing) > 0:
            print(
                f"⚠️  Dropping {len(other_missing)} samples from non-AudioSet datasets"
            )
            sampled_df = sampled_df[
                (sampled_df["dataset"] == "audioset") | sampled_df["file_exists"]
            ].copy()

        if len(audioset_missing) > 0:
            print(f"📥 {len(audioset_missing)} AudioSet files need to be downloaded")
            print(
                f"Sample missing files: {audioset_missing['filename'].head(5).apply(lambda x: Path(x).name).tolist()}"
            )

            download = input("\nDownload missing AudioSet files? (y/n): ").lower()
            if download == "y":
                cookies_path = input(
                    "Enter path to cookies.txt (or press Enter to skip): "
                ).strip()
                cookies = cookies_path if cookies_path else None

                # Split by train/eval based on the split column or path
                train_missing = audioset_missing[audioset_missing["split"] == "train"]
                val_missing = audioset_missing[audioset_missing["split"] == "val"]
                test_missing = audioset_missing[audioset_missing["split"] == "test"]

                # Download train files (train + val since both are from train split)
                train_and_val = pd.concat([train_missing, val_missing])
                if len(train_and_val) > 0:
                    output_dir = str(
                        Path(audio_base_path) / "audioset_strong" / "wavs" / "train"
                    )
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    print(
                        f"\nDownloading {len(train_and_val)} train files to: {output_dir}"
                    )
                    download_missing_files_from_df(
                        train_and_val, output_dir, "train", cookies
                    )

                # Download eval/test files
                if len(test_missing) > 0:
                    output_dir = str(
                        Path(audio_base_path) / "audioset_strong" / "wavs" / "eval"
                    )
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    print(
                        f"\nDownloading {len(test_missing)} eval files to: {output_dir}"
                    )
                    download_missing_files_from_df(
                        test_missing, output_dir, "eval", cookies
                    )

                # Recheck missing files
                sampled_df["file_exists"] = sampled_df["filename"].apply(
                    lambda f: Path(f).exists()
                )
                still_missing = sampled_df[~sampled_df["file_exists"]]

                if len(still_missing) > 0:
                    print(f"\n⚠️  Still missing {len(still_missing)} files")
                    cont = input("Continue training anyway? (y/n): ").lower()
                    if cont != "y":
                        print("Training aborted.")
                        return None
                    # Drop still missing samples
                    sampled_df = sampled_df[sampled_df["file_exists"]].copy()
                else:
                    print("\n✅ All files downloaded successfully!")
            else:
                cont = input("Continue training without downloading? (y/n): ").lower()
                if cont != "y":
                    print("Training aborted.")
                    return None
                # Drop missing samples
                sampled_df = sampled_df[sampled_df["file_exists"]].copy()

        # Clean up the temporary column
        sampled_df = sampled_df.drop(columns=["file_exists"])

        print(f"\n✅ Final dataset size: {len(sampled_df)} samples")
    else:
        print("✅ All audio files present!")

    # 6. Use existing splits if available, otherwise create new ones
    if sampled_df["split"].notna().all():
        print("\nUsing existing dataset splits...")
        train_df = sampled_df[sampled_df["split"] == "train"].copy()
        val_df = sampled_df[sampled_df["split"] == "val"].copy()
        test_df = sampled_df[sampled_df["split"] == "test"].copy()

    else:
        print("\nCreating train/val/test splits with stratification...")
        from sklearn.model_selection import train_test_split

        # Set random seed for reproducibility
        random_state = 42

        # First split: 70% train, 30% temp (val+test)
        train_df, temp_df = train_test_split(
            sampled_df,
            test_size=0.3,
            stratify=sampled_df["binary_label"],
            random_state=random_state,
        )
        # Second split: 15% val, 15% test
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            stratify=temp_df["binary_label"],
            random_state=random_state,
        )

        train_df["split"] = "train"
        val_df["split"] = "val"
        test_df["split"] = "test"

    # 7. Update label column to use binary labels
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

    # 8. Optional: Run hyperparameter optimization
    if optimize_hyperparams:
        from optimize_hyperparams import optimize as run_optuna, get_best_config
        import optuna

        study = run_optuna(train_df, val_df, test_df, n_trials=n_trials)
        config = get_best_config(study)
        print("\nUsing optimized hyperparameters for final training...")
        print(f"Best validation PR-AUC from optimization: {study.best_value:.4f}")

        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.write_image("opt_history.png")

        fig2 = optuna.visualization.plot_parallel_coordinate(study)
        fig2.write_image("parallel_coord.png")

        fig3 = optuna.visualization.plot_contour(study)
        fig3.write_image("contour.png")

        fig4 = optuna.visualization.plot_param_importances(study)
        fig4.write_image("param_importance.png")
    else:
        # 8. Create training configuration
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

    # 9. Train model using the lazy loader
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    model, hist1, hist2, test_results = train_plane_classifier(
        train_df, val_df, test_df, config=config
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(
        f"Model weights saved to: {config.checkpoint_dir}/best_model_phase2.weights.h5"
    )
    print(f"Final weights saved to: {config.checkpoint_dir}/final_model.weights.h5")
    print(f"Logs saved to: {config.log_dir}")
    print("\nTest Results:")
    print(f"  Loss:     {test_results[0]:.4f}")
    print(f"  Accuracy: {test_results[1]:.4f}")
    print("\nTo view training progress:")
    print(f"  tensorboard --logdir={config.log_dir}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train plane classifier")
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run Optuna hyperparameter optimization before training",
    )
    parser.add_argument(
        "--n-trials", type=int, default=20, help="Number of Optuna trials (default: 20)"
    )

    args = parser.parse_args()
    main(optimize_hyperparams=args.optimize, n_trials=args.n_trials)
