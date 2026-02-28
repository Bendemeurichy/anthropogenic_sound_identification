"""Sampler function that samples all samples from the datasets containing the class of interest and sample non-interest samples to the desired ratio."""

import pandas as pd

from .metadata_loader import load_metadata_datasets, split_seperation_classification


def get_coi(metadata_df: pd.DataFrame, target_class: list[str]) -> pd.DataFrame:
    """Get all samples containing the class of interest (COI).
    Args:
        metadata_df: DataFrame containing all dataset metadata.
        target_class: List of class names representing the class of interest.
    Returns:
        pandas.DataFrame: DataFrame containing all samples with the class of interest.
    """

    def _contains_target(labels):
        """Check if labels contain any target class, handling both list and string types."""
        if isinstance(labels, list):
            return any(target in labels for target in target_class)
        elif isinstance(labels, str):
            return labels in target_class
        else:
            return False

    # Check if any target class is in the label array for each row
    coi_df = metadata_df[metadata_df["label"].apply(_contains_target)].copy()

    # Debug info
    print(f"\nClass of Interest (COI) sampling:")
    print(f"  Total samples in metadata: {len(metadata_df)}")
    print(f"  COI samples found: {len(coi_df)}")
    print(f"  COI samples by split:")
    for split in ["train", "val", "test"]:
        split_count = len(coi_df[coi_df["split"] == split])
        print(f"    {split}: {split_count}")

    return coi_df


def sample_non_coi(
    metadata_df: pd.DataFrame,
    coi_df: pd.DataFrame,
    coi_ratio: float = 0.25,
) -> pd.DataFrame:
    """Sample non-class-of-interest (non-COI) samples to achieve the desired COI ratio.
    Args:
        metadata_df: DataFrame containing all dataset metadata.
        coi_df: DataFrame containing all samples with the class of interest.
        coi_ratio: Desired ratio of COI samples in the final sampled dataset.
    Returns:
        pandas.DataFrame: DataFrame containing sampled non-COI samples.
    Raises:
        ValueError: If not enough non-COI samples available in any split.
    """
    sampled_dfs = [coi_df]

    print(f"\nSampling non-COI with target ratio: {coi_ratio}")

    for split in ["train", "val", "test"]:
        coi_split_df = coi_df[coi_df["split"] == split]
        num_coi = len(coi_split_df)
        num_non_coi_needed = int(num_coi * ((1 - coi_ratio) / coi_ratio))

        non_coi_split_df = metadata_df[
            (metadata_df["split"] == split) & (~metadata_df.index.isin(coi_df.index))
        ]

        print(f"  {split}:")
        print(f"    COI samples: {num_coi}")
        print(f"    Non-COI needed: {num_non_coi_needed}")
        print(f"    Non-COI available: {len(non_coi_split_df)}")

        if len(non_coi_split_df) < num_non_coi_needed:
            raise ValueError(
                f"Insufficient non-COI samples in {split}: "
                f"need {num_non_coi_needed}, have {len(non_coi_split_df)}"
            )

        sampled_non_coi_split_df = non_coi_split_df.sample(
            n=num_non_coi_needed, random_state=42
        )
        sampled_dfs.append(sampled_non_coi_split_df)

    result_df = pd.concat(sampled_dfs, ignore_index=True)
    # keep a copy of whatever was in `label` before it gets converted to a
    # binary indicator – the validation/plotting code can then report which
    # original classes were confused.
    result_df["orig_label"] = result_df["label"]
    print(f"\nTotal sampled dataset: {len(result_df)} samples")
    print("  By split:")
    for split in ["train", "val", "test"]:
        split_count = len(result_df[result_df["split"] == split])
        print(f"    {split}: {split_count}")

    return result_df


def assemble_aerosonic_experiment(
    metadata_df: pd.DataFrame,
    include_freesound_bg: bool = True,
    include_aerosonic_bg: bool = True,
    include_esc50_bg: bool = False,
    random_state: int = 42,
) -> dict:
    """Assemble dataset pairs for the aerosonic experiment.

    Returns a dict with keys:
      - 'train_pairs': DataFrame of foreground/background pairs for training
      - 'aerosonic_test': aerosonic test-split metadata (for model testing)
      - 'risoux_test': risoux metadata (independent test set)

    Pairing policy: each aerosonic train foreground (plane) is paired with a
    randomly sampled background drawn from the requested background sources.
    """
    from sklearn.utils import resample

    rng = random_state

    # Foregrounds: aerosonic train planes
    aero = metadata_df[metadata_df["dataset"] == "aerosonicdb"].copy()
    aero_train_fg = aero[(aero["split"] == "train") & (aero["label"] == "plane")].copy()

    # aerosonic test split for final evaluation
    aerosonic_test = aero[aero["split"] == "test"].copy()

    # risoux independent test set (if present)
    risoux_test = metadata_df[metadata_df["dataset"] == "risoux_test"].copy()

    # Build background pool
    bg_datasets = []
    if include_aerosonic_bg:
        bg_datasets.append("aerosonicdb")
    if include_freesound_bg:
        bg_datasets.append("freesound")
    if include_esc50_bg:
        bg_datasets.append("esc50")

    bg_pool = pd.DataFrame()
    if bg_datasets:
        bg_pool = metadata_df[metadata_df["dataset"].isin(bg_datasets)].copy()
        # exclude plane foregrounds from background pool
        bg_pool = bg_pool[
            ~((bg_pool["dataset"] == "aerosonicdb") & (bg_pool["label"] == "plane"))
        ]

    if bg_pool.empty:
        raise ValueError("No background samples found for requested background sources")

    # For each foreground, sample a random background (with replacement if needed)
    n_fg = len(aero_train_fg)
    sampled_bg = resample(
        bg_pool, replace=(len(bg_pool) < n_fg), n_samples=n_fg, random_state=rng
    )

    pairs = aero_train_fg.reset_index(drop=True).copy()
    sampled_bg = sampled_bg.reset_index(drop=True).copy()

    # Compose paired DataFrame
    paired = pd.DataFrame(
        {
            "foreground_filename": pairs["filename"],
            "foreground_dataset": pairs.get("dataset", ["aerosonicdb"] * n_fg),
            "foreground_start": pairs.get("start_time"),
            "foreground_end": pairs.get("end_time"),
            "background_filename": sampled_bg["filename"],
            "background_dataset": sampled_bg["dataset"],
            "background_start": sampled_bg.get("start_time"),
            "background_end": sampled_bg.get("end_time"),
            "split": "train",
        }
    )

    return {
        "train_pairs": paired,
        "aerosonic_test": aerosonic_test,
        "risoux_test": risoux_test,
    }


def test_sampler():
    print("Loading dataset metadata...")
    # Use absolute path from project root
    project_root = Path(__file__).parent.parent.parent
    datasets_path = str(project_root / "data" / "metadata")
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
    target_classes = ["airplane", "Airplane"]

    # Train experiment test
    # target_classes = [
    #     "Rail transport",
    #     "Train",
    #     "Subway, metro, underground",
    #     "Railroad car, train wagon",
    #     "Train wheels squealing",
    #     "trian",
    #     "Rail_transport",
    #     "Subway_and_metro_and_underground",
    # ]

    print(f"\nTarget classes: {target_classes}")

    # 3. Sample data to get balanced dataset
    print("\nSampling data with class-of-interest ratio...")
    coi_df = get_coi(classification_metadata, target_classes)
    sampled_df = sample_non_coi(
        classification_metadata,
        coi_df,
        coi_ratio=0.25,  # Aim for 25% plane sounds
    )

    # 4. Preserve the original labels and then create binary labels: 1 for plane,
    #    0 for non-plane.  ``orig_label`` will be kept alongside the binary column.
    sampled_df["orig_label"] = sampled_df["label"]
    sampled_df["binary_label"] = sampled_df["label"].apply(
        lambda x: (
            1
            if (isinstance(x, list) and any(label in target_classes for label in x))
            or (isinstance(x, str) and x in target_classes)
            else 0
        )
    )

    # 6. Use existing splits if available, otherwise create new ones
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


if __name__ == "__main__":
    from pathlib import Path

    test_sampler()
