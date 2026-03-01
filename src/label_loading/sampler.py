"""Sampler function that samples all samples from the datasets containing the class of interest and sample non-interest samples to the desired ratio."""

import os
import struct

import numpy as np
import pandas as pd

from .metadata_loader import load_metadata_datasets, split_seperation_classification


def _get_wav_duration_fast(filepath: str) -> float:
    """Fast extraction of wav file duration by reading its header."""
    try:
        size = os.path.getsize(filepath)
        if size < 44:
            return 5.0

        with open(filepath, "rb") as f:
            header = f.read(44)

        if header[0:4] != b"RIFF" or header[8:12] != b"WAVE":
            return 5.0

        # simple assumption for standard 44-byte header:
        byte_rate = struct.unpack("<I", header[28:32])[0]
        if byte_rate > 0:
            return max(0.1, (size - 44) / byte_rate)
    except Exception:
        pass

    try:
        import torchaudio

        info = torchaudio.info(filepath)
        return info.num_frames / info.sample_rate
    except Exception:
        return 5.0


def add_durations(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure that the 'duration' column is populated."""
    df = df.copy()
    if "duration" not in df.columns:
        df["duration"] = pd.NA

    def get_duration(row):
        if pd.notna(row.get("duration")):
            return float(row["duration"])
        if pd.notna(row.get("start_time")) and pd.notna(row.get("end_time")):
            try:
                return float(row["end_time"]) - float(row["start_time"])
            except (ValueError, TypeError):
                pass
        return _get_wav_duration_fast(row["filename"])

    mask = df["duration"].isna()
    if mask.any():
        print(
            f"Calculating durations for {mask.sum()} files... (This may take a moment)"
        )
        try:
            from tqdm import tqdm

            tqdm.pandas(desc="Computing durations")
            df.loc[mask, "duration"] = df[mask].progress_apply(get_duration, axis=1)
        except ImportError:
            df.loc[mask, "duration"] = df[mask].apply(get_duration, axis=1)

    return df


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
    segment_duration: float = 5.0,
    segment_stride: float = 5.0,
) -> pd.DataFrame:
    """Sample non-class-of-interest (non-COI) samples to achieve the desired COI ratio.
    Args:
        metadata_df: DataFrame containing all dataset metadata.
        coi_df: DataFrame containing all samples with the class of interest.
        coi_ratio: Desired ratio of COI samples in the final sampled dataset.
        segment_duration: Length of segments in seconds.
        segment_stride: Stride between segments in seconds.
    Returns:
        pandas.DataFrame: DataFrame containing sampled non-COI samples.
    Raises:
        ValueError: If not enough non-COI samples available in any split.
    """
    metadata_df = add_durations(metadata_df)

    # Evaluate estimated segments for COI files to calculate the target non-COI segments
    coi_df = coi_df.copy()
    coi_df["duration"] = metadata_df.loc[coi_df.index, "duration"]

    def est_segments(dur):
        if pd.isna(dur):
            return 1
        dur = max(0.1, float(dur))
        if dur < segment_duration:
            return 1
        return 1 + int((dur - segment_duration) / segment_stride)

    metadata_df["est_segments"] = metadata_df["duration"].apply(est_segments)
    coi_df["est_segments"] = coi_df["duration"].apply(est_segments)

    sampled_dfs = [coi_df]

    print(f"\nSampling non-COI with target ratio: {coi_ratio}")

    for split in ["train", "val", "test"]:
        coi_split_df = coi_df[coi_df["split"] == split]
        num_coi_segments = coi_split_df["est_segments"].sum()
        num_non_coi_segments_needed = int(
            num_coi_segments * ((1 - coi_ratio) / coi_ratio)
        )

        non_coi_split_df = metadata_df[
            (metadata_df["split"] == split) & (~metadata_df.index.isin(coi_df.index))
        ]

        print(f"  {split}:")
        print(f"    COI files: {len(coi_split_df)} (Est. segments: {num_coi_segments})")
        print(f"    Non-COI segments needed: {num_non_coi_segments_needed}")

        shuffled_non_coi = non_coi_split_df.sample(frac=1, random_state=42)
        cumulative_segments = shuffled_non_coi["est_segments"].cumsum()

        mask = cumulative_segments <= num_non_coi_segments_needed
        num_files_needed = mask.sum() + 1

        if num_files_needed > len(shuffled_non_coi):
            num_files_needed = len(shuffled_non_coi)
            total_avail = (
                cumulative_segments.iloc[-1] if len(cumulative_segments) > 0 else 0
            )
            if total_avail < num_non_coi_segments_needed:
                print(
                    f"    Warning: Insufficient non-COI segments in {split}. Needed {num_non_coi_segments_needed}, have {total_avail} across {len(shuffled_non_coi)} files."
                )

        sampled_non_coi_split_df = shuffled_non_coi.head(num_files_needed)
        actual_segments_sampled = (
            sampled_non_coi_split_df["est_segments"].sum()
            if len(sampled_non_coi_split_df) > 0
            else 0
        )

        print(
            f"    Non-COI files sampled: {len(sampled_non_coi_split_df)} (Est. segments: {actual_segments_sampled})"
        )
        sampled_dfs.append(sampled_non_coi_split_df)

    result_df = pd.concat(sampled_dfs, ignore_index=True)
    # keep a copy of whatever was in `label` before it gets converted to a
    # binary indicator – the validation/plotting code can then report which
    # original classes were confused.
    result_df["orig_label"] = result_df["label"]

    print(f"\nTotal sampled dataset: {len(result_df)} files")
    total_segments = result_df["est_segments"].sum()
    print(f"Total estimated segments: {total_segments}")

    print("  By split:")
    for split in ["train", "val", "test"]:
        split_df = result_df[result_df["split"] == split]
        print(
            f"    {split}: {len(split_df)} files ({split_df['est_segments'].sum()} segments)"
        )

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
