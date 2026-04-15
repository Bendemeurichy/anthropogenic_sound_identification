"""
This module contains the functionality responsible for loading all the labels of the dataset.
Filenames can be used to sample data for finetuning.

Supports both:
1. File-based loading (original): Load audio from disk using file paths
2. WebDataset loading: Load audio from compressed tar shards

Usage:
    # File-based (original)
    df = load_metadata_datasets(datasets_path, audio_base_path)
    
    # WebDataset mode
    shard_paths = get_webdataset_paths(webdataset_dir, split="train")
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from .load_aerosonic_db import load_aerosonic_db
from .load_birdset import load_birdset
from .load_esc50 import load_esc50
from .load_freesound import load_freesound
from .load_risoux_test import load_risoux_test

# Dataset file configuration
DATASET_CONFIG = {
    "esc50": {
        "metadata": "esc50/esc50.csv",
    },
    "aerosonicdb": {
        "metadata": "aerosonicDB/sample_meta.csv",
    },
    "risoux_test": {
        "metadata": "risoux_test/annotations.csv",
    },
    "freesound": {
        "metadata": "freesound_curation/source_freesound_field_recordings_links_with_labels.csv",
    },
    "birdset": {
        "metadata": "birdset/annotations.csv",
    },
}

# Audio file path prefixes for each dataset
# Adjust these paths according to your actual data directory structure
DATASET_AUDIO_PATHS = {
    "esc50": {
        "base": "ESC-50-master/audio",  # Base path for ESC-50 audio files
    },
    "aerosonicdb": {
        "base": "AeroSonicDB/8371595/audio",  # Base path for AerosonicDB audio files
    },
    "risoux_test": {
        "base": "Risoux_test/10701274/audio_recordings",  # Base path for Risoux test audio files
    },
    "freesound": {
        "base": "Backgrounds/Background/Background",  # Base path for Freesound audio files
    },
    "birdset": {
        "base": "birdset",  # Base path for birdset audio files (to be configured)
    },
}


def add_audio_file_paths(df: pd.DataFrame, audio_base_path: str) -> pd.DataFrame:
    """Update filename column with full file paths based on dataset and split.

    Args:
        df: DataFrame with 'filename', 'dataset', and 'split' columns
        audio_base_path: Base path to the audio files directory (e.g., '/path/to/data/audio')

    Returns:
        DataFrame with updated 'filename' column containing full paths to audio files
    """
    from pathlib import Path

    # Create a copy to avoid modifying the original
    df = df.copy()

    # Build paths using a vectorized approach with groupby
    def build_path(row):
        dataset = row["dataset"]
        split = row["split"]
        filename = row["filename"]

        if dataset not in DATASET_AUDIO_PATHS:
            return str(Path(audio_base_path) / filename)

        dataset_config = DATASET_AUDIO_PATHS[dataset]
        base = dataset_config["base"]
        split_dir = dataset_config.get(split, "")

        if split_dir:
            return str(Path(audio_base_path) / base / split_dir / filename)
        else:
            return str(Path(audio_base_path) / base / filename)

    # Use vectorized operations by creating full paths with string formatting
    # This is more efficient than apply(axis=1) for large DataFrames
    for dataset in df["dataset"].unique():
        dataset_mask = df["dataset"] == dataset

        if dataset not in DATASET_AUDIO_PATHS:
            # Fallback: just use base path + filename
            base_path_obj = Path(audio_base_path)
            df.loc[dataset_mask, "filename"] = df.loc[dataset_mask, "filename"].apply(
                lambda fname: str(base_path_obj / fname)
            )
        else:
            dataset_config = DATASET_AUDIO_PATHS[dataset]
            base = dataset_config["base"]

            # Process each split within this dataset
            for split in df.loc[dataset_mask, "split"].unique():
                split_mask = dataset_mask & (df["split"] == split)
                split_dir = dataset_config.get(split, "")

                if split_dir:
                    base_path_obj = Path(audio_base_path) / base / split_dir
                else:
                    base_path_obj = Path(audio_base_path) / base

                # Use vectorized apply on filtered series
                df.loc[split_mask, "filename"] = df.loc[split_mask, "filename"].apply(
                    lambda fname: str(base_path_obj / fname)
                )

    return df


def load_metadata_datasets(datasets_path: str, audio_base_path: str) -> pd.DataFrame:
    """Loads labels from all datasets.
    Args:
        datasets_path: Path of directory containing label metadata files.
        audio_base_path: Base path to audio files. If provided, updates 'filename' column with full paths.
    Returns:
        pandas.DataFrame: DataFrame containing all labels.
    """
    esc50 = load_esc50(f"{datasets_path}/{DATASET_CONFIG['esc50']['metadata']}")
    print(f"Found {len(esc50)} samples in ESC-50 dataset.")
    print(f"For train set: {len(esc50[esc50['split'] == 'train'])} samples")
    print(f"For val set: {len(esc50[esc50['split'] == 'val'])} samples")
    print(f"For test set: {len(esc50[esc50['split'] == 'test'])} samples")

    #! Aerosonic for plane experiment
    aerosonic = load_aerosonic_db(
        f"{datasets_path}/{DATASET_CONFIG['aerosonicdb']['metadata']}"
    )
    print(f"Found {len(aerosonic)} samples in AerosonicDB dataset.")
    print(f"For train set: {len(aerosonic[aerosonic['split'] == 'train'])} samples")
    print(f"For val set: {len(aerosonic[aerosonic['split'] == 'val'])} samples")
    print(f"For test set: {len(aerosonic[aerosonic['split'] == 'test'])} samples")

    risoux = load_risoux_test(
        f"{datasets_path}/{DATASET_CONFIG['risoux_test']['metadata']}"
    )
    print("Risoux is only for testing, check this.")
    print(f"Found {len(risoux)} samples in Risoux test dataset.")
    print(f"For train set: {len(risoux[risoux['split'] == 'train'])} samples")
    print(f"For val set: {len(risoux[risoux['split'] == 'val'])} samples")
    print(f"For test set: {len(risoux[risoux['split'] == 'test'])} samples")

    freesound = load_freesound(
        audio_base_path=f"{audio_base_path}/{DATASET_AUDIO_PATHS['freesound']['base']}",
        metadata=f"{datasets_path}/{DATASET_CONFIG['freesound']['metadata']}",
    )
    print(f"Found {len(freesound)} background samples in Freesound dataset.")
    print(f"For train set: {len(freesound[freesound['split'] == 'train'])} samples")
    print(f"For val set: {len(freesound[freesound['split'] == 'val'])} samples")
    print(f"For test set: {len(freesound[freesound['split'] == 'test'])} samples")

    #! Birdset for bird separation experiment
    birdset = load_birdset(
        f"{datasets_path}/{DATASET_CONFIG['birdset']['metadata']}"
    )
    print(f"Found {len(birdset)} bird samples in Birdset dataset.")
    print(f"For train set: {len(birdset[birdset['split'] == 'train'])} samples")
    print(f"For val set: {len(birdset[birdset['split'] == 'val'])} samples")
    print(f"For test set: {len(birdset[birdset['split'] == 'test'])} samples")

    master_set = pd.concat([esc50, aerosonic, risoux, freesound, birdset], ignore_index=True)

    master_set = add_audio_file_paths(master_set, audio_base_path)

    return master_set


def split_seperation_classification(
    master_set: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split both train and val splits into 70-30 separation-classification sets
    while preserving train/val ratios in each to get 70-30 overall split.
    Args:
        master_set: DataFrame containing all dataset metadata.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: separation and classification DataFrames.
    """

    train_df = master_set[master_set["split"] == "train"].copy()
    val_df = master_set[master_set["split"] == "val"].copy()
    test_df = master_set[master_set["split"] == "test"].copy()

    # Shuffle and split train into 80-20 for separation vs classification
    train_shuffled = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_sep_size = int(0.8 * len(train_shuffled))
    separation_train = train_shuffled.iloc[:train_sep_size].copy()
    classification_train = train_shuffled.iloc[train_sep_size:].copy()

    # Shuffle and split val into 80-20 for separation vs classification
    val_shuffled = val_df.sample(frac=1, random_state=43).reset_index(drop=True)
    val_sep_size = int(0.8 * len(val_shuffled))
    separation_val = val_shuffled.iloc[:val_sep_size].copy()
    classification_val = val_shuffled.iloc[val_sep_size:].copy()

    # Combine train+val+test for each task
    separation_df = pd.concat(
        [separation_train, separation_val, test_df], ignore_index=True
    )
    classification_df = pd.concat(
        [classification_train, classification_val, test_df], ignore_index=True
    )

    return separation_df, classification_df


def test_load_metadata_datasets():
    datasets_path = "../../data/metadata"
    df = load_metadata_datasets(datasets_path)
    print(df.head())
    return df


if __name__ == "__main__":
    test_load_metadata_datasets()


# =============================================================================
# WebDataset Support
# =============================================================================


def get_webdataset_paths(
    webdataset_dir: Union[str, Path],
    split: str = "train",
    manifest_file: str = "manifest.json",
) -> str:
    """
    Get WebDataset shard path pattern for a given split.

    Args:
        webdataset_dir: Directory containing WebDataset shards
        split: Data split (train, val, test)
        manifest_file: Name of manifest file

    Returns:
        Brace-expansion path pattern for shards

    Example:
        >>> paths = get_webdataset_paths("/data/shards", "train")
        >>> print(paths)
        '/data/shards/train-{000000..000099}.tar'
    """
    webdataset_dir = Path(webdataset_dir)
    manifest_path = webdataset_dir / manifest_file

    if manifest_path.exists():
        # Use manifest for accurate shard counts
        with open(manifest_path) as f:
            manifest = json.load(f)

        if split not in manifest.get("splits", {}):
            raise ValueError(
                f"Split '{split}' not found in manifest. "
                f"Available: {list(manifest['splits'].keys())}"
            )

        shard_pattern = manifest["splits"][split].get("shard_pattern")
        if shard_pattern:
            return str(webdataset_dir / shard_pattern)

        # Fallback: calculate from num_shards
        num_shards = manifest["splits"][split]["num_shards"]
        return str(webdataset_dir / f"{split}-{{000000..{num_shards-1:06d}}}.tar")

    # No manifest: discover shards by listing directory
    import glob

    shard_files = sorted(glob.glob(str(webdataset_dir / f"{split}-*.tar")))
    if not shard_files:
        raise FileNotFoundError(
            f"No shards found for split '{split}' in {webdataset_dir}"
        )

    num_shards = len(shard_files)
    return str(webdataset_dir / f"{split}-{{000000..{num_shards-1:06d}}}.tar")


def get_all_webdataset_paths(
    webdataset_dir: Union[str, Path],
    splits: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Get WebDataset shard paths for multiple splits.

    Args:
        webdataset_dir: Directory containing WebDataset shards
        splits: List of splits to get (default: ['train', 'val', 'test'])

    Returns:
        Dictionary mapping split names to shard path patterns
    """
    if splits is None:
        splits = ["train", "val", "test"]

    paths = {}
    for split in splits:
        try:
            paths[split] = get_webdataset_paths(webdataset_dir, split)
        except (ValueError, FileNotFoundError):
            pass  # Split doesn't exist

    return paths


def load_webdataset_manifest(
    webdataset_dir: Union[str, Path],
    manifest_file: str = "manifest.json",
) -> Dict:
    """
    Load WebDataset manifest containing shard information.

    Args:
        webdataset_dir: Directory containing WebDataset shards
        manifest_file: Name of manifest file

    Returns:
        Manifest dictionary with shard information
    """
    manifest_path = Path(webdataset_dir) / manifest_file

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        return json.load(f)


def get_webdataset_info(webdataset_dir: Union[str, Path]) -> Dict:
    """
    Get summary information about a WebDataset.

    Args:
        webdataset_dir: Directory containing WebDataset shards

    Returns:
        Dictionary with dataset info (sample counts, sample rate, etc.)
    """
    try:
        manifest = load_webdataset_manifest(webdataset_dir)
        return {
            "total_samples": manifest.get("total_samples", 0),
            "total_shards": manifest.get("total_shards", 0),
            "target_sample_rate": manifest.get("target_sample_rate"),
            "splits": {
                name: {
                    "samples": info.get("num_samples", 0),
                    "shards": info.get("num_shards", 0),
                }
                for name, info in manifest.get("splits", {}).items()
            },
        }
    except FileNotFoundError:
        # No manifest, count shards manually
        import glob

        webdataset_dir = Path(webdataset_dir)
        info = {"splits": {}}

        for split in ["train", "val", "test"]:
            shards = glob.glob(str(webdataset_dir / f"{split}-*.tar"))
            if shards:
                info["splits"][split] = {"shards": len(shards)}

        info["total_shards"] = sum(
            s.get("shards", 0) for s in info["splits"].values()
        )
        return info
