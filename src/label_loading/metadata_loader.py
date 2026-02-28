"""
This module contains the functionality responsible for loading all the labels of the dataset.
Filenames can be used to sample data for finetuning.
"""

import pandas as pd

from .load_aerosonic_db import load_aerosonic_db
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
        "metadata": "freesound_curation/source_freesound_field_recordings_links.csv",
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

    #! Aerosonic for plane experiment
    aerosonic = load_aerosonic_db(
        f"{datasets_path}/{DATASET_CONFIG['aerosonicdb']['metadata']}"
    )
    risoux = load_risoux_test(
        f"{datasets_path}/{DATASET_CONFIG['risoux_test']['metadata']}"
    )

    freesound = load_freesound(
        audio_base_path=f"{audio_base_path}/{DATASET_AUDIO_PATHS['freesound']['base']}",
        metadata=f"{datasets_path}/{DATASET_CONFIG['freesound']['metadata']}",
    )

    master_set = pd.concat([esc50, aerosonic, risoux, freesound], ignore_index=True)

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
