"""
This module contains the functionality responsible for loading all the labels of the dataset.
Filenames can be used to sample data for finetuning.
"""

import pandas as pd
from load_audioset import load_audioset
from load_esc50 import load_esc50
from load_fsd50k import load_fsd50k
from load_sounddesc import load_sounddesc

# Dataset file configuration
DATASET_CONFIG = {
    "audioset": {
        "train": "audioset/audioset_train_strong.tsv",
        "eval": "audioset/audioset_eval_strong.tsv",
        "labels": "audioset/mid_to_display_name.tsv",
    },
    "esc50": {
        "metadata": "esc50/esc50.csv",
    },
    "fsd50k": {
        "dev": "fsd50k_labels/dev.csv",
        "eval": "fsd50k_labels/eval.csv",
    },
    "sounddesc": {
        "descriptions": "sounddesc/sounddescs_descriptions.pkl",
        "categories": "sounddesc/sounddescs_categories.pkl",
        "splits": "sounddesc/splits_sounddesc/group_filtered_split01",
    },
}

# Audio file path prefixes for each dataset
# Adjust these paths according to your actual data directory structure
DATASET_AUDIO_PATHS = {
    "audioset": {
        "base": "audioset_strong/wavs",  # Base path for AudioSet audio files
        "train": "train",  # train files in audioset/audio/train/
        "val": "train",  # val files also in train directory
        "test": "eval",  # test files in audioset/audio/eval/
    },
    "esc50": {
        "base": "ESC-50-master/audio",  # Base path for ESC-50 audio files
        "train": "",  # All files in esc50/audio/ directly
        "val": "",
        "test": "",
    },
    "fsd50k": {
        "base": "FSD50K/clips",  # Base path for FSD50K audio files
        "train": "dev",  # train/val in fsd50k/audio/dev_audio/
        "val": "dev",
        "test": "eval",  # test in fsd50k/audio/eval_audio/
    },
    "sounddesc": {
        "base": "SoundDesc/audios",  # Base path for SoundDesc audio files
        "train": "",  # All files in sounddesc/audio/ directly
        "val": "",
        "test": "",
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

    def construct_path(row):
        dataset = row["dataset"]
        split = row["split"]
        filename = row["filename"]

        if dataset not in DATASET_AUDIO_PATHS:
            # Fallback: just use base path + filename
            return str(Path(audio_base_path) / filename)

        dataset_config = DATASET_AUDIO_PATHS[dataset]
        base = dataset_config["base"]
        split_dir = dataset_config.get(split, "")

        # Construct the full path
        if split_dir:
            full_path = Path(audio_base_path) / base / split_dir / filename
        else:
            full_path = Path(audio_base_path) / base / filename

        return str(full_path)

    df["filename"] = df.apply(construct_path, axis=1)
    return df


def load_metadata_datasets(
    datasets_path: str, audio_base_path: str | None = None
) -> pd.DataFrame:
    """Loads labels from all datasets.
    Args:
        datasets_path: Path of directory containing label metadata files.
        audio_base_path: Base path to audio files. If provided, updates 'filename' column with full paths.
    Returns:
        pandas.DataFrame: DataFrame containing all labels.
    """
    audioset = load_audioset(
        f"{datasets_path}/{DATASET_CONFIG['audioset']['train']}",
        f"{datasets_path}/{DATASET_CONFIG['audioset']['eval']}",
        f"{datasets_path}/{DATASET_CONFIG['audioset']['labels']}",
    )
    esc50 = load_esc50(f"{datasets_path}/{DATASET_CONFIG['esc50']['metadata']}")
    fsd50k = load_fsd50k(
        f"{datasets_path}/{DATASET_CONFIG['fsd50k']['dev']}",
        f"{datasets_path}/{DATASET_CONFIG['fsd50k']['eval']}",
    )
    sounddesc = load_sounddesc(
        f"{datasets_path}/{DATASET_CONFIG['sounddesc']['descriptions']}",
        f"{datasets_path}/{DATASET_CONFIG['sounddesc']['categories']}",
        f"{datasets_path}/{DATASET_CONFIG['sounddesc']['splits']}",
    )

    master_set = pd.concat([audioset, esc50, fsd50k, sounddesc], ignore_index=True)

    # Add full file paths if audio_base_path is provided
    if audio_base_path:
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

    # Shuffle and split train into 70-30 for separation vs classification
    train_shuffled = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_sep_size = int(0.7 * len(train_shuffled))
    separation_train = train_shuffled.iloc[:train_sep_size].copy()
    classification_train = train_shuffled.iloc[train_sep_size:].copy()

    # Shuffle and split val into 70-30 for separation vs classification
    val_shuffled = val_df.sample(frac=1, random_state=43).reset_index(drop=True)
    val_sep_size = int(0.7 * len(val_shuffled))
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
