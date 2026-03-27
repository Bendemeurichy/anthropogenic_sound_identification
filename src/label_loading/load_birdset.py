"""Load bird COI samples from birdset. Samples should only be used as foreground samples."""

from typing import List, Optional

import numpy as np
import pandas as pd


def load_birdset(path: str) -> pd.DataFrame:
    """Load bird COI samples from birdset. Bird samples should only be used as foreground samples.

    Args:
        path: Path to the birdset annotations CSV file.
    Returns:
        A pandas DataFrame containing the metadata.
    """

    df = pd.read_csv(path, header=0)
    
    # Note: '????' species codes represent valid bird calls that couldn't be 
    # classified to a specific species by researchers. These are kept as valid samples.
    
    # Rename columns to match standard format
    df.rename(
        columns={
            "Filename": "filename",
            "Start Time (s)": "start_time",
            "End Time (s)": "end_time",
            "Species eBird Code": "species_code",
        },
        inplace=True,
    )

    # Calculate duration from start and end times
    df["duration"] = df["end_time"] - df["start_time"]

    # Add standard columns
    df["dataset"] = "birdset"
    df["label"] = "bird"  # All valid species become "bird" label
    
    # Convert time columns to strings (matching aerosonic format)
    df["start_time"] = df["start_time"].apply(lambda x: str(x))
    df["end_time"] = df["end_time"].apply(lambda x: str(x))

    # Create 80-10-10 train/val/test split at annotation level
    # Use random seed for reproducibility
    _rng = np.random.default_rng(42)
    shuffled_indices = _rng.permutation(len(df))
    
    train_size = int(0.8 * len(shuffled_indices))
    val_size = int(0.1 * len(shuffled_indices))
    
    train_idx = shuffled_indices[:train_size]
    val_idx = shuffled_indices[train_size:train_size + val_size]
    test_idx = shuffled_indices[train_size + val_size:]
    
    # Initialize all as train, then assign val and test
    df["split"] = "train"
    df.iloc[val_idx, df.columns.get_loc("split")] = "val"
    df.iloc[test_idx, df.columns.get_loc("split")] = "test"

    return df


def test_load_birdset():
    path = "/home/bendm/Thesis/project/code/data/metadata/birdset/annotations.csv"
    df = load_birdset(path)
    
    assert not df.empty, "DataFrame should not be empty"
    assert all(
        col in df.columns
        for col in [
            "filename",
            "start_time",
            "end_time",
            "duration",
            "label",
            "split",
            "dataset",
        ]
    ), "DataFrame should have the correct columns"
    
    assert df["dataset"].unique() == ["birdset"], (
        "Dataset column should only contain 'birdset'"
    )
    
    assert df["label"].unique() == ["bird"], (
        "Label column should only contain 'bird'"
    )

    # Verify splits exist
    assert set(df["split"].unique()) == {"train", "val", "test"}, (
        "Split column should contain 'train', 'val', and 'test'"
    )

    print("All tests passed for load_birdset!")
    print(f"\nTotal valid bird samples: {len(df)}")
    print(f"  Train: {len(df[df['split'] == 'train'])} samples")
    print(f"  Val:   {len(df[df['split'] == 'val'])} samples")
    print(f"  Test:  {len(df[df['split'] == 'test'])} samples")
    
    print(f"\nUnique audio files: {df['filename'].nunique()}")
    print(f"Unique bird species: {df['species_code'].nunique()}")
    
    print("\nFirst few samples:")
    print(df[["filename", "start_time", "end_time", "duration", "label", "split", "species_code"]].head(10))
    
    print("\nDuration statistics:")
    print(df["duration"].describe())


if __name__ == "__main__":
    test_load_birdset()
