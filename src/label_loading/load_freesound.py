"""Module that loads freesound audiosamples from currated directory. Labels for freesound samples should be retrieved from site and should be used as labels on top of 'background' label.
Files are only available trough directory exploration, but labels can be found in metadata csv by matching first part of filename before _ without leftpadding to index in csv.
Audio files are kept in separate directory so whole dir can be iterated to retrieve all samples.
"""

from pathlib import Path

import pandas as pd


def load_freesound(audio_base_path: str, metadata: str | None) -> pd.DataFrame:
    """Iterate through audio_base_path and retrieve all files, matching them to the metadata csv to get labels.
    Args:
        audio_base_path: Path to directory containing audio files and metadata csv.
        metadata: Path to metadata csv file containing labels for each sample. If None, all samples will be labeled as 'background'.
    Returns:
        pandas.DataFrame: DataFrame containing filename and labels for each sample.
    """
    base_path_obj = Path(audio_base_path)
    filenames = [
        str(Path(f).relative_to(base_path_obj)) for f in base_path_obj.rglob("*.wav")
    ]

    df = pd.DataFrame({"filename": filenames})

    if metadata is not None:
        df = resolve_labels(filenames, metadata)

    else:
        df["label"] = "background"

    df["dataset"] = "freesound"

    df["start_time"] = 0.0
    df["end_time"] = None

    # assign randomly 80-20 train-test split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_size = int(0.2 * len(df))
    df.loc[:test_size, "split"] = "test"
    df.loc[test_size:, "split"] = "train"

    # assign 80-20 train-validation split to training set randomly
    train_df = df[df["split"] == "train"].copy()
    train_df = train_df.sample(frac=1, random_state=43).reset_index(drop=True)
    val_size = int(0.2 * len(train_df))
    train_df.loc[:val_size, "split"] = "val"
    train_df.loc[val_size:, "split"] = "train"

    df.update(train_df)

    return df


def resolve_labels(filepaths: list[str], metadata: str | None) -> pd.DataFrame:
    """Resolve labels for a list of filenames based on the metadata csv.
    Args:
        filenames: List of audio filenames to resolve labels for.
        metadata: Path to metadata csv file containing labels for each sample. If None, all samples will be labeled as 'background'.
    Returns:
        pandas.DataFrame: DataFrame containing filename and labels for each sample.
    """
    metadata_df = pd.read_csv(metadata)

    labels = []
    for filepath in filepaths:
        label = _resolve_label(filepath, metadata_df)
        labels.append(label)

    df = pd.DataFrame({"filename": filepaths, "label": labels})

    return df


def _resolve_label(filepath: str, metadata: pd.DataFrame) -> list[str]:
    """Function that returns the labels of one single file based on the metadata.
    Args:
        filename: Name of the audio file, used to match with metadata.
        metadata: DataFrame containing the metadata for all samples, including labels.
    """

    filename = Path(filepath).name
    index = _get_index(filename)
    label = metadata.loc[metadata["index"] == index, "labels"].values
    return [label]


def _get_index(filename: str) -> int:
    """Helper function that extracts the index from the filename, which is used to match with the metadata.
    Args:
        filename: Name of the audio file, used to extract index.
    Returns:
        int: Index extracted from the filename.
    """
    padded_index = filename.split("_")[0]

    return int(padded_index.lstrip("0"))
