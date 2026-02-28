"""Load plane COI samples from aerosonicdb. Samples should only be used as foreground samples."""

import json
import os
from typing import List, Optional

import pandas as pd


def load_aerosonic_db(path: str) -> pd.DataFrame:
    """Load plane COI samples from aerosonicdb. Plane samples should only be used as foreground samples.

    Args:
        path: The containing the metadata csv of both foreground and background samples.
    Returns:
        A pandas DataFrame containing the metadata.
    """

    df = pd.read_csv(path, header=0)
    df.rename(
        columns={
            "train-test": "split",
        },
        inplace=True,
    )

    # Coerce to numeric (handles "1"/"0" strings), treat invalid/missing as 0 (background)
    cls = pd.to_numeric(df["class"], errors="coerce").fillna(0).astype(int)
    prefix = cls.map({1: "/1/", 0: "/0/"})

    # Build filename (keep existing behavior of leading slash)
    df["filename"] = prefix + df["filename"].astype(str)

    df["dataset"] = "aerosonicdb"
    df["label"] = df["class"].apply(lambda x: "plane" if x == 1 else "background")
    df["start_time"] = df["offset"].apply(lambda x: str(x))
    df["end_time"] = (df["offset"] + df["duration"]).apply(lambda x: str(x))

    return df


def test_load_aerosonic_db():
    path = "../../data/metadata/aerosonicDB/sample_meta.csv"
    df = load_aerosonic_db(path)
    assert not df.empty, "DataFrame should not be empty"
    assert any(
        col in df.columns
        for col in [
            "filename",
            "offset",
            "duration",
            "label",
            "split",
            "dataset",
            "start_time",
            "end_time",
        ]
    ), "DataFrame should have the correct columns"
    assert df["dataset"].unique() == ["aerosonicdb"], (
        "Dataset column should only contain 'aerosonicdb'"
    )
    assert set(df["label"].unique()) == {
        "plane",
        "background",
    }, "Label column should only contain 'plane' and 'background'"

    print("All tests passed for load_aerosonic_db!")
    print(df.head())


if __name__ == "__main__":
    test_load_aerosonic_db()
