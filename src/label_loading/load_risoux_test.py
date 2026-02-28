"""Load samples for the dataset in the Risoux forest, France.
Dataset is supposed to be independant testing dataset because of the weak labels."""

import pandas as pd


def load_risoux_test(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=0)
    class_cols = ["plane", "wind", "rain", "biophony", "silence"]
    # ensure columns exist and are integers (0/1)
    existing_cols = [c for c in class_cols if c in df.columns]
    if existing_cols:
        df[existing_cols] = df[existing_cols].fillna(0).astype(int)
        df["label"] = df[existing_cols].apply(
            lambda row: [col for col, val in row.items() if val == 1], axis=1
        )
    else:
        df["label"] = [[] for _ in range(len(df))]

    df["dataset"] = "risoux_test"
    df["start_time"] = "0.0"
    df["end_time"] = "unknown"
    df["split"] = "test"
    return df
