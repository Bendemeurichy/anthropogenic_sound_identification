import pandas as pd


def load_esc50(filepath) -> pd.DataFrame:
    """Loads ESC-50 labels from a CSV file.
    Args:
        filepath: Path of the ESC-50 label CSV file.
    Returns:
        pandas.DataFrame: DataFrame containing ESC-50 labels.
    """
    esc50_df = pd.read_csv(filepath)

    esc50_df = _map_snippet_timeframes(esc50_df)
    # map fold 1-4 to train, fold 5 to test
    esc50_df["split"] = esc50_df["fold"].apply(lambda x: "test" if x == 5 else "train")

    # split train split in 80-20 train val split
    train_df = esc50_df[esc50_df["split"] == "train"]
    val_size = int(0.2 * len(train_df))
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df.loc[:val_size, "split"] = "val"
    train_df.loc[val_size:, "split"] = "train"
    esc50_df.update(train_df)

    esc50_df = esc50_df.drop(columns=["take", "esc10", "fold", "src_file", "target"])

    # Transform to standard format
    esc50_df = esc50_df.rename(
        columns={
            "filename": "filename",
            "start": "start_time",
            "end": "end_time",
            "category": "label",
        }
    )

    esc50_df["caption"] = None  # No captions available

    esc50_df["dataset"] = "esc50"

    return esc50_df


def _map_snippet_timeframes(df: pd.DataFrame) -> pd.DataFrame:
    """Maps letter of recording order to start and end time in seconds.
    Args:
        df: DataFrame containing 'take' column with snippet letters.
    Returns:
        pandas.DataFrame: DataFrame with added 'start_time' and 'end_time' columns.
    """
    take_to_time = {
        "A": (0, 5),
        "B": (5, 10),
        "C": (10, 15),
        "D": (15, 20),
        "E": (20, 25),
        "F": (25, 30),
        "G": (30, 35),
        "H": (35, 40),
        "I": (40, 45),
        "J": (45, 50),
        "K": (50, 55),
        "L": (55, 60),
        "M": (60, 65),
        "N": (65, 70),
        "O": (70, 75),
        "P": (75, 80),
    }

    df["start_time"], df["end_time"] = zip(*df["take"].map(take_to_time))

    return df


def test_load_esc50():
    df = load_esc50("../../data/metadata/esc50/esc50.csv")
    assert not df.empty
    assert all(
        col in df.columns
        for col in ["filename", "start_time", "end_time", "label", "split", "caption"]
    )
    assert df.shape[0] == 2000
    print("ESC-50 loading test passed.")
    print(df.head())
    return df


if __name__ == "__main__":
    test_load_esc50()
