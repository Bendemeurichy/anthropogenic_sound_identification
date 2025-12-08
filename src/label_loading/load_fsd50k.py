import pandas as pd


def load_fsd50k(train_csv: str, eval_csv: str) -> pd.DataFrame:
    """Loads FSD50K labels from CSV files.
    Args:
        train_csv: Path of the FSD50K training label CSV file.
        eval_csv: Path of the FSD50K evaluation label CSV file.
        vocab_csv: Path of the FSD50K vocabulary CSV file.
    Returns:
        pandas.DataFrame: DataFrame containing FSD50K labels.
    """

    train_df = pd.read_csv(
        train_csv, names=["filename", "label", "mids", "split"], header=0
    )
    eval_df = pd.read_csv(eval_csv, names=["filename", "label", "mids"], header=0)

    eval_df["split"] = "test"

    all_labels_df = pd.concat([train_df, eval_df], ignore_index=True)
    all_labels_df["dataset"] = "fsd50k"
    all_labels_df["label"] = all_labels_df["label"].str.split(",")
    all_labels_df["caption"] = None  # No captions available
    all_labels_df["filename"] = all_labels_df["filename"].astype(str) + ".wav"
    all_labels_df["start_time"] = 0.0
    all_labels_df["end_time"] = None  # Variable length clips
    all_labels_df = all_labels_df.drop(columns=["mids"])
    return all_labels_df


def test_load_fsd50k():
    train_csv = "../../data/metadata/fsd50k_labels/dev.csv"
    eval_csv = "../../data/metadata/fsd50k_labels/eval.csv"
    df = load_fsd50k(train_csv, eval_csv)
    print(df.head())
    return df


if __name__ == "__main__":
    test_load_fsd50k()
