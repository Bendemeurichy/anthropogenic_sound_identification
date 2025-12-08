import pandas as pd


def load_audioset(train_set, test_set, display_names_path) -> pd.DataFrame:
    """Loads AudioSet labels from a CSV file.
    Args:
        filepath: Path of the AudioSet label CSV file.
    Returns:
        pandas.DataFrame: DataFrame containing AudioSet labels.
    """
    train_set_df = pd.read_csv(
        train_set,
        sep="\t",
        header=None,
        skiprows=1,
        names=["filename", "start_time", "end_time", "label", "split", "caption"],
        dtype={"start_time": str, "end_time": str},
    )
    train_set_df["split"] = "train"

    # assign 80-20 train-validation split to training set randomly
    train_set_df = train_set_df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_size = int(0.2 * len(train_set_df))
    train_set_df.loc[:val_size, "split"] = "val"
    train_set_df.loc[val_size:, "split"] = "train"

    test_set_df = pd.read_csv(
        test_set,
        sep="\t",
        header=None,
        skiprows=1,
        names=["filename", "start_time", "end_time", "label", "split", "caption"],
        dtype={"start_time": str, "end_time": str},
    )
    test_set_df["split"] = "test"

    display_names_df = pd.read_csv(
        display_names_path, sep="\t", header=None, names=["mid", "display_name"]
    )

    # TODO: add randomly generated captions based on labels

    all_labels_df = pd.concat([train_set_df, test_set_df], ignore_index=True)

    all_labels_df["filename"] = all_labels_df["filename"] + ".wav"

    # Create a dictionary from mid to display_name for mapping
    label_mapping = display_names_df.set_index("mid")["display_name"].to_dict()

    all_labels_df["label"] = (
        all_labels_df["label"].map(label_mapping).fillna(all_labels_df["label"])
    )

    all_labels_df["dataset"] = "audioset"

    all_labels_df["filename"] = (
        all_labels_df["filename"].str.split("_").str[:-1].str.join("_")
    )

    return all_labels_df


def test_load_audioset():
    train_set = "../../data/metadata/audioset/audioset_train_strong.tsv"
    test_set = "../../data/metadata/audioset/audioset_eval_strong.tsv"
    display_names_path = "../../data/metadata/audioset/mid_to_display_name.tsv"
    df = load_audioset(train_set, test_set, display_names_path)
    print(df.head())


if __name__ == "__main__":
    test_load_audioset()
