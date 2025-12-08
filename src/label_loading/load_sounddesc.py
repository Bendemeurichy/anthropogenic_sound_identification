import pandas as pd


def load_sounddesc(
    captions_pkl_file: str, label_pkl_file: str, splits_file_path: str
) -> pd.DataFrame:
    """Loads SoundDesc labels from a pickle file and splits from a text file.
    Args:
        captions_pkl_file: Path of the SoundDesc captions pickle file.
        label_pkl_file: Path of the SoundDesc label pickle file.
        splits_file_path: Path of the SoundDesc splits directory: contains train,val, test.
    Returns:
        pandas.DataFrame: DataFrame containing SoundDesc fragments.
    """

    captions_dict = pd.read_pickle(captions_pkl_file)
    sounddescs_captions_df = pd.DataFrame(
        list(captions_dict.items()), columns=["filename", "caption"]
    )

    labels_dict = pd.read_pickle(label_pkl_file)
    sounddescs_labels_df = pd.DataFrame(
        list(labels_dict.items()), columns=["filename", "label"]
    )
    sounddescs_df = sounddescs_captions_df.merge(
        sounddescs_labels_df, on="filename", how="inner"
    )

    train_split = splits_file_path + "/train_list.txt"
    val_split = splits_file_path + "/val_list.txt"
    test_split = splits_file_path + "/test_list.txt"

    with open(train_split, "r") as f:
        train_files = f.read().splitlines()
        sounddescs_split = pd.DataFrame({"filename": train_files, "split": "train"})
    with open(val_split, "r") as f:
        val_files = f.read().splitlines()
        val_split_df = pd.DataFrame({"filename": val_files, "split": "val"})
        sounddescs_split = pd.concat(
            [sounddescs_split, val_split_df], ignore_index=True
        )
    with open(test_split, "r") as f:
        test_files = f.read().splitlines()
        test_split_df = pd.DataFrame({"filename": test_files, "split": "test"})
        sounddescs_split = pd.concat(
            [sounddescs_split, test_split_df], ignore_index=True
        )

    print(f"Loaded {len(sounddescs_split)} filenames from splits.")

    sounddescs_df = sounddescs_df.merge(sounddescs_split, on="filename", how="inner")

    sounddescs_df["start_time"] = "0.0"
    sounddescs_df["end_time"] = None
    sounddescs_df["dataset"] = "sounddesc"
    sounddescs_df["filename"] = sounddescs_df["filename"].astype(str) + ".wav"

    return sounddescs_df


def test_load_sounddesc():
    captions_pkl_file = "../../data/metadata/sounddesc/sounddescs_descriptions.pkl"
    label_pkl_file = "../../data/metadata/sounddesc/sounddescs_categories.pkl"
    splits_file_path = (
        "../../data/metadata/sounddesc/splits_sounddesc/group_filtered_split01"
    )
    df = load_sounddesc(captions_pkl_file, label_pkl_file, splits_file_path)
    print(df.head())


if __name__ == "__main__":
    test_load_sounddesc()
