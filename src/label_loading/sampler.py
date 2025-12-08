"""Sampler function that samples all samples from the datasets containing the class of interest and sample non-interest samples to the desired ratio."""

import pandas as pd


def get_coi(metadata_df: pd.DataFrame, target_class: list[str]) -> pd.DataFrame:
    """Get all samples containing the class of interest (COI).
    Args:
        metadata_df: DataFrame containing all dataset metadata.
        target_class: List of class names representing the class of interest.
    Returns:
        pandas.DataFrame: DataFrame containing all samples with the class of interest.
    """
    # Check if any target class is in the label array for each row
    coi_df = metadata_df[
        metadata_df["label"].apply(
            lambda labels: any(target in labels for target in target_class)
        )
    ].copy()
    return coi_df


def sample_non_coi(
    metadata_df: pd.DataFrame,
    coi_df: pd.DataFrame,
    coi_ratio: float = 0.25,
) -> pd.DataFrame:
    """Sample non-class-of-interest (non-COI) samples to achieve the desired COI ratio.
    Args:
        metadata_df: DataFrame containing all dataset metadata.
        coi_df: DataFrame containing all samples with the class of interest.
        coi_ratio: Desired ratio of COI samples in the final sampled dataset.
    Returns:
        pandas.DataFrame: DataFrame containing sampled non-COI samples.
    Raises:
        ValueError: If not enough non-COI samples available in any split.
    """
    sampled_dfs = [coi_df]

    for split in ["train", "val", "test"]:
        coi_split_df = coi_df[coi_df["split"] == split]
        num_coi = len(coi_split_df)
        num_non_coi_needed = int(num_coi * ((1 - coi_ratio) / coi_ratio))

        non_coi_split_df = metadata_df[
            (metadata_df["split"] == split) & (~metadata_df.index.isin(coi_df.index))
        ]

        if len(non_coi_split_df) < num_non_coi_needed:
            raise ValueError(
                f"Insufficient non-COI samples in {split}: "
                f"need {num_non_coi_needed}, have {len(non_coi_split_df)}"
            )

        sampled_non_coi_split_df = non_coi_split_df.sample(
            n=num_non_coi_needed, random_state=42
        )
        sampled_dfs.append(sampled_non_coi_split_df)

    return pd.concat(sampled_dfs, ignore_index=True)
