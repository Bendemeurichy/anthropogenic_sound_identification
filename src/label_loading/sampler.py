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

    def _contains_target(labels):
        """Check if labels contain any target class, handling both list and string types."""
        if isinstance(labels, list):
            return any(target in labels for target in target_class)
        elif isinstance(labels, str):
            return labels in target_class
        else:
            return False

    # Check if any target class is in the label array for each row
    coi_df = metadata_df[metadata_df["label"].apply(_contains_target)].copy()

    # Debug info
    print(f"\nClass of Interest (COI) sampling:")
    print(f"  Total samples in metadata: {len(metadata_df)}")
    print(f"  COI samples found: {len(coi_df)}")
    print(f"  COI samples by split:")
    for split in ["train", "val", "test"]:
        split_count = len(coi_df[coi_df["split"] == split])
        print(f"    {split}: {split_count}")

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

    print(f"\nSampling non-COI with target ratio: {coi_ratio}")

    for split in ["train", "val", "test"]:
        coi_split_df = coi_df[coi_df["split"] == split]
        num_coi = len(coi_split_df)
        num_non_coi_needed = int(num_coi * ((1 - coi_ratio) / coi_ratio))

        non_coi_split_df = metadata_df[
            (metadata_df["split"] == split) & (~metadata_df.index.isin(coi_df.index))
        ]

        print(f"  {split}:")
        print(f"    COI samples: {num_coi}")
        print(f"    Non-COI needed: {num_non_coi_needed}")
        print(f"    Non-COI available: {len(non_coi_split_df)}")

        if len(non_coi_split_df) < num_non_coi_needed:
            raise ValueError(
                f"Insufficient non-COI samples in {split}: "
                f"need {num_non_coi_needed}, have {len(non_coi_split_df)}"
            )

        sampled_non_coi_split_df = non_coi_split_df.sample(
            n=num_non_coi_needed, random_state=42
        )
        sampled_dfs.append(sampled_non_coi_split_df)

    result_df = pd.concat(sampled_dfs, ignore_index=True)
    print(f"\nTotal sampled dataset: {len(result_df)} samples")
    print("  By split:")
    for split in ["train", "val", "test"]:
        split_count = len(result_df[result_df["split"] == split])
        print(f"    {split}: {split_count}")

    return result_df
