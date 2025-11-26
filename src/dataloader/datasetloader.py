"""
This module contains the DatasetLoader class responsible for loading all the labels of the dataset.
Filenames can be used to sample data for finetuning.
"""

import pandas as pd


class DatasetLoader:
    """Loads dataset labels and provides access to filenames."""

    def load_labels(self, filepath) -> pd.DataFrame:
        """Loads labels from all datasets.
        Args:
            filepath: Path of directory containing label metadata files.
        Returns:
            pandas.DataFrame: DataFrame containing all labels.
        """
