import pandas as pd

def load_trains(filepath):
    """load all wav files from directory for extra test-case.
        Args:
            filepath: Path to audio directory.
        Returns:
            pandas.DataFrame: DataFrame containing all audiopaths. All events are trains.
    """
    