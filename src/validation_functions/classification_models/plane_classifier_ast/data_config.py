"""Data loading configuration for PANN classifier"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DataLoaderConfig:
    """Configuration for PyTorch DataLoader"""

    # Column names
    filename_column: str = "filename"
    start_time_column: str = "start_time"
    end_time_column: str = "end_time"
    label_column: str = "label"
    split_column: str = "split"

    # Audio processing
    sample_rate: int = 32000  # PANN native sample rate
    audio_duration: float = 10.0  # Duration in seconds for each audio clip

    # How to handle long annotated segments
    split_long: bool = True  # If True, split long annotations into multiple clips
    min_clip_length: float = 0.5  # Minimum clip length in seconds

    # Batching / performance
    batch_size: int = 32
    num_workers: int = 4  # Number of parallel data loading workers
    pin_memory: bool = True  # Pin memory for faster GPU transfer
    shuffle_buffer: int = 10000  # Not used in PyTorch, kept for compatibility

    # Augmentation
    use_augmentation: bool = False
    aug_time_stretch_prob: float = 0.0
    aug_time_stretch_range: Tuple[float, float] = (0.9, 1.1)
    aug_noise_prob: float = 0.0
    aug_noise_stddev: float = 0.002
    aug_gain_prob: float = 0.0
    aug_gain_range: Tuple[float, float] = (0.8, 1.2)
