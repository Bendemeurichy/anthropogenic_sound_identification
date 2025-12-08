from __future__ import annotations

import typing as t
from dataclasses import dataclass


@dataclass
class DataLoaderConfig:
    # Column names
    filename_column: str = "filename"
    start_time_column: str = "start_time"
    end_time_column: str = "end_time"
    label_column: str = "label"
    split_column: str = "split"

    # Audio processing
    sample_rate: int = 16000
    audio_duration: float = 5.0

    # How to handle long annotated segments
    split_long: bool = True  # if True, split long annotations into multiple clips
    min_clip_length: float = (
        0.5  # seconds: if a remainder is shorter than this, still include (it will be padded)
    )

    # Batching / performance
    batch_size: int = 32
    shuffle_buffer: int = 10000

    # Augmentation
    use_augmentation: bool = False
    aug_time_stretch_prob: float = 0.0
    aug_time_stretch_range: t.Tuple[float, float] = (0.9, 1.1)
    aug_noise_prob: float = 0.0
    aug_noise_stddev: float = 0.002
    aug_gain_prob: float = 0.0
    aug_gain_range: t.Tuple[float, float] = (0.8, 1.2)
