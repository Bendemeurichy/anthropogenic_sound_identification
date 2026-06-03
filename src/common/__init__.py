"""
Common utilities for model training and evaluation.

This module provides shared components used across different separation models:
- COI training utilities (dataset, loss functions, batch preparation)
- Training utilities (logging, seeding, audio I/O)
"""

from .coi_training import (
    AudioAugmentations,
    COIAudioDataset,
    COIWeightedLoss,
    check_finite,
    create_coi_dataloader,
    normalize_tensor_wav,
    prepare_batch,
    prepare_batch_mono,
    prepare_batch_stereo,
    sisnr,
)

from .training_utils import (
    StepProgress,
    get_audio_info,
    progress_bar,
    robust_load_audio,
    set_seed,
)

__all__ = [
    "AudioAugmentations",
    "COIAudioDataset",
    "COIWeightedLoss",
    "StepProgress",
    "check_finite",
    "create_coi_dataloader",
    "get_audio_info",
    "normalize_tensor_wav",
    "prepare_batch",
    "prepare_batch_mono",
    "prepare_batch_stereo",
    "progress_bar",
    "robust_load_audio",
    "set_seed",
    "sisnr",
]
