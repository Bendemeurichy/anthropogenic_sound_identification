"""Audio augmentations for training data."""

import numpy as np
import torch

from src.common.augmentations import (
    add_noise,
    gain,
    low_pass_filter,
    time_shift,
    time_stretch,
)


class AudioAugmentations:
    """Audio augmentations for SuDoRM-RF training."""

    time_stretch = staticmethod(time_stretch)
    add_noise = staticmethod(add_noise)
    gain = staticmethod(gain)
    time_shift = staticmethod(time_shift)
    low_pass_filter = staticmethod(low_pass_filter)

    @staticmethod
    def random_augment(
        waveform: torch.Tensor, rng: np.random.Generator | None = None
    ) -> torch.Tensor:
        if rng is None:
            rng = np.random.default_rng()
        augmented = waveform.clone()

        if rng.random() < 0.5:
            augmented = AudioAugmentations.time_stretch(
                augmented, rng.uniform(0.9, 1.1)
            )
        if rng.random() < 0.7:
            augmented = AudioAugmentations.gain(augmented, rng.uniform(-6, 6))
        if rng.random() < 0.4:
            augmented = AudioAugmentations.add_noise(
                augmented, rng.uniform(0.001, 0.01)
            )
        if rng.random() < 0.5:
            max_shift = int(augmented.shape[-1] * 0.1)
            augmented = AudioAugmentations.time_shift(
                augmented, int(rng.integers(-max_shift, max_shift + 1))
            )
        if rng.random() < 0.3:
            augmented = AudioAugmentations.low_pass_filter(
                augmented, rng.uniform(0.6, 0.95)
            )

        return augmented
