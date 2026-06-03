"""Audio augmentations for training data."""

import numpy as np
import torch


class AudioAugmentations:
    """Audio augmentations for SuDoRM-RF training."""

    @staticmethod
    def time_stretch(waveform: torch.Tensor, rate: float) -> torch.Tensor:
        if rate == 1.0:
            return waveform
        orig_len = waveform.shape[-1]
        stretched = (
            torch.nn.functional.interpolate(
                waveform.unsqueeze(0).unsqueeze(0),
                scale_factor=1.0 / rate,
                mode="linear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )
        if stretched.shape[-1] > orig_len:
            return stretched[:orig_len]
        return torch.nn.functional.pad(stretched, (0, orig_len - stretched.shape[-1]))

    @staticmethod
    def add_noise(waveform: torch.Tensor, noise_level: float = 0.005) -> torch.Tensor:
        return waveform + torch.randn_like(waveform) * noise_level

    @staticmethod
    def gain(waveform: torch.Tensor, gain_db: float) -> torch.Tensor:
        return waveform * (10 ** (gain_db / 20.0))

    @staticmethod
    def time_shift(waveform: torch.Tensor, shift_samples: int) -> torch.Tensor:
        return torch.roll(waveform, shifts=shift_samples, dims=-1)

    @staticmethod
    def low_pass_filter(
        waveform: torch.Tensor, cutoff_ratio: float = 0.8
    ) -> torch.Tensor:
        if cutoff_ratio >= 1.0:
            return waveform
        fft = torch.fft.rfft(waveform)
        n_freqs = fft.shape[-1]
        cutoff_idx = int(n_freqs * cutoff_ratio)
        mask = torch.ones(n_freqs, device=waveform.device)
        rolloff_width = max(1, n_freqs // 20)
        for i in range(rolloff_width):
            if cutoff_idx + i < n_freqs:
                mask[cutoff_idx + i] = 1.0 - (i / rolloff_width)
        mask[cutoff_idx + rolloff_width :] = 0.0
        return torch.fft.irfft(fft * mask, n=waveform.shape[-1])

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
