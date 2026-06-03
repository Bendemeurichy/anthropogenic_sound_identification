"""
Audio augmentations for TUSS COI-separation training.

Provides both CPU and GPU-accelerated audio augmentation classes that apply
random time-stretching, gain, additive noise, time-shift, and low-pass
filtering to waveforms.

Core augmentation functions are imported from src.common.augmentations.
"""

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
    """CPU-based audio augmentations for per-item waveform processing."""

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
        aug = waveform.clone()
        if rng.random() < 0.5:
            aug = AudioAugmentations.time_stretch(aug, rng.uniform(0.9, 1.1))
        if rng.random() < 0.7:
            aug = AudioAugmentations.gain(aug, rng.uniform(-6, 6))
        if rng.random() < 0.4:
            aug = AudioAugmentations.add_noise(aug, rng.uniform(0.001, 0.01))
        if rng.random() < 0.5:
            max_shift = int(aug.shape[-1] * 0.1)
            aug = AudioAugmentations.time_shift(
                aug, int(rng.integers(-max_shift, max_shift + 1))
            )
        if rng.random() < 0.3:
            aug = AudioAugmentations.low_pass_filter(aug, rng.uniform(0.6, 0.95))
        return aug


class GpuAudioAugmentations:
    """GPU-accelerated audio augmentations for batch processing.

    All methods work on batched tensors (B, T) or (B, n_src, T) and run on GPU.
    Provides 10-100x speedup compared to CPU augmentations.
    """

    @staticmethod
    def time_stretch_batch(
        waveform: torch.Tensor, rate_range: tuple[float, float] = (0.9, 1.1)
    ) -> torch.Tensor:
        """Apply random time stretch to each sample in batch."""
        if waveform.dim() == 2:
            B, T = waveform.shape
            n_src = None
        else:
            B, n_src, T = waveform.shape
            waveform = waveform.reshape(B * n_src, T)

        rates = torch.empty(B if n_src is None else B * n_src, device=waveform.device).uniform_(*rate_range)

        result = []
        for i, rate in enumerate(rates):
            if abs(rate - 1.0) < 0.01:
                result.append(waveform[i])
                continue

            stretched = torch.nn.functional.interpolate(
                waveform[i].unsqueeze(0).unsqueeze(0),
                scale_factor=1.0 / rate.item(),
                mode="linear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

            if stretched.shape[-1] > T:
                result.append(stretched[:T])
            else:
                result.append(
                    torch.nn.functional.pad(stretched, (0, T - stretched.shape[-1]))
                )

        output = torch.stack(result, dim=0)
        if n_src is not None:
            output = output.reshape(B, n_src, T)
        return output

    @staticmethod
    def add_noise_batch(
        waveform: torch.Tensor, noise_level_range: tuple[float, float] = (0.001, 0.01)
    ) -> torch.Tensor:
        """Add white Gaussian noise to each sample in batch."""
        noise_levels = torch.empty(
            waveform.shape[0], device=waveform.device
        ).uniform_(*noise_level_range)

        if waveform.dim() == 3:
            noise_levels = noise_levels.unsqueeze(1).unsqueeze(2)
        else:
            noise_levels = noise_levels.unsqueeze(1)

        noise = torch.randn_like(waveform) * noise_levels
        return waveform + noise

    @staticmethod
    def gain_batch(
        waveform: torch.Tensor, gain_db_range: tuple[float, float] = (-6.0, 6.0)
    ) -> torch.Tensor:
        """Apply random gain to each sample in batch."""
        gain_db = torch.empty(
            waveform.shape[0], device=waveform.device
        ).uniform_(*gain_db_range)

        if waveform.dim() == 3:
            gain_linear = (10 ** (gain_db / 20.0)).unsqueeze(1).unsqueeze(2)
        else:
            gain_linear = (10 ** (gain_db / 20.0)).unsqueeze(1)

        return waveform * gain_linear

    @staticmethod
    def time_shift_batch(
        waveform: torch.Tensor, max_shift_ratio: float = 0.1
    ) -> torch.Tensor:
        """Apply random time shift to each sample in batch with zero-padding.

        Uses zero-padding instead of circular roll to avoid wraparound artifacts.
        Positive shift = delay (add silence at start), negative = advance (trim start).
        """
        T = waveform.shape[-1]
        max_shift = int(T * max_shift_ratio)

        shifts = torch.randint(
            -max_shift,
            max_shift + 1,
            (waveform.shape[0],),
            device=waveform.device,
        )

        result = []
        for i, shift in enumerate(shifts):
            shift_val = shift.item()
            if shift_val == 0:
                result.append(waveform[i])
            elif shift_val > 0:
                shifted = torch.zeros_like(waveform[i])
                shifted[..., shift_val:] = waveform[i, ..., :-shift_val]
                result.append(shifted)
            else:
                shifted = torch.zeros_like(waveform[i])
                shifted[..., :shift_val] = waveform[i, ..., -shift_val:]
                result.append(shifted)

        return torch.stack(result, dim=0)

    @staticmethod
    def low_pass_filter_batch(
        waveform: torch.Tensor, cutoff_ratio_range: tuple[float, float] = (0.6, 0.95)
    ) -> torch.Tensor:
        """Apply random low-pass filter to each sample in batch."""
        if waveform.dim() == 2:
            B, T = waveform.shape
            n_src = None
        else:
            B, n_src, T = waveform.shape
            waveform = waveform.reshape(B * n_src, T)

        cutoff_ratios = torch.empty(
            B if n_src is None else B * n_src, device=waveform.device
        ).uniform_(*cutoff_ratio_range)

        fft = torch.fft.rfft(waveform, dim=-1)
        n_freqs = fft.shape[-1]

        result = []
        for i, ratio in enumerate(cutoff_ratios):
            if ratio >= 0.99:
                result.append(waveform[i])
                continue

            cutoff_idx = int(n_freqs * ratio.item())
            mask = torch.ones(n_freqs, device=waveform.device)

            rolloff_width = max(1, n_freqs // 20)
            for j in range(rolloff_width):
                if cutoff_idx + j < n_freqs:
                    mask[cutoff_idx + j] = 1.0 - (j / rolloff_width)
            mask[cutoff_idx + rolloff_width :] = 0.0

            filtered = torch.fft.irfft(fft[i] * mask, n=T)
            result.append(filtered)

        output = torch.stack(result, dim=0)
        if n_src is not None:
            output = output.reshape(B, n_src, T)
        return output

    @staticmethod
    def random_augment_batch(
        waveform: torch.Tensor,
        time_stretch_prob: float = 0.5,
        gain_prob: float = 0.7,
        noise_prob: float = 0.4,
        shift_prob: float = 0.5,
        lpf_prob: float = 0.3,
    ) -> torch.Tensor:
        """Apply random combination of augmentations to batch.

        Args:
            waveform: (B, T) or (B, n_src, T) tensor on GPU
            *_prob: Probability of applying each augmentation

        Returns:
            Augmented waveform with same shape
        """
        aug = waveform.clone()

        if torch.rand(1).item() < time_stretch_prob:
            aug = GpuAudioAugmentations.time_stretch_batch(aug)

        if torch.rand(1).item() < gain_prob:
            aug = GpuAudioAugmentations.gain_batch(aug)

        if torch.rand(1).item() < noise_prob:
            aug = GpuAudioAugmentations.add_noise_batch(aug)

        if torch.rand(1).item() < shift_prob:
            aug = GpuAudioAugmentations.time_shift_batch(aug)

        if torch.rand(1).item() < lpf_prob:
            aug = GpuAudioAugmentations.low_pass_filter_batch(aug)

        return aug
