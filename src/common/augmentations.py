"""
Canonical audio augmentation functions for training data.

Consolidates duplicated implementations from:
- src/models/tuss/augmentations.py
- src/models/sudormrf/augmentations.py
- src/common/coi_training.py
- src/common/webdataset_utils.py

Each consumer wraps these in its own AudioAugmentations class for
backward compatibility.
"""

import torch


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


def add_noise(waveform: torch.Tensor, noise_level: float = 0.005) -> torch.Tensor:
    return waveform + torch.randn_like(waveform) * noise_level


def gain(waveform: torch.Tensor, gain_db: float) -> torch.Tensor:
    return waveform * (10 ** (gain_db / 20.0))


def time_shift(waveform: torch.Tensor, shift_samples: int) -> torch.Tensor:
    if shift_samples == 0:
        return waveform
    shifted = torch.zeros_like(waveform)
    if shift_samples > 0:
        shifted[..., shift_samples:] = waveform[..., :-shift_samples]
    else:
        shifted[..., :shift_samples] = waveform[..., -shift_samples:]
    return shifted


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
