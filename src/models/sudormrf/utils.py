"""
Training utilities for sudormrf: normalization, batch preparation, and validation.
"""

import torch

from .losses import SILENCE_ENERGY_EPS

ENERGY_EPS = 1e-8
NORMALIZE_MIN_STD = 1e-3
BG_SCALE_MIN = 0.1
BG_SCALE_MAX = 3.0


def normalize_tensor_wav(
    wav: torch.Tensor, eps: float = ENERGY_EPS, min_std: float = NORMALIZE_MIN_STD
) -> torch.Tensor:
    """Normalize waveform to zero mean and unit variance."""
    mean = wav.mean(dim=-1, keepdim=True)
    std = wav.std(dim=-1, keepdim=True)
    is_silent = std < min_std
    std_safe = torch.where(is_silent, torch.ones_like(std), std) + eps
    normalized = (wav - mean) / std_safe
    return torch.where(is_silent, torch.zeros_like(normalized), normalized)


def prepare_batch(
    sources: torch.Tensor, snr_range: tuple[float, float], deterministic: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare mixture and clean targets from source tensor.

    Follows the same convention as the base SuDORMRF training script:
      - The mixture input is normalized to zero-mean unit-variance.
      - Each target source is normalized independently to zero-mean unit-variance.
      - SI-SNR is scale-invariant, so additivity (sum(targets) == mixture) is
        NOT required and NOT enforced. Enforcing it (joint normalization) was
        found to create an easy local minimum where the model outputs the mixture
        for all heads and still achieves artificially high SI-SNR on training
        batches, while failing to separate on validation.

    Args:
        sources: (B, n_src, T) tensor with COI sources and background (last channel)
        snr_range: (min_snr, max_snr) in dB
        deterministic: If True, use linspace SNRs; otherwise random

    Returns:
        mixture: (B, T) normalized mixture (zero-mean, unit-variance)
        clean_wavs: (B, n_src, T) independently normalized target sources
    """
    B, n_src, T = sources.shape
    eps = ENERGY_EPS

    cois = [sources[:, i, :] for i in range(n_src - 1)]
    bg = sources[:, -1, :]
    total_coi = torch.stack(cois, dim=0).sum(dim=0)

    # SNR calculation
    if deterministic and B > 1:
        snr_db = torch.linspace(
            snr_range[0], snr_range[1], B, device=sources.device
        ).view(B, 1)
    else:
        snr_db = torch.zeros(B, 1, device=sources.device).uniform_(*snr_range)

    coi_power = total_coi.pow(2).mean(dim=-1, keepdim=True) + eps
    bg_power = bg.pow(2).mean(dim=-1, keepdim=True) + eps
    snr_linear = torch.pow(10.0, snr_db / 10.0)
    bg_scaling = torch.sqrt(coi_power / (bg_power * snr_linear + eps))

    # Don't scale if COI is silent
    silent_coi = coi_power < SILENCE_ENERGY_EPS
    bg_scaling = torch.where(silent_coi, torch.ones_like(bg_scaling), bg_scaling)
    bg_scaling = torch.clamp(bg_scaling, min=BG_SCALE_MIN, max=BG_SCALE_MAX)

    bg_scaled = bg * bg_scaling

    # ---- Normalize mixture input -------------------------------------------
    # The model input is the normalized mixture (zero-mean, unit-variance).
    raw_mixture = total_coi + bg_scaled  # (B, T)
    mixture = normalize_tensor_wav(raw_mixture)

    # ---- Independently normalize each target source ------------------------
    # Each target is normalized to zero-mean unit-variance independently.
    # Silent COI channels (background-only samples) are explicitly zeroed to
    # avoid phantom DC offsets from normalizing a zero signal.
    normalized_cois = []
    for c in cois:
        was_silent = c.pow(2).mean(dim=-1, keepdim=True) < SILENCE_ENERGY_EPS
        normed = normalize_tensor_wav(c)
        normed = torch.where(was_silent, torch.zeros_like(normed), normed)
        normalized_cois.append(normed)

    normalized_bg = normalize_tensor_wav(bg_scaled)

    clean_wavs = torch.stack(normalized_cois + [normalized_bg], dim=1)

    return mixture, clean_wavs


def check_finite(*tensors) -> bool:
    """Check if all tensors contain finite values."""
    return all(torch.isfinite(t).all() for t in tensors)
