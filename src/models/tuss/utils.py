"""
Training utilities for TUSS COI-separation training.

Provides audio normalisation, batch preparation (mixture creation with SNR
mixing), finite checks, and variable-prompt generation for the TUSS model.
"""

import numpy as np
import torch

from .losses import SILENCE_ENERGY_EPS

ENERGY_EPS: float = 1e-8
NORMALIZE_MIN_STD: float = 1e-3
BG_SCALE_MIN: float = 0.1
BG_SCALE_MAX: float = 3.0


def normalize_tensor_wav(
    wav: torch.Tensor,
    eps: float = ENERGY_EPS,
    min_std: float = NORMALIZE_MIN_STD,
) -> torch.Tensor:
    mean = wav.mean(dim=-1, keepdim=True)
    std = wav.std(dim=-1, keepdim=True)
    is_silent = std < min_std
    std_safe = torch.where(is_silent, torch.ones_like(std), std) + eps
    normalized = (wav - mean) / std_safe
    return torch.where(is_silent, torch.zeros_like(normalized), normalized)


def prepare_batch(
    sources: torch.Tensor,
    snr_range: tuple[float, float],
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build mixture and normalised clean sources from the raw source tensor.

    The mixture and all clean source references are normalised using the
    **same** statistics (mean and std of the raw mixture) so that
    ``sum(clean_sources) \\u2248 mixture`` is preserved.  This keeps the
    relative scale between sources consistent with what the model sees.

    Args:
        sources: (B, n_coi_classes + 1, T) \\u2013 COI tracks + background (last)
        snr_range: (min_snr_db, max_snr_db)
        deterministic: use linspace SNR for validation reproducibility
    Returns:
        mixture:    (B, T)
        clean_wavs: (B, n_coi_classes + 1, T)
    """
    B, n_src, T = sources.shape
    eps = ENERGY_EPS

    cois = [sources[:, i, :] for i in range(n_src - 1)]
    bg = sources[:, -1, :]
    total_coi = torch.stack(cois, dim=0).sum(dim=0)  # (B, T)

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
    silent_coi = coi_power < SILENCE_ENERGY_EPS
    bg_scaling = torch.where(silent_coi, torch.ones_like(bg_scaling), bg_scaling)
    bg_scaling = torch.clamp(bg_scaling, BG_SCALE_MIN, BG_SCALE_MAX)

    bg_scaled = bg * bg_scaling

    # ---- Joint normalisation ------------------------------------------------
    # Compute statistics from the raw mixture and apply the *same* transform
    # to every source so that additivity is preserved.
    raw_mixture = total_coi + bg_scaled  # (B, T)
    mix_mean = raw_mixture.mean(dim=-1, keepdim=True)  # (B, 1)
    mix_std = raw_mixture.std(dim=-1, keepdim=True)  # (B, 1)
    is_silent = mix_std < NORMALIZE_MIN_STD
    mix_std_safe = torch.where(is_silent, torch.ones_like(mix_std), mix_std) + eps

    mixture = (raw_mixture - mix_mean) / mix_std_safe
    mixture = torch.where(is_silent, torch.zeros_like(mixture), mixture)

    # Normalise each source with the mixture statistics.
    # Silent COI channels must be re-zeroed after normalization to avoid
    # phantom DC offsets that would trigger the full SNR loss instead of
    # the zero-reference penalty.
    norm_cois = []
    for c in cois:
        # Capture silence before normalization (per sample in the batch).
        was_silent = c.pow(2).mean(dim=-1, keepdim=True) < SILENCE_ENERGY_EPS  # (B, 1)
        normed = (c - mix_mean) / mix_std_safe
        normed = torch.where(is_silent, torch.zeros_like(normed), normed)
        # Re-zero channels whose source was silent (not merely a silent mixture).
        normed = torch.where(was_silent, torch.zeros_like(normed), normed)
        norm_cois.append(normed)

    norm_bg = (bg_scaled - mix_mean) / mix_std_safe
    norm_bg = torch.where(is_silent, torch.zeros_like(norm_bg), norm_bg)

    clean_wavs = torch.stack(norm_cois + [norm_bg], dim=1)  # (B, n_src, T)
    return mixture, clean_wavs


def check_finite(*tensors) -> bool:
    return all(torch.isfinite(t).all() for t in tensors)


def generate_variable_prompts(
    coi_prompts: list[str],
    bg_prompt: str,
    batch_size: int,
    dropout_prob: float = 0.5,
    min_coi: int = 0,
    rng: np.random.Generator | None = None,
) -> list[list[str]]:
    """Generate variable prompt configurations for a batch.

    Randomly drops COI prompts to create variable n_src configurations,
    similar to the base TUSS model's training strategy.

    Args:
        coi_prompts: List of COI prompt names (e.g., ["airplane", "birds"])
        bg_prompt: Background prompt name (e.g., "background")
        batch_size: Number of samples in batch
        dropout_prob: Probability of dropping each COI prompt
        min_coi: Minimum number of COI prompts to keep (0 = allow background-only)
        rng: Random number generator for reproducibility

    Returns:
        List of prompt lists, one per batch sample.
        Each inner list has variable length (min_coi+1 to len(coi_prompts)+1)

    Examples:
        >>> generate_variable_prompts(["airplane", "birds"], "background", 3, 0.5, 0)
        [["airplane", "background"],
         ["airplane", "birds", "background"],
         ["birds", "background"]]
    """
    if rng is None:
        rng = np.random.default_rng()

    n_coi = len(coi_prompts)
    min_coi = max(0, min(min_coi, n_coi))  # Clamp to valid range

    batch_prompts = []
    for _ in range(batch_size):
        # Randomly select which COI prompts to include
        keep_mask = rng.random(n_coi) > dropout_prob

        # Ensure minimum number of COI prompts
        n_kept = keep_mask.sum()
        if n_kept < min_coi:
            # Randomly enable additional prompts to reach minimum
            disabled_indices = np.where(~keep_mask)[0]
            enable_count = min_coi - n_kept
            enable_indices = rng.choice(
                disabled_indices, size=enable_count, replace=False
            )
            keep_mask[enable_indices] = True

        # Build prompt list for this sample
        sample_prompts = [coi_prompts[i] for i in range(n_coi) if keep_mask[i]]
        sample_prompts.append(bg_prompt)  # Always include background

        batch_prompts.append(sample_prompts)

    return batch_prompts


def select_sources_for_prompts(
    clean_wavs: torch.Tensor,
    all_coi_prompts: list[str],
    bg_prompt: str,
    selected_prompts: list[str],
) -> torch.Tensor:
    """Select and reorder source channels to match variable prompt configuration.

    Dropped COI sources are merged into the background channel so the model learns
    to separate only what's prompted and put everything else in the residual.

    Args:
        clean_wavs: (B, n_src_full, T) - full source tensor with all COI classes + background
        all_coi_prompts: Complete list of COI prompts (e.g., ["airplane", "birds"])
        bg_prompt: Background prompt name
        selected_prompts: Variable prompts for this batch (e.g., ["airplane", "background"])

    Returns:
        Tensor (B, n_src_selected, T) with sources matching selected_prompts order,
        where background includes any dropped COI sources.

    Example:
        clean_wavs.shape = (B, 3, T)  # [airplane, birds, background]
        all_coi_prompts = ["airplane", "birds"]
        selected_prompts = ["birds", "background"]

        Returns: (B, 2, T)  # [birds, (airplane+background)] with airplane merged to bg
    """
    B, n_src_full, T = clean_wavs.shape

    # Identify which COI prompts are ACTIVE (in selected_prompts)
    active_coi_indices = []
    for prompt in selected_prompts:
        if prompt != bg_prompt and prompt in all_coi_prompts:
            active_coi_indices.append(all_coi_prompts.index(prompt))

    # Start with original background channel
    bg_channel = clean_wavs[:, -1, :].clone()

    # Merge dropped COI sources into background
    # (any COI source that doesn't have a corresponding prompt)
    for i in range(len(all_coi_prompts)):
        if i not in active_coi_indices:
            bg_channel = bg_channel + clean_wavs[:, i, :]

    # Build output: active COI channels (in selected_prompts order) + merged background
    selected_channels = []
    for prompt in selected_prompts:
        if prompt == bg_prompt:
            selected_channels.append(bg_channel)
        elif prompt in all_coi_prompts:
            idx = all_coi_prompts.index(prompt)
            selected_channels.append(clean_wavs[:, idx, :])

    return torch.stack(selected_channels, dim=1)
