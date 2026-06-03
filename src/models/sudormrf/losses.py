"""
Loss functions for the SuDORMRF audio separation model.

Provides:
    - Scale-Invariant Signal-to-Noise Ratio (SI-SNR) metric
    - Class-of-Interest Weighted Loss for source separation training
"""

import torch

SILENCE_ENERGY_EPS = 1e-6
WEAK_TARGET_ENERGY_EPS = 1e-4


def sisnr(est: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute scale-invariant SNR (dB) per example.

    Args:
        est: (B, T) or (B, 1, T)
        target: (B, T) or (B, 1, T)
    Returns:
        sisnr_db: (B,) tensor of SI-SNR in dB, clamped to [-30, 30]
    """
    est, target = est.float(), target.float()
    if est.ndim == 3:
        est = est.squeeze(1)
    if target.ndim == 3:
        target = target.squeeze(1)

    # Zero-mean
    est_zm = est - est.mean(dim=-1, keepdim=True)
    target_zm = target - target.mean(dim=-1, keepdim=True)

    # Energy calculations
    T = target.shape[-1]
    min_energy = SILENCE_ENERGY_EPS
    est_energy = est_zm.pow(2).sum(dim=-1)
    target_energy = target_zm.pow(2).sum(dim=-1)

    target_is_zero = target_energy < (min_energy * T)
    target_is_weak = target_energy < (WEAK_TARGET_ENERGY_EPS * T)
    target_energy_safe = torch.clamp(target_energy, min=min_energy)

    # Projection
    s_target = (est_zm * target_zm).sum(dim=-1, keepdim=True) / (
        target_energy_safe.unsqueeze(-1) + eps
    )
    s_target = torch.clamp(s_target, min=-100.0, max=100.0)
    s_true = s_target * target_zm
    e_noise = est_zm - s_true

    # SI-SNR
    sisnr_lin = torch.clamp(
        s_true.pow(2).sum(dim=-1) / (e_noise.pow(2).sum(dim=-1) + eps),
        min=1e-10,
        max=1e10,
    )
    sisnr_db = 10.0 * torch.log10(sisnr_lin + eps)

    # Handle silent/weak targets
    # For truly silent targets, reward low output energy with scores up to 12 dB.
    #
    # Key design constraints:
    #   1. Standard SI-SNR gives zero gradient when target == 0 (s_true = 0 always),
    #      so we need a separate energy-based penalty for silent targets.
    #   2. Use MEAN energy (not sum) so the score is T-independent and comparable
    #      in magnitude to typical SI-SNR values.
    #   3. NO lower clamp here — the outer return clamp handles the reporting floor.
    #      A min clamp here would kill the gradient whenever the score falls below it,
    #      which for a unit-variance output (mean_energy ≈ 1) would be:
    #          -10 * log10(1 + eps) ≈ 0 dB  → NOT clamped → gradient flows ✓
    #      With the old sum-based formula and min=-30 clamp the score was always
    #      ≈ -60 dB → clamped to -30 → zero gradient on every background-only sample.
    est_mean_energy = est_zm.pow(2).mean(dim=-1)
    silence_score = torch.clamp(-10.0 * torch.log10(est_mean_energy + eps), max=12.0)
    sisnr_db = torch.where(target_is_zero, silence_score, sisnr_db)
    sisnr_db = torch.where(
        target_is_weak & ~target_is_zero, torch.clamp(sisnr_db, -20.0, 20.0), sisnr_db
    )

    return torch.clamp(sisnr_db, min=-30.0, max=30.0)


class COIWeightedLoss(torch.nn.Module):
    """Fixed-order, class-of-interest weighted SI-SNR loss.

    Works for any number of sources N >= 2:
        - Channels 0 .. N-2  are COI classes (all weighted equally by class_weight)
        - Channel  N-1        is background

    For N=2 (single-class) this is identical to the previous behaviour.
    """

    def __init__(self, class_weight: float = 1.5, eps: float = 1e-8):
        super().__init__()
        self.class_weight = float(class_weight)
        self.eps = float(eps)

    def forward(
        self, est_sources: torch.Tensor, target_sources: torch.Tensor
    ) -> torch.Tensor:
        if est_sources.ndim != 3 or target_sources.ndim != 3:
            raise ValueError("est_sources and target_sources must be (B, n_src, T)")

        n_src = est_sources.shape[1]
        if n_src < 2:
            raise ValueError(f"Expected at least 2 sources, got {n_src}")

        n_coi = n_src - 1

        if n_coi == 1:
            # Single-class: identical to previous behaviour — always head 0.
            coi_sisnr = sisnr(
                est_sources[:, 0, :], target_sources[:, 0, :], eps=self.eps
            )
        else:
            # Multi-class: route each sample to its active COI head only.
            #
            # Each batch item carries real audio in exactly one COI slot
            # (the slot matching its coi_class) and zeros everywhere else.
            # Averaging SI-SNR over all heads would dilute the meaningful
            # training signal with ill-defined zero-target terms; instead we
            # compute SI-SNR only for the head that actually holds real audio.
            #
            # For background-only samples (all COI targets are zero) there is
            # no active head, so the COI term contributes 0 to the loss —
            # those samples train only the background head.

            # Energy of each COI target channel: (B, n_coi)
            coi_target_energy = torch.stack(
                [
                    target_sources[:, i, :].pow(2).mean(dim=-1)
                    for i in range(n_coi)
                ],
                dim=1,
            )
            active_mask = coi_target_energy > SILENCE_ENERGY_EPS  # (B, n_coi)

            # SI-SNR for every COI head: (B, n_coi)
            all_coi_sisnr = torch.stack(
                [
                    sisnr(est_sources[:, i, :], target_sources[:, i, :], eps=self.eps)
                    for i in range(n_coi)
                ],
                dim=1,
            )

            # Weighted average: sum over active heads, divide by active count.
            # clamp(min=1) avoids div-by-zero for background-only samples
            # while keeping their numerator = 0 → coi_sisnr = 0.
            active_float = active_mask.float()
            active_count = active_float.sum(dim=1).clamp(min=1.0)  # (B,)
            coi_sisnr = (all_coi_sisnr * active_float).sum(dim=1) / active_count

        # --- Background head: always the last index ---
        bg_sisnr = sisnr(
            est_sources[:, -1, :],
            target_sources[:, -1, :],
            eps=self.eps,
        )

        weighted = (self.class_weight * coi_sisnr + bg_sisnr) / (
            self.class_weight + 1.0
        )
        return -weighted.mean()
