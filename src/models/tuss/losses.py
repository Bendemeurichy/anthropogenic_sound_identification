"""
Loss functions and learning-rate scheduling for TUSS COI-separation training.

Classes:
    COIWeightedSNRLoss – SNR loss with zero-reference handling, weighted towards
        COI (class-of-interest) heads.  Silent reference channels are detected
        via an energy threshold and excluded from the loss so they do not
        dominate the gradient.
    WarmupScheduler – Linear LR warm-up that ramps each parameter group from
        zero to its target learning rate over a configurable number of steps.
        After warmup it becomes a no-op so that ReduceLROnPlateau can own the
        learning rate without interference.
"""

import torch
import torch.optim as optim
import numpy as np

from loss_functions.snr import snr_with_zeroref_loss

# Energy threshold below which a channel is considered silent.
# Shared with prepare_batch in train.py.
SILENCE_ENERGY_EPS: float = 1e-6


class COIWeightedSNRLoss(torch.nn.Module):
    """SNR loss with zero-reference handling, weighted towards COI heads."""

    def __init__(
        self,
        n_src: int,
        coi_weight: float = 1.5,
        snr_max: int = 30,
        zero_ref_loss_weight: float = 0.1,
        eps: float = 1e-7,
        amp_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.n_src = n_src
        self.n_coi = n_src - 1  # last source is always background
        self.coi_weight = float(coi_weight)
        self.snr_max = snr_max
        self.zero_ref_loss_weight = zero_ref_loss_weight
        self.eps = eps
        # Stored so train_epoch / validate_epoch can read the dtype
        self._amp_dtype = amp_dtype

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Args:
            est: (B, n_src, T)
            ref: (B, n_src, T)
        Returns:
            scalar loss

        Note:
            When using variable prompts, n_src may differ from self.n_src.
            The function infers the actual n_src from the tensor shape.
        """
        # Infer actual n_src from tensor shape (supports variable prompts)
        actual_n_src = est.shape[1]
        actual_n_coi = actual_n_src - 1  # Last source is always background

        # snr_with_zeroref_loss returns (B, n_src) with solve_perm=False
        per_src = snr_with_zeroref_loss(
            est,
            ref,
            n_src=actual_n_src,  # Use actual n_src from tensor
            snr_max=self.snr_max,
            zero_ref_loss_weight=self.zero_ref_loss_weight,
            solve_perm=False,
            eps=self.eps,
        )  # (B, n_src)

        # COI heads are 0 … n_coi-1, background is last
        # Only average over ACTIVE COI classes (non-silent channels)
        # This prevents silent classes from dominating the loss gradient
        ref_power = (ref[:, :actual_n_coi] ** 2).mean(dim=-1)  # (B, actual_n_coi)
        is_active = ref_power > SILENCE_ENERGY_EPS  # (B, actual_n_coi)

        coi_losses = per_src[:, :actual_n_coi]  # (B, actual_n_coi)
        active_count = is_active.sum(dim=-1).clamp(min=1)  # (B,), prevent div by zero
        coi_loss = (coi_losses * is_active.float()).sum(dim=-1) / active_count  # (B,)

        bg_loss = per_src[:, -1]  # (B,)

        weighted = (self.coi_weight * coi_loss + bg_loss) / (self.coi_weight + 1.0)
        return weighted.mean()


class WarmupScheduler:
    """Linear warm-up that scales each param group from 0 → its target LR.

    After ``warmup_steps`` optimizer steps the scheduler becomes a no-op so
    that ``ReduceLROnPlateau`` can take over without interference.

    The per-group target LRs are captured from the optimizer at construction
    time, so differential LRs (e.g. backbone at 0.05× base) are respected.
    """

    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = max(1, int(warmup_steps))
        self._step_count = 0
        # Capture per-group target LRs before we zero them out.
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        # Start from zero so the first step sets lr = 1/warmup_steps × base.
        for g in optimizer.param_groups:
            g["lr"] = 0.0

    def step(self) -> None:
        """Advance by one optimizer step; no-op once warmup is complete."""
        self._step_count += 1
        if self._step_count <= self.warmup_steps:
            scale = self._step_count / self.warmup_steps
            for g, blr in zip(self.optimizer.param_groups, self.base_lrs):
                g["lr"] = blr * scale
        # After warmup: leave LR unchanged (ReduceLROnPlateau owns it).

    @property
    def done(self) -> bool:
        return self._step_count >= self.warmup_steps

    def state_dict(self) -> dict:
        return {"step_count": self._step_count, "base_lrs": self.base_lrs}

    def load_state_dict(self, state: dict) -> None:
        self._step_count = state["step_count"]
        self.base_lrs = state.get("base_lrs", self.base_lrs)
        # Restore the current LR position.
        if not self.done:
            scale = self._step_count / self.warmup_steps
            for g, blr in zip(self.optimizer.param_groups, self.base_lrs):
                g["lr"] = blr * scale
        else:
            # Warmup finished: restore base_lrs (ReduceLROnPlateau may have
            # reduced them further; its own state_dict handles that).
            for g, blr in zip(self.optimizer.param_groups, self.base_lrs):
                if g["lr"] == 0.0:
                    g["lr"] = blr
