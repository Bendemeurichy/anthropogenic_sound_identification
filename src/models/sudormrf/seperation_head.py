"""
Class of interest-specific separation head for use with original SuDoRM-RF models.
This module adapts the existing improved_sudormrf or groupcomm_sudormrf_v2 models.

Usage:
    from base.sudo_rm_rf.dnn.models.improved_sudormrf import SuDORMRF
    from seperation_head import wrap_model_for_coi

    # Load original model
    base_model = SuDORMRF(
        out_channels=256,
        in_channels=512,
        num_blocks=16,
        upsampling_depth=5,
        enc_kernel_size=21,
        enc_num_basis=512,
        num_sources=2
    )

    # Wrap with aircraft-specific head
    model = wrap_model_for_coi(base_model, coi='aircraft')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.audio import ScaleInvariantSinalNoiseRatio as SI_SNR
from base.sudo_rm_rf.sudo_rm_rf.dnn.models.groupcomm_sudormrf_v2 import (
    GroupCommSuDORMRFv2,
)
from base.sudo_rm_rf.sudo_rm_rf.dnn.models.improved_sudormrf import SuDORMRF


class COISeparationHead(nn.Module):
    """Seperation head specific for class of interest target audio seperation,
    replaces the original SuDoRM-RF final masking layer.
    Produces masks for class of interest and background.
    Args:
        nn (_type_): _description_
    """

    def __init__(self, in_channels, out_channels, n_src=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_src = n_src

        self.coi_branch = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv1d(in_channels, out_channels, 1),
            nn.PReLU(out_channels),
            nn.Conv1d(out_channels, out_channels, 3, padding=1, groups=out_channels),
            nn.Conv1d(out_channels, out_channels, 1),
        )

        self.background_branch = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv1d(in_channels, out_channels, 1),
            nn.PReLU(out_channels),
            nn.Conv1d(out_channels, out_channels, 3, padding=1, groups=out_channels),
            nn.Conv1d(out_channels, out_channels, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, T)
        Returns:
            masks: (B, n_src, C, T)
        """
        coi_mask = torch.relu(self.coi_branch(x))
        background_mask = torch.relu(self.background_branch(x))
        masks = torch.stack([coi_mask, background_mask], dim=1)
        return masks


def wrap_model_for_coi(model, replace_head=True, coi="aircraft"):
    """Wraps a SuDoRM-RF model with a COI-specific separation head.
    Args:
        model: SuDoRM-RF model instance
        replace_head: If True, replaces the existing separation head.
        coi: Class of interest type (for future use)
    Returns:
        model: Modified SuDoRM-RF model with COI-specific head
    """

    if replace_head:
        if isinstance(model, SuDORMRF) or isinstance(model, GroupCommSuDORMRFv2):
            in_channels = model.mask_net[-1].in_channels
            out_channels = model.mask_net[-1].out_channels
            n_src = 2  # COI and background

            model.mask_net = COISeparationHead(
                in_channels=in_channels, out_channels=out_channels, n_src=n_src
            )
        else:
            raise TypeError("Model type not supported for COI head replacement.")

    return model


class COILoss(nn.Module):
    """Loss function for class of interest separation.
    Combines SI-SNR for COI and background with class-aware weighting.
    """

    def __init__(self, class_weight=1.5, eps=1e-8):
        super().__init__()
        self.class_weight = class_weight
        self.eps = eps

    def forward(self, est_sources, target_sources):
        """
        Args:
            est_sources: (B, n_src, T)
            target_sources: (B, n_src, T)
        Returns:
            loss: Scalar loss value
        """
        coi_sisnr = SI_SNR(est_sources[:, 0, :], target_sources[:, 0, :], eps=self.eps)
        background_sisnr = SI_SNR(
            est_sources[:, 1, :], target_sources[:, 1, :], eps=self.eps
        )

        weighted_sisnr = (self.class_weight * coi_sisnr + background_sisnr) / (
            self.class_weight + 1
        )

        loss = -weighted_sisnr.mean()
        return loss
