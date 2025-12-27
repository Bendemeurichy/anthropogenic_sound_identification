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
from .base.sudo_rm_rf.dnn.models.groupcomm_sudormrf_v2 import (
    GroupCommSudoRmRf,
)
from .base.sudo_rm_rf.dnn.models.improved_sudormrf import SuDORMRF


class COISeparationHead(nn.Module):
    """Separation head specific for class of interest target audio separation.

    Replaces the original SuDoRM-RF final masking layer.
    Produces masks for class of interest and background.

    This head mimics the original mask_net structure: PReLU -> Conv1d
    The output is (B, n_src * enc_num_basis, T) which gets reshaped and
    passed through mask_nl_class (ReLU) in the base model's forward().

    We add semantic separation by having dedicated branches for COI vs background,
    but output in the same format as the original mask_net.
    """

    def __init__(self, in_channels, out_channels, n_src=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_src = n_src

        if n_src != 2:
            raise ValueError(
                "COISeparationHead only supports n_src=2 (COI + Background). "
                "Use multi_class_seperation for more sources."
            )

        # Shared feature extraction before branching (no activation - let branches handle it)
        self.shared_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv1d(in_channels, in_channels, 1),
        )

        # COI-specific branch - output is PRE-ReLU (ReLU applied by mask_nl_class)
        # Using linear output so mask_nl_class can apply ReLU properly
        self.coi_branch = nn.Sequential(
            nn.PReLU(in_channels),
            nn.Conv1d(in_channels, out_channels, 1),  # Linear projection to mask dim
        )

        # Background-specific branch - same structure
        self.background_branch = nn.Sequential(
            nn.PReLU(in_channels),
            nn.Conv1d(in_channels, out_channels, 1),  # Linear projection to mask dim
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, T) - bottleneck features from separation module (after PReLU from mask_net Sequential)
        Returns:
            masks: (B, n_src * out_channels, T) - Concatenated PRE-ACTIVATION masks
                   The base model will reshape to (B, n_src, out_channels, T) and apply ReLU.
        """
        shared = self.shared_conv(x)

        coi_mask = self.coi_branch(shared)  # (B, out_channels, T)
        bg_mask = self.background_branch(shared)  # (B, out_channels, T)

        # Concatenate: [COI, Background] along channel dimension
        # Output shape: (B, 2 * out_channels, T) = (B, n_src * enc_num_basis, T)
        masks = torch.cat([coi_mask, bg_mask], dim=1)
        return masks


def wrap_model_for_coi(model, replace_head=True):
    """Wraps a SuDoRM-RF model with a COI-specific separation head.

    Args:
        model: SuDoRM-RF model instance
        replace_head: If True, replaces the existing separation head.
    Returns:
        model: Modified SuDoRM-RF model with COI-specific head
    """

    if replace_head:
        if isinstance(model, SuDORMRF) or isinstance(model, GroupCommSudoRmRf):
            in_channels = model.out_channels  # output of bottleneck/separation module
            out_channels = model.enc_num_basis  # must match encoder basis for masking
            n_src = 2  # COI and background

            # Replace mask_net with our COI-specific head
            model.mask_net = nn.Sequential(
                nn.PReLU(),  # Keep the PReLU that was in the original mask_net
                COISeparationHead(
                    in_channels=in_channels, out_channels=out_channels, n_src=n_src
                ),
            )

            # Ensure model properties match
            model.num_sources = n_src

            # The decoder should NOT be changed from the original - it uses groups=1
            # which is correct for the SuDoRM-RF architecture.
            # The mask multiplication happens BEFORE the decoder in the forward pass.

        else:
            raise TypeError("Model type not supported for COI head replacement.")

    return model
