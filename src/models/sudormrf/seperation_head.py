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
    """Seperation head specific for class of interest target audio seperation,
    replaces the original SuDoRM-RF final masking layer.
    Produces masks for class of interest and background.
    """

    def __init__(self, in_channels, out_channels, n_src=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_src = n_src

        if n_src != 2:
            raise ValueError(
                "COISeparationHead only supports n_src=2 (COI + Background). Use multi_class_seperation for more sources."
            )

        self.coi_branch = self._make_branch(in_channels, out_channels)
        self.background_branch = self._make_branch(in_channels, out_channels)

    def _make_branch(self, in_chan, out_chan):
        return nn.Sequential(
            nn.Conv1d(in_chan, in_chan, 3, padding=1, groups=in_chan),
            nn.Conv1d(in_chan, out_chan, 1),
            nn.PReLU(out_chan),
            nn.Conv1d(out_chan, out_chan, 3, padding=1, groups=out_chan),
            nn.Conv1d(out_chan, out_chan, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, T)
        Returns:
            masks: (B, n_src * C, T) - Concatenated masks for SuDoRM-RF compatibility
        """
        coi_mask = torch.relu(self.coi_branch(x))
        background_mask = torch.relu(self.background_branch(x))
        # Concatenate along channel dimension to match SuDoRM-RF expectation
        # Shape becomes (B, 2 * out_channels, T)
        masks = torch.cat([coi_mask, background_mask], dim=1)
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
            in_channels = model.mask_net[-1].in_channels
            # Use enc_num_basis as out_channels for each source branch
            out_channels = model.enc_num_basis
            n_src = 2  # COI and background

            model.mask_net = COISeparationHead(
                in_channels=in_channels, out_channels=out_channels, n_src=n_src
            )

            # Ensure model properties match
            model.num_sources = n_src

            # Re-initialize decoder if needed (though usually base model is already 2 sources)
            if model.decoder.in_channels != model.enc_num_basis * n_src:
                model.decoder = nn.ConvTranspose1d(
                    in_channels=model.enc_num_basis * n_src,
                    out_channels=n_src,
                    output_padding=(model.enc_kernel_size // 2) - 1,
                    kernel_size=model.enc_kernel_size,
                    stride=model.enc_kernel_size // 2,
                    padding=model.enc_kernel_size // 2,
                    groups=1,
                    bias=False,
                )
                torch.nn.init.xavier_uniform_(model.decoder.weight)

        else:
            raise TypeError("Model type not supported for COI head replacement.")

    return model
