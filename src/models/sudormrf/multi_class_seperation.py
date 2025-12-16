"""
Multi-class Class-of-Interest (COI) separation head for SuDoRM-RF.

Supports multiple architectural strategies:
1. Shared background + per-class branches (memory efficient)
2. Fully independent branches (max expressiveness)
3. Hierarchical with shared low-level features (balanced)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base.sudo_rm_rf.dnn.models.groupcomm_sudormrf_v2 import (
    GroupCommSudoRmRf,
)
from .base.sudo_rm_rf.dnn.models.improved_sudormrf import SuDORMRF


class SharedBackgroundMultiCOI(nn.Module):
    """
    Multi-class COI separation with SHARED background branch.
    For scaling to multiple aircraft types: jet, propeller, helicopter, etc.
    """

    def __init__(self, in_channels, out_channels, n_coi_classes=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_coi_classes = n_coi_classes
        self.n_sources = n_coi_classes + 1  # COI classes + background

        # One branch per COI class
        self.coi_branches = nn.ModuleList(
            [self._make_branch(in_channels, out_channels) for _ in range(n_coi_classes)]
        )

        # Single shared background branch
        self.background_branch = self._make_branch(in_channels, out_channels)

    def _make_branch(self, in_chan, out_chan):
        """Create depthwise separable conv branch."""
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
            x: [B, in_channels, T]
        Returns:
            masks: [B, n_sources * out_channels, T]
                   sources = [coi_1, coi_2, ..., coi_n, background]
        """
        # process each COI class branch
        coi_masks = [torch.relu(branch(x)) for branch in self.coi_branches]

        bg_mask = torch.relu(self.background_branch(x))

        # Concatenate: COI classes first, then background
        # Shape: (B, n_sources * out_channels, T)
        masks = torch.cat(coi_masks + [bg_mask], dim=1)

        return masks


def wrap_model_for_multiclass(model, replace_head=True, n_coi_classes=3):
    """Wraps a SuDoRM-RF model with a Multi-class COI separation head.
    Args:
        model: SuDoRM-RF model instance
        replace_head: If True, replaces the existing separation head.
        n_coi_classes: Number of classes of interest.
    Returns:
        model: Modified SuDoRM-RF model with Multi-class COI head
    """
    if replace_head:
        if isinstance(model, SuDORMRF) or isinstance(model, GroupCommSuDORMRFv2):
            in_channels = model.mask_net[-1].in_channels
            out_channels = model.enc_num_basis
            n_src = n_coi_classes + 1

            # 1. Replace the mask network
            model.mask_net = SharedBackgroundMultiCOI(
                in_channels=in_channels,
                out_channels=out_channels,
                n_coi_classes=n_coi_classes,
            )

            # 2. Update the number of sources
            model.num_sources = n_src

            # 3. Re-initialize the decoder
            model.decoder = nn.ConvTranspose1d(
                in_channels=model.enc_num_basis * model.num_sources,
                out_channels=model.num_sources,
                output_padding=(model.enc_kernel_size // 2) - 1,
                kernel_size=model.enc_kernel_size,
                stride=model.enc_kernel_size // 2,
                padding=model.enc_kernel_size // 2,
                groups=1,
                bias=False,
            )
            torch.nn.init.xavier_uniform_(model.decoder.weight)

        else:
            raise TypeError(
                "Model type not supported for Multi-class COI head replacement."
            )

    return model
