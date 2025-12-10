"""
Multi-class Class-of-Interest (COI) separation head for SuDoRM-RF.

Supports multiple architectural strategies:
1. Shared background + per-class branches (memory efficient)
2. Fully independent branches (max expressiveness)
3. Hierarchical with shared low-level features (balanced)
"""

# TODO: Adapt to use proper base model imports and single seperation head

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            nn.PReLU(),
            nn.Conv1d(out_chan, out_chan, 3, padding=1, groups=out_chan),
            nn.Conv1d(out_chan, out_chan, 1),
        )

    def forward(self, x):
        """
        Args:
            x: [B, in_channels, T]
        Returns:
            masks: [B, n_sources, out_channels, T]
                   sources = [coi_1, coi_2, ..., coi_n, background]
        """
        # process each COI class branch
        coi_masks = [torch.relu(branch(x)) for branch in self.coi_branches]

        bg_mask = torch.relu(self.background_branch(x))

        # Stack: COI classes first, then background
        masks = torch.stack(coi_masks + [bg_mask], dim=1)

        return masks
