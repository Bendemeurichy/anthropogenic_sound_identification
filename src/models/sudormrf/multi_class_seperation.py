"""
Multi-class Class-of-Interest (COI) separation head for SuDoRM-RF.

Supports multiple architectural strategies:
1. Shared background + per-class branches (memory efficient)
2. Fully independent branches (max expressiveness)
3. Hierarchical with shared low-level features (balanced)
"""

import torch
import torch.nn as nn
from base.sudo_rm_rf.dnn.models.groupcomm_sudormrf_v2 import (
    GroupCommSudoRmRf,
)
from base.sudo_rm_rf.dnn.models.improved_sudormrf import SuDORMRF
from seperation_head import MaskEstimationBranch


class SharedBackgroundMultiCOI(nn.Module):
    """
    Multi-class COI separation with SHARED background branch.
    For scaling to multiple COI types (e.g. different aircraft classes, birds, etc.)

    IMPORTANT: Outputs PRE-ACTIVATION masks. The base model's mask_nl_class
    (ReLU) will be applied after the view reshape in the forward pass.

    Source ordering convention (must be consistent with AudioDataset and
    COIWeightedLoss):
        [coi_class_0, coi_class_1, ..., coi_class_{N-1}, background]
    Background is always the LAST channel.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_coi_classes=3,
        num_conv_blocks=0,
        upsampling_depth=4,
        expanded_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_coi_classes = n_coi_classes
        self.n_sources = n_coi_classes + 1  # COI classes + background

        # Shared feature extraction
        self.shared_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv1d(in_channels, in_channels, 1),
            nn.PReLU(in_channels),
        )

        # One branch per COI class, using MaskEstimationBranch so num_conv_blocks
        # and expanded_channels are respected just like the single-class path.
        self.coi_branches = nn.ModuleList(
            [
                MaskEstimationBranch(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    num_blocks=num_conv_blocks,
                    upsampling_depth=upsampling_depth,
                    expanded_channels=expanded_channels,
                )
                for _ in range(n_coi_classes)
            ]
        )

        # Single shared background branch
        self.background_branch = MaskEstimationBranch(
            in_channels=in_channels,
            out_channels=out_channels,
            num_blocks=num_conv_blocks,
            upsampling_depth=upsampling_depth,
            expanded_channels=expanded_channels,
        )

    def forward(self, x):
        """
        Args:
            x: [B, in_channels, T]
        Returns:
            masks: [B, n_sources * out_channels, T] - PRE-ACTIVATION masks
                   sources = [coi_0, coi_1, ..., coi_{N-1}, background]
        """
        shared = self.shared_conv(x)

        # Process each COI class branch (no activation - let base model handle it)
        coi_masks = [branch(shared) for branch in self.coi_branches]
        bg_mask = self.background_branch(shared)

        # Concatenate: COI classes first, then background (LAST)
        # Shape: (B, n_sources * out_channels, T)
        masks = torch.cat(coi_masks + [bg_mask], dim=1)

        return masks


def wrap_model_for_multiclass(
    model,
    replace_head=True,
    n_coi_classes=3,
    num_conv_blocks=0,
    upsampling_depth=None,
    expanded_channels=None,
):
    """Wraps a SuDoRM-RF model with a Multi-class COI separation head.

    Fixes applied vs. the original implementation:
    - Decoder is rebuilt to match the new n_src (Bug 1 fix).
    - MaskEstimationBranch is used so num_conv_blocks / expanded_channels are
      respected consistently with the single-class path (Bug 5 fix).

    Args:
        model: SuDoRM-RF model instance (SuDORMRF or GroupCommSudoRmRf)
        replace_head: If True, replaces the existing separation head.
        n_coi_classes: Number of classes of interest (background is +1).
        num_conv_blocks: UConvBlocks per branch (0 = simple lightweight head).
        upsampling_depth: Depth for UConvBlocks; defaults to model.upsampling_depth.
        expanded_channels: Expanded dim for UConvBlocks; defaults to model.in_channels.
    Returns:
        model: Modified SuDoRM-RF model with Multi-class COI head.
    """
    if not replace_head:
        return model

    # Validate model type by class name (handles pickled checkpoints from
    # different module paths gracefully).
    valid_class_names = {"SuDORMRF", "GroupCommSudoRmRf"}
    if type(model).__name__ not in valid_class_names:
        raise TypeError(
            f"Model type {type(model).__name__} not supported for Multi-class COI head "
            f"replacement. Expected one of: {valid_class_names}."
        )

    in_channels = model.out_channels   # output of bottleneck/separation module
    out_channels = model.enc_num_basis  # must match encoder basis for masking
    n_src = n_coi_classes + 1

    if upsampling_depth is None:
        upsampling_depth = model.upsampling_depth
    if expanded_channels is None:
        expanded_channels = model.in_channels  # typically 512

    # ------------------------------------------------------------------
    # Replace mask_net
    # ------------------------------------------------------------------
    model.mask_net = nn.Sequential(
        nn.PReLU(),
        SharedBackgroundMultiCOI(
            in_channels=in_channels,
            out_channels=out_channels,
            n_coi_classes=n_coi_classes,
            num_conv_blocks=num_conv_blocks,
            upsampling_depth=upsampling_depth,
            expanded_channels=expanded_channels,
        ),
    )

    # Update the number of sources so the base model's forward() reshapes
    # x correctly: x.view(B, num_sources, enc_num_basis, T)
    model.num_sources = n_src

    # ------------------------------------------------------------------
    # Replace decoder (Bug 1 fix)
    #
    # The decoder's in_channels must equal n_src * enc_num_basis (times
    # in_audio_channels for GroupCommSudoRmRf).  The original decoder was
    # created with num_sources=2; leaving it unchanged causes a shape
    # mismatch at runtime for any n_coi_classes != 1.
    # ------------------------------------------------------------------
    in_audio_channels = getattr(model, "in_audio_channels", 1)
    model.decoder = nn.ConvTranspose1d(
        in_channels=out_channels * n_src * in_audio_channels,
        out_channels=n_src * in_audio_channels,
        output_padding=(model.enc_kernel_size // 2) - 1,
        kernel_size=model.enc_kernel_size,
        stride=model.enc_kernel_size // 2,
        padding=model.enc_kernel_size // 2,
        groups=1,
        bias=False,
    )
    torch.nn.init.xavier_uniform_(model.decoder.weight)

    return model
