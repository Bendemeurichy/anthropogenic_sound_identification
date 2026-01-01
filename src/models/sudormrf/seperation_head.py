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
from .base.sudo_rm_rf.dnn.models.improved_sudormrf import (
    SuDORMRF,
    UConvBlock,
)


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

    def __init__(
        self,
        in_channels,
        out_channels,
        n_src=2,
        num_conv_blocks=0,
        upsampling_depth=4,
        *args,
        **kwargs,
    ):
        """
        Args:
            in_channels: Number of input channels from bottleneck (out_channels from model)
            out_channels: Number of output channels (enc_num_basis)
            n_src: Number of sources (must be 2)
            num_conv_blocks: Number of UConvBlocks per branch for feature extraction.
                           If 0 (default), uses simple PReLU + Conv1d structure.
                           If > 0, uses UConvBlocks for class-specific feature extraction.
            upsampling_depth: Upsampling depth for UConvBlocks (default 4)
        """
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_src = n_src
        self.num_conv_blocks = num_conv_blocks
        self.upsampling_depth = upsampling_depth

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

        if num_conv_blocks == 0:
            # Original simple architecture
            # COI-specific branch - output is PRE-ReLU (ReLU applied by mask_nl_class)
            # Using linear output so mask_nl_class can apply ReLU properly
            self.coi_branch = nn.Sequential(
                nn.PReLU(in_channels),
                nn.Conv1d(
                    in_channels, out_channels, 1
                ),  # Linear projection to mask dim
            )

            # Background-specific branch - same structure
            self.background_branch = nn.Sequential(
                nn.PReLU(in_channels),
                nn.Conv1d(
                    in_channels, out_channels, 1
                ),  # Linear projection to mask dim
            )
        else:
            # Enhanced architecture with multiple UConvBlocks for feature extraction
            self.coi_branch = self._build_uconv_branch(
                in_channels, out_channels, num_conv_blocks
            )
            self.background_branch = self._build_uconv_branch(
                in_channels, out_channels, num_conv_blocks
            )

    def _build_uconv_branch(self, in_channels, out_channels, num_blocks):
        """Build a branch with multiple UConvBlocks for class-specific feature extraction.

        Uses UConvBlock from the base SuDO-RM-RF model, which performs successive
        downsampling and upsampling to analyze features at multiple resolutions.

        Args:
            in_channels: Input channels (out_channels from model)
            out_channels: Output channels (enc_num_basis)
            num_blocks: Number of UConvBlocks
        Returns:
            nn.Sequential: Complete branch
        """
        layers = []

        # Initial activation
        layers.append(nn.PReLU(in_channels))

        # Add UConvBlocks - same structure as base model's separation module
        # Note: UConvBlock expects out_channels as first param, in_channels as second
        for i in range(num_blocks):
            layers.append(
                UConvBlock(
                    out_channels=in_channels,  # Keep same channels throughout
                    in_channels=in_channels,  # Internal processing channels
                    upsampling_depth=self.upsampling_depth,
                )
            )

        # Final projection to output dimension (no activation - will be applied by mask_nl_class)
        layers.append(nn.Conv1d(in_channels, out_channels, 1))

        return nn.Sequential(*layers)

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


def wrap_model_for_coi(
    model, replace_head=True, num_conv_blocks=0, upsampling_depth=None
):
    """Wraps a SuDoRM-RF model with a COI-specific separation head.

    Args:
        model: SuDoRM-RF model instance
        replace_head: If True, replaces the existing separation head.
        num_conv_blocks: Number of UConvBlocks per branch for feature extraction.
                        0 = simple architecture (default), >0 = enhanced architecture with UConvBlocks
        upsampling_depth: Upsampling depth for UConvBlocks (default: use model's upsampling_depth)
    Returns:
        model: Modified SuDoRM-RF model with COI-specific head
    """

    if replace_head:
        if isinstance(model, SuDORMRF) or isinstance(model, GroupCommSudoRmRf):
            in_channels = model.out_channels  # output of bottleneck/separation module
            out_channels = model.enc_num_basis  # must match encoder basis for masking
            n_src = 2  # COI and background

            # Use model's upsampling_depth if not specified
            if upsampling_depth is None:
                upsampling_depth = model.upsampling_depth

            # Replace mask_net with our COI-specific head
            model.mask_net = nn.Sequential(
                nn.PReLU(),  # Keep the PReLU that was in the original mask_net
                COISeparationHead(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_src=n_src,
                    num_conv_blocks=num_conv_blocks,
                    upsampling_depth=upsampling_depth,
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
