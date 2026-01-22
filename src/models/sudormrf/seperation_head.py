"""
Class of interest-specific separation head for use with original SuDoRM-RF models.

This module provides a separation head that guarantees consistent output head assignment:
    - Head 0 (COI_HEAD_INDEX): Always outputs the Class of Interest (e.g., airplane noise)
    - Head 1 (BACKGROUND_HEAD_INDEX): Always outputs background/non-COI audio

This fixed assignment is critical for downstream processing that relies on knowing
which output channel contains the separated airplane audio.

Usage:
    from base.sudo_rm_rf.dnn.models.improved_sudormrf import SuDORMRF
    from seperation_head import wrap_model_for_coi, COI_HEAD_INDEX, BACKGROUND_HEAD_INDEX

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
    model = wrap_model_for_coi(base_model)

    # After inference:
    estimated_sources = model(mixture)  # Shape: (B, 2, T)
    airplane_audio = estimated_sources[:, COI_HEAD_INDEX, :]      # Always airplane
    background_audio = estimated_sources[:, BACKGROUND_HEAD_INDEX, :]  # Always background
"""

from typing import Optional

import torch
import torch.nn as nn

from .base.sudo_rm_rf.dnn.models.groupcomm_sudormrf_v2 import (
    GroupCommSudoRmRf,
)
from .base.sudo_rm_rf.dnn.models.improved_sudormrf import (
    SuDORMRF,
    UConvBlock,
)

# =============================================================================
# HEAD INDEX CONSTANTS - Use these for consistent access to model outputs
# =============================================================================
COI_HEAD_INDEX = 0  # Class of Interest (airplane) is ALWAYS at index 0
BACKGROUND_HEAD_INDEX = 1  # Background audio is ALWAYS at index 1
NUM_SOURCES = 2  # Total number of output sources


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable convolution for efficient feature processing.

    Consists of:
    1. Depthwise conv: applies a single filter per input channel
    2. Pointwise conv: 1x1 conv to mix channel information

    This is more parameter-efficient than standard convolution while
    maintaining good representational capacity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class MaskEstimationBranch(nn.Module):
    """Single branch for estimating separation masks.

    This branch learns to produce masks specific to one source type
    (either COI or background). Using dedicated branches allows the
    network to learn source-specific features.

    Architecture options:
    - Simple mode (num_blocks=0): Lightweight with depthwise-separable conv
    - Enhanced mode (num_blocks>0): Uses UConvBlocks for multi-resolution analysis
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 0,
        upsampling_depth: int = 4,
    ):
        """
        Args:
            in_channels: Number of input channels (from bottleneck)
            out_channels: Number of output channels (enc_num_basis for masking)
            num_blocks: Number of UConvBlocks. 0 = simple mode, >0 = enhanced mode
            upsampling_depth: Upsampling depth for UConvBlocks (only used if num_blocks > 0)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks

        if num_blocks == 0:
            # Simple but effective architecture
            # Uses depthwise-separable conv for efficiency with PReLU activation
            self.branch = nn.Sequential(
                nn.PReLU(in_channels),
                DepthwiseSeparableConv1d(
                    in_channels, in_channels, kernel_size=3, padding=1
                ),
                nn.PReLU(in_channels),
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
            )
        else:
            # Enhanced architecture with UConvBlocks for multi-resolution processing
            layers: list[nn.Module] = [nn.PReLU(in_channels)]

            for _ in range(num_blocks):
                layers.append(
                    UConvBlock(
                        out_channels=in_channels,
                        in_channels=in_channels,
                        upsampling_depth=upsampling_depth,
                    )
                )

            # Final projection to output dimension
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))

            self.branch = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, in_channels, T)
        Returns:
            Mask logits (B, out_channels, T) - PRE-ACTIVATION
        """
        return self.branch(x)


class COISeparationHead(nn.Module):
    """Separation head for Class-of-Interest (COI) audio separation.

    This head replaces the original SuDoRM-RF mask network to provide
    dedicated processing branches for COI (airplane) and background audio.

    CRITICAL: Output ordering is FIXED and GUARANTEED:
        - Channel 0: COI (airplane) mask
        - Channel 1: Background mask

    This ordering is maintained through the forward pass and must NOT be
    changed, as downstream processing depends on it.

    The head outputs PRE-ACTIVATION masks. The base model's mask_nl_class
    (ReLU) will be applied after reshaping in the forward pass.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_src: int = NUM_SOURCES,
        num_conv_blocks: int = 0,
        upsampling_depth: int = 4,
    ):
        """
        Args:
            in_channels: Input channels from separation module (model.out_channels)
            out_channels: Output channels per source (model.enc_num_basis)
            n_src: Number of sources (must be 2 for COI + Background)
            num_conv_blocks: Number of UConvBlocks per branch
                            0 = simple lightweight architecture
                            >0 = enhanced architecture with multi-resolution processing
            upsampling_depth: Upsampling depth for UConvBlocks (if num_conv_blocks > 0)
        """
        super().__init__()

        if n_src != NUM_SOURCES:
            raise ValueError(
                f"COISeparationHead only supports n_src={NUM_SOURCES} (COI + Background). "
                f"Got n_src={n_src}. Use multi_class_seperation for more sources."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_src = n_src
        self.num_conv_blocks = num_conv_blocks
        self.upsampling_depth = upsampling_depth

        # Shared feature extraction before branching
        # This processes common features before source-specific branches
        self.shared_features = nn.Sequential(
            DepthwiseSeparableConv1d(
                in_channels, in_channels, kernel_size=3, padding=1
            ),
            nn.PReLU(in_channels),
        )

        # COI-specific branch (airplane noise) - ALWAYS outputs at HEAD 0
        self.coi_branch = MaskEstimationBranch(
            in_channels=in_channels,
            out_channels=out_channels,
            num_blocks=num_conv_blocks,
            upsampling_depth=upsampling_depth,
        )

        # Background-specific branch - ALWAYS outputs at HEAD 1
        self.background_branch = MaskEstimationBranch(
            in_channels=in_channels,
            out_channels=out_channels,
            num_blocks=num_conv_blocks,
            upsampling_depth=upsampling_depth,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with GUARANTEED output ordering.

        Args:
            x: Bottleneck features from separation module (B, in_channels, T)
               Note: The PReLU from mask_net Sequential is applied before this

        Returns:
            masks: Concatenated PRE-ACTIVATION masks (B, n_src * out_channels, T)
                   Layout: [COI_mask, Background_mask] along channel dimension

                   After reshape in base model: (B, n_src, out_channels, T)
                   Where:
                       masks[:, COI_HEAD_INDEX, :, :] = COI (airplane) mask
                       masks[:, BACKGROUND_HEAD_INDEX, :, :] = Background mask
        """
        # Extract shared features
        shared = self.shared_features(x)

        # Generate source-specific masks
        # IMPORTANT: COI must come FIRST to ensure consistent head assignment
        coi_mask = self.coi_branch(shared)  # (B, out_channels, T)
        background_mask = self.background_branch(shared)  # (B, out_channels, T)

        # Concatenate masks: COI FIRST (index 0), then Background (index 1)
        # This ordering is CRITICAL and must not be changed!
        # Shape: (B, 2 * out_channels, T) = (B, n_src * enc_num_basis, T)
        masks = torch.cat([coi_mask, background_mask], dim=1)

        return masks

    def extra_repr(self) -> str:
        """String representation for printing model architecture."""
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"n_src={self.n_src}, "
            f"num_conv_blocks={self.num_conv_blocks}, "
            f"upsampling_depth={self.upsampling_depth}"
        )


def wrap_model_for_coi(
    model: nn.Module,
    replace_head: bool = True,
    num_conv_blocks: int = 0,
    upsampling_depth: Optional[int] = None,
) -> nn.Module:
    """Wrap a SuDoRM-RF model with a COI-specific separation head.

    This function modifies the model's mask_net to use our COISeparationHead,
    which guarantees consistent output ordering:
        - output[:, 0, :] = COI (airplane) audio
        - output[:, 1, :] = Background audio

    Args:
        model: SuDoRM-RF model instance (SuDORMRF or GroupCommSudoRmRf)
        replace_head: If True, replaces the existing separation head
        num_conv_blocks: Number of UConvBlocks per branch for feature extraction
                        0 = simple architecture (default, lightweight)
                        >0 = enhanced architecture with UConvBlocks
        upsampling_depth: Upsampling depth for UConvBlocks
                         If None, uses the model's upsampling_depth

    Returns:
        model: Modified SuDoRM-RF model with COI-specific head

    Raises:
        TypeError: If model type is not supported

    Example:
        >>> base_model = SuDORMRF(...)
        >>> model = wrap_model_for_coi(base_model, num_conv_blocks=2)
        >>> outputs = model(mixture)
        >>> airplane = outputs[:, COI_HEAD_INDEX, :]
        >>> background = outputs[:, BACKGROUND_HEAD_INDEX, :]
    """
    if not replace_head:
        return model

    # Validate model type
    if not isinstance(model, (SuDORMRF, GroupCommSudoRmRf)):
        raise TypeError(
            f"Model type {type(model).__name__} not supported for COI head replacement. "
            f"Expected SuDORMRF or GroupCommSudoRmRf."
        )

    # Extract model parameters
    in_channels = model.out_channels  # Output of bottleneck/separation module
    out_channels = model.enc_num_basis  # Must match encoder basis for masking

    # Use model's upsampling_depth if not specified
    if upsampling_depth is None:
        upsampling_depth = model.upsampling_depth

    # Replace mask_net with our COI-specific head
    # Structure: PReLU -> COISeparationHead
    # The PReLU is kept from the original architecture for compatibility
    model.mask_net = nn.Sequential(
        nn.PReLU(),
        COISeparationHead(
            in_channels=in_channels,
            out_channels=out_channels,
            n_src=NUM_SOURCES,
            num_conv_blocks=num_conv_blocks,
            upsampling_depth=upsampling_depth,
        ),
    )

    # Update model's num_sources to match our head
    model.num_sources = NUM_SOURCES

    # Note: The decoder is NOT modified. It uses groups=1 which is correct
    # for the SuDoRM-RF architecture. The mask multiplication happens
    # BEFORE the decoder in the forward pass.

    return model


def get_head_indices():
    """Get the head indices for COI and background.

    Use this function to get indices programmatically rather than
    hardcoding values, ensuring consistency across the codebase.

    Returns:
        tuple: (coi_index, background_index)

    Example:
        >>> coi_idx, bg_idx = get_head_indices()
        >>> airplane = model_output[:, coi_idx, :]
        >>> background = model_output[:, bg_idx, :]
    """
    return COI_HEAD_INDEX, BACKGROUND_HEAD_INDEX


def verify_head_assignment(model_output: torch.Tensor) -> bool:
    """Verify that the model output has the expected shape for head assignment.

    This is a utility function for debugging and validation.

    Args:
        model_output: Output tensor from the wrapped model

    Returns:
        bool: True if the output has the expected shape (B, NUM_SOURCES, T)

    Example:
        >>> output = model(mixture)
        >>> assert verify_head_assignment(output), "Unexpected output shape!"
    """
    if model_output.ndim != 3:
        return False
    if model_output.shape[1] != NUM_SOURCES:
        return False
    return True
