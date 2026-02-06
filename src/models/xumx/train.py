"""
Training script for xumx-sliCQ model adapted for COI (Class of Interest) separation.

This script trains the xumx-sliCQ model for single-target + residue separation,
using the shared COI training utilities from src/common/coi_training.py.

Expected dataframe structure:
    - 'filename': path to wav file
    - 'split': train/val/test
    - 'label': 1 for COI, 0 for background (non-COI)
"""

import argparse
import copy
import gc
import json
import random
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tqdm

# Add parent directories to path for imports
_src_root = Path(__file__).parent.parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

# Import shared COI training utilities
from common.coi_training import (
    COIWeightedLoss,
    check_finite,
    create_coi_dataloader,
    prepare_batch,
    sisnr,
)

# Import xumx components
from models.xumx.base.xumx_slicq_v2 import model as xumx_model
from models.xumx.base.xumx_slicq_v2 import transforms

# =============================================================================
# Model Modifications for 2-source separation
# =============================================================================


class COISlicedUnmixCDAE(nn.Module):
    """Modified CDAE for 2-source separation (COI + background)."""

    def __init__(
        self,
        slicq_sample_input,
        hidden_size_1: int = 50,
        hidden_size_2: int = 51,
        freq_filter_small: int = 1,
        freq_filter_medium: int = 3,
        freq_filter_large: int = 5,
        freq_thresh_small: int = 10,
        freq_thresh_medium: int = 20,
        time_filter_2: int = 4,
        realtime: bool = False,
        input_mean=None,
        input_scale=None,
        num_sources: int = 2,
    ):
        super().__init__()

        self.num_sources = num_sources

        (
            nb_samples,
            nb_channels,
            nb_f_bins,
            nb_slices,
            nb_t_bins,
        ) = slicq_sample_input.shape

        if nb_f_bins < freq_thresh_small:
            freq_filter = freq_filter_small
        elif nb_f_bins < freq_thresh_medium:
            freq_filter = freq_filter_medium
        else:
            freq_filter = freq_filter_large

        window = nb_t_bins
        hop = window // 2

        if realtime:
            first_conv_module = xumx_model._CausalConv2d
        else:
            first_conv_module = nn.Conv2d

        # Create CDAEs for each source (COI + background)
        self.cdaes = nn.ModuleList()
        for _ in range(num_sources):
            encoder = [
                first_conv_module(
                    nb_channels,
                    hidden_size_1,
                    (freq_filter, window),
                    stride=(1, hop),
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_size_1),
                nn.ReLU(),
                nn.Conv2d(
                    hidden_size_1,
                    hidden_size_2,
                    (freq_filter, time_filter_2),
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_size_2),
                nn.ReLU(),
            ]

            decoder = [
                nn.ConvTranspose2d(
                    hidden_size_2,
                    hidden_size_1,
                    (freq_filter, time_filter_2),
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_size_1),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    hidden_size_1,
                    nb_channels,
                    (freq_filter, window),
                    stride=(1, hop),
                    bias=True,
                ),
                nn.Sigmoid(),
            ]

            self.cdaes.append(nn.Sequential(*encoder, *decoder))

        self.mask = True
        self.realtime = realtime
        self.nb_f_bins = nb_f_bins
        self.nb_slices = nb_slices
        self.nb_t_bins = nb_t_bins

        if input_mean is not None:
            input_mean = torch.from_numpy(-input_mean).float()
        else:
            input_mean = torch.zeros(nb_f_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(1.0 / input_scale).float()
        else:
            input_scale = torch.ones(nb_f_bins)

        self.input_mean = nn.Parameter(input_mean)
        self.input_scale = nn.Parameter(input_scale)

    def forward(self, xcomplex, x, skip_phase_reconstruction: bool = False):
        """Forward pass.

        Args:
            xcomplex: Complex NSGT coefficients (B, C, F, S, T, 2)
            x: Magnitude of NSGT coefficients (B, C, F, S, T)
            skip_phase_reconstruction: If True, skip Wiener filtering and return
                magnitude estimates with simple phase mixing. Useful for training.

        Returns:
            ret: Separated source estimates
            ret_masks: Soft masks for each source
        """
        mix = x.detach().clone()
        x_shape = x.shape
        nb_samples, nb_channels, nb_f_bins, nb_slices, nb_t_bins = x_shape

        ret = torch.zeros(
            (self.num_sources, *x_shape),
            device=x.device,
            dtype=x.dtype,
        )
        ret_masks = torch.zeros(
            (self.num_sources, *x_shape),
            device=x.device,
            dtype=x.dtype,
        )

        x = torch.flatten(x, start_dim=-2, end_dim=-1)

        # Normalize input
        x = x.permute(0, 1, 3, 2)
        x = x + self.input_mean
        x = x * self.input_scale
        x = x.permute(0, 1, 3, 2)

        for i, cdae in enumerate(self.cdaes):
            x_tmp = x.clone()
            for layer in cdae:
                x_tmp = layer(x_tmp)

            # Crop if necessary
            x_tmp = x_tmp[..., :nb_f_bins, : nb_slices * nb_t_bins]
            x_tmp = x_tmp.reshape(x_shape)

            ret_masks[i] = x_tmp.clone()

            if self.mask:
                x_tmp = x_tmp * mix

            ret[i] = x_tmp

        # Phase reconstruction - use simple phase mixing for training
        # The original wiener filtering is hardcoded for 4 sources
        if skip_phase_reconstruction:
            # Return magnitude estimates directly - caller will handle phase
            # Apply mixture phase to magnitude estimates
            ret = self._apply_mixture_phase(xcomplex, ret)
        else:
            # Use simple phasemix for any number of sources
            ret = self._apply_mixture_phase(xcomplex, ret)

        return ret, ret_masks

    def _apply_mixture_phase(self, xcomplex, ymag):
        """Apply mixture phase to magnitude estimates.

        Args:
            xcomplex: Complex mixture (B, C, F, S, T, 2) - real/imag stacked
            ymag: Magnitude estimates (num_sources, B, C, F, S, T)

        Returns:
            Complex estimates (num_sources, B, C, F, S, T, 2)
        """
        # Compute phase from mixture
        phase = torch.atan2(xcomplex[..., 1], xcomplex[..., 0])

        # Apply phase to each source magnitude
        ycomplex = torch.zeros((*ymag.shape, 2), device=ymag.device, dtype=ymag.dtype)
        ycomplex[..., 0] = ymag * torch.cos(phase)
        ycomplex[..., 1] = ymag * torch.sin(phase)

        return ycomplex


class COIUnmix(nn.Module):
    """Unmix model for 2-source COI separation."""

    def __init__(
        self,
        jagged_slicq_sample_input,
        realtime: bool = False,
        input_means=None,
        input_scales=None,
        num_sources: int = 2,
    ):
        super().__init__()

        self.num_sources = num_sources
        self.sliced_umx = nn.ModuleList()

        for i, C_block in enumerate(jagged_slicq_sample_input):
            input_mean = input_means[i] if input_means else None
            input_scale = input_scales[i] if input_scales else None

            self.sliced_umx.append(
                COISlicedUnmixCDAE(
                    C_block,
                    realtime=realtime,
                    input_mean=input_mean,
                    input_scale=input_scale,
                    num_sources=num_sources,
                )
            )

    def forward(
        self, Xcomplex, return_masks=False, skip_phase_reconstruction: bool = False
    ):
        """Forward pass through all sliCQ blocks.

        Args:
            Xcomplex: List of complex NSGT blocks
            return_masks: Whether to return soft masks
            skip_phase_reconstruction: Skip expensive phase reconstruction

        Returns:
            Ycomplex: List of separated source estimates per block
            Ymasks: (optional) List of soft masks per block
        """
        from models.xumx.base.xumx_slicq_v2.phase import abs_of_real_complex

        Ycomplex = [None] * len(Xcomplex)
        Ymasks = [None] * len(Xcomplex)

        for i, Xblock in enumerate(Xcomplex):
            Ycomplex_block, Ymask_block = self.sliced_umx[i](
                Xblock,
                abs_of_real_complex(Xblock),
                skip_phase_reconstruction=skip_phase_reconstruction,
            )
            Ycomplex[i] = Ycomplex_block
            Ymasks[i] = Ymask_block

        if return_masks:
            return Ycomplex, Ymasks
        return Ycomplex


# =============================================================================
# Statistics Calculation
# =============================================================================


def get_statistics(encoder, dataset, time_blocks, max_samples=100, quiet=False):
    """Calculate dataset statistics for input normalization."""
    import sklearn.preprocessing

    nsgt, _, cnorm = encoder

    nsgt_cpu = copy.deepcopy(nsgt).to("cpu")
    cnorm_cpu = copy.deepcopy(cnorm).to("cpu")

    scalers = [sklearn.preprocessing.StandardScaler() for _ in range(time_blocks)]

    pbar = tqdm.tqdm(range(min(len(dataset), max_samples)), disable=quiet)
    for ind in pbar:
        mixture, _ = dataset[ind]
        pbar.set_description("Computing dataset statistics")

        # Get NSGT of mixture
        X = cnorm_cpu(nsgt_cpu(mixture[None, ...]))

        for i, X_block in enumerate(X):
            X_block_flat = np.squeeze(
                torch.flatten(X_block, start_dim=-2, end_dim=-1)
                .mean(1, keepdim=False)
                .permute(0, 2, 1)
                .numpy(),
                axis=0,
            )
            scalers[i].partial_fit(X_block_flat)

    std = [
        np.maximum(scaler.scale_, 1e-4 * np.max(scaler.scale_)) for scaler in scalers
    ]
    return [scaler.mean_ for scaler in scalers], std


# =============================================================================
# Training Loop
# =============================================================================


def train_epoch(
    model,
    encoder,
    dataloader,
    optimizer,
    criterion,
    device,
    snr_range: tuple = (-5, 5),
    use_amp: bool = True,
    quiet: bool = False,
):
    """Train for one epoch."""
    nsgt, insgt, cnorm = encoder

    model.train()
    running_loss = 0.0
    n_samples = 0

    amp_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_amp and device.type == "cuda"
        else nullcontext()
    )

    pbar = tqdm.tqdm(dataloader, desc="Training", leave=False, disable=quiet)
    for mixture, sources in pbar:
        # Move to device
        sources = sources.to(device)

        # Prepare batch with SNR mixing
        mixture, clean_wavs = prepare_batch(sources, snr_range, deterministic=False)

        if not check_finite(mixture, clean_wavs):
            continue

        with amp_ctx:
            # Transform to sliCQ domain
            Xcomplex = nsgt(mixture)

            # Forward pass - skip phase reconstruction during training for efficiency
            Ycomplex_ests, Ymasks = model(
                Xcomplex, return_masks=True, skip_phase_reconstruction=True
            )

            # Compute loss in frequency domain (on magnitude)
            loss = torch.tensor(0.0, device=device)

            # Transform targets and compute MSE loss on complex coefficients
            for src_idx in range(clean_wavs.shape[1]):
                Ycomplex_target = nsgt(clean_wavs[:, src_idx, :, :])
                for block_idx in range(len(Ycomplex_ests)):
                    est_block = Ycomplex_ests[block_idx][src_idx]
                    tgt_block = Ycomplex_target[block_idx]
                    # Both are (B, C, F, S, T, 2) - compute MSE on complex values
                    loss = loss + torch.nn.functional.mse_loss(est_block, tgt_block)

            # Mask regularization (masks should sum to ~1)
            for block_idx in range(len(Ymasks)):
                mask_sum = Ymasks[block_idx].sum(dim=0)
                loss = loss + 0.1 * torch.nn.functional.mse_loss(
                    mask_sum, torch.ones_like(mask_sum)
                )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        B = mixture.shape[0]
        running_loss += float(loss.item()) * B
        n_samples += B

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(n_samples, 1)


@torch.no_grad()
def validate_epoch(
    model,
    encoder,
    dataloader,
    criterion,
    device,
    snr_range: tuple = (-5, 5),
    use_amp: bool = True,
    quiet: bool = False,
):
    """Validate for one epoch."""
    nsgt, insgt, cnorm = encoder

    model.eval()
    running_loss = 0.0
    n_samples = 0
    all_coi_sisnr = []
    all_bg_sisnr = []

    amp_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_amp and device.type == "cuda"
        else nullcontext()
    )

    pbar = tqdm.tqdm(dataloader, desc="Validation", leave=False, disable=quiet)
    for mixture, sources in pbar:
        sources = sources.to(device)
        mixture, clean_wavs = prepare_batch(sources, snr_range, deterministic=True)

        if not check_finite(mixture, clean_wavs):
            continue

        with amp_ctx:
            Xcomplex = nsgt(mixture)
            # Use phase reconstruction for validation to get better estimates
            Ycomplex_ests, _ = model(
                Xcomplex, return_masks=True, skip_phase_reconstruction=False
            )

            # Reconstruct time domain signals for SI-SNR evaluation
            y_ests = []
            for src_idx in range(clean_wavs.shape[1]):
                src_blocks = [
                    Ycomplex_ests[b][src_idx] for b in range(len(Ycomplex_ests))
                ]
                y_est = insgt(src_blocks, mixture.shape[-1])
                y_ests.append(y_est)
            y_ests = torch.stack(y_ests, dim=1)

            # Time domain loss using shared criterion
            loss = criterion(y_ests, clean_wavs)

            # Compute SI-SNR metrics
            # Average over channels for stereo
            y_ests_mono = y_ests.mean(dim=2) if y_ests.ndim == 4 else y_ests
            clean_mono = clean_wavs.mean(dim=2) if clean_wavs.ndim == 4 else clean_wavs

            coi_sisnr_val = sisnr(y_ests_mono[:, 0], clean_mono[:, 0]).mean()
            bg_sisnr_val = sisnr(y_ests_mono[:, -1], clean_mono[:, -1]).mean()

            all_coi_sisnr.append(float(coi_sisnr_val.item()))
            all_bg_sisnr.append(float(bg_sisnr_val.item()))

        B = mixture.shape[0]
        running_loss += float(loss.item()) * B
        n_samples += B

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            coi=f"{coi_sisnr_val.item():.2f}",
            bg=f"{bg_sisnr_val.item():.2f}",
        )

    if all_coi_sisnr:
        print(
            f"  Val SI-SNR - COI: {np.mean(all_coi_sisnr):.2f} dB, "
            f"BG: {np.mean(all_bg_sisnr):.2f} dB"
        )

    return running_loss / max(n_samples, 1)


# =============================================================================
# Main Training Function
# =============================================================================


def train(
    df_path: str,
    checkpoint_dir: str = "checkpoints",
    sample_rate: int = 44100,
    segment_length: float = 2.0,
    snr_range: tuple = (-5, 5),
    batch_size: int = 16,
    num_epochs: int = 500,
    lr: float = 0.001,
    patience: int = 100,
    device: str = "cuda",
    num_workers: int = 4,
    seed: int = 42,
    fscale: str = "bark",
    fbins: int = 262,
    fmin: float = 32.9,
    realtime: bool = False,
    use_amp: bool = True,
    quiet: bool = False,
    debug: bool = False,
):
    """Main training function."""
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = Path(checkpoint_dir) / timestamp
    checkpoint_path.mkdir(exist_ok=True, parents=True)

    # Device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataloaders using shared module
    print("Creating dataloaders...")
    train_loader, train_dataset = create_coi_dataloader(
        df_path=df_path,
        split="train",
        batch_size=batch_size,
        sample_rate=sample_rate,
        segment_length=segment_length,
        snr_range=snr_range,
        stereo=True,  # xumx uses stereo
        num_workers=num_workers,
        seed=seed,
    )

    # Setup NSGT transforms
    print("Setting up NSGT transforms...")
    nsgt_base = transforms.NSGTBase(
        fscale,
        fbins,
        fmin,
        fgamma=0.0,
        fs=sample_rate,
        device=device,
    )

    nsgt, insgt = transforms.make_filterbanks(nsgt_base, sample_rate=sample_rate)
    cnorm = transforms.ComplexNorm()

    nsgt = nsgt.to(device)
    insgt = insgt.to(device)
    cnorm = cnorm.to(device)

    encoder = (nsgt, insgt, cnorm)

    # Get sample input for model initialization
    jagged_slicq, _ = nsgt_base.predict_input_size(batch_size, 2, segment_length)
    jagged_slicq_cnorm = cnorm(jagged_slicq)
    n_blocks = len(jagged_slicq)

    # Calculate statistics
    if debug:
        scaler_mean = None
        scaler_std = None
    else:
        print("Computing dataset statistics...")
        scaler_mean, scaler_std = get_statistics(
            encoder, train_dataset, n_blocks, quiet=quiet
        )

    # Create model
    print("Creating model...")
    model = COIUnmix(
        jagged_slicq_cnorm,
        realtime=realtime,
        input_means=scaler_mean,
        input_scales=scaler_std,
        num_sources=2,  # COI + background
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Setup training
    criterion = COIWeightedLoss(class_weight=1.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.3, patience=30, cooldown=10
    )

    # Training loop
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss = train_epoch(
            model,
            encoder,
            train_loader,
            optimizer,
            criterion,
            device,
            snr_range=snr_range,
            use_amp=use_amp,
            quiet=quiet,
        )
        history["train_loss"].append(train_loss)

        # Validation
        val_loader, _ = create_coi_dataloader(
            df_path=df_path,
            split="val",
            batch_size=1,
            sample_rate=sample_rate,
            segment_length=segment_length,
            snr_range=snr_range,
            stereo=True,
            num_workers=num_workers,
            seed=seed,
        )

        val_loss = validate_epoch(
            model,
            encoder,
            val_loader,
            criterion,
            device,
            snr_range=snr_range,
            use_amp=use_amp,
            quiet=quiet,
        )
        del val_loader
        gc.collect()

        history["val_loss"].append(val_loss)
        print(f"Train: {train_loss:.4f}, Val: {val_loss:.4f}")

        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "history": history,
                },
                checkpoint_path / "best_model.pt",
            )
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            epochs_without_improvement += 1

        # Checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_val_loss,
            },
            checkpoint_path / "checkpoint.pt",
        )

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping after {epoch} epochs")
            break

        gc.collect()
        torch.cuda.empty_cache()

    # Save history
    with open(checkpoint_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining completed! Best val loss: {best_val_loss:.4f}")
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description="Train xumx for COI separation")
    parser.add_argument(
        "--df-path", type=str, required=True, help="Path to dataset CSV"
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/xumx")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--segment-length", type=float, default=2.0)
    parser.add_argument("--snr-min", type=float, default=-5)
    parser.add_argument("--snr-max", type=float, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fscale", type=str, default="bark")
    parser.add_argument("--fbins", type=int, default=262)
    parser.add_argument("--fmin", type=float, default=32.9)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    train(
        df_path=args.df_path,
        checkpoint_dir=args.checkpoint_dir,
        sample_rate=args.sample_rate,
        segment_length=args.segment_length,
        snr_range=(args.snr_min, args.snr_max),
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        fscale=args.fscale,
        fbins=args.fbins,
        fmin=args.fmin,
        realtime=args.realtime,
        use_amp=not args.no_amp,
        quiet=args.quiet,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
