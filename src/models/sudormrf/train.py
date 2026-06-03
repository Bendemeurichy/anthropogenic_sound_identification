"""
Training script for sudormrf model with custom separation head and loss function.
Expected dataframe structure:
    - 'filename': path to wav file
    - 'split': train/val/test
    - 'label': 1 for aircraft (COI), 0 for background (non-COI)
"""

import argparse
import gc
import io
import json
import math
import os
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

# Under pythonw there is no console and sys.stdout/stderr are None.
# Wrap only when the underlying buffer actually exists.
if sys.stdout is not None and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", line_buffering=True
    )
if sys.stderr is not None and hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", line_buffering=True
    )

# Pin to single GPU before importing torch (prevents multi-GPU OOM issues)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader

from common.training_utils import (
    _ensure_autoflush,
    _is_tty,
    _redirect_to_log,
    progress_bar,
    set_seed,
)


# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))  # src dir


from base.sudo_rm_rf.dnn.models.groupcomm_sudormrf_v2 import GroupCommSudoRmRf
from base.sudo_rm_rf.dnn.models.improved_sudormrf import SuDORMRF

from seperation_head import (
    BACKGROUND_HEAD_INDEX,
    COI_HEAD_INDEX,
    wrap_model_for_coi,
)

BACKGROUND_HEAD_INDEX: int = BACKGROUND_HEAD_INDEX
COI_HEAD_INDEX: int = COI_HEAD_INDEX


from config import Config
from multi_class_seperation import wrap_model_for_multiclass

from label_loading.metadata_loader import (
    load_metadata_datasets,
    split_seperation_classification,
)
from label_loading.sampler import get_coi, sample_non_coi

from .losses import COIWeightedLoss, SILENCE_ENERGY_EPS, WEAK_TARGET_ENERGY_EPS, sisnr
from .utils import BG_SCALE_MAX, BG_SCALE_MIN, ENERGY_EPS, NORMALIZE_MIN_STD, check_finite, normalize_tensor_wav, prepare_batch

from .dataset import AudioDataset, _worker_init_fn

LOSS_EPS = 1e-8

# =============================================================================
# Training and Validation Loops
# =============================================================================





def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    clip_grad_norm: float = 5.0,
    grad_accum_steps: int = 1,
    use_amp: bool = True,
    snr_range: tuple[float, float] = (-5.0, 5.0),
    scaler: "torch.cuda.amp.GradScaler | None" = None,
    scheduler: optim.lr_scheduler._LRScheduler | None = None,
) -> tuple[float, int, list[float]]:
    """Train for one epoch."""
    model.train()
    running_loss, n_samples = 0.0, 0
    grad_norms: list[float] = []

    use_amp = use_amp and str(device).startswith("cuda")
    if scaler is None and use_amp:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.float16, enabled=True)
        if use_amp
        else nullcontext()
    )

    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    has_pending_grads = False

    pbar = progress_bar(dataloader, desc="Training")
    for step_idx, sources in enumerate(pbar, start=1):
        sources = sources.to(device, non_blocking=True)
        mixture, clean_wavs = prepare_batch(sources, snr_range, deterministic=False)
        B = sources.shape[0]
        del sources  # Free; clean_wavs has the processed copy

        if not check_finite(mixture, clean_wavs):
            del mixture, clean_wavs
            grad_norms.append(float("nan"))
            continue

        # Forward pass + loss inside autocast to avoid fp16->fp32 duplication
        with autocast_ctx:
            outputs = model(mixture.unsqueeze(1))

            if not check_finite(outputs):
                del outputs  # Free old computation graph before retry
                if use_amp:
                    torch.cuda.empty_cache()
                    with torch.amp.autocast("cuda", enabled=False):
                        outputs = model(mixture.unsqueeze(1).float())
                    if not check_finite(outputs):
                        del outputs, mixture, clean_wavs
                        grad_norms.append(float("nan"))
                        torch.cuda.empty_cache()
                        continue
                else:
                    # No retry available without AMP; skip this batch
                    del mixture, clean_wavs
                    grad_norms.append(float("nan"))
                    continue

            loss = criterion(outputs.float(), clean_wavs.float())

        del outputs  # Graph retained via loss; free the direct reference
        del mixture, clean_wavs

        if not check_finite(loss):
            del loss
            grad_norms.append(float("nan"))
            continue

        # Backward
        loss_scaled = loss / grad_accum_steps
        if use_amp:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        # Extract scalar before freeing loss tensors
        batch_loss = float(loss.item())
        del loss, loss_scaled

        # Optimizer step
        if step_idx % grad_accum_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)

            # Check gradients
            grads_ok = all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )

            if grads_ok:
                optimizer_step += 1

                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_grad_norm
                )
                grad_norms.append(float(total_norm.item()))

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()
            else:
                grad_norms.append(float("nan"))
                if use_amp:
                    try:
                        scaler.update()
                    except Exception:
                        pass

            optimizer.zero_grad(set_to_none=True)
            has_pending_grads = False
        else:
            has_pending_grads = True

        # Metrics
        running_loss += batch_loss * B
        n_samples += B

        pbar.set_postfix(
            loss=f"{batch_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}"
        )

        # Periodically release cached CUDA memory to prevent fragmentation buildup
        if use_amp and step_idx % 50 == 0:
            torch.cuda.empty_cache()

    # Flush any remaining accumulated gradients from the last partial cycle
    if has_pending_grads:
        if use_amp:
            scaler.unscale_(optimizer)

        grads_ok = all(
            torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None
        )

        if grads_ok:
            optimizer_step += 1

            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_grad_norm
            )
            grad_norms.append(float(total_norm.item()))

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if scheduler is not None:
                scheduler.step()
        else:
            grad_norms.append(float("nan"))
            if use_amp:
                try:
                    scaler.update()
                except Exception:
                    pass

        optimizer.zero_grad(set_to_none=True)

    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    # Print gradient stats
    valid_norms = [n for n in grad_norms if not np.isnan(n)]
    if valid_norms:
        print(
            f"  Gradient norms - avg: {np.mean(valid_norms):.4f}, max: {np.max(valid_norms):.4f}"
        )

    return running_loss / max(n_samples, 1), optimizer_step, grad_norms


def validate_epoch(
    model,
    dataloader,
    criterion,
    device,
    use_amp: bool = True,
    snr_range: tuple[float, float] = (-5.0, 5.0),
) -> float:
    """Validate for one epoch."""
    model.eval()
    running_loss, n_samples = 0.0, 0
    all_coi_present_sisnr: list[float] = []
    all_coi_absent_sisnr: list[float] = []
    all_bg_sisnr: list[float] = []

    use_amp = use_amp and str(device).startswith("cuda")
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.float16, enabled=True)
        if use_amp
        else nullcontext()
    )

    pbar = progress_bar(dataloader, desc="Validation")
    with torch.no_grad():
        for sources in pbar:
            sources = sources.to(device, non_blocking=True)
            mixture, clean_wavs = prepare_batch(sources, snr_range, deterministic=True)
            B = sources.shape[0]
            del sources

            if not check_finite(mixture, clean_wavs):
                del mixture, clean_wavs
                continue

            with autocast_ctx:
                outputs = model(mixture.unsqueeze(1))
                loss = criterion(outputs.float(), clean_wavs.float())

            batch_loss = float(loss.item())
            running_loss += batch_loss * B
            n_samples += B

            # SI-SNR metrics - separate COI-present from COI-absent samples
            try:
                n_src = outputs.shape[1]
                n_coi = n_src - 1

                if n_coi == 1:
                    # Single-class: unchanged.
                    coi_per_sample = sisnr(outputs[:, 0], clean_wavs[:, 0])  # (B,)
                    coi_target_energy = clean_wavs[:, 0].pow(2).mean(dim=-1)  # (B,)
                else:
                    # Multi-class: route each sample to its active COI head.
                    # (B, n_coi)
                    coi_energies = torch.stack(
                        [clean_wavs[:, i].pow(2).mean(dim=-1) for i in range(n_coi)],
                        dim=1,
                    )
                    active_mask = coi_energies > SILENCE_ENERGY_EPS  # (B, n_coi)
                    all_coi_sisnr = torch.stack(
                        [sisnr(outputs[:, i], clean_wavs[:, i]) for i in range(n_coi)],
                        dim=1,
                    )  # (B, n_coi)
                    active_float = active_mask.float()
                    active_count = active_float.sum(dim=1).clamp(min=1.0)
                    coi_per_sample = (
                        (all_coi_sisnr * active_float).sum(dim=1) / active_count
                    )  # (B,)
                    # Use max energy over COI heads so that any active class
                    # correctly trips the coi_present_mask.
                    coi_target_energy = coi_energies.max(dim=1).values  # (B,)

                coi_present_mask = coi_target_energy > SILENCE_ENERGY_EPS  # (B,)

                if coi_present_mask.any():
                    all_coi_present_sisnr.append(
                        float(coi_per_sample[coi_present_mask].mean().item())
                    )
                if (~coi_present_mask).any():
                    all_coi_absent_sisnr.append(
                        float(coi_per_sample[~coi_present_mask].mean().item())
                    )

                bg_sisnr = sisnr(outputs[:, -1], clean_wavs[:, -1]).mean()
                all_bg_sisnr.append(float(bg_sisnr.item()))

                # Progress bar shows the COI-present SI-SNR when available,
                # falling back to the full batch average otherwise
                coi_display = (
                    coi_per_sample[coi_present_mask].mean()
                    if coi_present_mask.any()
                    else coi_per_sample.mean()
                )
                pbar.set_postfix(
                    loss=f"{batch_loss:.4f}",
                    coi=f"{coi_display.item():.2f}",
                    bg=f"{bg_sisnr.item():.2f}",
                )
            except Exception:
                pbar.set_postfix(loss=f"{batch_loss:.4f}")

            del outputs, loss, mixture, clean_wavs
            # Explicitly free intermediate metric tensors so GPU memory is
            # released promptly rather than waiting for the next iteration to
            # rebind the variables.
            try:
                del coi_per_sample, coi_target_energy, coi_present_mask, bg_sisnr
            except NameError:
                pass  # metrics block was skipped via exception

    if all_bg_sisnr:
        coi_present_str = (
            f"{np.mean(all_coi_present_sisnr):.2f} dB"
            if all_coi_present_sisnr
            else "n/a"
        )
        coi_absent_str = (
            f"{np.mean(all_coi_absent_sisnr):.2f} dB" if all_coi_absent_sisnr else "n/a"
        )
        print(
            f"  Val SI-SNR - COI (present): {coi_present_str}, "
            f"COI (absent/suppression): {coi_absent_str}, "
            f"BG: {np.mean(all_bg_sisnr):.2f} dB"
        )

    return running_loss / max(n_samples, 1)


# =============================================================================
# Data Loading and Model Creation
# =============================================================================


def create_dataloader(config: Config, split: str) -> tuple[DataLoader, AudioDataset]:
    """Create dataloader for specified split.
    
    Supports both file-based and WebDataset loading modes based on config.
    """
    # Check if we should use WebDataset
    use_webdataset = getattr(config.data, "use_webdataset", False)
    raw_webdataset_path = str(getattr(config.data, "webdataset_path", "") or "")
    webdataset_path = os.path.expanduser(os.path.expandvars(raw_webdataset_path)).strip()
    
    if use_webdataset:
        if not webdataset_path:
            raise ValueError("webdataset_path must be set when use_webdataset=True")

        from src.common.webdataset_utils import COIWebDatasetWrapper
        from src.label_loading.metadata_loader import get_webdataset_paths

        # Persist the resolved path so checkpoints/logging reflect the actual HPC path.
        config.data.webdataset_path = webdataset_path

        print(f"Using WebDataset loading from: {webdataset_path}")

        # Prefer a concrete list of shard files over brace-expansion patterns.
        # This avoids silent empty datasets when environment-variable expansion or
        # brace expansion is not handled the same way across systems.
        _tar_pattern = get_webdataset_paths(webdataset_path, split)
        tar_paths = sorted(str(p) for p in Path(webdataset_path).glob(f"{split}-*.tar"))
        if not tar_paths:
            raise FileNotFoundError(
                f"No {split} shards found in resolved WebDataset directory: "
                f"{webdataset_path} (pattern from manifest: {_tar_pattern})"
            )
        print(f"  Resolved {len(tar_paths)} {split} shard(s)")

        # Get target_classes for filtering
        target_classes = getattr(config.data, "target_classes", [])
        if not target_classes:
            print("  Warning: No target_classes specified, using all label==1 as COI")
        else:
            print(f"  Filtering COI by labels: {target_classes}")

        # Get seed for reproducibility
        seed = getattr(config.training, "seed", 42)

        dataset = COIWebDatasetWrapper(
            tar_paths=tar_paths,
            split=split,
            target_sr=config.data.sample_rate,
            segment_length=config.data.segment_length,
            snr_range=tuple(config.data.snr_range),
            n_coi_classes=config.data.n_coi_classes,
            shuffle=(split == "train"),
            augment=(split == "train"),
            stereo=False,
            background_only_prob=(
                getattr(config.data, "background_only_prob", 0.0)
                if split == "train"
                else 0.0
            ),
            target_classes=target_classes,
            dataset_filter=None,  # Could be added to config if needed
            coi_ratio=0.25,  # Could be added to config if needed
            seed=seed,
        )

        num_workers = int(getattr(config.training, "num_workers", 0))
        pin_memory = (
            getattr(config.training, "pin_memory", False) and torch.cuda.is_available()
        )

        # WebDataset is an IterableDataset - different DataLoader settings
        loader = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            # No shuffle for IterableDataset - handled internally
            # No drop_last for IterableDataset
        )

        return loader, dataset
    
    # File-based mode (original behavior)
    # Read a minimal probe of the header to discover which optional columns
    # (start_time, end_time, duration, …) are actually present in the CSV so
    # that AudioDataset can use file-level bounds for segmentation.
    header_cols = pd.read_csv(config.data.df_path, nrows=0).columns.tolist()

    usecols = ["filename", "label", "split"]
    for optional_col in ("start_time", "end_time", "duration", "est_segments"):
        if optional_col in header_cols:
            usecols.append(optional_col)
    if getattr(config.data, "n_coi_classes", 1) > 1 and "coi_class" in header_cols:
        usecols.append("coi_class")

    df = pd.read_csv(config.data.df_path, usecols=usecols)
    df["label"] = df["label"].astype("uint8")
    df["split"] = df["split"].astype("category")
    if "coi_class" in df.columns:
        df["coi_class"] = df["coi_class"].astype("category")

    dataset = AudioDataset(
        df,
        split=split,
        sample_rate=config.data.sample_rate,
        segment_length=config.data.segment_length,
        snr_range=tuple(config.data.snr_range),
        n_coi_classes=config.data.n_coi_classes,
        augment=(split == "train"),
        background_only_prob=(
            getattr(config.data, "background_only_prob", 0.0)
            if split == "train"
            else 0.0
        ),
        background_mix_n=getattr(config.data, "background_mix_n", 2),
        augment_multiplier=getattr(config.data, "augment_multiplier", 1),
        seed=getattr(config.training, "seed", 42),
        multi_coi_prob=getattr(config.data, "multi_coi_prob", 0.0),
    )

    num_workers = int(getattr(config.training, "num_workers", 0))
    pin_memory = (
        getattr(config.training, "pin_memory", False) and torch.cuda.is_available()
    )

    prefetch_factor = (
        int(getattr(config.training, "prefetch_factor", 2)) if num_workers > 0 else None
    )

    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=(split == "train"),
        drop_last=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        prefetch_factor=prefetch_factor,
        worker_init_fn=_worker_init_fn if (num_workers > 0 and split == "train") else None,
    )

    del df
    gc.collect()
    return loader, dataset


def create_model(config: Config):
    """Create and configure model."""
    ModelClass = SuDORMRF if config.model.type == "improved" else GroupCommSudoRmRf
    base_model = ModelClass(
        out_channels=config.model.out_channels,
        in_channels=config.model.in_channels,
        num_blocks=config.model.num_blocks,
        upsampling_depth=config.model.upsampling_depth,
        enc_kernel_size=config.model.enc_kernel_size,
        enc_num_basis=config.model.enc_num_basis,
        num_sources=2,
    )

    # Add compatibility attribute
    if not hasattr(base_model, "n_least_samples_req"):
        base_model.n_least_samples_req = (config.model.enc_kernel_size // 2) * (
            2**config.model.upsampling_depth
        )

    # Wrap model
    if config.data.n_coi_classes > 1:
        print(
            f"Wrapping model for Multi-class separation with {config.data.n_coi_classes} classes "
            f"({config.model.num_head_conv_blocks} head blocks)"
        )
        expanded_channels = getattr(config.model, "expanded_channels", None)
        model = wrap_model_for_multiclass(
            base_model,
            n_coi_classes=config.data.n_coi_classes,
            num_conv_blocks=config.model.num_head_conv_blocks,
            upsampling_depth=config.model.upsampling_depth,
            expanded_channels=expanded_channels,
        )
    else:
        print(
            f"Wrapping model for Single COI separation ({config.model.num_head_conv_blocks} head blocks)"
        )
        # Use expanded_channels from config if available, otherwise default to None (uses model.in_channels)
        expanded_channels = getattr(config.model, "expanded_channels", None)
        model = wrap_model_for_coi(
            base_model,
            num_conv_blocks=config.model.num_head_conv_blocks,
            upsampling_depth=config.model.upsampling_depth,
            expanded_channels=expanded_channels,
        )

    if not hasattr(model, "n_least_samples_req"):
        model.n_least_samples_req = base_model.n_least_samples_req

    # Move to device
    try:
        model = model.to(config.training.device)
    except Exception as e:
        print(f"Error moving to {config.training.device}: {e}. Using CPU.")
        model = model.to("cpu")

    # Compile if requested
    if hasattr(torch, "compile") and getattr(config.training, "compile_model", False):
        backend = getattr(config.training, "compile_backend", "inductor")
        print(f"Compiling model with {backend} backend...")
        try:
            model = torch.compile(model, backend=backend)
        except Exception as e:
            print(f"Warning: torch.compile failed ({e})")

    return model


# =============================================================================
# Main Training Function
# =============================================================================


def train(config: Config, timestamp: str | None = None):
    """Main training function."""
    seed = getattr(config.training, "seed", 42)
    set_seed(seed)
    print(f"Random seed: {seed}")

    # Setup directories
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(config.training.checkpoint_dir) / timestamp
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    config.save(checkpoint_dir / "config.yaml")

    # When running detached ensure output reaches the log file promptly.
    if not _is_tty():
        if sys.stdout is None:
            # pythonw launched without -RedirectStandardOutput: create our own log.
            log_path = checkpoint_dir / "train.log"
            _redirect_to_log(log_path)
            print(
                f"=== training log  started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==="
            )
            print(f"    checkpoint dir : {checkpoint_dir}")
            print(f"    log file       : {log_path}")
        else:
            # stdout already redirected by the caller (e.g. Start-Process
            # -RedirectStandardOutput): just make sure it flushes every line
            # instead of buffering in 8 KB blocks.
            _ensure_autoflush()
            print(
                f"=== training log  started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==="
            )
            print(f"    checkpoint dir : {checkpoint_dir}")

    # Create model and data
    print("Creating model...")
    model = create_model(config)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    print("Creating train dataloader...")
    train_loader, train_dataset = create_dataloader(config, "train")

    print("Creating val dataloader...")
    val_loader, val_dataset = create_dataloader(config, "val")

    # Setup training
    criterion = COIWeightedLoss(
        class_weight=getattr(config.training, "class_weight", 1.5)
    )
    base_lr = float(config.training.lr)
    optimizer = optim.AdamW(
        model.parameters(), lr=base_lr, weight_decay=float(config.training.weight_decay)
    )

    warmup_steps = int(getattr(config.training, "warmup_steps", 300))
    # Estimate total optimizer steps for cosine schedule
    steps_per_epoch = max(
        1, len(train_loader) // max(1, config.training.grad_accum_steps)
    )
    total_steps = steps_per_epoch * config.training.num_epochs

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        # Cosine decay from 1.0 to 0.01 after warmup
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Create GradScaler once so its internal state persists across epochs
    use_amp = getattr(config.training, "use_amp", True) and str(
        config.training.device
    ).startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    validate_every_n = int(getattr(config.training, "validate_every_n_epochs", 1))

    # Training loop state (may be overwritten by a resumed checkpoint below)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    global_step = 0  # cumulative optimizer steps across all epochs
    history = {"train_loss": [], "val_loss": [], "grad_norms": []}
    start_epoch = 1

    last_ckpt_path = checkpoint_dir / "last_checkpoint.pt"
    best_ckpt_path = checkpoint_dir / "best_model.pt"
    if last_ckpt_path.exists():
        print(f"\nResuming from checkpoint: {last_ckpt_path}")
        ckpt = torch.load(
            last_ckpt_path, map_location=config.training.device, weights_only=False
        )
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_val_loss = ckpt["best_val_loss"]
        epochs_without_improvement = ckpt["epochs_without_improvement"]
        history = ckpt["history"]
        print(
            f"  Resumed at epoch {start_epoch}, global_step {global_step}, "
            f"best_val_loss {best_val_loss:.4f}"
        )
    elif best_ckpt_path.exists():
        print(
            f"\nNo last_checkpoint.pt found. Resuming from best_model.pt: {best_ckpt_path}"
        )
        ckpt = torch.load(
            best_ckpt_path, map_location=config.training.device, weights_only=False
        )
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_val_loss = ckpt["val_loss"]
        # best_model.pt was saved at the best epoch, so reset counter to 0
        epochs_without_improvement = 0
        if "history" in ckpt:
            history = ckpt["history"]
        print(
            f"  Fast-forwarding scheduler {global_step} steps to restore LR position..."
        )
        for _ in range(global_step):
            scheduler.step()
        print(
            f"  Resumed at epoch {start_epoch}, global_step {global_step}, "
            f"best_val_loss {best_val_loss:.4f}"
        )
    else:
        print("No checkpoint found — starting from scratch.")

    for epoch in range(start_epoch, config.training.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.training.num_epochs}")

        # Re-seed dataset RNG for augmentation diversity each epoch
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)

        train_loss, epoch_steps, epoch_grad_norms = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            config.training.device,
            clip_grad_norm=config.training.clip_grad_norm,
            grad_accum_steps=getattr(config.training, "grad_accum_steps", 1),
            use_amp=getattr(config.training, "use_amp", True),
            snr_range=tuple(config.data.snr_range),
            scaler=scaler,
            scheduler=scheduler,
        )
        global_step += epoch_steps
        history["train_loss"].append(train_loss)
        valid_norms = [n for n in epoch_grad_norms if not np.isnan(n)]
        history["grad_norms"].append(
            np.mean(valid_norms) if valid_norms else float("nan")
        )

        # Validation
        run_validation = (
            (epoch % validate_every_n == 0)
            or (epoch == 1)
            or (epoch == config.training.num_epochs)
        )

        if run_validation:
            # Reset val dataset RNG for deterministic background pairing each time
            if hasattr(val_loader.dataset, "set_epoch"):
                val_loader.dataset.set_epoch(0)  # always same seed for comparable val
            val_loss = validate_epoch(
                model,
                val_loader,
                criterion,
                config.training.device,
                use_amp=getattr(config.training, "use_amp", True),
                snr_range=tuple(config.data.snr_range),
            )
            history["val_loss"].append(val_loss)
            print(f"Train: {train_loss:.4f}, Val: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "config": config.to_dict(),
                        "history": history,
                    },
                    checkpoint_dir / "best_model.pt",
                )
                print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= config.training.patience:
                print(f"\nEarly stopping after {epoch} epochs")
                break
        else:
            history["val_loss"].append(None)
            print(f"Train: {train_loss:.4f}")

        last_ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
        }
        if scaler is not None:
            last_ckpt["scaler_state_dict"] = scaler.state_dict()
        torch.save(last_ckpt, last_ckpt_path)

    # Save history
    with open(checkpoint_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Convert negative SI-SNR to positive for reporting
    best_val_sisnr = -best_val_loss
    print(f"\nTraining completed! Best val loss: {best_val_loss:.4f}")
    print(f"Best Val SI-SNR: {best_val_sisnr:.2f} dB")


def main():
    parser = argparse.ArgumentParser(description="Train aircraft sound separation")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to an existing checkpoint directory to resume training from "
            "(e.g. checkpoints/20240101_120000). The config.yaml and "
            "separation_dataset.csv inside that directory will be reused."
        ),
    )
    args = parser.parse_args()

    if args.resume:
        resume_dir = Path(args.resume)
        if not resume_dir.is_dir():
            raise FileNotFoundError(f"Resume directory not found: {resume_dir}")
        if not (resume_dir / "last_checkpoint.pt").exists():
            if not (resume_dir / "best_model.pt").exists():
                raise FileNotFoundError(
                    f"Neither last_checkpoint.pt nor best_model.pt found in {resume_dir}. "
                    "Cannot resume — did the first epoch complete?"
                )
            print(
                f"  No last_checkpoint.pt found in {resume_dir}; "
                "will fall back to best_model.pt (scheduler will be reconstructed)."
            )

        print(f"Resuming run from: {resume_dir}")
        config = Config.from_yaml(str(resume_dir / "config.yaml"))
        # Override config with any flags from the user-supplied config file,
        # but keep df_path pointing at the already-sampled dataset.
        user_config = Config.from_yaml(str(Path(args.config)))
        config.data.df_path = str(resume_dir / "separation_dataset.csv")
        # Allow the user to tweak training hyper-params (lr, epochs, …) via
        # the --config file while still reusing the existing dataset/run.
        config.training = user_config.training

        timestamp = resume_dir.name
        print(f"  Using dataset: {config.data.df_path}")
        train(config, timestamp=timestamp)
        return

    config = Config.from_yaml(str(Path(args.config)))
    print("Configuration:")
    print(f"  Model: {config.model.type} ({config.model.num_blocks} blocks)")
    print(f"  Device: {config.training.device}")

    use_webdataset = getattr(config.data, "use_webdataset", False)
    webdataset_path = getattr(config.data, "webdataset_path", "")

    if use_webdataset:
        # WebDataset mode: Skip CSV loading, go straight to training
        if not webdataset_path:
            raise ValueError(
                "webdataset_path must be set when use_webdataset=True. "
                "Add webdataset_path to your training_config.yaml"
            )
        
        print("\n" + "=" * 70)
        print("WebDataset Mode Enabled")
        print("=" * 70)
        print(f"  WebDataset path: {webdataset_path}")
        print("  Skipping CSV dataset creation and metadata loading")
        print("  Data will be loaded directly from tar shards")
        print("=" * 70 + "\n")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = Path(config.training.checkpoint_dir) / timestamp
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        config.data.df_path = str(checkpoint_dir / "webdataset_placeholder.csv")
        
        # Save config and start training
        train(config, timestamp=timestamp)
        return

    print("\nLoading dataset metadata...")
    project_root = Path(__file__).parent.parent.parent.parent
    datasets_path = str(project_root / "data")
    audio_base_path = str(project_root.parent / "datasets")

    all_metadata = load_metadata_datasets(datasets_path, audio_base_path)
    separation_metadata, _ = split_seperation_classification(all_metadata)

    print(f"Loaded {len(all_metadata)} total samples")
    print(f"Using {len(separation_metadata)} for separation training (80%)")

    if config.data.n_coi_classes > 1:
        coi_class_cfgs = config.data.coi_classes
        if not coi_class_cfgs:
            raise ValueError(
                "n_coi_classes > 1 but coi_classes is empty in the config. "
                "Add one entry per COI class (labels + dataset) to coi_classes."
            )
        if len(coi_class_cfgs) != config.data.n_coi_classes:
            raise ValueError(
                f"n_coi_classes={config.data.n_coi_classes} but "
                f"coi_classes has {len(coi_class_cfgs)} entries. "
                "Provide exactly one entry per COI class."
            )

        # Flat union of all labels — used for non-COI exclusion in sample_non_coi
        all_labels = [lbl for cc in coi_class_cfgs for lbl in cc.labels]
        config.data.target_classes = all_labels  # sync so checkpoint config is self-consistent
        target_classes = all_labels
        print(f"\nTarget classes (union): {target_classes}")

        # Per-class COI sampling with optional dataset filter
        print("\nSampling COI data per class...")
        class_coi_dfs = []
        for class_idx, cc in enumerate(coi_class_cfgs):
            name = cc.name or (cc.labels[0] if cc.labels else f"class_{class_idx}")
            class_coi_df = get_coi(separation_metadata, cc.labels)
            if cc.dataset:
                mask = class_coi_df["dataset"].str.contains(
                    cc.dataset, case=False, na=False
                )
                n_dropped = int((~mask).sum())
                if n_dropped:
                    print(
                        f"  [{name}] Dropped {n_dropped} samples not matching "
                        f"dataset filter '{cc.dataset}'"
                    )
                class_coi_df = class_coi_df[mask].copy()
            else:
                class_coi_df = class_coi_df.copy()
            class_coi_df["coi_class"] = class_idx
            print(f"  COI class {class_idx} ({name!r}): {len(class_coi_df)} samples")
            class_coi_dfs.append(class_coi_df)

        all_coi_df = pd.concat(class_coi_dfs, ignore_index=True)

        print("\nSampling non-COI data...")
        sampled_df = sample_non_coi(
            separation_metadata, all_coi_df, target_class=target_classes, coi_ratio=0.25
        )
        # sample_non_coi preserves coi_class for COI rows; fill -1 for non-COI
        sampled_df["coi_class"] = sampled_df["coi_class"].fillna(-1).astype(int)

    else:
        # Single-class mode: use target_classes directly.
        target_classes = getattr(config.data, "target_classes", None)
        if not target_classes:
            raise ValueError(
                "No target_classes specified in config.data – please add a list of "
                "labels to the YAML configuration"
            )
        print(f"\nTarget classes: {target_classes}")

        print("\nSampling data...")
        coi_df = get_coi(separation_metadata, target_classes)
        sampled_df = sample_non_coi(
            separation_metadata, coi_df, target_class=target_classes, coi_ratio=0.25
        )

    # Binary labels (orig_label was already set by sample_non_coi)
    sampled_df["label"] = sampled_df["orig_label"].apply(
        lambda x: (
            1
            if (isinstance(x, list) and any(lbl in target_classes for lbl in x))
            or (isinstance(x, str) and x in target_classes)
            else 0
        )
    )

    # Check files
    print("\nChecking audio files...")
    sampled_df["file_exists"] = sampled_df["filename"].apply(lambda f: Path(f).exists())
    if (~sampled_df["file_exists"]).any():
        missing = (~sampled_df["file_exists"]).sum()
        print(f"⚠️  Found {missing} missing files, dropping...")
        for dataset in ["esc50", "aerosonicdb", "freesound", "risoux_test"]:
            if (
                ~sampled_df["file_exists"] & sampled_df["dataset"].str.contains(dataset)
            ).any():
                print(
                    f"  - {dataset}: {(~sampled_df['file_exists'] & sampled_df['dataset'].str.contains(dataset)).sum()} missing"
                )
                print(
                    f"  - file format of missing file is {sampled_df[~sampled_df['file_exists'] & sampled_df['dataset'].str.contains(dataset)]['filename'].iloc[0]}"
                )
        sampled_df = sampled_df[sampled_df["file_exists"]]
    sampled_df = sampled_df.drop(columns=["file_exists"])
    print(f"✅ Final dataset: {len(sampled_df)} samples")

    # Stats
    print("\nDataset splits:")
    for split in ["train", "val", "test"]:
        split_df = sampled_df[sampled_df["split"] == split]
        print(
            f"  {split}: {len(split_df)} (COI: {(split_df['label'] == 1).sum()}, non-COI: {(split_df['label'] == 0).sum()})"
        )

    # Save dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(config.training.checkpoint_dir) / timestamp
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    sampled_df["label"] = sampled_df["label"].astype("uint8")
    sampled_df["split"] = sampled_df["split"].astype("category")
    df_path = checkpoint_dir / "separation_dataset.csv"
    sampled_df.to_csv(df_path, index=False)
    print(f"\nSaved dataset to: {df_path}")

    config.data.df_path = str(df_path)

    # Cleanup and train
    del all_metadata, separation_metadata, sampled_df
    gc.collect()

    train(config, timestamp=timestamp)


if __name__ == "__main__":
    main()
