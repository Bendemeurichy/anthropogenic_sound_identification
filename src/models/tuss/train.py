"""
Training script for TUSS model fine-tuned for COI separation.

Loads the pretrained TUSS backbone, injects new learnable prompt vectors for each
COI class (e.g. "airplane", "train", "bird") and a background prompt ("background"),
warm-starting them from the existing "sfx" / "sfxbg" vectors.  The full network is
then fine-tuned on a COI dataset using snr_with_zeroref_loss so that absent sources
(zero-energy targets) are handled gracefully.

Config is read from training_config.yaml in the same directory.
Pass --device / --gpu on the command line to override the device without editing the YAML.

Dataset expectations (same CSV format as sudormrf):
    - 'filename' : path to wav file
    - 'split'    : train / val / test
    - 'label'    : list or string of semantic class labels
    - 'coi_class': integer index (0 … n_coi_classes-1)  [added by this script]
"""

import argparse
import gc
import io
import json
import os
import sys
from contextlib import nullcontext
from .config import Config, DataConfig, ModelConfig, TrainingConfig, resolve_device
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchaudio
import yaml
from torch.utils.data import DataLoader

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

# ---------------------------------------------------------------------------
# Resolve paths so imports work regardless of working directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).parent.resolve()
_BASE_DIR = _SCRIPT_DIR / "base"
_SRC_DIR = _SCRIPT_DIR.parent.parent  # code/src

for _p in [str(_BASE_DIR), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.training_utils import (
    _ensure_autoflush,
    _is_tty,
    _redirect_to_log,
    progress_bar,
    set_seed,
)
from loss_functions.snr import snr_with_zeroref_loss
from nets.model_wrapper import SeparationModel

from label_loading.metadata_loader import (
    load_metadata_datasets,
    split_seperation_classification,
)
from label_loading.sampler import get_coi, sample_non_coi

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOSS_EPS = 1e-8
SILENCE_ENERGY_EPS = 1e-6
WEAK_TARGET_ENERGY_EPS = 1e-4

CONFIG_PATH = _SCRIPT_DIR / "training_config.yaml"

from .dataset import AudioDataset
from .augmentations import AudioAugmentations, GpuAudioAugmentations
from .losses import COIWeightedSNRLoss, WarmupScheduler
from .utils import (
    BG_SCALE_MAX,
    BG_SCALE_MIN,
    ENERGY_EPS,
    NORMALIZE_MIN_STD,
    check_finite,
    generate_variable_prompts,
    normalize_tensor_wav,
    prepare_batch,
    select_sources_for_prompts,
)


# =============================================================================
# Training and validation loops
# =============================================================================


def train_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    criterion: COIWeightedSNRLoss,
    prompts_batch_template: list[list[str]],
    device,
    clip_grad_norm: float = 5.0,
    grad_accum_steps: int = 1,
    use_amp: bool = True,
    snr_range: tuple[float, float] = (-5.0, 5.0),
    scaler: torch.amp.GradScaler | None = None,
    scheduler=None,
    # Variable prompts configuration (like base TUSS training)
    variable_prompts: bool = False,
    coi_prompts: list[str] | None = None,
    bg_prompt: str | None = None,
    prompt_dropout_prob: float = 0.5,
    min_coi_prompts: int = 0,
    epoch_seed: int = 0,
    # GPU augmentation settings
    use_gpu_augmentations: bool = True,
    gpu_aug_time_stretch_prob: float = 0.5,
    gpu_aug_gain_prob: float = 0.7,
    gpu_aug_noise_prob: float = 0.4,
    gpu_aug_shift_prob: float = 0.5,
    gpu_aug_lpf_prob: float = 0.3,
) -> tuple[float, int, list[float]]:
    model.train()
    running_loss, n_samples = 0.0, 0
    grad_norms: list[float] = []
    use_amp = use_amp and str(device).startswith("cuda")
    amp_dtype = getattr(criterion, "_amp_dtype", torch.bfloat16)
    needs_scaler = use_amp and amp_dtype == torch.float16
    if scaler is None and needs_scaler:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    elif not needs_scaler:
        scaler = None  # GradScaler is unnecessary for bf16
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=amp_dtype, enabled=True)
        if use_amp
        else nullcontext()
    )

    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    has_pending_grads = False
    
    # Initialize RNG for variable prompts (deterministic per epoch)
    prompt_rng = np.random.default_rng(epoch_seed) if variable_prompts else None

    pbar = progress_bar(dataloader, desc="Training")
    for step_idx, sources in enumerate(pbar, start=1):
        sources = sources.to(device, non_blocking=True)
        
        # Apply GPU augmentations (10-100x faster than CPU)
        # Only augment COI channels (not background), matching original behavior
        if use_gpu_augmentations and str(device).startswith("cuda"):
            # sources shape: (B, n_coi_classes + 1, T)
            # COI channels: sources[:, :-1]  (all except last)
            # Background:   sources[:, -1:]  (last channel only)
            coi_sources = sources[:, :-1, :]  # (B, n_coi, T)
            bg_sources = sources[:, -1:, :]   # (B, 1, T)

            # Bug-fix: record which COI channels are silent (absent class) BEFORE
            # augmentation.  add_noise_batch injects Gaussian noise at amplitude
            # 0.001–0.01, giving per-sample energy ≈ 1e-6 – right at
            # SILENCE_ENERGY_EPS.  Without this guard, ~50 % of absent-class
            # channels cross the threshold and are incorrectly treated as active
            # sources by snr_with_zeroref_loss and COIWeightedSNRLoss.
            # Shape: (B, n_coi, 1) – True where the channel was all-zeros.
            silent_coi_mask = (
                coi_sources.pow(2).mean(dim=-1, keepdim=True) < SILENCE_ENERGY_EPS
            )

            # Augment only COI channels
            coi_sources = GpuAudioAugmentations.random_augment_batch(
                coi_sources,
                time_stretch_prob=gpu_aug_time_stretch_prob,
                gain_prob=gpu_aug_gain_prob,
                noise_prob=gpu_aug_noise_prob,
                shift_prob=gpu_aug_shift_prob,
                lpf_prob=gpu_aug_lpf_prob,
            )

            # Restore exact zeros for channels that were silent before augmentation
            # so that downstream silence detection (SILENCE_ENERGY_EPS threshold)
            # continues to work correctly.
            coi_sources = torch.where(
                silent_coi_mask.expand_as(coi_sources),
                torch.zeros_like(coi_sources),
                coi_sources,
            )

            # Recombine: augmented COI + original background
            sources = torch.cat([coi_sources, bg_sources], dim=1)
        
        mixture, clean_wavs = prepare_batch(sources, snr_range, deterministic=False)
        B = sources.shape[0]
        del sources

        if not check_finite(mixture, clean_wavs):
            del mixture, clean_wavs
            grad_norms.append(float("nan"))
            continue

        # Build per-batch prompts list
        # With variable_prompts: generate new prompt config for each batch
        # Without variable_prompts: use fixed template (backward compatible)
        if variable_prompts and coi_prompts is not None and bg_prompt is not None:
            # Generate single prompt configuration for this batch
            # All samples in batch use same prompts (required by TUSS architecture)
            batch_prompt_config = generate_variable_prompts(
                coi_prompts, bg_prompt, batch_size=1,
                dropout_prob=prompt_dropout_prob,
                min_coi=min_coi_prompts,
                rng=prompt_rng
            )[0]
            prompts = [batch_prompt_config] * B
            
            # CRITICAL: Select corresponding ground truth channels to match variable prompts
            # Dropped COI sources are merged into background so model learns to separate
            # only what's prompted and put everything else in the residual.
            # Example:
            #   clean_wavs from dataset: (B, 3, T) = [airplane, birds, background]
            #   If prompts = ["birds", "background"], output: (B, 2, T) = [birds, (airplane+background)]
            clean_wavs = select_sources_for_prompts(
                clean_wavs, coi_prompts, bg_prompt, batch_prompt_config
            )
        else:
            # Fixed prompts (original behavior)
            prompts = prompts_batch_template[:B]

        with autocast_ctx:
            outputs = model(mixture, prompts)

            if not check_finite(outputs):
                del outputs
                if use_amp and scaler is not None:
                    torch.cuda.empty_cache()
                    with torch.amp.autocast("cuda", enabled=False):
                        outputs = model(mixture.float(), prompts)
                    if not check_finite(outputs):
                        del outputs, mixture, clean_wavs
                        optimizer.zero_grad(set_to_none=True)
                        grad_norms.append(float("nan"))
                        torch.cuda.empty_cache()
                        continue
                else:
                    del mixture, clean_wavs
                    optimizer.zero_grad(set_to_none=True)
                    grad_norms.append(float("nan"))
                    continue

            # Trim/pad outputs to match reference length (STFT/iSTFT may change length)
            T_ref = clean_wavs.shape[-1]
            T_est = outputs.shape[-1]
            if T_est > T_ref:
                outputs = outputs[..., :T_ref]
            elif T_est < T_ref:
                clean_wavs = clean_wavs[..., :T_est]

            loss = criterion(outputs.float(), clean_wavs.float())
        
        # Track per-class active samples for visibility (before deletion)
        active_class_counts = None
        actual_n_coi = clean_wavs.shape[1] - 1  # Infer from tensor (last channel is background)
        if step_idx % 10 == 0 and actual_n_coi >= 1:
            with torch.no_grad():
                ref_power = (clean_wavs[:, :actual_n_coi] ** 2).mean(dim=-1)
                is_active = ref_power > SILENCE_ENERGY_EPS
                active_class_counts = is_active.sum(dim=0).cpu()  # (actual_n_coi,)

        del outputs, mixture, clean_wavs

        if not check_finite(loss):
            del loss
            optimizer.zero_grad(set_to_none=True)
            grad_norms.append(float("nan"))
            continue

        loss_scaled = loss / grad_accum_steps
        if use_amp and scaler is not None:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        batch_loss = float(loss.item())
        del loss, loss_scaled

        if step_idx % grad_accum_steps == 0:
            if use_amp and scaler is not None:
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
                if use_amp and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            else:
                grad_norms.append(float("nan"))
                if use_amp and scaler is not None:
                    try:
                        scaler.update()
                    except Exception:
                        pass

            optimizer.zero_grad(set_to_none=True)
            has_pending_grads = False
        else:
            has_pending_grads = True

        running_loss += batch_loss * B
        n_samples += B
        
        # Show class distribution in progress bar
        postfix = {"loss": f"{batch_loss:.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"}
        if active_class_counts is not None:
            for cls_i in range(len(active_class_counts)):
                postfix[f"c{cls_i}"] = int(active_class_counts[cls_i])
        pbar.set_postfix(postfix)

        # Reduce cache clearing frequency for better performance
        # Only clear every 100 steps instead of 20 (cache clearing is slow)
        if str(device).startswith("cuda") and step_idx % 100 == 0:
            torch.cuda.empty_cache()

    # Flush remaining accumulated gradients
    if has_pending_grads:
        if use_amp and scaler is not None:
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
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
        else:
            grad_norms.append(float("nan"))
            if use_amp and scaler is not None:
                try:
                    scaler.update()
                except Exception:
                    pass
        optimizer.zero_grad(set_to_none=True)

    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    valid_norms = [n for n in grad_norms if not np.isnan(n)]
    if valid_norms:
        print(
            f"  Gradient norms – avg: {np.mean(valid_norms):.4f}, max: {np.max(valid_norms):.4f}"
        )

    return running_loss / max(n_samples, 1), optimizer_step, grad_norms


def validate_epoch(
    model,
    dataloader: DataLoader,
    criterion: COIWeightedSNRLoss,
    prompts_batch_template: list[list[str]],
    device,
    use_amp: bool = True,
    snr_range: tuple[float, float] = (-5.0, 5.0),
) -> float:
    model.eval()
    running_loss, n_samples = 0.0, 0
    val_step = 0
    per_class_sisnr: list[list[float]] = [[] for _ in range(criterion.n_coi)]
    per_class_counts: list[int] = [0 for _ in range(criterion.n_coi)]
    bg_sisnr_vals: list[float] = []
    bg_sisnr_bg_only: list[float] = []  # Background SI-SNR on background-only samples
    bg_sisnr_mixed: list[float] = []  # Background SI-SNR on mixed samples
    bg_energy_ratios: list[float] = []  # BG energy / total energy ratio
    
    # Track per-class losses for class-balanced validation loss
    per_class_loss_sum: list[float] = [0.0 for _ in range(criterion.n_coi)]
    per_class_loss_counts: list[int] = [0 for _ in range(criterion.n_coi)]
    bg_loss_sum = 0.0
    bg_loss_count = 0

    use_amp = use_amp and str(device).startswith("cuda")
    amp_dtype = getattr(criterion, "_amp_dtype", torch.bfloat16)
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=amp_dtype, enabled=True)
        if use_amp
        else nullcontext()
    )

    pbar = progress_bar(dataloader, desc="Validation")
    with torch.no_grad():
        for sources in pbar:
            val_step += 1
            sources = sources.to(device, non_blocking=True)
            mixture, clean_wavs = prepare_batch(sources, snr_range, deterministic=True)
            B = sources.shape[0]
            del sources

            if not check_finite(mixture, clean_wavs):
                del mixture, clean_wavs
                continue

            prompts = prompts_batch_template[:B]

            with autocast_ctx:
                outputs = model(mixture, prompts)
                # Trim/pad outputs to match reference length (STFT/iSTFT may change length)
                T_ref = clean_wavs.shape[-1]
                T_est = outputs.shape[-1]
                if T_est > T_ref:
                    outputs = outputs[..., :T_ref]
                elif T_est < T_ref:
                    clean_wavs = clean_wavs[..., :T_est]

                loss = criterion(outputs.float(), clean_wavs.float())

            batch_loss = float(loss.item())
            running_loss += batch_loss * B
            n_samples += B
            
            # Accumulate per-class losses for class-balanced validation
            # This ensures equal weight for each class regardless of sample count
            with torch.no_grad():
                n_src = outputs.shape[1]
                # Compute per-source losses (same as in COIWeightedSNRLoss but without weighting)
                from loss_functions.snr import snr_with_zeroref_loss
                per_src_losses = snr_with_zeroref_loss(
                    outputs.float(),
                    clean_wavs.float(),
                    n_src=n_src,
                    snr_max=criterion.snr_max,
                    zero_ref_loss_weight=criterion.zero_ref_loss_weight,
                    solve_perm=False,
                    eps=criterion.eps,
                )  # (B, n_src)
                
                # Track COI losses per class (only for active samples)
                for cls_i in range(n_src - 1):
                    ref_power = (clean_wavs[:, cls_i] ** 2).mean(dim=-1)  # (B,)
                    is_active = ref_power > SILENCE_ENERGY_EPS  # (B,)
                    if is_active.any():
                        active_losses = per_src_losses[:, cls_i][is_active]
                        per_class_loss_sum[cls_i] += active_losses.sum().item()
                        per_class_loss_counts[cls_i] += is_active.sum().item()
                
                # Track background loss (always active)
                bg_loss_sum += per_src_losses[:, -1].sum().item()
                bg_loss_count += B

            # Per-class SI-SNR for reporting
            try:
                from loss_functions.si_snr import si_snr_loss

                n_src = outputs.shape[1]
                
                # Check if any COI is present (to identify background-only samples)
                any_coi_present = False
                for cls_i in range(n_src - 1):
                    ref_energy = clean_wavs[:, cls_i].pow(2).mean(dim=-1)
                    present = ref_energy > SILENCE_ENERGY_EPS
                    if present.any():
                        any_coi_present = True
                        snr_val = -si_snr_loss(
                            outputs[:, cls_i : cls_i + 1],
                            clean_wavs[:, cls_i : cls_i + 1],
                            solve_perm=False,
                        )
                        per_class_sisnr[cls_i].append(
                            float(snr_val[present].mean().item())
                        )
                        per_class_counts[cls_i] += present.sum().item()
                
                # Background metrics
                bg_snr = -si_snr_loss(
                    outputs[:, -1:], clean_wavs[:, -1:], solve_perm=False
                )
                bg_snr_mean = float(bg_snr.mean().item())
                bg_sisnr_vals.append(bg_snr_mean)
                
                # Track background-only vs mixed samples separately
                if any_coi_present:
                    bg_sisnr_mixed.append(bg_snr_mean)
                else:
                    bg_sisnr_bg_only.append(bg_snr_mean)
                
                # Track energy ratio: BG / (COI + BG)
                coi_energy = outputs[:, :-1].pow(2).sum(dim=(1, 2))  # Sum over all COI sources
                bg_energy = outputs[:, -1].pow(2).sum(dim=1)
                total_energy = coi_energy + bg_energy + 1e-8
                energy_ratio = (bg_energy / total_energy).mean().item()
                bg_energy_ratios.append(energy_ratio)
                
                pbar.set_postfix(loss=f"{batch_loss:.4f}")
            except Exception:
                pbar.set_postfix(loss=f"{batch_loss:.4f}")

            # Reduce cache clearing frequency for better performance
            del outputs, loss, mixture, clean_wavs
            if str(device).startswith("cuda") and val_step % 100 == 0:
                torch.cuda.empty_cache()

    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    # Compute class-balanced validation loss
    # Average per-class losses, then average across classes (not samples)
    # This gives equal weight to each class regardless of validation set imbalance
    per_class_avg_losses = []
    for cls_i in range(criterion.n_coi):
        if per_class_loss_counts[cls_i] > 0:
            avg_loss = per_class_loss_sum[cls_i] / per_class_loss_counts[cls_i]
            per_class_avg_losses.append(avg_loss)
    
    if bg_loss_count > 0:
        bg_avg_loss = bg_loss_sum / bg_loss_count
        per_class_avg_losses.append(bg_avg_loss)
    
    # Class-balanced loss: equal weight to each class
    class_balanced_loss = (
        np.mean(per_class_avg_losses) if per_class_avg_losses else 0.0
    )
    
    # Also compute the standard sample-weighted loss for comparison
    sample_weighted_loss = running_loss / max(n_samples, 1)

    if bg_sisnr_vals:
        class_strs = []
        for i, v in enumerate(per_class_sisnr):
            if v:
                avg_snr = np.mean(v)
                count = per_class_counts[i]
                class_strs.append(f"cls{i}: {avg_snr:.2f} dB [{count} samples]")
            else:
                class_strs.append(f"cls{i}: n/a")
        
        bg_overall = np.mean(bg_sisnr_vals)
        print(
            f"  Val SI-SNR – {', '.join(class_strs)}, BG: {bg_overall:.2f} dB"
        )
        
        # Enhanced background metrics
        if bg_sisnr_bg_only:
            print(f"    BG (background-only samples): {np.mean(bg_sisnr_bg_only):.2f} dB [{len(bg_sisnr_bg_only)} batches]")
        if bg_sisnr_mixed:
            print(f"    BG (mixed samples): {np.mean(bg_sisnr_mixed):.2f} dB [{len(bg_sisnr_mixed)} batches]")
        if bg_energy_ratios:
            print(f"    BG energy ratio (BG/total): {np.mean(bg_energy_ratios):.3f}")
    
    # Print both loss metrics for transparency
    print(f"  Val Loss (class-balanced): {class_balanced_loss:.6f}")
    print(f"  Val Loss (sample-weighted): {sample_weighted_loss:.6f}")
    print(f"  Class sample counts: {per_class_loss_counts}, BG: {bg_loss_count}")

    return class_balanced_loss


# =============================================================================
# Data loading
# =============================================================================


def create_dataloader(config: Config, split: str) -> tuple[DataLoader, AudioDataset]:
    """Create dataloader for specified split.

    Supports both file-based and WebDataset loading modes based on config.
    """
    # Check if we should use WebDataset
    use_webdataset = getattr(config.data, "use_webdataset", False)
    raw_webdataset_path = str(getattr(config.data, "webdataset_path", "") or "")
    webdataset_path = os.path.expanduser(
        os.path.expandvars(raw_webdataset_path)
    ).strip()

    if use_webdataset:
        if not webdataset_path:
            raise ValueError("webdataset_path must be set when use_webdataset=True")

        from src.common.webdataset_utils import COIWebDatasetWrapper
        from src.label_loading.metadata_loader import get_webdataset_paths

        print(f"Using WebDataset loading from: {webdataset_path}")

        n_coi = len(config.model.coi_prompts)
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

        # Disable CPU augmentations if GPU augmentations are enabled (10-100x faster)
        # GPU augmentations are applied in the training loop instead
        use_gpu_aug = getattr(config.training, "use_gpu_augmentations", True)
        cpu_augment = (split == "train") and not use_gpu_aug
        
        dataset = COIWebDatasetWrapper(
            tar_paths=tar_paths,
            split=split,
            target_sr=config.data.sample_rate,
            segment_length=config.data.segment_length,
            snr_range=tuple(config.data.snr_range),
            n_coi_classes=n_coi,
            shuffle=(split == "train"),
            augment=cpu_augment,  # Only use CPU augmentations if GPU augmentations are disabled
            stereo=False,
            background_only_prob=(
                config.data.background_only_prob if split == "train" else 0.0
            ),
            target_classes=target_classes,
            dataset_filter=None,  # Could be added to config if needed
            coi_ratio=0.25,  # Could be added to config if needed
            seed=seed,
            multi_coi_prob=(
                getattr(config.data, "multi_coi_prob", 0.0) if split == "train" else 0.0
            ),
            balance_classes=getattr(config.data, "balance_classes", False),
            coi_class_multipliers=(
                getattr(config.data, "coi_class_multipliers", None)
            ),
        )

        num_workers = config.training.num_workers
        pin_memory = config.training.pin_memory and torch.cuda.is_available()

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
    header_cols = pd.read_csv(config.data.df_path, nrows=0).columns.tolist()
    usecols = ["filename", "label", "split", "coi_class"]
    for opt in ("start_time", "end_time", "duration"):
        if opt in header_cols:
            usecols.append(opt)
    usecols = [c for c in usecols if c in header_cols]

    df = pd.read_csv(config.data.df_path, usecols=usecols)
    df["label"] = df["label"].astype("uint8")
    df["split"] = df["split"].astype("category")
    df["coi_class"] = df["coi_class"].astype("int16")

    n_coi = len(config.model.coi_prompts)
    dataset = AudioDataset(
        df,
        split=split,
        sample_rate=config.data.sample_rate,
        segment_length=config.data.segment_length,
        snr_range=tuple(config.data.snr_range),
        n_coi_classes=n_coi,
        augment=(split == "train"),
        background_only_prob=(
            config.data.background_only_prob if split == "train" else 0.0
        ),
        background_mix_n=config.data.background_mix_n,
        augment_multiplier=config.data.augment_multiplier,
        multi_coi_prob=getattr(config.data, "multi_coi_prob", 0.3),
    )

    num_workers = config.training.num_workers
    pin_memory = config.training.pin_memory and torch.cuda.is_available()

    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=(split == "train"),
        drop_last=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0 and split == "train"),
    )
    del df
    gc.collect()
    return loader, dataset





def create_model(
    config: Config, resume_ckpt_path: str | None = None
) -> tuple[torch.nn.Module, dict[str, list]]:
    """Build SeparationModel, load pretrained weights, inject new prompt vectors.

    Args:
        config: Training configuration.
        resume_ckpt_path: Path to a fine-tuned checkpoint (.pt file) to resume from.
            When provided, existing prompts from the checkpoint are preserved and
            only NEW prompts (not in the checkpoint) are injected.

    Returns:
        Tuple of:
            - model: The created separation model
            - param_groups: Dict with keys 'new_prompts', 'continuing_prompts',
                           'frozen_prompts', 'backbone' mapping to parameter lists
    """
    pretrained_path = config.model.pretrained_path
    if pretrained_path is not None:
        pretrained_path = _SCRIPT_DIR / pretrained_path

    # Track which prompts already exist in a checkpoint we're resuming from
    existing_prompts_in_ckpt = set()
    resume_state_dict = None

    if resume_ckpt_path and Path(resume_ckpt_path).is_file():
        print(f"Preparing to resume from fine-tuned checkpoint: {resume_ckpt_path}")
        resume_ckpt = torch.load(
            resume_ckpt_path, map_location="cpu", weights_only=False
        )
        resume_state_dict = resume_ckpt.get("model_state_dict", {})

        # Extract existing prompt names from checkpoint
        for key in resume_state_dict.keys():
            if key.startswith("separator.prompts."):
                prompt_name = key.replace("separator.prompts.", "", 1)
                existing_prompts_in_ckpt.add(prompt_name)

        if existing_prompts_in_ckpt:
            print(
                f"  Found {len(existing_prompts_in_ckpt)} existing prompts in checkpoint: {sorted(existing_prompts_in_ckpt)}"
            )

    if pretrained_path is not None and Path(pretrained_path).exists():
        hparams_file = Path(pretrained_path) / "hparams.yaml"
        with open(hparams_file) as f:
            hparams = yaml.safe_load(f)

        model = SeparationModel(
            encoder_name=hparams["encoder_name"],
            encoder_conf=hparams["encoder_conf"],
            decoder_name=hparams["decoder_name"],
            decoder_conf=hparams["decoder_conf"],
            separator_name=hparams["model_name"],
            separator_conf=hparams["model_conf"],
            css_conf=hparams["css_conf"],
            variance_normalization=hparams.get("variance_normalization", True),
        )

        # Only load base pretrained weights if we're NOT resuming from a fine-tuned checkpoint
        if resume_state_dict is None:
            ckpt_path = Path(pretrained_path) / "checkpoints" / "model.pth"
            print(f"Loading pretrained weights from {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            # Strip 'model.' prefix that Lightning checkpoints add
            state_dict = {
                (k[len("model.") :] if k.startswith("model.") else k): v
                for k, v in state_dict.items()
            }
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  Missing keys (expected for new prompts): {missing}")
            if unexpected:
                print(f"  Unexpected keys: {unexpected}")

        emb_dim = hparams["model_conf"]["conf_cross_prompt_module"]["emb_dim"]
    else:
        print("No pretrained model found – initialising from scratch.")
        model = SeparationModel(
            encoder_name=config.model.encoder_name,
            encoder_conf=config.model.encoder_conf,
            decoder_name=config.model.decoder_name,
            decoder_conf=config.model.decoder_conf,
            separator_name=config.model.separator_name,
            separator_conf=config.model.separator_conf,
            css_conf=config.model.css_conf,
            variance_normalization=config.model.variance_normalization,
        )
        emb_dim = config.model.separator_conf["conf_cross_prompt_module"]["emb_dim"]

    # ------------------------------------------------------------------ #
    # Inject new prompt vectors for each COI class and background.        #
    # When resuming, only inject prompts that DON'T exist in the ckpt.    #
    # Warm-start from the acoustically closest pretrained prompt.         #
    # ------------------------------------------------------------------ #
    prompts_dict = model.separator.prompts

    def _get_init_vector(init_from: str) -> torch.Tensor:
        if init_from in prompts_dict:
            return prompts_dict[init_from].data.clone()
        return torch.randn(emb_dim, 1, 1) * 0.02

    # CRITICAL: First inject ALL prompts from checkpoint so they can be loaded
    # PyTorch's ParameterDict won't auto-create keys during load_state_dict
    if resume_state_dict is not None:
        print(f"Pre-injecting prompts from checkpoint for loading...")
        for key in resume_state_dict.keys():
            if key.startswith("separator.prompts."):
                prompt_name = key.replace("separator.prompts.", "", 1)
                if prompt_name not in prompts_dict:
                    # Create placeholder parameter with correct shape from checkpoint
                    saved_tensor = resume_state_dict[key]
                    prompts_dict[prompt_name] = torch.nn.Parameter(
                        torch.zeros_like(saved_tensor)
                    )
                    print(f"  Pre-injected prompt '{prompt_name}' from checkpoint")

    new_prompts = config.model.coi_prompts + [config.model.bg_prompt]
    init_sources = [config.model.coi_prompt_init_from] * len(
        config.model.coi_prompts
    ) + [config.model.bg_prompt_init_from]

    newly_injected = []
    for prompt_name, init_from in zip(new_prompts, init_sources):
        # Skip injection if this prompt will be loaded from the resume checkpoint
        if prompt_name in existing_prompts_in_ckpt:
            print(f"  Prompt '{prompt_name}' exists in checkpoint – will be loaded")
            continue

        if prompt_name not in prompts_dict:
            init_val = _get_init_vector(init_from)
            # Add noise so each new COI prompt starts from a different position
            # even if they all init from the same source (e.g., "sfx")
            # Increased from 0.15 to 0.50 based on divergence analysis:
            #   - At 0.15: prompts stayed 84% similar to init source, only 65% similar to each other
            #   - Target: 30-45% inter-prompt similarity for excellent separation
            #   - 0.50 provides strong initial divergence while maintaining pretrained benefits
            noise = torch.randn_like(init_val) * 0.50
            prompts_dict[prompt_name] = torch.nn.Parameter(init_val + noise)
            newly_injected.append(prompt_name)
            print(f"  Injected NEW prompt '{prompt_name}' (init from '{init_from}')")
        else:
            print(f"  Prompt '{prompt_name}' already exists – keeping pretrained value")

    # Now load the resume checkpoint weights (if provided)
    # This will load existing prompts but leave newly injected ones untouched
    if resume_state_dict is not None:
        print(
            f"Loading checkpoint weights (newly injected prompts will be preserved)..."
        )
        missing, unexpected = model.load_state_dict(resume_state_dict, strict=False)
        if newly_injected:
            print(
                f"  ✓ {len(newly_injected)} new prompt(s) preserved: {newly_injected}"
            )
        expected_missing = [f"separator.prompts.{p}" for p in newly_injected]
        unexpected_missing = [k for k in missing if k not in expected_missing]
        if unexpected_missing:
            print(f"  ⚠ Unexpected missing keys: {unexpected_missing}")
        if unexpected:
            print(f"  ⚠ Unexpected keys: {unexpected}")

    # ------------------------------------------------------------------ #
    # Optional: freeze backbone, only update the prompt embeddings        #
    # ------------------------------------------------------------------ #
    if config.model.freeze_backbone:
        frozen, trainable = 0, 0
        for name, param in model.named_parameters():
            if "prompts" in name or "sos_token" in name:
                param.requires_grad_(True)
                trainable += param.numel()
            else:
                param.requires_grad_(False)
                frozen += param.numel()
        print(
            f"  Backbone frozen: {frozen / 1e6:.2f}M params frozen, "
            f"{trainable / 1e6:.2f}M trainable (prompts only)"
        )
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f"  Full fine-tune: {total / 1e6:.2f}M parameters")

    device = config.training.device
    print(f"\nMoving model to device: {device}")
    try:
        model = model.to(device)
        print(f"✓ Model successfully moved to {device}")
    except Exception as e:
        print(f"❌ Cannot move to {device}: {e} – falling back to CPU")
        model = model.to("cpu")
        config.training.device = "cpu"

    # ------------------------------------------------------------------ #
    # Collect parameter groups for differential learning rates            #
    # ------------------------------------------------------------------ #
    param_groups = {
        "new_prompts": [],
        "continuing_prompts": [],
        "frozen_prompts": [],
        "backbone": [],
    }

    if resume_ckpt_path:
        checkpoint_info = get_prompts_from_checkpoint(resume_ckpt_path)
        checkpoint_prompts = set(checkpoint_info.get("prompts_in_state", {}).keys())
        new_p, continuing_p, frozen_p = classify_prompts_by_training_strategy(
            config.model.coi_prompts, config.model.bg_prompt, checkpoint_prompts
        )

        # Collect parameters for each prompt category
        prompts_dict = model.separator.prompts

        for prompt_name in new_p:
            if prompt_name in prompts_dict:
                param_groups["new_prompts"].append(prompts_dict[prompt_name])

        for prompt_name in continuing_p:
            if prompt_name in prompts_dict:
                param_groups["continuing_prompts"].append(prompts_dict[prompt_name])

        for prompt_name in frozen_p:
            if prompt_name in prompts_dict:
                prompts_dict[prompt_name].requires_grad_(False)
                param_groups["frozen_prompts"].append(prompts_dict[prompt_name])

        # Collect backbone parameters (everything except prompts)
        for name, param in model.named_parameters():
            if "prompts" not in name and param.requires_grad:
                param_groups["backbone"].append(param)
    else:
        # No checkpoint: all config prompts are "new", no frozen prompts
        prompts_dict = model.separator.prompts
        for prompt_name in config.model.coi_prompts + [config.model.bg_prompt]:
            if prompt_name in prompts_dict:
                param_groups["new_prompts"].append(prompts_dict[prompt_name])

        for name, param in model.named_parameters():
            if "prompts" not in name and param.requires_grad:
                param_groups["backbone"].append(param)

    return model, param_groups


# =============================================================================
# Checkpoint validation utilities
# =============================================================================


def get_prompts_from_checkpoint(checkpoint_path: str | Path) -> dict:
    """Extract prompt information from a TUSS checkpoint.

    Returns:
        Dictionary with:
            - 'coi_prompts': List of COI class prompts
            - 'bg_prompt': Background prompt name
            - 'all_prompts_meta': All prompt names from metadata
            - 'prompts_in_state': Dict mapping prompt names to tensor shapes
            - 'checkpoint_info': Dict with epoch, global_step, val_loss
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        return {
            "coi_prompts": [],
            "bg_prompt": "",
            "all_prompts_meta": [],
            "prompts_in_state": {},
            "checkpoint_info": {"epoch": "N/A", "global_step": "N/A", "val_loss": "N/A"},
        }

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        coi_prompts = ckpt.get("coi_prompts", [])
        bg_prompt = ckpt.get("bg_prompt", "")
        all_prompts_meta = ckpt.get("all_prompts", [])

        model_state = ckpt.get("model_state_dict", {})
        prompts_in_state = {}
        prompt_prefix = "separator.prompts."
        for key, value in model_state.items():
            if key.startswith(prompt_prefix):
                prompt_name = key.replace(prompt_prefix, "", 1)
                prompts_in_state[prompt_name] = value.shape

        return {
            "coi_prompts": coi_prompts,
            "bg_prompt": bg_prompt,
            "all_prompts_meta": all_prompts_meta,
            "prompts_in_state": prompts_in_state,
            "checkpoint_info": {
                "epoch": ckpt.get("epoch", "N/A"),
                "global_step": ckpt.get("global_step", "N/A"),
                "val_loss": ckpt.get("val_loss", "N/A"),
            },
        }
    except Exception as e:
        print(f"⚠ Warning: Could not read prompts from checkpoint: {e}")
        return {
            "coi_prompts": [],
            "bg_prompt": "",
            "all_prompts_meta": [],
            "prompts_in_state": {},
            "checkpoint_info": {"epoch": "N/A", "global_step": "N/A", "val_loss": "N/A"},
        }


def validate_prompts_against_checkpoint(
    config_prompts: list[str],
    bg_prompt: str,
    checkpoint_path: str | None,
) -> tuple[list[str], list[str]]:
    """Validate config prompts against an existing checkpoint.

    Args:
        config_prompts: COI prompts from config
        bg_prompt: Background prompt from config
        checkpoint_path: Path to checkpoint to resume from

    Returns:
        Tuple of:
            - existing_prompts: List of prompts that already exist in checkpoint
            - new_prompts: List of prompts that are new (will be injected)
    """
    if not checkpoint_path:
        return [], config_prompts + [bg_prompt]

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        old_coi_prompts = ckpt.get("coi_prompts", [])
        old_bg_prompt = ckpt.get("bg_prompt", bg_prompt)

        if old_coi_prompts:
            # Strict validation: Check prefix matches
            if len(config_prompts) < len(old_coi_prompts):
                raise ValueError(
                    f"Config has fewer COI prompts ({len(config_prompts)}) than checkpoint ({len(old_coi_prompts)})."
                )
            for i, p in enumerate(old_coi_prompts):
                if config_prompts[i] != p:
                    raise ValueError(
                        f"Prompt order mismatch! Checkpoint prompt {i} is '{p}', but config prompt {i} is '{config_prompts[i]}'. You must append new prompts at the end."
                    )

            if bg_prompt != old_bg_prompt:
                raise ValueError(
                    f"Background prompt mismatch! Checkpoint uses '{old_bg_prompt}', config uses '{bg_prompt}'."
                )

        checkpoint_prompts = (
            set(old_coi_prompts + [old_bg_prompt])
            if old_coi_prompts
            else set(get_prompts_from_checkpoint(checkpoint_path).get("prompts_in_state", {}).keys())
        )
    except Exception as e:
        if isinstance(e, ValueError):
            raise e
        print(
            f"⚠ Warning: Could not read strict prompts from checkpoint, falling back: {e}"
        )
        checkpoint_prompts = set(get_prompts_from_checkpoint(checkpoint_path).get("prompts_in_state", {}).keys())

    if not checkpoint_prompts:
        # Checkpoint doesn't have prompts or couldn't be read
        return [], config_prompts + [bg_prompt]

    all_config_prompts = set(config_prompts + [bg_prompt])
    existing = sorted(all_config_prompts & checkpoint_prompts)
    new = sorted(all_config_prompts - checkpoint_prompts)
    frozen = sorted(checkpoint_prompts - all_config_prompts)

    print("\n" + "=" * 70)
    print("PROMPT VALIDATION")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Prompts in checkpoint: {sorted(checkpoint_prompts)}")
    print(f"Prompts in config: {sorted(all_config_prompts)}")

    if existing:
        print(f"\n✓ Already exist in checkpoint ({len(existing)}): {existing}")
        print(f"  → Will CONTINUE training at REDUCED LR (fine-tune further)")
    if new:
        print(f"\n+ NEW prompts ({len(new)}): {new}")
        print(f"  → Will be injected and trained from scratch")
    if frozen:
        print(f"\n❄️  Prompts in checkpoint but not in config ({len(frozen)}): {frozen}")
        print(f"  → Will be FROZEN (no training)")

    if not new and existing:
        print(f"\n📌 Training mode: CONTINUE FINE-TUNING")
        print(
            f"   All prompts exist in checkpoint - will continue training with more data"
        )
    elif new and existing:
        print(f"\n📌 Training mode: EXTEND (add new classes)")
        print(f"   Existing prompts will continue fine-tuning")
        print(f"   New prompts will learn from scratch")
    elif new and not existing:
        print(f"\n📌 Training mode: FRESH START")
        print(f"   All prompts are new - training from base model")

    print("=" * 70 + "\n")
    return existing, new


def classify_prompts_by_training_strategy(
    config_prompts: list[str],
    bg_prompt: str,
    checkpoint_prompts: set[str],
) -> tuple[list[str], list[str], list[str]]:
    """Classify prompts into three training strategies.

    Args:
        config_prompts: COI prompts from training config
        bg_prompt: Background prompt from config
        checkpoint_prompts: Prompts found in checkpoint

    Returns:
        Tuple of:
            - new_prompts: In config but not checkpoint (train at full LR)
            - continuing_prompts: In both config and checkpoint (train at reduced LR)
            - frozen_prompts: In checkpoint but not config (freeze, no training)
    """
    all_config = set(config_prompts + [bg_prompt])

    new_prompts = sorted(all_config - checkpoint_prompts)
    continuing_prompts = sorted(all_config & checkpoint_prompts)
    frozen_prompts = sorted(checkpoint_prompts - all_config)

    return new_prompts, continuing_prompts, frozen_prompts


# =============================================================================
# Seed
# =============================================================================


# =============================================================================
# Main training function
# =============================================================================


def train(config: Config, timestamp: str | None = None):
    """Main training function.

    Supports three training modes:
    1. Fresh start: No resume_from, trains all prompts from base model
    2. Continue fine-tuning: Resume with same prompts, continue training
    3. Extend: Resume and add NEW prompts to existing checkpoint

    Args:
        config: Training configuration
        timestamp: Optional timestamp string for checkpoint directory naming
    """
    seed = config.training.seed
    set_seed(seed)
    print(f"Seed: {seed}")

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

    # Validate prompts against checkpoint if resuming
    resume_path = getattr(config.training, "resume_from", "") or ""

    if resume_path:
        existing, new = validate_prompts_against_checkpoint(
            config.model.coi_prompts,
            config.model.bg_prompt,
            resume_path,
        )

        if existing and not new:
            print("ℹ️  Continuing fine-tuning on existing prompts with more training")
        elif new and existing:
            print(
                f"ℹ️  Extending model: keeping {len(existing)} existing + adding {len(new)} new prompts"
            )
        elif new and not existing:
            print("ℹ️  Starting fresh with new prompts")

    print("\nCreating model …")
    model, param_groups = create_model(
        config, resume_ckpt_path=resume_path if resume_path else None
    )

    n_coi = len(config.model.coi_prompts)
    n_src = n_coi + 1  # COI classes + background

    # Build the static prompts list used at every forward pass:
    # each sample in the batch sees the same n_src queries simultaneously.
    all_prompts = config.model.coi_prompts + [config.model.bg_prompt]
    # Pre-allocate a large template; we slice to [:B] at runtime.
    _MAX_BATCH = 256
    prompts_template = [list(all_prompts)] * _MAX_BATCH

    print(f"\n📋 Model will use {n_src} outputs: {all_prompts}")

    # Print parameter group information
    if (
        param_groups["new_prompts"]
        or param_groups["continuing_prompts"]
        or param_groups["frozen_prompts"]
    ):
        print("\n" + "=" * 70)
        print("PARAMETER GROUPS")
        print("=" * 70)

        if param_groups["new_prompts"]:
            names = [
                name
                for name, p in model.separator.prompts.items()
                if any(p is param for param in param_groups["new_prompts"])
            ]
            n_params = sum(p.numel() for p in param_groups["new_prompts"])
            print(f"\n🆕 New prompts (full LR: {config.training.lr:.1e}):")
            print(f"   {', '.join(names)}")
            print(f"   Total: {n_params:,} parameters")

        if param_groups["continuing_prompts"]:
            names = [
                name
                for name, p in model.separator.prompts.items()
                if any(p is param for param in param_groups["continuing_prompts"])
            ]
            n_params = sum(p.numel() for p in param_groups["continuing_prompts"])
            reduced_lr = (
                config.training.lr * config.training.existing_prompt_lr_multiplier
            )
            print(f"\n🔄 Continuing prompts (reduced LR: {reduced_lr:.1e}):")
            print(f"   {', '.join(names)}")
            print(f"   Total: {n_params:,} parameters")

        if param_groups["frozen_prompts"]:
            names = [
                name
                for name, p in model.separator.prompts.items()
                if any(p is param for param in param_groups["frozen_prompts"])
            ]
            n_params = sum(p.numel() for p in param_groups["frozen_prompts"])
            print(f"\n❄️  Frozen prompts (no training):")
            print(f"   {', '.join(names)}")
            print(f"   Total: {n_params:,} parameters")

        if param_groups["backbone"]:
            n_params = sum(p.numel() for p in param_groups["backbone"])
            print(f"\n🏗️  Backbone (LR: {config.training.lr:.1e}):")
            print(f"   Total: {n_params/1e6:.1f}M parameters")

        print("=" * 70 + "\n")

    print("\nCreating data loaders …")
    train_loader, train_dataset = create_dataloader(config, "train")
    val_loader, val_dataset = create_dataloader(config, "val")

    # ---- Resolve AMP dtype --------------------------------------------------
    _amp_dtype_str = getattr(config.training, "amp_dtype", "bf16").lower()
    if _amp_dtype_str in ("bf16", "bfloat16"):
        amp_dtype = torch.bfloat16
    elif _amp_dtype_str in ("fp16", "float16"):
        amp_dtype = torch.float16
    else:
        print(f"Unknown amp_dtype '{_amp_dtype_str}' – defaulting to bf16")
        amp_dtype = torch.bfloat16

    criterion = COIWeightedSNRLoss(
        n_src=n_src,
        coi_weight=config.training.coi_weight,
        snr_max=config.training.snr_max,
        zero_ref_loss_weight=config.training.zero_ref_loss_weight,
        amp_dtype=amp_dtype,
    )

    base_lr = float(config.training.lr)
    weight_decay = float(config.training.weight_decay)
    existing_lr_mult = float(config.training.existing_prompt_lr_multiplier)

    # Build parameter groups with different learning rates
    optimizer_param_groups = []

    if param_groups["new_prompts"]:
        optimizer_param_groups.append(
            {
                "params": param_groups["new_prompts"],
                "lr": base_lr,
                "name": "new_prompts",
            }
        )

    if param_groups["continuing_prompts"]:
        optimizer_param_groups.append(
            {
                "params": param_groups["continuing_prompts"],
                "lr": base_lr * existing_lr_mult,
                "name": "continuing_prompts",
            }
        )

    if param_groups["backbone"]:
        if config.model.freeze_backbone:
            # Backbone is frozen, don't add to optimizer
            pass
        else:
            backbone_lr_mult = getattr(config.training, "backbone_lr_multiplier", 0.05)
            optimizer_param_groups.append(
                {"params": param_groups["backbone"], "lr": base_lr * backbone_lr_mult, "name": "backbone"}
            )

    # If no parameter groups (shouldn't happen), fall back to old behavior
    if not optimizer_param_groups:
        optimizer_param_groups = [
            {
                "params": filter(lambda p: p.requires_grad, model.parameters()),
                "name": "default",
            }
        ]

    optimizer = optim.AdamW(optimizer_param_groups, weight_decay=weight_decay)

    warmup_steps = int(config.training.warmup_steps)
    stabilization_epochs = getattr(config.training, "stabilization_epochs", 0)
    steps_per_epoch = max(
        1, len(train_loader) // max(1, config.training.grad_accum_steps)
    )

    # Warm-up scheduler: linearly ramps each group from 0 → its target LR.
    # Becomes a no-op once warmup_steps are done so ReduceLROnPlateau can own
    # the LR afterwards without interference.
    warmup_scheduler = WarmupScheduler(optimizer, warmup_steps)

    # ReduceLROnPlateau: reduces LR when val loss stops improving.
    # Mirrors the base TUSS training config (patience=5, factor=0.5).
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=int(getattr(config.training, "scheduler_patience", 5)),
        factor=float(getattr(config.training, "scheduler_factor", 0.5)),
        min_lr=float(getattr(config.training, "scheduler_min_lr", 1e-7)),
    )

    use_amp = config.training.use_amp and str(config.training.device).startswith("cuda")
    # GradScaler is only needed for fp16; bf16 has sufficient dynamic range.
    _amp_backend = str(config.training.device).split(":")[0]
    needs_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(_amp_backend, enabled=True) if needs_scaler else None

    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Device: {config.training.device}")
    print(f"AMP enabled: {use_amp}")
    print(f"AMP dtype: {amp_dtype}")
    print(f"GradScaler: {scaler is not None}")
    if str(config.training.device).startswith("cuda"):
        if torch.cuda.is_available():
            device_idx = (
                int(config.training.device.split(":")[1])
                if ":" in config.training.device
                else 0
            )
            print(f"GPU: {torch.cuda.get_device_name(device_idx)}")
            print(
                f"GPU memory: {torch.cuda.get_device_properties(device_idx).total_memory / 1e9:.2f} GB"
            )
        else:
            print(
                "⚠️  WARNING: CUDA device specified but torch.cuda.is_available() = False"
            )
    print("=" * 70 + "\n")

    # ---- Resume from checkpoint ---------------------------------------------
    start_epoch = 1
    validate_every_n = int(config.training.validate_every_n_epochs)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    global_step = 0
    history: dict = {"train_loss": [], "val_loss": [], "grad_norms": []}

    resume_path = getattr(config.training, "resume_from", "") or ""
    if resume_path and Path(resume_path).is_file():
        print(f"Resuming training state from checkpoint: {resume_path}")
        ckpt = torch.load(
            resume_path, map_location=config.training.device, weights_only=False
        )
        # Model weights were already loaded in create_model(), only load optimizer/scheduler/history

        # Try to load optimizer state, but skip if parameter groups have changed
        # (happens when extending model with new prompts)
        has_new_prompts = bool(param_groups["new_prompts"])
        has_frozen_prompts = bool(param_groups["frozen_prompts"])
        # Only consider it "extending" if there are truly NEW prompts being injected
        # (not just continuing to train existing prompts)
        is_extending = has_new_prompts or has_frozen_prompts

        if is_extending:
            print("  ⚠️  Model is being extended with new/frozen prompts")
            print("     Skipping optimizer state loading (will start fresh)")
            print(
                "     This is expected and safe - Adam will build new momentum for all parameters"
            )
        else:
            # Continue fine-tuning mode: same prompts as before
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                print("  ✓ Loaded optimizer state")
            except Exception as e:
                print(f"  ⚠️  Could not load optimizer state: {e}")
                print("     Starting with fresh optimizer state")

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))

        # Reset best_val_loss ONLY when extending with new prompts (not when continuing all prompts)
        # Otherwise new prompts won't get a chance to save checkpoints
        # But if all prompts are continuing, we should keep the previous best_val_loss
        if has_new_prompts:
            # There are new prompts being injected - reset best_val_loss
            best_val_loss = float("inf")
            epochs_without_improvement = 0
            print("  ⚠️  Resetting best_val_loss to inf (new prompts need to learn)")
            print(
                f"     Previous checkpoint had val_loss: {ckpt.get('val_loss', 'N/A')}"
            )
        else:
            # All prompts are continuing - keep previous best_val_loss
            best_val_loss = float(ckpt.get("val_loss", float("inf")))

        history = ckpt.get("history", history)

        # Restore scheduler states so the LR curve continues seamlessly.
        if is_extending:
            print(
                "  ⚠️  Starting fresh schedulers (will apply warmup from beginning)"
            )
        else:
            if "warmup_scheduler_state_dict" in ckpt:
                warmup_scheduler.load_state_dict(ckpt["warmup_scheduler_state_dict"])
                print("  ✓ Loaded warmup scheduler state")
            else:
                # Legacy checkpoint: replay warmup steps manually.
                print("  ⚠ No warmup scheduler state – replaying warmup steps …")
                for _ in range(min(global_step, warmup_steps)):
                    warmup_scheduler.step()

            if "plateau_scheduler_state_dict" in ckpt:
                plateau_scheduler.load_state_dict(ckpt["plateau_scheduler_state_dict"])
                print("  ✓ Loaded plateau scheduler state")
            elif "scheduler_state_dict" in ckpt:
                # Legacy LambdaLR checkpoint: nothing meaningful to restore for
                # ReduceLROnPlateau, just log and continue.
                print("  ⚠ Legacy LambdaLR scheduler state found – skipping (not compatible with ReduceLROnPlateau)")

        # Restore GradScaler state (only relevant for fp16)
        if scaler is not None and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        print(
            f"  Resumed at epoch {start_epoch}, global_step {global_step}, "
            f"best_val_loss {best_val_loss:.4f}, "
            f"lr {optimizer.param_groups[0]['lr']:.2e}"
        )
    elif resume_path:
        print(f"resume_from path not found: {resume_path} – starting fresh")

    for epoch in range(start_epoch, config.training.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.training.num_epochs}")

        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)

        train_loss, epoch_steps, epoch_grad_norms = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            prompts_template,
            config.training.device,
            clip_grad_norm=config.training.clip_grad_norm,
            grad_accum_steps=config.training.grad_accum_steps,
            use_amp=config.training.use_amp,
            snr_range=tuple(config.data.snr_range),
            scaler=scaler,
            scheduler=warmup_scheduler,
            # Variable prompts configuration
            variable_prompts=config.training.variable_prompts,
            coi_prompts=config.model.coi_prompts,
            bg_prompt=config.model.bg_prompt,
            prompt_dropout_prob=config.training.prompt_dropout_prob,
            min_coi_prompts=config.training.min_coi_prompts,
            epoch_seed=config.training.seed + epoch,
            # GPU augmentation settings
            use_gpu_augmentations=getattr(config.training, "use_gpu_augmentations", True),
            gpu_aug_time_stretch_prob=getattr(config.training, "gpu_aug_time_stretch_prob", 0.5),
            gpu_aug_gain_prob=getattr(config.training, "gpu_aug_gain_prob", 0.7),
            gpu_aug_noise_prob=getattr(config.training, "gpu_aug_noise_prob", 0.4),
            gpu_aug_shift_prob=getattr(config.training, "gpu_aug_shift_prob", 0.5),
            gpu_aug_lpf_prob=getattr(config.training, "gpu_aug_lpf_prob", 0.3),
        )
        global_step += epoch_steps
        history["train_loss"].append(train_loss)
        valid_norms = [n for n in epoch_grad_norms if not np.isnan(n)]
        history["grad_norms"].append(
            np.mean(valid_norms) if valid_norms else float("nan")
        )

        run_val = (
            epoch % validate_every_n == 0
            or epoch == 1
            or epoch == config.training.num_epochs
        )

        if run_val:
            if hasattr(val_loader.dataset, "set_epoch"):
                val_loader.dataset.set_epoch(0)

            val_loss = validate_epoch(
                model,
                val_loader,
                criterion,
                prompts_template,
                config.training.device,
                use_amp=config.training.use_amp,
                snr_range=tuple(config.data.snr_range),
            )
            history["val_loss"].append(val_loss)
            print(f"Train: {train_loss:.4f}  Val: {val_loss:.4f}")

            # ReduceLROnPlateau: step with the validation loss so LR is
            # reduced whenever improvement stalls.  Only starts acting once
            # the warmup is complete to avoid fighting the ramp.
            if warmup_scheduler.done:
                plateau_scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"  LR after plateau step: {current_lr:.2e}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                ckpt_payload = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "warmup_scheduler_state_dict": warmup_scheduler.state_dict(),
                    "plateau_scheduler_state_dict": plateau_scheduler.state_dict(),
                    "val_loss": val_loss,
                    "config": config.to_dict(),
                    "history": history,
                    "coi_prompts": config.model.coi_prompts,
                    "bg_prompt": config.model.bg_prompt,
                    "all_prompts": all_prompts,
                }
                if scaler is not None:
                    ckpt_payload["scaler_state_dict"] = scaler.state_dict()
                torch.save(ckpt_payload, checkpoint_dir / "best_model.pt")
                print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= config.training.patience:
                print(f"\nEarly stopping after {epoch} epochs")
                break
        else:
            history["val_loss"].append(None)
            print(f"Train: {train_loss:.4f}")

    with open(checkpoint_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Convert TUSS loss to SNR for Optuna (approximate, loss is ~negative SNR-like metric)
    # TUSS uses snr_with_zeroref_loss which is a positive loss value (lower = better)
    # Negate it to get an approximate SNR for comparison
    best_val_snr = -best_val_loss
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best Val SNR: {best_val_snr:.2f} dB")


# =============================================================================
# Entry point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train the TUSS model for COI sound separation.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--device",
        "-d",
        type=str,
        default=None,
        metavar="DEVICE",
        help=(
            "Device to train on.  Overrides training_config.yaml.\n"
            "Examples:\n"
            "  --device cuda        (use default / best GPU)\n"
            "  --device cuda:0      (first GPU)\n"
            "  --device cuda:1      (second GPU)\n"
            "  --device cpu         (force CPU)"
        ),
    )
    device_group.add_argument(
        "--gpu",
        type=int,
        default=None,
        metavar="INDEX",
        help="GPU index shorthand.  --gpu 1 is equivalent to --device cuda:1.",
    )
    args = parser.parse_args()

    print(f"Loading config from {CONFIG_PATH}")
    config = Config.from_yaml(CONFIG_PATH)

    # Apply CLI device override (--gpu takes precedence when both somehow set,
    # but argparse enforces mutual exclusion so only one can be present).
    if args.gpu is not None:
        config.training.device = resolve_device(args.gpu)
    elif args.device is not None:
        config.training.device = resolve_device(args.device)
    else:
        # Validate / normalise whatever is in the YAML
        config.training.device = resolve_device(config.training.device)

    # Claim the target GPU as the process-default CUDA device immediately,
    # before any CUDA API call (including manual_seed_all and cudnn
    # benchmarking).  Without this, PyTorch lazily initialises a CUDA context
    # on cuda:0, leaving a resident allocation on that device for the entire
    # process lifetime even when training on a different GPU.
    if config.training.device.startswith("cuda:"):
        torch.cuda.set_device(int(config.training.device.split(":")[1]))

    print(f"Device:      {config.training.device}")
    print(f"COI prompts: {config.model.coi_prompts}")
    print(f"BG prompt:   {config.model.bg_prompt}")

    # ------------------------------------------------------------------ #
    # Check if using WebDataset mode                                       #
    # ------------------------------------------------------------------ #
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

        # Set dummy df_path (not used in WebDataset mode but needed for validation)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = Path(config.training.checkpoint_dir) / timestamp
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        config.data.df_path = str(checkpoint_dir / "webdataset_placeholder.csv")

        # Save config and start training
        train(config, timestamp=timestamp)
        return

    # ------------------------------------------------------------------ #
    # CSV-based mode: Load dataset metadata                               #
    # ------------------------------------------------------------------ #
    print("\nLoading dataset metadata …")
    project_root = _SCRIPT_DIR.parent.parent.parent  # code/
    datasets_path = str(project_root / "data")
    audio_base_path = str(project_root.parent / "datasets")

    all_metadata = load_metadata_datasets(datasets_path, audio_base_path)
    separation_metadata, _ = split_seperation_classification(all_metadata)
    print(
        f"Loaded {len(all_metadata)} total samples, "
        f"using {len(separation_metadata)} for separation"
    )

    # Normalise target_classes to a list-of-lists (one list per COI class)
    target_classes = config.data.target_classes
    if not target_classes:
        raise ValueError(
            "target_classes is empty – add class label lists to training_config.yaml"
        )

    # Support a flat list (single class) or a list of lists (multi-class)
    if isinstance(target_classes[0], str):
        target_classes = [target_classes]

    n_coi = len(config.model.coi_prompts)
    if len(target_classes) != n_coi:
        raise ValueError(
            f"target_classes has {len(target_classes)} groups but "
            f"coi_prompts has {n_coi} entries - they must match."
        )

    print(f"\nTarget classes ({n_coi} groups):")
    for i, (labels, prompt) in enumerate(zip(target_classes, config.model.coi_prompts)):
        print(f"  [{i}] {labels} -> prompt='{prompt}'")

    # Build one combined set of all COI labels for get_coi / sample_non_coi
    all_coi_labels = [lbl for group in target_classes for lbl in group]
    print("\nSampling dataset …")
    coi_df = get_coi(separation_metadata, all_coi_labels)
    sampled_df = sample_non_coi(separation_metadata, coi_df, coi_ratio=0.25)

    sampled_df["orig_label"] = sampled_df["label"]

    # Binary label: 1 if file belongs to any COI group
    def _is_coi(x):
        lbl_list = x if isinstance(x, list) else [x]
        return int(any(lbl in all_coi_labels for lbl in lbl_list))

    sampled_df["label"] = sampled_df["label"].apply(_is_coi)

    # coi_class: integer index into coi_prompts (NaN / -1 for non-COI files)
    def _coi_class(x):
        lbl_list = x if isinstance(x, list) else [x]
        for cls_idx, group in enumerate(target_classes):
            if any(lbl in group for lbl in lbl_list):
                return cls_idx
        return -1  # non-COI

    sampled_df["coi_class"] = sampled_df["orig_label"].apply(_coi_class)

    # Drop files that are missing on disk
    sampled_df["file_exists"] = sampled_df["filename"].apply(lambda f: Path(f).exists())
    n_missing = (~sampled_df["file_exists"]).sum()
    if n_missing:
        print(f"Dropping {n_missing} missing files")
        sampled_df = sampled_df[sampled_df["file_exists"]]
    sampled_df = sampled_df.drop(columns=["file_exists"])
    print(f"! Final dataset: {len(sampled_df)} samples")

    print("\nDataset splits:")
    for split in ["train", "val", "test"]:
        sdf = sampled_df[sampled_df["split"] == split]
        print(
            f"  {split}: {len(sdf)}  "
            f"(COI: {(sdf['label'] == 1).sum()}, non-COI: {(sdf['label'] == 0).sum()})"
        )
        for cls_idx, prompt in enumerate(config.model.coi_prompts):
            n = ((sdf["label"] == 1) & (sdf["coi_class"] == cls_idx)).sum()
            print(f"    class {cls_idx} '{prompt}': {n}")

    # Save dataset CSV alongside the checkpoint
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(config.training.checkpoint_dir) / timestamp
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    sampled_df["label"] = sampled_df["label"].astype("uint8")
    sampled_df["split"] = sampled_df["split"].astype("category")
    sampled_df["coi_class"] = sampled_df["coi_class"].astype("int16")
    df_path = checkpoint_dir / "separation_dataset.csv"
    sampled_df.to_csv(df_path, index=False)
    print(f"\nSaved dataset to: {df_path}")

    config.data.df_path = str(df_path)
    config.data.target_classes = target_classes  # normalised form

    del all_metadata, separation_metadata, coi_df, sampled_df
    gc.collect()

    train(config, timestamp=timestamp)


if __name__ == "__main__":
    main()
