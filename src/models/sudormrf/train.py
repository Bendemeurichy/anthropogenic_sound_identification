"""
Training script for sudormrf model with custom separation head and loss function.
Expected dataframe structure:
    - 'filename': path to wav file
    - 'split': train/val/test
    - 'label': 1 for aircraft (COI), 0 for background (non-COI)
"""

import argparse
import gc
import json
import math
import os
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

# Pin to single GPU before importing torch (prevents multi-GPU OOM issues)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from base.sudo_rm_rf.dnn.models.groupcomm_sudormrf_v2 import GroupCommSudoRmRf
from base.sudo_rm_rf.dnn.models.improved_sudormrf import SuDORMRF

# Check for environment variable to use old separation head
USE_OLD_SEPARATION_HEAD = os.environ.get("USE_OLD_SEPARATION_HEAD", "0") == "1"
if USE_OLD_SEPARATION_HEAD:
    from seperation_head_old import wrap_model_for_coi

    _COI_HEAD_INDEX, _BACKGROUND_HEAD_INDEX = 0, 1
else:
    from seperation_head import (
        BACKGROUND_HEAD_INDEX as _BACKGROUND_HEAD_INDEX,
    )
    from seperation_head import (
        COI_HEAD_INDEX as _COI_HEAD_INDEX,
    )
    from seperation_head import (
        wrap_model_for_coi,
    )

COI_HEAD_INDEX: int = _COI_HEAD_INDEX
BACKGROUND_HEAD_INDEX: int = _BACKGROUND_HEAD_INDEX

from config import Config
from multi_class_seperation import wrap_model_for_multiclass

from label_loading.metadata_loader import (
    load_metadata_datasets,
    split_seperation_classification,
)
from label_loading.sampler import get_coi, sample_non_coi

LOSS_EPS = 1e-8
ENERGY_EPS = 1e-8
NORMALIZE_MIN_STD = 1e-3
SILENCE_ENERGY_EPS = 1e-6
WEAK_TARGET_ENERGY_EPS = 1e-4
BG_SCALE_MIN = 0.1
BG_SCALE_MAX = 3.0
RESAMPLER_CACHE_MAX = 8


# =============================================================================
# Loss Functions
# =============================================================================


def sisnr(est: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute scale-invariant SNR (dB) per example.

    Args:
        est: (B, T) or (B, 1, T)
        target: (B, T) or (B, 1, T)
    Returns:
        sisnr_db: (B,) tensor of SI-SNR in dB, clamped to [-30, 30]
    """
    est, target = est.float(), target.float()
    if est.ndim == 3:
        est = est.squeeze(1)
    if target.ndim == 3:
        target = target.squeeze(1)

    # Zero-mean
    est_zm = est - est.mean(dim=-1, keepdim=True)
    target_zm = target - target.mean(dim=-1, keepdim=True)

    # Energy calculations
    T = target.shape[-1]
    min_energy = SILENCE_ENERGY_EPS
    est_energy = est_zm.pow(2).sum(dim=-1)
    target_energy = target_zm.pow(2).sum(dim=-1)

    target_is_zero = target_energy < (min_energy * T)
    target_is_weak = target_energy < (WEAK_TARGET_ENERGY_EPS * T)
    target_energy_safe = torch.clamp(target_energy, min=min_energy)

    # Projection
    s_target = (est_zm * target_zm).sum(dim=-1, keepdim=True) / (
        target_energy_safe.unsqueeze(-1) + eps
    )
    s_target = torch.clamp(s_target, min=-100.0, max=100.0)
    s_true = s_target * target_zm
    e_noise = est_zm - s_true

    # SI-SNR
    sisnr_lin = torch.clamp(
        s_true.pow(2).sum(dim=-1) / (e_noise.pow(2).sum(dim=-1) + eps),
        min=1e-10,
        max=1e10,
    )
    sisnr_db = 10.0 * torch.log10(sisnr_lin + eps)

    # Handle silent/weak targets
    # For truly silent targets, reward low output energy with scores up to 12 dB
    # This ensures background-only samples can achieve comparable loss to COI samples
    # Cap at 12 dB to stay balanced with class_weight=1.2 and typical COI SI-SNR (~10-12 dB)
    silence_score = torch.clamp(
        -10.0 * torch.log10(est_energy / min_energy + eps), min=-30.0, max=12.0
    )
    sisnr_db = torch.where(target_is_zero, silence_score, sisnr_db)
    sisnr_db = torch.where(
        target_is_weak & ~target_is_zero, torch.clamp(sisnr_db, -20.0, 20.0), sisnr_db
    )

    return torch.clamp(sisnr_db, min=-30.0, max=30.0)


class COIWeightedLoss(torch.nn.Module):
    """Fixed-order, class-of-interest weighted SI-SNR loss."""

    def __init__(self, class_weight: float = 1.5, eps: float = 1e-8):
        super().__init__()
        self.class_weight = float(class_weight)
        self.eps = float(eps)

    def forward(
        self, est_sources: torch.Tensor, target_sources: torch.Tensor
    ) -> torch.Tensor:
        if est_sources.ndim != 3 or target_sources.ndim != 3:
            raise ValueError("est_sources and target_sources must be (B, n_src, T)")

        # COI SI-SNR (using head index constants)
        coi_sisnr = sisnr(
            est_sources[:, COI_HEAD_INDEX, :],
            target_sources[:, COI_HEAD_INDEX, :],
            eps=self.eps,
        )

        # Background SI-SNR (using head index constants)
        bg_sisnr = sisnr(
            est_sources[:, BACKGROUND_HEAD_INDEX, :],
            target_sources[:, BACKGROUND_HEAD_INDEX, :],
            eps=self.eps,
        )

        weighted = (self.class_weight * coi_sisnr + bg_sisnr) / (
            self.class_weight + 1.0
        )
        return -weighted.mean()


# =============================================================================
# Audio Augmentations
# =============================================================================


class AudioAugmentations:
    """Audio augmentations for training data."""

    @staticmethod
    def time_stretch(waveform: torch.Tensor, rate: float) -> torch.Tensor:
        if rate == 1.0:
            return waveform
        orig_len = waveform.shape[-1]
        stretched = (
            torch.nn.functional.interpolate(
                waveform.unsqueeze(0).unsqueeze(0),
                scale_factor=1.0 / rate,
                mode="linear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )
        if stretched.shape[-1] > orig_len:
            return stretched[:orig_len]
        return torch.nn.functional.pad(stretched, (0, orig_len - stretched.shape[-1]))

    @staticmethod
    def add_noise(waveform: torch.Tensor, noise_level: float = 0.005) -> torch.Tensor:
        return waveform + torch.randn_like(waveform) * noise_level

    @staticmethod
    def gain(waveform: torch.Tensor, gain_db: float) -> torch.Tensor:
        return waveform * (10 ** (gain_db / 20.0))

    @staticmethod
    def time_shift(waveform: torch.Tensor, shift_samples: int) -> torch.Tensor:
        return torch.roll(waveform, shifts=shift_samples, dims=-1)

    @staticmethod
    def low_pass_filter(
        waveform: torch.Tensor, cutoff_ratio: float = 0.8
    ) -> torch.Tensor:
        if cutoff_ratio >= 1.0:
            return waveform
        fft = torch.fft.rfft(waveform)
        n_freqs = fft.shape[-1]
        cutoff_idx = int(n_freqs * cutoff_ratio)
        mask = torch.ones(n_freqs, device=waveform.device)
        rolloff_width = max(1, n_freqs // 20)
        for i in range(rolloff_width):
            if cutoff_idx + i < n_freqs:
                mask[cutoff_idx + i] = 1.0 - (i / rolloff_width)
        mask[cutoff_idx + rolloff_width :] = 0.0
        return torch.fft.irfft(fft * mask, n=waveform.shape[-1])

    @staticmethod
    def random_augment(
        waveform: torch.Tensor, rng: np.random.Generator | None = None
    ) -> torch.Tensor:
        if rng is None:
            rng = np.random.default_rng()
        augmented = waveform.clone()

        if rng.random() < 0.5:
            augmented = AudioAugmentations.time_stretch(
                augmented, rng.uniform(0.9, 1.1)
            )
        if rng.random() < 0.7:
            augmented = AudioAugmentations.gain(augmented, rng.uniform(-6, 6))
        if rng.random() < 0.4:
            augmented = AudioAugmentations.add_noise(
                augmented, rng.uniform(0.001, 0.01)
            )
        if rng.random() < 0.5:
            max_shift = int(augmented.shape[-1] * 0.1)
            augmented = AudioAugmentations.time_shift(
                augmented, int(rng.integers(-max_shift, max_shift + 1))
            )
        if rng.random() < 0.3:
            augmented = AudioAugmentations.low_pass_filter(
                augmented, rng.uniform(0.6, 0.95)
            )

        return augmented


# =============================================================================
# Dataset
# =============================================================================


class AudioDataset(Dataset):
    """Dataset for audio separation training."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        split: str = "train",
        sample_rate: int = 16000,
        segment_length: float = 5.0,
        snr_range: tuple = (-5, 5),
        n_coi_classes: int = 1,
        augment: bool = True,
        segment_stride: float | None = None,
        background_only_prob: float = 0.0,
        background_mix_n: int = 2,
        augment_multiplier: int = 1,
    ):
        self.split = split
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_length * sample_rate)
        self.snr_range = snr_range
        self.n_coi_classes = n_coi_classes
        self.augment = augment and split == "train"
        self.augment_multiplier = int(augment_multiplier) if self.augment else 1
        self.background_only_prob = max(0.0, float(background_only_prob))
        self.background_mix_n = int(background_mix_n)
        self.segment_stride_samples = int(
            (segment_stride or segment_length) * sample_rate
        )

        self._rng = np.random.default_rng(42)
        self._resamplers: dict[tuple[int, int], torchaudio.transforms.Resample] = {}
        self._resampler_cache_max = int(RESAMPLER_CACHE_MAX)

        # Filter and extract file lists
        if split == "test":
            split_df = dataframe.iloc[0:0]
        else:
            split_df = dataframe[dataframe["split"] == split]

        coi_mask = split_df["label"] == 1
        self.coi_files = split_df.loc[coi_mask, "filename"].tolist()
        self.non_coi_files = split_df.loc[~coi_mask, "filename"].tolist()

        # File to class mapping for multi-class
        self.file_to_class = {}
        if n_coi_classes > 1 and "coi_class" in split_df.columns:
            self.file_to_class = dict(zip(split_df["filename"], split_df["coi_class"]))

        # Precompute segments for validation/test
        self.coi_segments = self._compute_segments(split_df)

        # Background-only sample count (for both train and val)
        self._extra_background_count = 0
        if self.coi_files and self.background_only_prob > 0.0:
            if split == "train":
                self._extra_background_count = int(
                    self.background_only_prob
                    * len(self.coi_files)
                    * self.augment_multiplier
                    + 0.5
                )
            elif split == "val":
                # Proportional to COI segments so validation covers silence suppression
                self._extra_background_count = int(
                    self.background_only_prob * len(self.coi_segments) + 0.5
                )

        # Print stats
        if n_coi_classes > 1 and "coi_class" in split_df.columns:
            coi_by_class = [
                len(split_df[(split_df["label"] == 1) & (split_df["coi_class"] == i)])
                for i in range(n_coi_classes)
            ]
            print(
                f"{split} set: {coi_by_class} per class, {len(self.non_coi_files)} non-COI"
            )
        else:
            print(
                f"{split} set: {len(self.coi_files)} COI, {len(self.non_coi_files)} non-COI"
            )
            if self.augment_multiplier > 1:
                print(
                    f"  → With {self.augment_multiplier}x augmentation: {len(self.coi_files) * self.augment_multiplier} effective samples"
                )
            if self._extra_background_count > 0:
                print(
                    f"  → With {self._extra_background_count} background-only samples"
                )

    def set_epoch(self, epoch: int):
        """Re-seed the internal RNG for a new epoch to ensure augmentation diversity."""
        self._rng = np.random.default_rng(42 + epoch)

    def _compute_segments(
        self, split_df: pd.DataFrame
    ) -> list[tuple[str, int, int | None, int]]:
        """Compute segments for each COI file."""
        segments = []
        for filepath in self.coi_files:
            class_idx = int(self.file_to_class.get(filepath, 0))
            try:
                info = torchaudio.info(filepath)
                orig_sr = info.sample_rate
                num_frames = int(info.num_frames)
                seg_frames = max(
                    1, int(self.segment_samples * orig_sr / self.sample_rate)
                )
                stride_frames = max(
                    1, int(self.segment_stride_samples * orig_sr / self.sample_rate)
                )

                n_segs = (
                    1
                    if num_frames <= seg_frames
                    else 1 + max(0, (num_frames - seg_frames) // stride_frames)
                )
                for s in range(n_segs):
                    segments.append(
                        (filepath, s * stride_frames, seg_frames, class_idx)
                    )
            except Exception:
                segments.append((filepath, 0, None, class_idx))
        return segments

    def __len__(self):
        if self.split == "train":
            if self.coi_files:
                return (
                    len(self.coi_files) * self.augment_multiplier
                    + self._extra_background_count
                )
            return len(self.non_coi_files)
        return len(self.coi_segments) + self._extra_background_count

    def _load_audio(
        self, filepath: str, frame_offset: int = 0, num_frames: int | None = None
    ) -> torch.Tensor:
        """Load and preprocess audio segment."""
        try:
            info = torchaudio.info(filepath)
            orig_sr = info.sample_rate
            seg_frames = num_frames or max(
                1, int(self.segment_samples * orig_sr / self.sample_rate)
            )
            waveform, sr = torchaudio.load(
                filepath, frame_offset=int(frame_offset), num_frames=int(seg_frames)
            )
        except Exception:
            waveform, sr = torchaudio.load(filepath)
            orig_sr = sr

        # Resample if needed
        if sr != self.sample_rate:
            key = (sr, self.sample_rate)
            if key not in self._resamplers:
                if len(self._resamplers) >= self._resampler_cache_max:
                    self._resamplers.pop(next(iter(self._resamplers)))
                self._resamplers[key] = torchaudio.transforms.Resample(
                    sr, self.sample_rate
                )
            waveform = self._resamplers[key](waveform)

        # Mono and pad/trim
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        if waveform.shape[0] < self.segment_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.segment_samples - waveform.shape[0])
            )
        return waveform[: self.segment_samples]

    def __getitem__(self, idx):
        background = None

        if self.split == "train":
            coi_count = len(self.coi_files)
            effective_coi_count = coi_count * self.augment_multiplier

            if coi_count > 0 and idx < effective_coi_count:
                # COI sample
                actual_idx = idx % coi_count
                augment_variant = idx // coi_count
                coi_file = self.coi_files[actual_idx]
                class_idx = int(self.file_to_class.get(coi_file, 0))

                # Random offset for training
                try:
                    info = torchaudio.info(coi_file)
                    seg_frames = max(
                        1,
                        int(self.segment_samples * info.sample_rate / self.sample_rate),
                    )
                    max_offset = max(0, int(info.num_frames) - seg_frames)
                    frame_offset = (
                        int(self._rng.integers(0, max_offset + 1))
                        if max_offset > 0
                        else 0
                    )
                    coi_audio = self._load_audio(coi_file, frame_offset, seg_frames)
                except Exception:
                    coi_audio = self._load_audio(coi_file)

                if self.augment and augment_variant > 0:
                    coi_audio = AudioAugmentations.random_augment(coi_audio, self._rng)

                sources = (
                    [torch.zeros_like(coi_audio) for _ in range(self.n_coi_classes)]
                    if self.n_coi_classes > 1
                    else []
                )
                if self.n_coi_classes > 1:
                    sources[class_idx] = coi_audio
                else:
                    sources = [coi_audio]
            else:
                # Background-only sample - create mixture of multiple background sources
                idxs = self._rng.choice(
                    len(self.non_coi_files),
                    size=max(1, self.background_mix_n),
                    replace=True,
                )
                background_sources = [
                    self._load_audio(self.non_coi_files[int(i)]) for i in idxs
                ]
                background = torch.stack(background_sources).sum(dim=0)
                # COI sources are all zeros (no aircraft present)
                sources = [
                    torch.zeros_like(background) for _ in range(self.n_coi_classes)
                ]
        else:
            # Validation/Test
            if idx < len(self.coi_segments):
                filepath, frame_offset, num_frames, class_idx = self.coi_segments[idx]
                coi_audio = self._load_audio(filepath, frame_offset, num_frames)
                sources = (
                    [torch.zeros_like(coi_audio) for _ in range(self.n_coi_classes)]
                    if self.n_coi_classes > 1
                    else [coi_audio]
                )
                if self.n_coi_classes > 1:
                    sources[class_idx] = coi_audio
            else:
                # Background-only validation sample
                idxs = self._rng.choice(
                    len(self.non_coi_files),
                    size=max(1, self.background_mix_n),
                    replace=True,
                )
                background = torch.stack(
                    [self._load_audio(self.non_coi_files[int(i)]) for i in idxs]
                ).sum(dim=0)
                sources = [
                    torch.zeros_like(background) for _ in range(self.n_coi_classes)
                ]

        # Add background
        if background is None:
            background = self._load_audio(
                self.non_coi_files[int(self._rng.integers(len(self.non_coi_files)))]
            )
        sources.append(background)

        sources_tensor = torch.stack(sources, dim=0)
        return sources_tensor


# =============================================================================
# Training Utilities
# =============================================================================


def normalize_tensor_wav(
    wav: torch.Tensor, eps: float = ENERGY_EPS, min_std: float = NORMALIZE_MIN_STD
) -> torch.Tensor:
    """Normalize waveform to zero mean and unit variance."""
    mean = wav.mean(dim=-1, keepdim=True)
    std = wav.std(dim=-1, keepdim=True)
    is_silent = std < min_std
    std_safe = torch.where(is_silent, torch.ones_like(std), std) + eps
    normalized = (wav - mean) / std_safe
    return torch.where(is_silent, torch.zeros_like(normalized), normalized)


def prepare_batch(
    sources: torch.Tensor, snr_range: tuple[float, float], deterministic: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare mixture and clean targets from source tensor.

    Args:
        sources: (B, n_src, T) tensor with COI sources and background (last channel)
        snr_range: (min_snr, max_snr) in dB
        deterministic: If True, use linspace SNRs; otherwise random

    Returns:
        mixture: (B, T) normalized mixture
        clean_wavs: (B, n_src, T) independently normalized sources
    """
    B, n_src, T = sources.shape
    eps = ENERGY_EPS

    cois = [sources[:, i, :] for i in range(n_src - 1)]
    bg = sources[:, -1, :]
    total_coi = torch.stack(cois, dim=0).sum(dim=0)

    # SNR calculation
    if deterministic and B > 1:
        snr_db = torch.linspace(
            snr_range[0], snr_range[1], B, device=sources.device
        ).view(B, 1)
    else:
        snr_db = torch.zeros(B, 1, device=sources.device).uniform_(*snr_range)

    coi_power = total_coi.pow(2).mean(dim=-1, keepdim=True) + eps
    bg_power = bg.pow(2).mean(dim=-1, keepdim=True) + eps
    snr_linear = torch.pow(10.0, snr_db / 10.0)
    bg_scaling = torch.sqrt(coi_power / (bg_power * snr_linear + eps))

    # Don't scale if COI is silent
    silent_coi = coi_power < SILENCE_ENERGY_EPS
    bg_scaling = torch.where(silent_coi, torch.ones_like(bg_scaling), bg_scaling)
    bg_scaling = torch.clamp(bg_scaling, min=BG_SCALE_MIN, max=BG_SCALE_MAX)

    bg_scaled = bg * bg_scaling
    mixture = normalize_tensor_wav(
        total_coi + bg_scaled, eps=eps, min_std=NORMALIZE_MIN_STD
    )

    # Normalize each source independently
    normalized_cois = [
        normalize_tensor_wav(c, eps=eps, min_std=NORMALIZE_MIN_STD) for c in cois
    ]
    normalized_bg = normalize_tensor_wav(bg_scaled, eps=eps, min_std=NORMALIZE_MIN_STD)
    clean_wavs = torch.stack(normalized_cois + [normalized_bg], dim=1)

    return mixture, clean_wavs


def check_finite(*tensors) -> bool:
    """Check if all tensors contain finite values."""
    return all(torch.isfinite(t).all() for t in tensors)


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
    scaler: torch.amp.GradScaler | None = None,
    scheduler: optim.lr_scheduler._LRScheduler | None = None,
) -> tuple[float, int, list[float]]:
    """Train for one epoch."""
    model.train()
    running_loss, n_samples = 0.0, 0
    grad_norms: list[float] = []

    use_amp = use_amp and str(device).startswith("cuda")
    if scaler is None and use_amp:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.float16, enabled=True)
        if use_amp
        else nullcontext()
    )

    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    has_pending_grads = False

    pbar = tqdm(dataloader, desc="Training", leave=False, ascii=True, ncols=100)
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
                        optimizer.zero_grad(set_to_none=True)
                        grad_norms.append(float("nan"))
                        torch.cuda.empty_cache()
                        continue
                else:
                    # No retry available without AMP; skip this batch
                    del mixture, clean_wavs
                    optimizer.zero_grad(set_to_none=True)
                    grad_norms.append(float("nan"))
                    continue

            loss = criterion(outputs.float(), clean_wavs.float())

        del outputs  # Graph retained via loss; free the direct reference
        del mixture, clean_wavs

        if not check_finite(loss):
            del loss
            optimizer.zero_grad(set_to_none=True)
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
    all_coi_sisnr, all_bg_sisnr = [], []

    use_amp = use_amp and str(device).startswith("cuda")
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.float16, enabled=True)
        if use_amp
        else nullcontext()
    )

    pbar = tqdm(dataloader, desc="Validation", leave=False, ascii=True, ncols=100)
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

            # SI-SNR metrics
            try:
                n_src = outputs.shape[1]
                coi_sisnr = torch.stack(
                    [sisnr(outputs[:, i], clean_wavs[:, i]) for i in range(n_src - 1)]
                ).mean()
                bg_sisnr = sisnr(outputs[:, -1], clean_wavs[:, -1]).mean()
                all_coi_sisnr.append(float(coi_sisnr.item()))
                all_bg_sisnr.append(float(bg_sisnr.item()))
                pbar.set_postfix(
                    loss=f"{batch_loss:.4f}",
                    coi=f"{coi_sisnr.item():.2f}",
                    bg=f"{bg_sisnr.item():.2f}",
                )
            except Exception:
                pbar.set_postfix(loss=f"{batch_loss:.4f}")

            del outputs, loss, mixture, clean_wavs

    if all_coi_sisnr and all_bg_sisnr:
        print(
            f"  Val SI-SNR - COI: {np.mean(all_coi_sisnr):.2f} dB, BG: {np.mean(all_bg_sisnr):.2f} dB"
        )

    return running_loss / max(n_samples, 1)


# =============================================================================
# Data Loading and Model Creation
# =============================================================================


def create_dataloader(config: Config, split: str) -> tuple[DataLoader, AudioDataset]:
    """Create dataloader for specified split."""
    usecols = ["filename", "label", "split"]
    if getattr(config.data, "n_coi_classes", 1) > 1:
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
            if split in ("train")
            else 0.0
        ),
        background_mix_n=getattr(config.data, "background_mix_n", 2),
        augment_multiplier=getattr(config.data, "augment_multiplier", 1),
    )

    num_workers = int(getattr(config.training, "num_workers", 0))
    pin_memory = (
        getattr(config.training, "pin_memory", False) and torch.cuda.is_available()
    )

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
            f"Wrapping model for Multi-class separation with {config.data.n_coi_classes} classes."
        )
        model = wrap_model_for_multiclass(
            base_model, n_coi_classes=config.data.n_coi_classes
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


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


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
        model.parameters(), lr=base_lr, weight_decay=config.training.weight_decay
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
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    validate_every_n = int(getattr(config.training, "validate_every_n_epochs", 1))

    # Training loop
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    global_step = 0
    history = {"train_loss": [], "val_loss": [], "grad_norms": []}

    for epoch in range(1, config.training.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.training.num_epochs}")

        # Re-seed dataset RNG for augmentation diversity each epoch
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)

        train_loss, global_step, epoch_grad_norms = train_epoch(
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

    # Save history
    with open(checkpoint_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining completed! Best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train aircraft sound separation")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    args = parser.parse_args()

    config = Config.from_yaml(Path(args.config))
    print("Configuration:")
    print(f"  Model: {config.model.type} ({config.model.num_blocks} blocks)")
    print(f"  Device: {config.training.device}")

    # Load dataset metadata
    print("\nLoading dataset metadata...")
    project_root = Path(__file__).parent.parent.parent.parent
    datasets_path = str(project_root / "data")
    audio_base_path = str(project_root.parent / "datasets")

    all_metadata = load_metadata_datasets(datasets_path, audio_base_path)
    separation_metadata, _ = split_seperation_classification(all_metadata)

    print(f"Loaded {len(all_metadata)} total samples")
    print(f"Using {len(separation_metadata)} for separation training (70%)")

    # Target classes are now specified in the configuration YAML.  Read them
    # from the loaded config object rather than hard-coding them here.
    target_classes = getattr(config.data, "target_classes", None)
    if not target_classes:
        raise ValueError(
            "No target_classes specified in config.data – please add a list of "
            "labels to the YAML configuration"
        )
    print(f"\nTarget classes: {target_classes}")

    # Sample data
    print("\nSampling data...")
    coi_df = get_coi(separation_metadata, target_classes)
    sampled_df = sample_non_coi(separation_metadata, coi_df, coi_ratio=0.25)

    # Binary labels
    sampled_df["label"] = sampled_df["label"].apply(
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
    del all_metadata, separation_metadata, coi_df, sampled_df
    gc.collect()

    train(config, timestamp=timestamp)


if __name__ == "__main__":
    main()
