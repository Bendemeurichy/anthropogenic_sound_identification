"""
Shared training utilities for COI (Class of Interest) separation models.

This module provides common components used across different separation models
(sudormrf, xumx, clapsep) for single-target + residue separation:
- AudioDataset: Loads COI and background audio, creates mixtures
- Loss functions: SI-SNR based losses
- Batch preparation: SNR-based mixing and normalization
- Audio augmentations

Expected dataframe structure:
    - 'filename': path to wav file
    - 'split': train/val/test
    - 'label': 1 for COI, 0 for background (non-COI)
    - 'coi_class' (optional): class index for multi-class COI
"""

from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset

# =============================================================================
# Loss Functions
# =============================================================================


def sisnr(est: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute scale-invariant SNR (dB) per example.

    Args:
        est: (B, T) or (B, 1, T) estimated signal
        target: (B, T) or (B, 1, T) target signal

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
    min_energy = 1e-6
    est_energy = est_zm.pow(2).sum(dim=-1)
    target_energy = target_zm.pow(2).sum(dim=-1)

    target_is_zero = target_energy < (min_energy * T)
    target_is_weak = target_energy < (1e-4 * T)
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
    silence_score = torch.clamp(
        -10.0 * torch.log10(est_energy / min_energy + eps), min=-30.0, max=0.0
    )
    sisnr_db = torch.where(target_is_zero, silence_score, sisnr_db)
    sisnr_db = torch.where(
        target_is_weak & ~target_is_zero, torch.clamp(sisnr_db, -20.0, 20.0), sisnr_db
    )

    return torch.clamp(sisnr_db, min=-30.0, max=30.0)


class COIWeightedLoss(nn.Module):
    """Fixed-order, class-of-interest weighted SI-SNR loss.

    Expects sources ordered as: [COI_0, ..., COI_n, background]
    where background is always the last channel.
    """

    def __init__(self, class_weight: float = 1.5, eps: float = 1e-8):
        """
        Args:
            class_weight: Weight for COI sources relative to background.
                         Higher values emphasize COI reconstruction.
            eps: Small value for numerical stability.
        """
        super().__init__()
        self.class_weight = float(class_weight)
        self.eps = float(eps)

    def forward(
        self, est_sources: torch.Tensor, target_sources: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            est_sources: (B, n_src, T) or (B, n_src, C, T) estimated sources
            target_sources: (B, n_src, T) or (B, n_src, C, T) target sources

        Returns:
            Scalar loss value (negative weighted SI-SNR mean)
        """
        # Handle 4D tensors (with channel dimension) by averaging over channels
        if est_sources.ndim == 4:
            est_sources = est_sources.mean(dim=2)
            target_sources = target_sources.mean(dim=2)

        if est_sources.ndim != 3 or target_sources.ndim != 3:
            raise ValueError("est_sources and target_sources must be (B, n_src, T)")

        n_src = est_sources.shape[1]
        n_coi = n_src - 1

        # COI SI-SNR (all heads except last)
        if n_coi > 0:
            coi_sisnrs = torch.stack(
                [
                    sisnr(est_sources[:, i, :], target_sources[:, i, :], eps=self.eps)
                    for i in range(n_coi)
                ],
                dim=0,
            ).mean(dim=0)
        else:
            coi_sisnrs = torch.zeros(est_sources.shape[0], device=est_sources.device)

        # Background SI-SNR (last head)
        bg_sisnr = sisnr(est_sources[:, -1, :], target_sources[:, -1, :], eps=self.eps)

        weighted = (self.class_weight * coi_sisnrs + bg_sisnr) / (
            self.class_weight + 1.0
        )
        return -weighted.mean()


# =============================================================================
# Audio Augmentations
# =============================================================================


class AudioAugmentations:
    """Audio augmentations for training data."""

    @staticmethod
    def time_stretch(
        wav: torch.Tensor, rate: float = 1.0, sample_rate: int = 16000
    ) -> torch.Tensor:
        """Apply time stretching."""
        if abs(rate - 1.0) < 0.01:
            return wav
        try:
            stretched = torchaudio.functional.speed(
                wav.unsqueeze(0), sample_rate, rate
            )[0]
            if stretched.shape[-1] < wav.shape[-1]:
                stretched = torch.nn.functional.pad(
                    stretched, (0, wav.shape[-1] - stretched.shape[-1])
                )
            return stretched[..., : wav.shape[-1]]
        except Exception:
            return wav

    @staticmethod
    def add_noise(wav: torch.Tensor, noise_level: float = 0.005) -> torch.Tensor:
        """Add Gaussian noise."""
        return wav + noise_level * torch.randn_like(wav)

    @staticmethod
    def gain(wav: torch.Tensor, gain_db: float = 0.0) -> torch.Tensor:
        """Apply gain in dB."""
        return wav * (10 ** (gain_db / 20))

    @staticmethod
    def time_shift(wav: torch.Tensor, shift_samples: int = 0) -> torch.Tensor:
        """Circular time shift."""
        return torch.roll(wav, shifts=shift_samples, dims=-1)

    @staticmethod
    def low_pass_filter(
        wav: torch.Tensor, cutoff_freq: float = 8000, sample_rate: int = 16000
    ) -> torch.Tensor:
        """Apply low-pass filter."""
        try:
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
                squeeze = True
            else:
                squeeze = False
            filtered = torchaudio.functional.lowpass_biquad(
                wav, sample_rate, cutoff_freq
            )
            return filtered.squeeze(0) if squeeze else filtered
        except Exception:
            return wav

    @staticmethod
    def random_augment(
        wav: torch.Tensor,
        rng: np.random.Generator,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """Apply random augmentation chain.

        Args:
            wav: Input waveform (T,) or (C, T)
            rng: NumPy random generator for reproducibility
            sample_rate: Audio sample rate

        Returns:
            Augmented waveform
        """
        if rng.random() < 0.3:
            rate = rng.uniform(0.9, 1.1)
            wav = AudioAugmentations.time_stretch(wav, rate, sample_rate)
        if rng.random() < 0.3:
            noise_level = rng.uniform(0.001, 0.01)
            wav = AudioAugmentations.add_noise(wav, noise_level)
        if rng.random() < 0.5:
            gain_db = rng.uniform(-6, 6)
            wav = AudioAugmentations.gain(wav, gain_db)
        if rng.random() < 0.3:
            shift = int(rng.uniform(-0.1, 0.1) * wav.shape[-1])
            wav = AudioAugmentations.time_shift(wav, shift)
        if rng.random() < 0.2:
            cutoff = rng.uniform(4000, 8000)
            wav = AudioAugmentations.low_pass_filter(wav, cutoff, sample_rate)
        return wav


# =============================================================================
# Dataset
# =============================================================================


class COIAudioDataset(Dataset):
    """Dataset for COI audio separation training.

    Loads COI (class of interest) and non-COI (background) audio files,
    creates mixtures for training separation models.

    Returns:
        tuple: (mixture, sources_tensor) where:
            - mixture: (C, T) or (T,) depending on stereo flag
            - sources_tensor: (n_src, C, T) or (n_src, T) with COI sources
              followed by background as the last channel
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        split: str = "train",
        sample_rate: int = 16000,
        segment_length: float = 5.0,
        snr_range: tuple = (-5, 5),
        n_coi_classes: int = 1,
        augment: bool = True,
        segment_stride: Optional[float] = None,
        background_only_prob: float = 0.0,
        background_mix_n: int = 2,
        augment_multiplier: int = 1,
        stereo: bool = False,
        seed: int = 42,
    ):
        """
        Args:
            dataframe: DataFrame with 'filename', 'split', 'label' columns
            split: Data split ('train', 'val', 'test')
            sample_rate: Target sample rate
            segment_length: Length of audio segments in seconds
            snr_range: (min, max) SNR range for mixing in dB
            n_coi_classes: Number of COI classes (1 for binary)
            augment: Whether to apply augmentations (only for train)
            segment_stride: Stride for segmentation (defaults to segment_length)
            background_only_prob: Probability of background-only samples
            background_mix_n: Number of backgrounds to mix for bg-only samples
            augment_multiplier: Each COI sample seen this many times per epoch
            stereo: Whether to output stereo (2-channel) audio
            seed: Random seed
        """
        self.split = split
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_length * sample_rate)
        self.snr_range = snr_range
        self.n_coi_classes = n_coi_classes
        self.augment = augment and split == "train"
        self.augment_multiplier = int(augment_multiplier) if self.augment else 1
        self.background_only_prob = max(0.0, float(background_only_prob))
        self.background_mix_n = int(background_mix_n)
        self.segment_stride = segment_stride or segment_length
        self.stereo = stereo

        self._rng = np.random.default_rng(seed)
        self._resamplers: dict[tuple[int, int], torchaudio.transforms.Resample] = {}

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

        # Background-only sample count
        self._extra_background_count = 0
        if split == "train" and self.coi_files and self.background_only_prob > 0.0:
            self._extra_background_count = int(
                self.background_only_prob
                * len(self.coi_files)
                * self.augment_multiplier
                + 0.5
            )

        # Precompute segments for validation/test, or for long files in train
        self.coi_segments = self._compute_segments(split_df)

        if self.split == "train":
            self.coi_segments_train_files = [seg[0] for seg in self.coi_segments]

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
                f"{split} set: {len(self.coi_files)} unique COI files ({len(self.coi_segments)} segments), {len(self.non_coi_files)} non-COI files"
            )
            if self.split == "train" and self.augment_multiplier > 1:
                print(
                    f"  → With {self.augment_multiplier}x augmentation: "
                    f"{len(self.coi_segments_train_files) * self.augment_multiplier} effective epoch steps"
                )

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
                    1, int(self.segment_stride * orig_sr / self.sample_rate)
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
            if self.coi_segments_train_files:
                return (
                    len(self.coi_segments_train_files) * self.augment_multiplier
                    + self._extra_background_count
                )
            return len(self.non_coi_files)
        return len(self.coi_segments)

    def _load_audio(
        self,
        filepath: str,
        frame_offset: int | None = None,
        num_frames: int | None = None,
    ) -> torch.Tensor:
        """Load and preprocess audio segment.

        Returns:
            Tensor of shape (C, T) if stereo else (T,)
        """
        try:
            info = torchaudio.info(filepath)
            orig_sr = info.sample_rate
            seg_frames = num_frames or max(
                1, int(self.segment_samples * orig_sr / self.sample_rate)
            )

            if frame_offset is None:
                max_offset = max(0, int(info.num_frames) - seg_frames)
                offset = (
                    int(np.random.randint(0, max_offset + 1)) if max_offset > 0 else 0
                )
            else:
                offset = int(frame_offset)

            waveform, sr = torchaudio.load(
                filepath, frame_offset=offset, num_frames=int(seg_frames)
            )
        except Exception:
            waveform, sr = torchaudio.load(filepath)
            orig_sr = sr

            seg_frames = num_frames or max(
                1, int(self.segment_samples * orig_sr / self.sample_rate)
            )
            num_actual_frames = waveform.shape[-1]

            if frame_offset is None:
                max_offset = max(0, num_actual_frames - seg_frames)
                offset = (
                    int(np.random.randint(0, max_offset + 1)) if max_offset > 0 else 0
                )
            else:
                offset = int(frame_offset)

            waveform = waveform[..., offset : offset + seg_frames]

        # Resample if needed
        if sr != self.sample_rate:
            key = (sr, self.sample_rate)
            if key not in self._resamplers:
                self._resamplers[key] = torchaudio.transforms.Resample(
                    sr, self.sample_rate
                )
            waveform = self._resamplers[key](waveform)

        # Handle channels
        if self.stereo:
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            elif waveform.shape[0] > 2:
                waveform = waveform[:2, :]
        else:
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.squeeze(0)

        # Pad/trim to segment length
        if waveform.shape[-1] < self.segment_samples:
            pad_size = self.segment_samples - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        waveform = waveform[..., : self.segment_samples]

        return waveform

    def _create_empty_source(self) -> torch.Tensor:
        """Create an empty (silent) source tensor."""
        if self.stereo:
            return torch.zeros(2, self.segment_samples)
        return torch.zeros(self.segment_samples)

    def __getitem__(self, idx):
        background = None

        if self.split == "train":
            coi_count = len(self.coi_segments_train_files)
            effective_coi_count = coi_count * self.augment_multiplier

            if coi_count > 0 and idx < effective_coi_count:
                # COI sample
                actual_idx = idx % coi_count
                augment_variant = idx // coi_count
                coi_file = self.coi_segments_train_files[actual_idx]
                class_idx = int(self.file_to_class.get(coi_file, 0))

                # Random offset for training
                coi_audio = self._load_audio(coi_file, frame_offset=None)

                if self.augment and augment_variant > 0:
                    coi_audio = AudioAugmentations.random_augment(
                        coi_audio, self._rng, self.sample_rate
                    )

                sources = (
                    [self._create_empty_source() for _ in range(self.n_coi_classes)]
                    if self.n_coi_classes > 1
                    else []
                )
                if self.n_coi_classes > 1:
                    sources[class_idx] = coi_audio
                else:
                    sources = [coi_audio]
            else:
                # Background-only sample
                idxs = np.random.choice(
                    len(self.non_coi_files), size=max(1, self.background_mix_n)
                )
                background = torch.stack(
                    [
                        self._load_audio(self.non_coi_files[int(i)], frame_offset=None)
                        for i in idxs
                    ]
                ).sum(dim=0)
                sources = [
                    self._create_empty_source() for _ in range(self.n_coi_classes)
                ]
        else:
            # Validation/Test
            filepath, frame_offset, num_frames, class_idx = self.coi_segments[idx]
            coi_audio = self._load_audio(filepath, frame_offset, num_frames)
            sources = (
                [self._create_empty_source() for _ in range(self.n_coi_classes)]
                if self.n_coi_classes > 1
                else [coi_audio]
            )
            if self.n_coi_classes > 1:
                sources[class_idx] = coi_audio

        # Add background
        if background is None:
            background = self._load_audio(
                self.non_coi_files[np.random.randint(len(self.non_coi_files))],
                frame_offset=None,
            )
        sources.append(background)

        # Stack: (n_src, ...) where ... is (C, T) or (T)
        sources_tensor = torch.stack(sources, dim=0)
        # Mixture: sum over sources
        mixture = sources_tensor.sum(dim=0)

        return mixture, sources_tensor


# =============================================================================
# Batch Preparation
# =============================================================================


def normalize_tensor_wav(
    wav: torch.Tensor, eps: float = 1e-8, min_std: float = 1e-4
) -> torch.Tensor:
    """Normalize waveform to zero mean and unit variance.

    Args:
        wav: Input waveform, last dimension is time
        eps: Small value for numerical stability
        min_std: Minimum std to consider non-silent

    Returns:
        Normalized waveform (silent signals returned as zeros)
    """
    mean = wav.mean(dim=-1, keepdim=True)
    std = wav.std(dim=-1, keepdim=True)
    is_silent = std < min_std
    std_safe = torch.where(is_silent, torch.ones_like(std), std) + eps
    normalized = (wav - mean) / std_safe
    return torch.where(is_silent, torch.zeros_like(normalized), normalized)


def prepare_batch_mono(
    sources: torch.Tensor,
    snr_range: tuple[float, float],
    deterministic: bool = False,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare mixture and targets from source tensor (mono version).

    Args:
        sources: (B, n_src, T) tensor with COI sources and background (last)
        snr_range: (min_snr, max_snr) in dB
        deterministic: If True, use linspace SNRs; otherwise random

    Returns:
        mixture: (B, T) normalized mixture
        clean_wavs: (B, n_src, T) independently normalized sources
    """
    B, n_src, T = sources.shape

    cois = [sources[:, i, :] for i in range(n_src - 1)]
    bg = sources[:, -1, :]
    total_coi = torch.stack(cois, dim=0).sum(dim=0) if cois else torch.zeros_like(bg)

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
    silent_coi = coi_power < 1e-8
    bg_scaling = torch.where(silent_coi, torch.ones_like(bg_scaling), bg_scaling)
    bg_scaling = torch.clamp(bg_scaling, min=0.1, max=3.0)

    bg_scaled = bg * bg_scaling
    mixture = normalize_tensor_wav(total_coi + bg_scaled, eps=eps, min_std=1e-3)

    # Normalize each source independently
    normalized_cois = [normalize_tensor_wav(c, eps=eps, min_std=1e-3) for c in cois]
    normalized_bg = normalize_tensor_wav(bg_scaled, eps=eps, min_std=1e-3)
    clean_wavs = torch.stack(normalized_cois + [normalized_bg], dim=1)

    return mixture, clean_wavs


def prepare_batch_stereo(
    sources: torch.Tensor,
    snr_range: tuple[float, float],
    deterministic: bool = False,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare mixture and targets from source tensor (stereo version).

    Args:
        sources: (B, n_src, C, T) tensor with COI sources and background (last)
        snr_range: (min_snr, max_snr) in dB
        deterministic: If True, use linspace SNRs; otherwise random

    Returns:
        mixture: (B, C, T) normalized mixture
        clean_wavs: (B, n_src, C, T) independently normalized sources
    """
    B, n_src, C, T = sources.shape

    cois = [sources[:, i, :, :] for i in range(n_src - 1)]
    bg = sources[:, -1, :, :]
    total_coi = torch.stack(cois, dim=0).sum(dim=0) if cois else torch.zeros_like(bg)

    # SNR calculation - average power across channels
    if deterministic and B > 1:
        snr_db = torch.linspace(
            snr_range[0], snr_range[1], B, device=sources.device
        ).view(B, 1, 1)
    else:
        snr_db = torch.zeros(B, 1, 1, device=sources.device).uniform_(*snr_range)

    coi_power = total_coi.pow(2).mean(dim=(-2, -1), keepdim=True) + eps
    bg_power = bg.pow(2).mean(dim=(-2, -1), keepdim=True) + eps
    snr_linear = torch.pow(10.0, snr_db / 10.0)
    bg_scaling = torch.sqrt(coi_power / (bg_power * snr_linear + eps))

    silent_coi = coi_power < 1e-8
    bg_scaling = torch.where(silent_coi, torch.ones_like(bg_scaling), bg_scaling)
    bg_scaling = torch.clamp(bg_scaling, min=0.1, max=3.0)

    bg_scaled = bg * bg_scaling
    mixture = normalize_tensor_wav(total_coi + bg_scaled, eps=eps, min_std=1e-3)

    normalized_cois = [normalize_tensor_wav(c, eps=eps, min_std=1e-3) for c in cois]
    normalized_bg = normalize_tensor_wav(bg_scaled, eps=eps, min_std=1e-3)
    clean_wavs = torch.stack(normalized_cois + [normalized_bg], dim=1)

    return mixture, clean_wavs


def prepare_batch(
    sources: torch.Tensor,
    snr_range: tuple[float, float],
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare mixture and targets from source tensor (auto-detects mono/stereo).

    Args:
        sources: (B, n_src, T) or (B, n_src, C, T) tensor
        snr_range: (min_snr, max_snr) in dB
        deterministic: If True, use linspace SNRs

    Returns:
        mixture: (B, T) or (B, C, T) normalized mixture
        clean_wavs: (B, n_src, T) or (B, n_src, C, T) normalized sources
    """
    if sources.ndim == 3:
        return prepare_batch_mono(sources, snr_range, deterministic)
    elif sources.ndim == 4:
        return prepare_batch_stereo(sources, snr_range, deterministic)
    else:
        raise ValueError(f"Expected 3D or 4D sources tensor, got {sources.ndim}D")


def check_finite(*tensors) -> bool:
    """Check if all tensors contain finite values."""
    return all(torch.isfinite(t).all() for t in tensors)


# =============================================================================
# DataLoader Utilities
# =============================================================================


def create_coi_dataloader(
    df_path: str,
    split: str,
    batch_size: int,
    sample_rate: int = 16000,
    segment_length: float = 5.0,
    snr_range: tuple = (-5, 5),
    n_coi_classes: int = 1,
    background_only_prob: float = 0.0,
    background_mix_n: int = 2,
    augment_multiplier: int = 1,
    stereo: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    seed: int = 42,
) -> tuple:
    """Create dataloader for COI separation training.

    Args:
        df_path: Path to CSV with 'filename', 'split', 'label' columns
        split: Data split ('train', 'val', 'test')
        batch_size: Batch size
        sample_rate: Target sample rate
        segment_length: Segment length in seconds
        snr_range: SNR range for mixing
        n_coi_classes: Number of COI classes
        background_only_prob: Probability of background-only samples
        background_mix_n: Backgrounds to mix for bg-only samples
        augment_multiplier: Augmentation multiplier
        stereo: Whether to use stereo audio
        num_workers: DataLoader workers
        pin_memory: Whether to pin memory
        seed: Random seed

    Returns:
        tuple: (DataLoader, COIAudioDataset)
    """
    import gc

    from torch.utils.data import DataLoader

    usecols = ["filename", "label", "split"]
    if n_coi_classes > 1:
        usecols.append("coi_class")

    df = pd.read_csv(df_path, usecols=usecols)
    df["label"] = df["label"].astype("uint8")
    df["split"] = df["split"].astype("category")
    if "coi_class" in df.columns:
        df["coi_class"] = df["coi_class"].astype("category")

    dataset = COIAudioDataset(
        df,
        split=split,
        sample_rate=sample_rate,
        segment_length=segment_length,
        snr_range=snr_range,
        n_coi_classes=n_coi_classes,
        augment=(split == "train"),
        background_only_prob=background_only_prob if split == "train" else 0.0,
        background_mix_n=background_mix_n,
        augment_multiplier=augment_multiplier,
        stereo=stereo,
        seed=seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        drop_last=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=(num_workers > 0 and split == "train"),
    )

    del df
    gc.collect()
    return loader, dataset
