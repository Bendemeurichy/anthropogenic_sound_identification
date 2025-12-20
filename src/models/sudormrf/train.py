"""
Training script for sudormrf model with custom serperation head and loss function.
- Uses PITLossWrapper for automatic permutation handling
Expected dataframe structure:
    - 'filename': path to wav file
    - 'split': train/val/test
    - 'label': 1 for aircraft (COI), 0 for background (non-COI)
"""

import os
import gc
import json
from datetime import datetime

# Pin to single GPU before importing torch (prevents multi-GPU OOM issues)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import torch

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from contextlib import nullcontext

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .base.sudo_rm_rf.dnn.models.improved_sudormrf import SuDORMRF
from base.sudo_rm_rf.dnn.models.groupcomm_sudormrf_v2 import (
    GroupCommSudoRmRf,
)
from .seperation_head import wrap_model_for_coi
from .multi_class_seperation import wrap_model_for_multiclass
from .config import Config


from label_loading.sampler import get_coi, sample_non_coi
from label_loading.metadata_loader import (
    load_metadata_datasets,
    split_seperation_classification,
)

# Small epsilon added to losses to avoid exact-zero divisions
LOSS_EPS = 1e-8


def sisnr(est: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute scale-invariant SNR (dB) per example.

    Args:
        est: (B, T) or (B, 1, T)
        target: (B, T) or (B, 1, T)
    Returns:
        sisnr_db: (B,) tensor of SI-SNR in dB
    """
    # ensure shape (B, T)
    if est.ndim == 3:
        est = est.squeeze(1)
    if target.ndim == 3:
        target = target.squeeze(1)

    # zero-mean
    est_zm = est - est.mean(dim=-1, keepdim=True)
    target_zm = target - target.mean(dim=-1, keepdim=True)

    # Detect silent targets and estimates (prevent numerical instability)
    target_energy = target_zm.pow(2).sum(dim=-1)
    est_energy = est_zm.pow(2).sum(dim=-1)
    min_energy_threshold = 1e-6
    silent_target_mask = target_energy < min_energy_threshold
    silent_est_mask = est_energy < min_energy_threshold

    # projection of est onto target
    s_target = (est_zm * target_zm).sum(dim=-1, keepdim=True) / (
        target_zm.pow(2).sum(dim=-1, keepdim=True) + eps
    )
    s_true = s_target * target_zm
    e_noise = est_zm - s_true

    # energies
    true_energy = s_true.pow(2).sum(dim=-1)
    noise_energy = e_noise.pow(2).sum(dim=-1) + eps

    sisnr_lin = true_energy / noise_energy
    sisnr_db = 10.0 * torch.log10(sisnr_lin + eps)

    # Handle silent targets:
    # - If target is silent AND estimate is silent: REWARD (correct behavior) -> +20 dB
    # - If target is silent BUT estimate has energy: PENALIZE based on estimate energy
    # This trains the model to output silence when there's no COI present
    both_silent = silent_target_mask & silent_est_mask
    target_silent_est_loud = silent_target_mask & ~silent_est_mask

    # Reward for correctly predicting silence (bounded positive SI-SNR)
    silence_reward = torch.full_like(sisnr_db, 20.0)

    # Penalty proportional to estimate energy when target is silent
    # -10 * log10(est_energy) gives large negative values for loud estimates
    # Clamp to [-30, 0] to avoid extreme gradients
    silence_penalty = -10.0 * torch.log10(est_energy + eps)
    silence_penalty = torch.clamp(silence_penalty, min=-30.0, max=0.0)

    # Apply: reward when both silent, penalize when only target is silent
    sisnr_db = torch.where(both_silent, silence_reward, sisnr_db)
    sisnr_db = torch.where(target_silent_est_loud, silence_penalty, sisnr_db)

    return sisnr_db


class COIWeightedLoss(torch.nn.Module):
    """Fixed-order, class-of-interest weighted SI-SNR loss.

    This compares `est[:,0]` to `target[:,0]` (COI) and `est[:,1]` to
    `target[:,1]` (background) and returns a negative weighted SI-SNR.

    For background-only samples (where COI target is silent), the SI-SNR
    function returns a penalty based on estimate energy, training the model
    to output silence from the COI head when there's no plane present.
    """

    def __init__(self, class_weight: float = 1.5, eps: float = 1e-8):
        super().__init__()
        self.class_weight = float(class_weight)
        self.eps = float(eps)

    def forward(
        self, est_sources: torch.Tensor, target_sources: torch.Tensor
    ) -> torch.Tensor:
        # Expect (B, n_src, T)
        if est_sources.ndim != 3 or target_sources.ndim != 3:
            raise ValueError("est_sources and target_sources must be (B, n_src, T)")

        # Compute per-example SI-SNR (dB) for each source using stateless sisnr
        # sisnr() handles silent targets by penalizing non-zero estimates
        coi_sisnr = sisnr(est_sources[:, 0, :], target_sources[:, 0, :], eps=self.eps)
        bg_sisnr = sisnr(est_sources[:, 1, :], target_sources[:, 1, :], eps=self.eps)

        # Always use weighted average (COI term will push toward silence when target is silent)
        weighted = (self.class_weight * coi_sisnr + bg_sisnr) / (
            self.class_weight + 1.0
        )
        loss = -weighted.mean()
        return loss


class AudioAugmentations:
    """Audio augmentations for increasing effective dataset size.

    Each augmentation is applied randomly with the specified probability.
    """

    @staticmethod
    def time_stretch(waveform: torch.Tensor, rate: float) -> torch.Tensor:
        """Time stretch without changing pitch (simplified via resampling)."""
        if rate == 1.0:
            return waveform
        orig_len = waveform.shape[-1]
        # Use simple interpolation for time stretching
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
        # Adjust length back to original
        if stretched.shape[-1] > orig_len:
            stretched = stretched[:orig_len]
        elif stretched.shape[-1] < orig_len:
            stretched = torch.nn.functional.pad(
                stretched, (0, orig_len - stretched.shape[-1])
            )
        return stretched

    @staticmethod
    def pitch_shift_approx(waveform: torch.Tensor, semitones: float) -> torch.Tensor:
        """Approximate pitch shift via time stretch + resample."""
        # This is a simplified approximation - proper pitch shift needs librosa/torchaudio-sox
        rate = 2 ** (semitones / 12.0)
        return AudioAugmentations.time_stretch(waveform, rate)

    @staticmethod
    def add_noise(waveform: torch.Tensor, noise_level: float = 0.005) -> torch.Tensor:
        """Add Gaussian noise."""
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise

    @staticmethod
    def gain(waveform: torch.Tensor, gain_db: float) -> torch.Tensor:
        """Apply gain in dB."""
        return waveform * (10 ** (gain_db / 20.0))

    @staticmethod
    def time_shift(waveform: torch.Tensor, shift_samples: int) -> torch.Tensor:
        """Circular time shift."""
        return torch.roll(waveform, shifts=shift_samples, dims=-1)

    @staticmethod
    def low_pass_filter(
        waveform: torch.Tensor, cutoff_ratio: float = 0.8
    ) -> torch.Tensor:
        """Simple low-pass filter via FFT."""
        if cutoff_ratio >= 1.0:
            return waveform
        fft = torch.fft.rfft(waveform)
        n_freqs = fft.shape[-1]
        cutoff_idx = int(n_freqs * cutoff_ratio)
        # Smooth rolloff
        mask = torch.ones(n_freqs, device=waveform.device)
        rolloff_width = max(1, n_freqs // 20)
        for i in range(rolloff_width):
            if cutoff_idx + i < n_freqs:
                mask[cutoff_idx + i] = 1.0 - (i / rolloff_width)
        mask[cutoff_idx + rolloff_width :] = 0.0
        filtered_fft = fft * mask
        return torch.fft.irfft(filtered_fft, n=waveform.shape[-1])

    @staticmethod
    def random_augment(
        waveform: torch.Tensor, rng: np.random.Generator | None = None
    ) -> torch.Tensor:
        """Apply a random combination of augmentations."""
        if rng is None:
            rng = np.random.default_rng()

        # Randomly select which augmentations to apply
        augmented = waveform.clone()

        # Time stretch (0.9 - 1.1x)
        if rng.random() < 0.5:
            rate = rng.uniform(0.9, 1.1)
            augmented = AudioAugmentations.time_stretch(augmented, rate)

        # Gain (-6 to +6 dB)
        if rng.random() < 0.7:
            gain_db = rng.uniform(-6, 6)
            augmented = AudioAugmentations.gain(augmented, gain_db)

        # Add noise
        if rng.random() < 0.4:
            noise_level = rng.uniform(0.001, 0.01)
            augmented = AudioAugmentations.add_noise(augmented, noise_level)

        # Time shift (up to 10% of length)
        if rng.random() < 0.5:
            max_shift = int(augmented.shape[-1] * 0.1)
            shift = int(rng.integers(-max_shift, max_shift + 1))
            augmented = AudioAugmentations.time_shift(augmented, shift)

        # Low-pass filter
        if rng.random() < 0.3:
            cutoff = rng.uniform(0.6, 0.95)
            augmented = AudioAugmentations.low_pass_filter(augmented, cutoff)

        return augmented


class AudioDataset(Dataset):
    """pytorch dataset handler for wav files."""

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
        # Only keep necessary columns to reduce memory
        # Silently skip test split (no samples loaded)
        if split == "test":
            split_df = dataframe.iloc[0:0][["filename", "label"]].copy()
            if "coi_class" in dataframe.columns:
                split_df["coi_class"] = dataframe.iloc[0:0]["coi_class"]
            split_df = split_df.reset_index(drop=True)
        else:
            split_df = dataframe[dataframe["split"] == split][
                ["filename", "label"]
            ].copy()
            if "coi_class" in dataframe.columns:
                split_df["coi_class"] = dataframe.loc[split_df.index, "coi_class"]
            split_df = split_df.reset_index(drop=True)

        self.sample_rate = sample_rate
        self.segment_samples = int(segment_length * sample_rate)
        self.snr_range = snr_range
        self.n_coi_classes = n_coi_classes
        self.augment = augment
        # Augmentation multiplier: how many augmented versions per COI sample
        # Set to 3 to get 3x augmentations per COI sample
        self.augment_multiplier = (
            int(augment_multiplier) if split == "train" and augment else 1
        )
        # Random generator for reproducible augmentations
        self._rng = np.random.default_rng(42)
        # Probability (0..1) of returning a background-only example during
        # training. Background-only examples have zeroed COI targets and the
        # background in the last source slot.
        self.background_only_prob = float(background_only_prob)
        # How many background files to mix together for background-only examples
        self.background_mix_n = int(background_mix_n)

        # Cache resamplers by (orig_sr -> target_sr) to avoid repeatedly creating modules
        self._resamplers: dict[tuple[int, int], torchaudio.transforms.Resample] = {}

        # Store only file paths as lists instead of DataFrames (more memory efficient)
        coi_mask = split_df["label"] == 1
        self.coi_files = split_df.loc[coi_mask, "filename"].tolist()
        self.non_coi_files = split_df.loc[~coi_mask, "filename"].tolist()

        # Compute number of extra background-only samples to append per epoch.
        # Interpret `background_only_prob` as a ratio of COI-count to add as
        # extra background-only examples (e.g. 0.25 means 25% extra backgrounds).
        if len(self.coi_files) > 0:
            self._extra_background_count = int(
                self.background_only_prob * len(self.coi_files) + 0.5
            )
        else:
            # If no COI files, we will iterate over backgrounds only.
            self._extra_background_count = 0

        # Segment stride (seconds). If None, default to non-overlapping windows
        # equal to `segment_length`.
        self.segment_stride = (
            segment_stride if segment_stride is not None else segment_length
        )

        # Precompute available COI segments (file, frame_offset, num_frames)
        # so that each epoch can iterate over all segments deterministically for
        # validation/test. For training we will instead produce one random-offset
        # mixture per COI file in __getitem__.
        self.coi_segments: list[tuple[str, int, int]] = []
        for filepath in self.coi_files:
            try:
                info = torchaudio.info(filepath)
                orig_sr = info.sample_rate
                num_frames_orig = int(info.num_frames)

                # Convert target segment and stride to original-sr frames
                seg_frames_orig = max(
                    1, int(self.segment_samples * orig_sr / self.sample_rate)
                )
                stride_frames_orig = max(
                    1, int(self.segment_stride * orig_sr / self.sample_rate)
                )

                if num_frames_orig <= 0:
                    n_segs = 1
                else:
                    if num_frames_orig <= seg_frames_orig:
                        n_segs = 1
                    else:
                        # cover the file with sliding windows using stride
                        n_segs = 1 + max(
                            0, (num_frames_orig - seg_frames_orig) // stride_frames_orig
                        )

                for s in range(n_segs):
                    offset = s * stride_frames_orig
                    # ensure we don't go past file end; `load_and_preprocess` will pad
                    self.coi_segments.append((filepath, offset, seg_frames_orig))
            except Exception:
                # If we cannot obtain file info at init time, do NOT load the
                # full file here (would spike memory). Instead, defer any
                # expensive operations to __getitem__ where audio is loaded on
                # demand. Treat the file as having one segment and let
                # `load_and_preprocess` determine actual frames when called.
                self.coi_segments.append((filepath, 0, None))

        if n_coi_classes > 1:
            # Store file lists per class instead of DataFrames
            coi_class_col = (
                split_df["coi_class"]
                if "coi_class" in split_df.columns
                else pd.Series(0, index=split_df.index)
            )
            self.coi_by_class = [
                split_df.loc[
                    (split_df["label"] == 1) & (coi_class_col == i),
                    "filename",
                ].tolist()
                for i in range(n_coi_classes)
            ]
            print(
                f"{split} set: {[len(c) for c in self.coi_by_class]} per class, "
                f"{len(self.non_coi_files)} non-COI"
            )
        else:
            print(
                f"{split} set: {len(self.coi_files)} COI, {len(self.non_coi_files)} non-COI"
            )
            if self.augment_multiplier > 1:
                print(
                    f"  → With {self.augment_multiplier}x augmentation: "
                    f"{len(self.coi_files) * self.augment_multiplier} effective COI samples"
                )

        # Clear the temporary DataFrame
        del split_df
        gc.collect()

    def __len__(self):
        if self.split == "train":
            # Each COI file is seen `augment_multiplier` times per epoch with
            # different random augmentations. Background-only examples are added
            # on top based on `background_only_prob`.
            if len(self.coi_files) > 0:
                base_coi_samples = len(self.coi_files) * self.augment_multiplier
                return base_coi_samples + self._extra_background_count
            # Fallback: no COI files -> iterate over backgrounds
            return len(self.non_coi_files)
        return len(self.coi_segments)

    def load_and_preprocess(
        self, filepath: str, frame_offset: int = 0, num_frames: int | None = None
    ) -> torch.Tensor:
        """Load only the needed segment from disk to reduce RAM spikes.

        Args:
            filepath: path to audio file
            frame_offset: offset in original-file frames (used with torchaudio.load)
            num_frames: number of frames to load at original sampling rate. If
                None, computed from target `segment_samples` and file sample rate.
        """

        try:
            info = torchaudio.info(filepath)
            orig_sr = info.sample_rate
            total_frames = int(info.num_frames)
        except Exception:
            # Fallback: torchaudio.info can fail on some backends/codecs
            waveform, orig_sr = torchaudio.load(filepath)
            if orig_sr != self.sample_rate:
                key = (orig_sr, self.sample_rate)
                resampler = self._resamplers.get(key)
                if resampler is None:
                    resampler = torchaudio.transforms.Resample(
                        orig_sr, self.sample_rate
                    )
                    self._resamplers[key] = resampler
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            waveform = waveform.squeeze(0)
            if waveform.shape[0] < self.segment_samples:
                waveform = torch.nn.functional.pad(
                    waveform, (0, self.segment_samples - waveform.shape[0])
                )
            return waveform[: self.segment_samples]

        # Decide how many frames to load at original sample rate
        segment_frames_orig = int(self.segment_samples * orig_sr / self.sample_rate)
        segment_frames_orig = max(segment_frames_orig, 1)

        # If caller provided num_frames use it, otherwise default to segment_frames_orig
        if num_frames is None:
            num_frames_to_load = segment_frames_orig
        else:
            num_frames_to_load = int(num_frames)

        # Load only a segment from disk at the requested offset
        waveform, sr = torchaudio.load(
            filepath, frame_offset=int(frame_offset), num_frames=int(num_frames_to_load)
        )

        if sr != self.sample_rate:
            key = (sr, self.sample_rate)
            resampler = self._resamplers.get(key)
            if resampler is None:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                self._resamplers[key] = resampler
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = waveform.squeeze(0)

        # Enforce exact segment length at target SR (pad or trim)
        if waveform.shape[0] < self.segment_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.segment_samples - waveform.shape[0])
            )
        else:
            waveform = waveform[: self.segment_samples]

        return waveform

    def create_mixture(self, source, noise, snr_db):
        """Create a mixture of source and noise at a given SNR."""
        source_power = torch.mean(source**2)
        noise_power = torch.mean(noise**2)

        snr_linear = 10 ** (snr_db / 10)
        scaling_factor = torch.sqrt(source_power / (snr_linear * noise_power + 1e-8))

        scaled_noise = noise * scaling_factor
        mixture = source + scaled_noise

        return mixture

    def normalize(self, waveform):
        """Normalize waveform to have zero mean and unit variance."""
        mean = torch.mean(waveform)
        std = torch.std(waveform) + 1e-8
        normalized_waveform = (waveform - mean) / std
        return normalized_waveform

    def __getitem__(self, idx):
        # Select COI class
        if self.n_coi_classes == 1:
            # Binary case: for TRAIN create one random-offset mixture per COI file.
            if self.split == "train":
                # With augment_multiplier, the effective dataset size is:
                #   len(coi_files) * augment_multiplier + _extra_background_count
                # Map idx back to actual COI file index
                coi_count = len(self.coi_files)
                effective_coi_count = coi_count * self.augment_multiplier

                if coi_count > 0 and idx < effective_coi_count:
                    # Map augmented index back to original file
                    actual_coi_idx = idx % coi_count
                    augment_variant = (
                        idx // coi_count
                    )  # Which augmentation variant (0, 1, 2, ...)
                    coi_file = self.coi_files[actual_coi_idx]

                    # Determine original file info to compute valid offsets
                    try:
                        file_info = torchaudio.info(coi_file)
                        orig_sr = int(file_info.sample_rate)
                        total_frames = int(file_info.num_frames)

                        # Number of frames to load at original SR corresponding to segment length
                        segment_frames_orig = int(
                            self.segment_samples * orig_sr / self.sample_rate
                        )
                        segment_frames_orig = max(segment_frames_orig, 1)

                        max_offset = max(0, total_frames - segment_frames_orig)
                        frame_offset = (
                            int(np.random.randint(0, max_offset + 1))
                            if max_offset > 0
                            else 0
                        )

                        # Load the COI segment at the random offset
                        coi_audio = self.load_and_preprocess(
                            coi_file,
                            frame_offset=frame_offset,
                            num_frames=segment_frames_orig,
                        )
                    except Exception:
                        coi_audio = self.load_and_preprocess(coi_file)

                    # Apply augmentation if this is an augmented variant (variant > 0)
                    # or randomly for variant 0 when augment is enabled
                    if (
                        self.augment
                        and self.augment_multiplier > 1
                        and augment_variant > 0
                    ):
                        coi_audio = AudioAugmentations.random_augment(
                            coi_audio, self._rng
                        )

                    sources = [coi_audio]
                else:
                    # Background-only extra sample: mix `background_mix_n` random
                    # non-COI files together.
                    mix_n = max(1, int(self.background_mix_n))
                    idxs = np.random.choice(len(self.non_coi_files), size=mix_n)
                    bg_list = []
                    for i in idxs:
                        nf = self.non_coi_files[int(i)]
                        bg_list.append(self.load_and_preprocess(nf))

                    background = torch.stack(bg_list, dim=0).sum(dim=0)
                    coi_audio = torch.zeros_like(background)
                    sources = [coi_audio]
            else:
                # Use the precomputed segment mapping for validation/test
                filepath, frame_offset, num_frames = self.coi_segments[idx]
                coi_audio = self.load_and_preprocess(
                    filepath, frame_offset=frame_offset, num_frames=num_frames
                )
                sources = [coi_audio]
        else:
            # Multi-class case: randomly select one COI class
            class_idx = np.random.randint(0, self.n_coi_classes)
            coi_idx = np.random.randint(0, len(self.coi_by_class[class_idx]))
            coi_file = self.coi_by_class[class_idx][coi_idx]
            coi_audio = self.load_and_preprocess(coi_file)

            # Create sources list with zeros for other classes
            sources = [torch.zeros_like(coi_audio) for _ in range(self.n_coi_classes)]
            sources[class_idx] = coi_audio

        # Sample background if not already prepared (background-only branch)
        if "background" not in locals():
            noncoi_idx = np.random.randint(0, len(self.non_coi_files))
            noncoi_file = self.non_coi_files[noncoi_idx]
            background = self.load_and_preprocess(noncoi_file)

        # Create mixture from sources and background. Do not pre-normalize
        # individual sources here — instead normalize the whole mixture (and
        # targets) using the mixture mean/std so training matches inference
        # where inputs are normalized per-chunk before passing to the model.
        total_coi = torch.stack(sources).sum(dim=0)
        snr_db = np.random.uniform(*self.snr_range)

        # Scale background for target SNR. If total COI is all zeros (i.e.
        # background-only example) then keep the background unscaled so the
        # mixture is simply the background.
        coi_power = torch.mean(total_coi**2) + 1e-8
        bg_power = torch.mean(background**2) + 1e-8
        snr_linear = 10 ** (snr_db / 10)
        if torch.allclose(total_coi, torch.zeros_like(total_coi)):
            scaled_background = background
            mixture = scaled_background
        else:
            scaling_factor = torch.sqrt(coi_power / (snr_linear * bg_power))
            scaled_background = background * scaling_factor

            # Mixture is now sum of sources (in original waveform scale)
            mixture = total_coi + scaled_background

        # Update background to the scaled version (what's actually in mixture)
        sources.append(scaled_background)
        sources_tensor = torch.stack(sources, dim=0)

        # Normalize each waveform independently (matching original SuDORMRF)
        def normalize_wav(wav):
            mean = wav.mean()
            std = wav.std() + 1e-8
            return (wav - mean) / std

        mixture = normalize_wav(mixture)
        sources_tensor = torch.stack([normalize_wav(s) for s in sources_tensor], dim=0)

        return mixture, sources_tensor


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    clip_grad_norm=5.0,
    *,
    grad_accum_steps: int = 1,
    use_amp: bool = True,
    warmup_steps: int = 0,
    global_step: int = 0,
    base_lr: float = 0.001,
):
    """Train for one epoch.

    Args:
        warmup_steps: Number of steps to linearly warmup LR from 0 to base_lr
        global_step: Current global step count (for warmup across epochs)
        base_lr: Target learning rate after warmup

    Returns:
        tuple: (epoch_loss, final_global_step, grad_norm_history)
    """
    model.train()
    running_loss = 0.0
    n_samples = 0
    grad_norms: list[float] = []  # Track gradient norms for monitoring

    grad_accum_steps = max(int(grad_accum_steps), 1)
    use_amp = bool(use_amp) and (str(device).startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
        if use_amp
        else nullcontext()
    )

    progress_bar = tqdm(dataloader, desc="Training", leave=False, ascii=True, ncols=100)
    optimizer.zero_grad(set_to_none=True)
    step_idx = 0
    current_step = global_step  # Track global step for warmup

    # Define normalization function once outside the loop for efficiency
    def normalize_tensor_wav(wav: torch.Tensor) -> torch.Tensor:
        """Normalize waveforms to zero mean and unit variance per sample."""
        mean = wav.mean(dim=-1, keepdim=True)
        std = wav.std(dim=-1, keepdim=True) + 1e-8
        return (wav - mean) / std

    for step_idx, (mixtures, sources) in enumerate(progress_bar, start=1):
        current_step += 1

        # --- Learning rate warmup ---
        if warmup_steps > 0 and current_step <= warmup_steps:
            warmup_lr = base_lr * (current_step / warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

        # mixtures: (B, T), sources: (B, n_src, T)
        # Sources are pre-normalized in the dataset, but we'll re-mix online
        sources = sources.to(device, non_blocking=True)

        # --- Online mixing: shuffle sources within batch for augmentation ---
        # This creates new source combinations each batch, improving generalization
        # and is memory-efficient (standard technique in source separation)
        B, n_src, T = sources.shape
        denom_eps = 1e-8

        # Store original energies to preserve SNR distribution
        energies = torch.sum(sources**2, dim=-1, keepdim=True)  # (B, 2, 1)

        # Randomly permute batch indices for each source independently
        idx_coi = torch.randperm(B, device=device)
        idx_bg = torch.randperm(B, device=device)

        # Create new source combinations by shuffling
        new_coi = sources[idx_coi, 0, :]  # (B, T)
        new_bg = sources[idx_bg, 1, :]  # (B, T)

        # Rescale shuffled sources to match original energy distribution
        # This preserves the SNR statistics from the dataset
        new_coi = new_coi * torch.sqrt(
            energies[:, 0] / ((new_coi**2).sum(-1, keepdim=True) + denom_eps)
        )
        new_bg = new_bg * torch.sqrt(
            energies[:, 1] / ((new_bg**2).sum(-1, keepdim=True) + denom_eps)
        )

        # Create mixture and normalize all waveforms consistently
        m1wavs = normalize_tensor_wav(new_coi + new_bg)  # (B, T)
        clean_wavs = torch.stack(
            [normalize_tensor_wav(new_coi), normalize_tensor_wav(new_bg)], dim=1
        )  # (B, 2, T)

        with autocast_ctx:
            # Model expects (B, 1, T) input, outputs (B, n_src, T)
            rec_sources_wavs = model(m1wavs.unsqueeze(1))
            loss = criterion(rec_sources_wavs, clean_wavs)
            loss = loss.float()
            loss_to_backprop = loss / grad_accum_steps

            # Diagnostic: compute per-source SI-SNR (dB) for monitoring
            try:
                coi_sisnr_batch = sisnr(
                    rec_sources_wavs[:, 0, :], clean_wavs[:, 0, :], eps=LOSS_EPS
                )
                bg_sisnr_batch = sisnr(
                    rec_sources_wavs[:, 1, :], clean_wavs[:, 1, :], eps=LOSS_EPS
                )
                coi_sisnr_mean = float(coi_sisnr_batch.detach().cpu().mean().item())
                bg_sisnr_mean = float(bg_sisnr_batch.detach().cpu().mean().item())
            except Exception:
                coi_sisnr_mean = float("nan")
                bg_sisnr_mean = float("nan")

        if use_amp:
            assert scaler is not None
            scaler.scale(loss_to_backprop).backward()
        else:
            loss_to_backprop.backward()

        if (step_idx % grad_accum_steps) == 0:
            if use_amp:
                assert scaler is not None
                scaler.unscale_(optimizer)
            grads_finite = True
            for p in model.parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        grads_finite = False
                        break
            loss_finite = torch.isfinite(loss)
            if not grads_finite or not loss_finite:
                if use_amp:
                    try:
                        scaler.update()
                    except Exception:
                        pass
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"Warning: non-finite loss or gradients at step {current_step}; skipping optimizer step."
                )
                grad_norms.append(float("nan"))
            else:
                # Track gradient norm BEFORE clipping for monitoring
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_grad_norm
                )
                grad_norms.append(float(total_norm.item()))

                if use_amp:
                    assert scaler is not None
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        batch_size = B  # Use B from online mixing
        running_loss += float(loss.detach().cpu().item()) * batch_size
        n_samples += batch_size

        # Show current LR during warmup for visibility
        current_lr = optimizer.param_groups[0]["lr"]
        progress_bar.set_postfix(
            loss=float(loss.detach().cpu().item()),
            coi=f"{coi_sisnr_mean:.3f}",
            bg=f"{bg_sisnr_mean:.3f}",
            lr=f"{current_lr:.2e}",
        )

        # Free memory explicitly
        del (
            sources,
            m1wavs,
            clean_wavs,
            rec_sources_wavs,
            loss,
            loss_to_backprop,
        )

    # Flush any remaining accumulated gradients
    if step_idx != 0 and (step_idx % grad_accum_steps) != 0:
        if use_amp:
            assert scaler is not None
            scaler.unscale_(optimizer)

        # Check grads for finiteness before final update
        grads_finite = True
        for p in model.parameters():
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    grads_finite = False
                    break

        if not grads_finite:
            if use_amp:
                try:
                    scaler.update()
                except Exception:
                    pass
            optimizer.zero_grad(set_to_none=True)
            print(
                "Warning: non-finite gradients detected; skipping final optimizer step."
            )
            grad_norms.append(float("nan"))
        else:
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_grad_norm
            )
            grad_norms.append(float(total_norm.item()))
            if use_amp:
                assert scaler is not None
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    # Clear GPU cache after epoch
    if device != "cpu":
        torch.cuda.empty_cache()

    epoch_loss = running_loss / n_samples

    # Compute gradient norm statistics for monitoring
    valid_norms = [n for n in grad_norms if not np.isnan(n)]
    if valid_norms:
        avg_grad_norm = np.mean(valid_norms)
        max_grad_norm = np.max(valid_norms)
        print(
            f"  Gradient norms - avg: {avg_grad_norm:.4f}, max: {max_grad_norm:.4f}, nan_count: {len(grad_norms) - len(valid_norms)}"
        )

    return epoch_loss, current_step, grad_norms


@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device, *, use_amp: bool = True):
    model.eval()
    running_loss = 0.0
    n_samples = 0
    # Track SI-SNR metrics across all batches for epoch-level reporting
    all_coi_sisnr: list[float] = []
    all_bg_sisnr: list[float] = []

    use_amp = bool(use_amp) and (str(device).startswith("cuda"))
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
        if use_amp
        else nullcontext()
    )

    progress_bar = tqdm(
        dataloader, desc="Validation", leave=False, ascii=True, ncols=100
    )
    for mixtures, sources in progress_bar:
        mixtures = mixtures.to(device, non_blocking=True)
        sources = sources.to(device, non_blocking=True)

        with autocast_ctx:
            estimates = model(mixtures.unsqueeze(1))
            # Cast to FP32 for loss computation to ensure numerical accuracy
            loss = criterion(estimates.float(), sources.float())

        batch_size = mixtures.size(0)
        running_loss += float(loss.detach().cpu().item()) * batch_size
        n_samples += batch_size

        # Compute SI-SNR metrics in FP32 for accuracy
        try:
            estimates_fp32 = estimates.float()
            sources_fp32 = sources.float()
            coi_sisnr_batch = sisnr(
                estimates_fp32[:, 0, :], sources_fp32[:, 0, :], eps=LOSS_EPS
            )
            bg_sisnr_batch = sisnr(
                estimates_fp32[:, 1, :], sources_fp32[:, 1, :], eps=LOSS_EPS
            )
            coi_sisnr_mean = float(coi_sisnr_batch.detach().cpu().mean().item())
            bg_sisnr_mean = float(bg_sisnr_batch.detach().cpu().mean().item())
            all_coi_sisnr.append(coi_sisnr_mean)
            all_bg_sisnr.append(bg_sisnr_mean)
        except Exception:
            coi_sisnr_mean = float("nan")
            bg_sisnr_mean = float("nan")

        progress_bar.set_postfix(
            loss=float(loss.detach().cpu().item()),
            coi=f"{coi_sisnr_mean:.3f}",
            bg=f"{bg_sisnr_mean:.3f}",
        )

        # Free memory explicitly
        del mixtures, sources, estimates, loss

    # Clear GPU cache after validation
    if device != "cpu":
        torch.cuda.empty_cache()

    epoch_loss = running_loss / n_samples

    # Report epoch-level SI-SNR metrics
    if all_coi_sisnr and all_bg_sisnr:
        avg_coi_sisnr = np.mean(all_coi_sisnr)
        avg_bg_sisnr = np.mean(all_bg_sisnr)
        print(f"  Val SI-SNR - COI: {avg_coi_sisnr:.2f} dB, BG: {avg_bg_sisnr:.2f} dB")

    return epoch_loss


def create_dataloaders(config: Config):
    """Create train and validation dataloaders."""
    # Load minimal columns for train dataset creation
    usecols = ["filename", "label", "split"]
    if getattr(config.data, "n_coi_classes", 1) > 1:
        usecols.append("coi_class")

    df = pd.read_csv(config.data.df_path, usecols=usecols)
    # Optimize dtypes
    df["label"] = df["label"].astype("uint8")
    df["split"] = df["split"].astype("category")
    if "coi_class" in df.columns:
        df["coi_class"] = df["coi_class"].astype("category")

    # Create train dataset only (val loader created on demand)
    train_dataset = AudioDataset(
        df,
        split="train",
        sample_rate=config.data.sample_rate,
        segment_length=config.data.segment_length,
        snr_range=tuple(config.data.snr_range),
        n_coi_classes=config.data.n_coi_classes,
        augment=True,
        background_only_prob=getattr(config.data, "background_only_prob", 0.0),
        background_mix_n=getattr(config.data, "background_mix_n", 2),
        augment_multiplier=getattr(config.data, "augment_multiplier", 1),
    )

    # Memory-optimized DataLoader settings: default to 0 workers
    num_workers = int(getattr(config.training, "num_workers", 0))
    pin_memory = (
        bool(getattr(config.training, "pin_memory", False))
        and torch.cuda.is_available()
    )

    loader_kwargs = {
        "batch_size": config.training.batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            getattr(config.training, "persistent_workers", True)
        )
        loader_kwargs["prefetch_factor"] = int(
            getattr(config.training, "prefetch_factor", 2)
        )

    # Free the dataframe before creating the loader

    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=True, **loader_kwargs
    )

    return train_loader


def create_val_dataloader(config: Config) -> tuple[DataLoader, "AudioDataset"]:
    """Create validation dataloader on demand to avoid holding val dataset in memory.

    Returns:
        tuple: (val_loader, val_dataset) - Both must be deleted by caller to free memory.
    """
    usecols = ["filename", "label", "split"]
    if getattr(config.data, "n_coi_classes", 1) > 1:
        usecols.append("coi_class")

    df = pd.read_csv(config.data.df_path, usecols=usecols)
    df["label"] = df["label"].astype("uint8")
    df["split"] = df["split"].astype("category")
    if "coi_class" in df.columns:
        df["coi_class"] = df["coi_class"].astype("category")

    val_dataset = AudioDataset(
        df,
        split="val",
        sample_rate=config.data.sample_rate,
        segment_length=config.data.segment_length,
        snr_range=tuple(config.data.snr_range),
        n_coi_classes=config.data.n_coi_classes,
        augment=False,
        background_only_prob=0.0,
    )

    num_workers = int(getattr(config.training, "num_workers", 0))
    pin_memory = (
        bool(getattr(config.training, "pin_memory", False))
        and torch.cuda.is_available()
    )
    loader_kwargs = {
        "batch_size": config.training.batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = False
        loader_kwargs["prefetch_factor"] = 1

    del df
    gc.collect()

    val_loader = DataLoader(
        val_dataset, shuffle=False, drop_last=False, **loader_kwargs
    )
    gc.collect()
    # Return both loader and dataset so caller can explicitly delete both
    return val_loader, val_dataset


def get_model_args(model):
    """Extract model arguments for checkpoint saving."""
    return {
        "in_channels": getattr(model, "in_channels", None),
        "out_channels": getattr(model, "out_channels", None),
        "num_blocks": getattr(model, "num_blocks", None),
    }


def create_model(config: Config):
    """Create and wrap model with aircraft separation head."""
    if config.model.type == "improved":
        base_model = SuDORMRF(
            out_channels=config.model.out_channels,
            in_channels=config.model.in_channels,
            num_blocks=config.model.num_blocks,
            upsampling_depth=config.model.upsampling_depth,
            enc_kernel_size=config.model.enc_kernel_size,
            enc_num_basis=config.model.enc_num_basis,
            num_sources=2,
        )
    else:

        base_model = GroupCommSudoRmRf(
            out_channels=config.model.out_channels,
            in_channels=config.model.in_channels,
            num_blocks=config.model.num_blocks,
            upsampling_depth=config.model.upsampling_depth,
            enc_kernel_size=config.model.enc_kernel_size,
            enc_num_basis=config.model.enc_num_basis,
            num_sources=2,
        )

    if config.data.n_coi_classes > 1:
        print(
            f"Wrapping model for Multi-class separation with {config.data.n_coi_classes} classes."
        )
        model = wrap_model_for_multiclass(
            base_model, n_coi_classes=config.data.n_coi_classes
        )
    else:
        print("Wrapping model for Single COI separation.")
        model = wrap_model_for_coi(base_model)

    try:
        model = model.to(config.training.device)
    except Exception as e:
        print(f"Error moving model to device {config.training.device}: {e}")
        print("Moving model to CPU instead.")
        model = model.to("cpu")

    # Compile model for faster training (requires PyTorch 2.0+)
    # Note: torch.compile with inductor backend can be slow on WSL
    # Use 'eager' backend or disable compilation if experiencing slowness
    if hasattr(torch, "compile") and config.training.compile_model:
        backend = getattr(config.training, "compile_backend", "inductor")
        print(f"Compiling model with torch.compile() using '{backend}' backend...")
        model = torch.compile(model, backend=backend)
    elif hasattr(torch, "compile"):
        print("Model compilation disabled (set compile_model=True in config to enable)")

    return model


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(config: Config, timestamp: str | None = None):
    """Main training function."""
    # Set seed for reproducibility
    seed = getattr(config.training, "seed", 42)
    set_seed(seed)
    print(f"Random seed set to: {seed}")

    # Setup - create timestamped subdirectory
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(config.training.checkpoint_dir) / timestamp
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Save config
    config.save(checkpoint_dir / "config.yaml")

    print("Creating train dataloader...")
    train_loader = create_dataloaders(config)
    gc.collect()  # Clean up after dataloader creation

    print("Creating model...")
    model = create_model(config)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.2f}M")

    # Setup training
    # Use fixed-order COI-weighted SI-SNR loss (no PIT) to preserve semantic heads
    criterion = COIWeightedLoss(
        class_weight=getattr(config.training, "class_weight", 1.5)
    )
    base_lr = float(config.training.lr)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Get warmup and validation frequency settings
    warmup_steps = int(getattr(config.training, "warmup_steps", 500))
    validate_every_n = int(getattr(config.training, "validate_every_n_epochs", 1))

    if warmup_steps > 0:
        print(f"Learning rate warmup: {warmup_steps} steps")
    if validate_every_n > 1:
        print(f"Validation frequency: every {validate_every_n} epochs")

    # Training loop
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    global_step = 0  # Track steps across epochs for warmup
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "grad_norms": [],
    }

    for epoch in range(1, config.training.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.training.num_epochs}")

        train_loss, global_step, epoch_grad_norms = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            config.training.device,
            config.training.clip_grad_norm,
            grad_accum_steps=getattr(config.training, "grad_accum_steps", 1),
            use_amp=getattr(config.training, "use_amp", True),
            warmup_steps=warmup_steps,
            global_step=global_step,
            base_lr=base_lr,
        )
        history["train_loss"].append(train_loss)
        # Store mean grad norm per epoch for history
        valid_norms = [n for n in epoch_grad_norms if not np.isnan(n)]
        history["grad_norms"].append(
            np.mean(valid_norms) if valid_norms else float("nan")
        )

        # Validation: run every N epochs, or always on first/last epoch
        run_validation = (
            epoch % validate_every_n == 0
            or epoch == 1
            or epoch == config.training.num_epochs
        )

        if run_validation:
            # Create validation dataloader on demand to save memory
            val_loader, val_dataset = create_val_dataloader(config)
            val_loss = validate_epoch(
                model,
                val_loader,
                criterion,
                config.training.device,
                use_amp=getattr(config.training, "use_amp", True),
            )
            # Free validation loader AND dataset memory immediately
            del val_loader, val_dataset
            gc.collect()
            if config.training.device != "cpu":
                torch.cuda.empty_cache()
            history["val_loss"].append(val_loss)

            print(f"Train: {train_loss:.4f}, Val: {val_loss:.4f}")

            scheduler.step(val_loss)
        else:
            # No validation this epoch - use last known val_loss for logging
            val_loss = history["val_loss"][-1] if history["val_loss"] else float("inf")
            history["val_loss"].append(
                val_loss
            )  # Repeat last value for history alignment
            print(f"Train: {train_loss:.4f} (skipping validation)")

        # Save checkpoint only when validation was run
        if run_validation:
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                epochs_without_improvement = 0

                checkpoint = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "model_args": get_model_args(model),
                    "config": config.to_dict(),
                    "history": history,
                }
                torch.save(checkpoint, checkpoint_dir / "best_model.pt")
                print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= config.training.patience:
                print(f"\nEarly stopping after {epoch} epochs")
                break

    # Save training history
    with open(checkpoint_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining completed! Best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train aircraft sound separation")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    args = parser.parse_args()

    config_path = (
        Path(args.config)
        if isinstance(args.config, str)
        else Path("./training_config.yaml")
    )

    # Load config
    config = Config.from_yaml(config_path)
    print("Configuration:")
    print(f"  Model: {config.model.type} ({config.model.num_blocks} blocks)")
    print(f"  Device: {config.training.device}")

    # 1. Load all dataset metadata (same as plane classifier)
    print("\nLoading dataset metadata...")
    project_root = Path(__file__).parent.parent.parent.parent
    datasets_path = str(project_root / "data")
    audio_base_path = str(project_root.parent / "datasets")

    all_metadata = load_metadata_datasets(datasets_path, audio_base_path)

    # 2. Get the separation half (70%) of the data
    separation_metadata, _ = split_seperation_classification(all_metadata)

    print(f"Loaded {len(all_metadata)} total samples from all datasets")
    print(f"Using {len(separation_metadata)} samples for separation training (70%)")
    print(f"Datasets included: {separation_metadata['dataset'].unique()}")

    # 3. Define target classes (plane-related sounds) - same as plane classifier
    target_classes = [
        "airplane",
        "Aircraft",
        "Fixed-wing aircraft, airplane",
        "Aircraft engine",
        "Fixed-wing_aircraft_and_airplane",
    ]

    print(f"\nTarget classes: {target_classes}")

    # 4. Sample data to get balanced dataset
    print("\nSampling data with class-of-interest ratio...")
    coi_df = get_coi(separation_metadata, target_classes)
    sampled_df = sample_non_coi(
        separation_metadata,
        coi_df,
        coi_ratio=0.25,
    )

    # 5. Create binary labels: 1 for COI (plane), 0 for non-COI (background)
    sampled_df["label"] = sampled_df["label"].apply(
        lambda x: (
            1
            if (isinstance(x, list) and any(label in target_classes for label in x))
            or (isinstance(x, str) and x in target_classes)
            else 0
        )
    )

    # 6. Check for missing files
    print("\nChecking for missing audio files...")
    sampled_df["file_exists"] = sampled_df["filename"].apply(lambda f: Path(f).exists())
    missing_mask = ~sampled_df["file_exists"]

    if missing_mask.any():
        missing_count = missing_mask.sum()
        print(
            f"⚠️  Found {missing_count} missing files out of {len(sampled_df)} total samples"
        )
        print("Dropping samples with missing files...")
        sampled_df = sampled_df[sampled_df["file_exists"]].copy()

    sampled_df = sampled_df.drop(columns=["file_exists"])
    print(f"✅ Final dataset size: {len(sampled_df)} samples")

    # 7. Print dataset statistics
    print("\nDataset splits:")
    for split in ["train", "val", "test"]:
        split_df = sampled_df[sampled_df["split"] == split]
        coi_count = (split_df["label"] == 1).sum()
        non_coi_count = (split_df["label"] == 0).sum()
        print(
            f"  {split}: {len(split_df)} samples (COI: {coi_count}, non-COI: {non_coi_count})"
        )

    # 8. Save the prepared dataframe for reproducibility (with optimized dtypes)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(config.training.checkpoint_dir) / timestamp
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    # Downcast dtypes before saving
    sampled_df["label"] = sampled_df["label"].astype("uint8")
    if "coi_class" in sampled_df.columns:
        sampled_df["coi_class"] = sampled_df["coi_class"].astype("category")
    sampled_df["split"] = sampled_df["split"].astype("category")
    df_save_path = checkpoint_dir / "separation_dataset.csv"
    sampled_df.to_csv(df_save_path, index=False)
    print(f"\nSaved prepared dataset to: {df_save_path}")

    # 9. Override the config's df_path with our prepared dataframe path
    config.data.df_path = str(df_save_path)
    print(f"  Data: {config.data.df_path}")

    # 10. Clean up large DataFrames before training
    del all_metadata, separation_metadata, coi_df, sampled_df
    gc.collect()
    print("Cleaned up metadata DataFrames from memory.")

    # 11. Train
    train(config, timestamp=timestamp)


if __name__ == "__main__":
    main()
