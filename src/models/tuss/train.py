"""
Training script for TUSS model fine-tuned for COI separation.

Loads the pretrained TUSS backbone, injects new learnable prompt vectors for each
COI class (e.g. "airplane", "train", "bird") and a background prompt ("background"),
warm-starting them from the existing "sfx" / "sfxbg" vectors.  The full network is
then fine-tuned on a COI dataset using snr_with_zeroref_loss so that absent sources
(zero-energy targets) are handled gracefully.

Config is read from training_config.yml in the same directory – no CLI args needed.

Dataset expectations (same CSV format as sudormrf):
    - 'filename' : path to wav file
    - 'split'    : train / val / test
    - 'label'    : list or string of semantic class labels
    - 'coi_class': integer index (0 … n_coi_classes-1)  [added by this script]
"""

import gc
import json
import math
import os
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchaudio
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Resolve paths so imports work regardless of working directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).parent.resolve()
_BASE_DIR = _SCRIPT_DIR / "base"
_SRC_DIR = _SCRIPT_DIR.parent.parent  # code/src

for _p in [str(_BASE_DIR), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
ENERGY_EPS = 1e-8
NORMALIZE_MIN_STD = 1e-3
SILENCE_ENERGY_EPS = 1e-6
WEAK_TARGET_ENERGY_EPS = 1e-4
BG_SCALE_MIN = 0.1
BG_SCALE_MAX = 3.0
RESAMPLER_CACHE_MAX = 8

CONFIG_PATH = _SCRIPT_DIR / "training_config.yml"


# =============================================================================
# Configuration dataclasses
# =============================================================================


@dataclass
class DataConfig:
    df_path: str = "data/aircraft_data.csv"
    sample_rate: int = 48000
    segment_length: float = 6.0
    snr_range: list = field(default_factory=lambda: [-5, 5])
    # Nested list: target_classes[i] is a list of label strings mapping to
    # coi_prompts[i].  A flat list is also accepted (treated as one class).
    target_classes: list = field(default_factory=list)
    background_only_prob: float = 0.15
    background_mix_n: int = 2
    augment_multiplier: int = 2


@dataclass
class ModelConfig:
    pretrained_path: str = "base/pretrained_models/tuss.medium.2-4src"
    coi_prompts: list = field(default_factory=lambda: ["airplane", "train", "bird"])
    bg_prompt: str = "background"
    coi_prompt_init_from: str = "sfx"
    bg_prompt_init_from: str = "sfxbg"
    freeze_backbone: bool = False
    # From-scratch architecture (used when pretrained_path is null)
    encoder_name: str = "stft"
    encoder_conf: dict = field(default_factory=dict)
    decoder_name: str = "stft"
    decoder_conf: dict = field(default_factory=dict)
    separator_name: str = "tuss"
    separator_conf: dict = field(default_factory=dict)
    css_conf: dict = field(default_factory=dict)
    variance_normalization: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 2
    grad_accum_steps: int = 8
    use_amp: bool = True
    num_epochs: int = 200
    lr: float = 5e-5
    weight_decay: float = 1e-2
    num_workers: int = 4
    pin_memory: bool = True
    clip_grad_norm: float = 5.0
    patience: int = 30
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda"
    coi_weight: float = 1.5
    snr_max: int = 30
    zero_ref_loss_weight: float = 0.1
    warmup_steps: int = 300
    validate_every_n_epochs: int = 1
    seed: int = 42


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)
        data_cfg = DataConfig(**raw.get("data", {}))
        model_cfg = ModelConfig(**raw.get("model", {}))
        training_cfg = TrainingConfig(**raw.get("training", {}))
        return cls(data=data_cfg, model=model_cfg, training=training_cfg)

    def save(self, path: Path):
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    def to_dict(self):
        return asdict(self)


# =============================================================================
# Audio file info
# =============================================================================


def _get_audio_info(filepath: str) -> tuple[int, int]:
    """Return (sample_rate, num_frames) using the best available backend."""
    if hasattr(torchaudio, "info"):
        try:
            info = torchaudio.info(filepath)
            return int(info.sample_rate), int(info.num_frames)
        except Exception:
            pass
    try:
        import soundfile as sf

        info = sf.info(filepath)
        return int(info.samplerate), int(info.frames)
    except Exception:
        pass
    raise RuntimeError(f"Cannot read audio info for {filepath}")


# =============================================================================
# Loss
# =============================================================================


class COIWeightedSNRLoss(torch.nn.Module):
    """SNR loss with zero-reference handling, weighted towards COI heads."""

    def __init__(
        self,
        n_src: int,
        coi_weight: float = 1.5,
        snr_max: int = 30,
        zero_ref_loss_weight: float = 0.1,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.n_src = n_src
        self.n_coi = n_src - 1  # last source is always background
        self.coi_weight = float(coi_weight)
        self.snr_max = snr_max
        self.zero_ref_loss_weight = zero_ref_loss_weight
        self.eps = eps

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Args:
            est: (B, n_src, T)
            ref: (B, n_src, T)
        Returns:
            scalar loss
        """
        # snr_with_zeroref_loss returns (B, n_src) with solve_perm=False
        per_src = snr_with_zeroref_loss(
            est,
            ref,
            n_src=self.n_src,
            snr_max=self.snr_max,
            zero_ref_loss_weight=self.zero_ref_loss_weight,
            solve_perm=False,
            eps=self.eps,
        )  # (B, n_src)

        # COI heads are 0 … n_coi-1, background is last
        coi_loss = per_src[:, : self.n_coi].mean(dim=-1)  # (B,)
        bg_loss = per_src[:, -1]  # (B,)

        weighted = (self.coi_weight * coi_loss + bg_loss) / (self.coi_weight + 1.0)
        return weighted.mean()


# =============================================================================
# Audio augmentations
# =============================================================================


class AudioAugmentations:
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
        aug = waveform.clone()
        if rng.random() < 0.5:
            aug = AudioAugmentations.time_stretch(aug, rng.uniform(0.9, 1.1))
        if rng.random() < 0.7:
            aug = AudioAugmentations.gain(aug, rng.uniform(-6, 6))
        if rng.random() < 0.4:
            aug = AudioAugmentations.add_noise(aug, rng.uniform(0.001, 0.01))
        if rng.random() < 0.5:
            max_shift = int(aug.shape[-1] * 0.1)
            aug = AudioAugmentations.time_shift(
                aug, int(rng.integers(-max_shift, max_shift + 1))
            )
        if rng.random() < 0.3:
            aug = AudioAugmentations.low_pass_filter(aug, rng.uniform(0.6, 0.95))
        return aug


# =============================================================================
# Dataset
# =============================================================================


class AudioDataset(Dataset):
    """Multi-class COI audio separation dataset.

    Returns a (n_coi_classes + 1, T) source tensor per item:
        sources[:n_coi_classes]  – one track per COI class (most are silent)
        sources[-1]              – a single background / non-COI track

    The mixture and normalisation are produced later in prepare_batch so that
    they can be done efficiently on GPU.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        split: str = "train",
        sample_rate: int = 48000,
        segment_length: float = 6.0,
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
        self._resampler_cache_max = RESAMPLER_CACHE_MAX
        self._file_native_info: dict[str, tuple[int, int]] = {}

        if split == "test":
            split_df = dataframe.iloc[0:0]
        else:
            split_df = dataframe[dataframe["split"] == split]

        coi_mask = split_df["label"] == 1
        self.non_coi_files = split_df.loc[~coi_mask, "filename"].tolist()

        # Per-class file lists (index = coi_class)
        self.coi_files_by_class: list[list[str]] = [[] for _ in range(n_coi_classes)]
        for _, row in split_df[coi_mask].iterrows():
            cls_idx = int(row.get("coi_class", 0))
            if 0 <= cls_idx < n_coi_classes:
                self.coi_files_by_class[cls_idx].append(row["filename"])

        # Flattened list used for segment pre-computation
        self.coi_files = [f for cls in self.coi_files_by_class for f in cls]
        self.file_to_class = {}
        for cls_idx, files in enumerate(self.coi_files_by_class):
            for f in files:
                self.file_to_class[f] = cls_idx

        # Bounds
        self.file_to_bounds: dict[str, tuple[float, float | None]] = {}
        if "start_time" in split_df.columns or "end_time" in split_df.columns:
            for _, row in split_df.iterrows():
                fname = row["filename"]
                try:
                    st = (
                        float(row["start_time"])
                        if pd.notna(row.get("start_time"))
                        else 0.0
                    )
                except (ValueError, TypeError):
                    st = 0.0
                try:
                    et = (
                        float(row["end_time"])
                        if pd.notna(row.get("end_time"))
                        else None
                    )
                except (ValueError, TypeError):
                    et = None
                self.file_to_bounds[fname] = (st, et)

        self.coi_segments = self._compute_segments(split_df)
        if self.split == "train":
            self.coi_segments_train = list(self.coi_segments)

        self._extra_background_count = 0
        if self.coi_files and self.background_only_prob > 0.0:
            base = len(
                self.coi_segments_train if split == "train" else self.coi_segments
            )
            multiplier = self.augment_multiplier if split == "train" else 1
            self._extra_background_count = int(
                self.background_only_prob * base * multiplier + 0.5
            )

        print(
            f"{split}: {len(self.coi_files)} COI files ({len(self.coi_segments)} segments), "
            f"{len(self.non_coi_files)} non-COI files"
        )
        if split == "train" and self._extra_background_count:
            print(f"  + {self._extra_background_count} background-only steps")
        per_class = [len(cls) for cls in self.coi_files_by_class]
        print(f"  COI files per class: {per_class}")

    def set_epoch(self, epoch: int):
        self._rng = np.random.default_rng(42 + epoch)

    def _compute_segments(
        self, split_df: pd.DataFrame
    ) -> list[tuple[str, int, int, int]]:
        segments = []
        failures = 0
        for filepath in self.coi_files:
            class_idx = self.file_to_class.get(filepath, 0)
            bounds = self.file_to_bounds.get(filepath, (0.0, None))
            start_sec, end_sec = bounds
            try:
                orig_sr, num_frames = _get_audio_info(filepath)
                self._file_native_info[filepath] = (orig_sr, num_frames)
            except Exception as exc:
                failures += 1
                if failures <= 5:
                    print(f"  ⚠ info() failed for {filepath}: {exc}")
                orig_sr = self.sample_rate
                num_frames = (
                    int(end_sec * orig_sr)
                    if end_sec is not None
                    else self.segment_samples
                )
                self._file_native_info[filepath] = (orig_sr, num_frames)

            seg_frames = max(1, int(self.segment_samples * orig_sr / self.sample_rate))
            stride_frames = max(
                1, int(self.segment_stride_samples * orig_sr / self.sample_rate)
            )
            start_frame = int(start_sec * orig_sr)
            end_frame = int(end_sec * orig_sr) if end_sec is not None else num_frames
            end_frame = min(end_frame, num_frames)
            valid = max(0, end_frame - start_frame)
            n_segs = (
                1
                if valid <= seg_frames
                else 1 + max(0, (valid - seg_frames) // stride_frames)
            )
            for s in range(n_segs):
                segments.append(
                    (filepath, start_frame + s * stride_frames, seg_frames, class_idx)
                )
        if failures:
            print(f"  ⚠ info() failed for {failures}/{len(self.coi_files)} files")
        return segments

    def __len__(self):
        if self.split == "train":
            base = (
                len(self.coi_segments_train) * self.augment_multiplier
                if self.coi_segments_train
                else 0
            )
            return base + self._extra_background_count
        return len(self.coi_segments) + self._extra_background_count

    def _load_audio(
        self,
        filepath: str,
        frame_offset: int | None = None,
        num_frames: int | None = None,
    ) -> torch.Tensor:
        bounds = self.file_to_bounds.get(filepath, (0.0, None))
        start_sec, end_sec = bounds

        cached = self._file_native_info.get(filepath)
        if cached is None:
            try:
                orig_sr, total_frames = _get_audio_info(filepath)
                self._file_native_info[filepath] = (orig_sr, total_frames)
            except Exception:
                try:
                    waveform, sr = torchaudio.load(
                        filepath, num_frames=self.segment_samples * 2
                    )
                except Exception:
                    return torch.zeros(self.segment_samples)
                waveform = (
                    waveform.mean(0) if waveform.shape[0] > 1 else waveform.squeeze(0)
                )
                self._file_native_info[filepath] = (sr, waveform.shape[-1])
                cached = (sr, waveform.shape[-1])

        if cached is None:
            cached = self._file_native_info[filepath]
        orig_sr, total_frames = cached

        seg_frames = num_frames or max(
            1, int(self.segment_samples * orig_sr / self.sample_rate)
        )
        start_frame = int(start_sec * orig_sr)
        end_frame = int(end_sec * orig_sr) if end_sec is not None else total_frames
        end_frame = min(end_frame, total_frames)

        if frame_offset is None:
            max_off = max(start_frame, end_frame - seg_frames)
            offset = (
                int(self._rng.integers(start_frame, max_off + 1))
                if max_off > start_frame
                else start_frame
            )
        else:
            offset = int(frame_offset)

        offset = max(0, min(offset, max(0, total_frames - seg_frames)))

        try:
            waveform, sr = torchaudio.load(
                filepath, frame_offset=offset, num_frames=int(seg_frames)
            )
        except Exception:
            try:
                waveform, sr = torchaudio.load(filepath, num_frames=int(seg_frames))
            except Exception:
                return torch.zeros(self.segment_samples)

        if sr != self.sample_rate:
            key = (sr, self.sample_rate)
            if key not in self._resamplers:
                if len(self._resamplers) >= self._resampler_cache_max:
                    self._resamplers.pop(next(iter(self._resamplers)))
                self._resamplers[key] = torchaudio.transforms.Resample(
                    sr, self.sample_rate
                )
            waveform = self._resamplers[key](waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        if waveform.shape[0] < self.segment_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.segment_samples - waveform.shape[0])
            )
        return waveform[: self.segment_samples]

    def _jittered_offset(
        self, base_offset: int, seg_frames: int | None, filepath: str
    ) -> int:
        if seg_frames is None:
            return base_offset
        cached = self._file_native_info.get(filepath)
        if cached is None:
            return base_offset
        orig_sr, total_frames = cached
        bounds = self.file_to_bounds.get(filepath, (0.0, None))
        start_sec, end_sec = bounds
        start_frame = int(start_sec * orig_sr)
        end_frame = int(end_sec * orig_sr) if end_sec is not None else total_frames
        end_frame = min(end_frame, total_frames)
        half_stride = max(
            1, int(self.segment_stride_samples * orig_sr / self.sample_rate) // 2
        )
        jitter = int(self._rng.integers(-half_stride, half_stride + 1))
        offset = base_offset + jitter
        offset = max(start_frame, offset)
        offset = min(offset, max(start_frame, end_frame - seg_frames))
        return offset

    def __getitem__(self, idx) -> torch.Tensor:
        background = None

        if self.split == "train":
            coi_count = len(self.coi_segments_train)
            effective_coi_count = coi_count * self.augment_multiplier

            if coi_count > 0 and idx < effective_coi_count:
                actual_idx = idx % coi_count
                augment_variant = idx // coi_count
                filepath, base_offset, seg_frames, class_idx = self.coi_segments_train[
                    actual_idx
                ]
                offset = self._jittered_offset(base_offset, seg_frames, filepath)
                coi_audio = self._load_audio(
                    filepath, frame_offset=offset, num_frames=seg_frames
                )
                if self.augment and augment_variant > 0:
                    coi_audio = AudioAugmentations.random_augment(coi_audio, self._rng)

                sources = [
                    torch.zeros(self.segment_samples) for _ in range(self.n_coi_classes)
                ]
                sources[class_idx] = coi_audio
            else:
                # Background-only sample
                idxs = self._rng.choice(
                    len(self.non_coi_files),
                    size=max(1, self.background_mix_n),
                    replace=True,
                )
                bg_parts = [self._load_audio(self.non_coi_files[int(i)]) for i in idxs]
                background = torch.stack(bg_parts).sum(0)
                sources = [
                    torch.zeros(self.segment_samples) for _ in range(self.n_coi_classes)
                ]
        else:
            if idx < len(self.coi_segments):
                filepath, frame_offset, num_frames, class_idx = self.coi_segments[idx]
                coi_audio = self._load_audio(filepath, frame_offset, num_frames)
                sources = [
                    torch.zeros(self.segment_samples) for _ in range(self.n_coi_classes)
                ]
                sources[class_idx] = coi_audio
            else:
                idxs = self._rng.choice(
                    len(self.non_coi_files),
                    size=max(1, self.background_mix_n),
                    replace=True,
                )
                background = torch.stack(
                    [self._load_audio(self.non_coi_files[int(i)]) for i in idxs]
                ).sum(0)
                sources = [
                    torch.zeros(self.segment_samples) for _ in range(self.n_coi_classes)
                ]

        if background is None:
            bg_idx = int(self._rng.integers(len(self.non_coi_files)))
            background = self._load_audio(self.non_coi_files[bg_idx])

        sources.append(background)
        return torch.stack(sources, dim=0)  # (n_coi_classes + 1, T)


# =============================================================================
# Training utilities
# =============================================================================


def normalize_tensor_wav(
    wav: torch.Tensor, eps: float = ENERGY_EPS, min_std: float = NORMALIZE_MIN_STD
) -> torch.Tensor:
    mean = wav.mean(dim=-1, keepdim=True)
    std = wav.std(dim=-1, keepdim=True)
    is_silent = std < min_std
    std_safe = torch.where(is_silent, torch.ones_like(std), std) + eps
    normalized = (wav - mean) / std_safe
    return torch.where(is_silent, torch.zeros_like(normalized), normalized)


def prepare_batch(
    sources: torch.Tensor,
    snr_range: tuple[float, float],
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build mixture and normalised clean sources from the raw source tensor.

    Args:
        sources: (B, n_coi_classes + 1, T) – COI tracks + background (last)
        snr_range: (min_snr_db, max_snr_db)
        deterministic: use linspace SNR for validation reproducibility
    Returns:
        mixture:    (B, T)
        clean_wavs: (B, n_coi_classes + 1, T)
    """
    B, n_src, T = sources.shape
    eps = ENERGY_EPS

    cois = [sources[:, i, :] for i in range(n_src - 1)]
    bg = sources[:, -1, :]
    total_coi = torch.stack(cois, dim=0).sum(dim=0)  # (B, T)

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
    silent_coi = coi_power < SILENCE_ENERGY_EPS
    bg_scaling = torch.where(silent_coi, torch.ones_like(bg_scaling), bg_scaling)
    bg_scaling = torch.clamp(bg_scaling, BG_SCALE_MIN, BG_SCALE_MAX)

    bg_scaled = bg * bg_scaling
    mixture = normalize_tensor_wav(total_coi + bg_scaled)

    norm_cois = [normalize_tensor_wav(c) for c in cois]
    norm_bg = normalize_tensor_wav(bg_scaled)
    clean_wavs = torch.stack(norm_cois + [norm_bg], dim=1)  # (B, n_src, T)
    return mixture, clean_wavs


def check_finite(*tensors) -> bool:
    return all(torch.isfinite(t).all() for t in tensors)


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
) -> tuple[float, int, list[float]]:
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
        del sources

        if not check_finite(mixture, clean_wavs):
            del mixture, clean_wavs
            grad_norms.append(float("nan"))
            continue

        # Build per-batch prompts list; length == B, each sub-list has n_src strings
        prompts = prompts_batch_template[:B]

        with autocast_ctx:
            outputs = model(mixture, prompts)

            if not check_finite(outputs):
                del outputs
                if use_amp:
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

            loss = criterion(outputs.float(), clean_wavs.float())

        del outputs, mixture, clean_wavs

        if not check_finite(loss):
            del loss
            optimizer.zero_grad(set_to_none=True)
            grad_norms.append(float("nan"))
            continue

        loss_scaled = loss / grad_accum_steps
        if use_amp:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        batch_loss = float(loss.item())
        del loss, loss_scaled

        if step_idx % grad_accum_steps == 0:
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
            has_pending_grads = False
        else:
            has_pending_grads = True

        running_loss += batch_loss * B
        n_samples += B
        pbar.set_postfix(
            loss=f"{batch_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}"
        )

        if use_amp and step_idx % 50 == 0:
            torch.cuda.empty_cache()

    # Flush remaining accumulated gradients
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
    per_class_sisnr: list[list[float]] = [[] for _ in range(criterion.n_coi)]
    bg_sisnr_vals: list[float] = []

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

            prompts = prompts_batch_template[:B]

            with autocast_ctx:
                outputs = model(mixture, prompts)
                loss = criterion(outputs.float(), clean_wavs.float())

            batch_loss = float(loss.item())
            running_loss += batch_loss * B
            n_samples += B

            # Per-class SI-SNR for reporting
            try:
                from loss_functions.si_snr import si_snr_loss

                n_src = outputs.shape[1]
                for cls_i in range(n_src - 1):
                    ref_energy = clean_wavs[:, cls_i].pow(2).mean(dim=-1)
                    present = ref_energy > SILENCE_ENERGY_EPS
                    if present.any():
                        snr_val = -si_snr_loss(
                            outputs[:, cls_i : cls_i + 1],
                            clean_wavs[:, cls_i : cls_i + 1],
                            solve_perm=False,
                        )
                        per_class_sisnr[cls_i].append(
                            float(snr_val[present].mean().item())
                        )
                bg_snr = -si_snr_loss(
                    outputs[:, -1:], clean_wavs[:, -1:], solve_perm=False
                )
                bg_sisnr_vals.append(float(bg_snr.mean().item()))
                pbar.set_postfix(loss=f"{batch_loss:.4f}")
            except Exception:
                pbar.set_postfix(loss=f"{batch_loss:.4f}")

            del outputs, loss, mixture, clean_wavs

    if bg_sisnr_vals:
        class_strs = [
            f"cls{i}: {np.mean(v):.2f} dB" if v else f"cls{i}: n/a"
            for i, v in enumerate(per_class_sisnr)
        ]
        print(
            f"  Val SI-SNR – {', '.join(class_strs)}, BG: {np.mean(bg_sisnr_vals):.2f} dB"
        )

    return running_loss / max(n_samples, 1)


# =============================================================================
# Data loading
# =============================================================================


def create_dataloader(config: Config, split: str) -> tuple[DataLoader, AudioDataset]:
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


# =============================================================================
# Model creation
# =============================================================================


def create_model(config: Config) -> torch.nn.Module:
    """Build SeparationModel, load pretrained weights, inject new prompt vectors."""
    pretrained_path = config.model.pretrained_path
    if pretrained_path is not None:
        pretrained_path = _SCRIPT_DIR / pretrained_path

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

        ckpt_path = Path(pretrained_path) / "checkpoints" / "model.pth"
        print(f"Loading pretrained weights from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")
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
    # Warm-start from the acoustically closest pretrained prompt.         #
    # ------------------------------------------------------------------ #
    prompts_dict = model.separator.prompts

    def _get_init_vector(init_from: str) -> torch.Tensor:
        if init_from in prompts_dict:
            return prompts_dict[init_from].data.clone()
        return torch.randn(emb_dim, 1, 1) * 0.02

    new_prompts = config.model.coi_prompts + [config.model.bg_prompt]
    init_sources = [config.model.coi_prompt_init_from] * len(
        config.model.coi_prompts
    ) + [config.model.bg_prompt_init_from]

    for prompt_name, init_from in zip(new_prompts, init_sources):
        if prompt_name not in prompts_dict:
            init_val = _get_init_vector(init_from)
            # Small noise so each new COI prompt starts from a slightly
            # different position even if they all init from "sfx"
            noise = torch.randn_like(init_val) * 0.001
            prompts_dict[prompt_name] = torch.nn.Parameter(init_val + noise)
            print(f"  Injected new prompt '{prompt_name}' (init from '{init_from}')")
        else:
            print(f"  Prompt '{prompt_name}' already exists – keeping pretrained value")

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
    try:
        model = model.to(device)
    except Exception as e:
        print(f"Cannot move to {device}: {e} – falling back to CPU")
        model = model.to("cpu")
        config.training.device = "cpu"

    return model


# =============================================================================
# Seed
# =============================================================================


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =============================================================================
# Main training function
# =============================================================================


def train(config: Config, timestamp: str | None = None):
    seed = config.training.seed
    set_seed(seed)
    print(f"Seed: {seed}")

    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(config.training.checkpoint_dir) / timestamp
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    config.save(checkpoint_dir / "config.yaml")

    print("Creating model …")
    model = create_model(config)

    n_coi = len(config.model.coi_prompts)
    n_src = n_coi + 1  # COI classes + background

    # Build the static prompts list used at every forward pass:
    # each sample in the batch sees the same n_src queries simultaneously.
    all_prompts = config.model.coi_prompts + [config.model.bg_prompt]
    # Pre-allocate a large template; we slice to [:B] at runtime.
    _MAX_BATCH = 256
    prompts_template = [list(all_prompts)] * _MAX_BATCH

    print("Creating data loaders …")
    train_loader, train_dataset = create_dataloader(config, "train")
    val_loader, val_dataset = create_dataloader(config, "val")

    criterion = COIWeightedSNRLoss(
        n_src=n_src,
        coi_weight=config.training.coi_weight,
        snr_max=config.training.snr_max,
        zero_ref_loss_weight=config.training.zero_ref_loss_weight,
    )

    base_lr = float(config.training.lr)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=base_lr,
        weight_decay=float(config.training.weight_decay),
    )

    warmup_steps = int(config.training.warmup_steps)
    steps_per_epoch = max(
        1, len(train_loader) // max(1, config.training.grad_accum_steps)
    )
    total_steps = steps_per_epoch * config.training.num_epochs

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = config.training.use_amp and str(config.training.device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    validate_every_n = int(config.training.validate_every_n_epochs)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    global_step = 0
    history: dict = {"train_loss": [], "val_loss": [], "grad_norms": []}

    for epoch in range(1, config.training.num_epochs + 1):
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
            scheduler=scheduler,
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
                        "coi_prompts": config.model.coi_prompts,
                        "bg_prompt": config.model.bg_prompt,
                        "all_prompts": all_prompts,
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

    with open(checkpoint_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


# =============================================================================
# Entry point
# =============================================================================


def main():
    print(f"Loading config from {CONFIG_PATH}")
    config = Config.from_yaml(CONFIG_PATH)

    print(f"Device:      {config.training.device}")
    print(f"COI prompts: {config.model.coi_prompts}")
    print(f"BG prompt:   {config.model.bg_prompt}")

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
            "target_classes is empty – add class label lists to training_config.yml"
        )

    # Support a flat list (single class) or a list of lists (multi-class)
    if isinstance(target_classes[0], str):
        target_classes = [target_classes]

    n_coi = len(config.model.coi_prompts)
    if len(target_classes) != n_coi:
        raise ValueError(
            f"target_classes has {len(target_classes)} groups but "
            f"coi_prompts has {n_coi} entries – they must match."
        )

    print(f"\nTarget classes ({n_coi} groups):")
    for i, (labels, prompt) in enumerate(zip(target_classes, config.model.coi_prompts)):
        print(f"  [{i}] {labels} → prompt='{prompt}'")

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
        print(f"⚠️  Dropping {n_missing} missing files")
        sampled_df = sampled_df[sampled_df["file_exists"]]
    sampled_df = sampled_df.drop(columns=["file_exists"])
    print(f"✅ Final dataset: {len(sampled_df)} samples")

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
