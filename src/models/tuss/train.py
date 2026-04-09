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
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchaudio
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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
# Logging helpers for detached / pythonw runs
# ---------------------------------------------------------------------------


class _AutoFlushStream:
    """Wraps a file object and flushes after every write."""

    def __init__(self, f):
        self._f = f

    def write(self, text):
        self._f.write(text)
        self._f.flush()

    def flush(self):
        self._f.flush()

    def __getattr__(self, name):
        return getattr(self._f, name)


def _redirect_to_log(log_path: Path) -> None:
    """Redirect sys.stdout and sys.stderr to *log_path* (append mode).

    Only used when stdout is truly absent (pythonw launched without
    -RedirectStandardOutput).  When the caller has already redirected stdout
    to a file we leave that handle in place and just ensure it auto-flushes.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "a", encoding="utf-8", buffering=1)
    stream = _AutoFlushStream(fh)
    sys.stdout = stream  # type: ignore[assignment]
    sys.stderr = stream  # type: ignore[assignment]


def _ensure_autoflush() -> None:
    """Wrap sys.stdout/stderr with _AutoFlushStream if they exist but are not
    a TTY.  This prevents Python's default block-buffering from holding output
    in an 8 KB buffer when stdout is redirected to a file (e.g. via
    Start-Process -RedirectStandardOutput)."""
    if sys.stdout is not None and not _is_tty():
        sys.stdout = _AutoFlushStream(sys.stdout)  # type: ignore[assignment]
    if sys.stderr is not None and not _is_tty():
        sys.stderr = _AutoFlushStream(sys.stderr)  # type: ignore[assignment]


def _is_tty() -> bool:
    """Return True only when stdout is an interactive terminal."""
    try:
        return sys.stdout is not None and sys.stdout.isatty()
    except Exception:
        return False


class StepProgress:
    """Minimal tqdm replacement that writes plain-text progress lines.

    Used automatically when stdout is not a TTY (e.g. pythonw or a log file
    redirect).  Prints one line every *log_every_pct* percent of steps so the
    log file stays readable without being flooded.
    """

    def __init__(
        self,
        iterable,
        desc: str = "",
        total: int | None = None,
        log_every_pct: float = 5.0,
    ):
        self._it = iterable
        self._desc = desc
        try:
            self._total = total if total is not None else len(iterable)
        except TypeError:
            self._total = None
        self._n = 0
        self._postfix: dict = {}
        self._start = time.time()
        if self._total:
            self._log_every = max(1, int(self._total * log_every_pct / 100))
        else:
            self._log_every = 10  # fallback when length is unknown

    # Allow `with StepProgress(...) as p:` usage (mirrors tqdm interface)
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def __iter__(self):
        for item in self._it:
            yield item
            self._n += 1
            should_log = (self._n % self._log_every == 0) or (
                self._total is not None and self._n == self._total
            )
            if should_log:
                self._print_line()

    def _print_line(self):
        elapsed = time.time() - self._start
        rate = self._n / elapsed if elapsed > 0 else 0.0
        if self._total:
            pct = 100.0 * self._n / self._total
            eta = (self._total - self._n) / rate if rate > 0 else 0.0
            progress = f"{self._n}/{self._total} ({pct:.0f}%) eta {eta:.0f}s"
        else:
            progress = f"{self._n} steps"
        postfix = "  ".join(f"{k}={v}" for k, v in self._postfix.items())
        sep = "  |  " if postfix else ""
        print(f"  [{self._desc}] {progress}  {elapsed:.0f}s elapsed{sep}{postfix}")

    def set_postfix(self, refresh=True, **kwargs):  # noqa: ARG002
        self._postfix = {k: v for k, v in kwargs.items()}


def progress_bar(iterable, desc: str = "", total: int | None = None, **tqdm_kwargs):
    """Return a tqdm bar when interactive, StepProgress otherwise."""
    if _is_tty():
        return tqdm(
            iterable,
            desc=desc,
            total=total,
            leave=False,
            ascii=True,
            ncols=100,
            **tqdm_kwargs,
        )
    return StepProgress(iterable, desc=desc, total=total)


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

CONFIG_PATH = _SCRIPT_DIR / "training_config.yaml"


# =============================================================================
# Device resolution
# =============================================================================


def resolve_device(device: str | int) -> str:
    """Return a concrete device string from a flexible specification.

    Accepted forms
    ──────────────
    "cuda"      → "cuda:<index of best GPU>" (or "cpu" if none available)
    "cuda:N"    → validated, falls back to "cpu" if GPU N is absent
    N  (int)    → "cuda:N"  (e.g. 0, 1, 2 …)
    "cpu"       → "cpu"

    The resolved string is always safe to pass to ``tensor.to(device)``.
    """
    if isinstance(device, int):
        device = f"cuda:{device}"

    device = str(device).strip().lower()

    if device == "cpu":
        return "cpu"

    if device == "cuda":
        if not torch.cuda.is_available():
            print("No CUDA device found – falling back to CPU")
            return "cpu"
        idx = torch.cuda.current_device()
        return f"cuda:{idx}"

    if device.startswith("cuda:"):
        if not torch.cuda.is_available():
            print("No CUDA device found – falling back to CPU")
            return "cpu"
        try:
            idx = int(device.split(":")[1])
        except ValueError:
            print(f"Invalid device string '{device}' – falling back to cuda:0")
            idx = 0
        n_gpus = torch.cuda.device_count()
        if idx >= n_gpus:
            print(
                f"GPU {idx} requested but only {n_gpus} GPU(s) available – "
                f"falling back to cuda:0"
            )
            idx = 0
        return f"cuda:{idx}"

    print(f"Unrecognised device '{device}' – falling back to CPU")
    return "cpu"


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
    amp_dtype: str = "bf16"  # "bf16" (recommended, matches pretrained) or "fp16"
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
    resume_from: str = ""  # path to checkpoint .pt file to resume training
    seed: int = 42
    existing_prompt_lr_multiplier: float = 0.1  # LR multiplier for prompts that exist in checkpoint


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
        amp_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.n_src = n_src
        self.n_coi = n_src - 1  # last source is always background
        self.coi_weight = float(coi_weight)
        self.snr_max = snr_max
        self.zero_ref_loss_weight = zero_ref_loss_weight
        self.eps = eps
        # Stored so train_epoch / validate_epoch can read the dtype
        self._amp_dtype = amp_dtype

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
                    print(f"info() failed for {filepath}: {exc}")
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

    The mixture and all clean source references are normalised using the
    **same** statistics (mean and std of the raw mixture) so that
    ``sum(clean_sources) ≈ mixture`` is preserved.  This keeps the
    relative scale between sources consistent with what the model sees.

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

    # ---- Joint normalisation ------------------------------------------------
    # Compute statistics from the raw mixture and apply the *same* transform
    # to every source so that additivity is preserved.
    raw_mixture = total_coi + bg_scaled  # (B, T)
    mix_mean = raw_mixture.mean(dim=-1, keepdim=True)  # (B, 1)
    mix_std = raw_mixture.std(dim=-1, keepdim=True)  # (B, 1)
    is_silent = mix_std < NORMALIZE_MIN_STD
    mix_std_safe = torch.where(is_silent, torch.ones_like(mix_std), mix_std) + eps

    mixture = (raw_mixture - mix_mean) / mix_std_safe
    mixture = torch.where(is_silent, torch.zeros_like(mixture), mixture)

    # Normalise each clean source with the mixture's mean/std.
    # Each source is zero-centred around its own mean (so DC offsets don't
    # leak between sources) but scaled by the shared std.
    norm_cois = []
    for c in cois:
        c_mean = c.mean(dim=-1, keepdim=True)
        normed = (c - c_mean) / mix_std_safe
        normed = torch.where(is_silent, torch.zeros_like(normed), normed)
        norm_cois.append(normed)

    bg_mean = bg_scaled.mean(dim=-1, keepdim=True)
    norm_bg = (bg_scaled - bg_mean) / mix_std_safe
    norm_bg = torch.where(is_silent, torch.zeros_like(norm_bg), norm_bg)

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

    pbar = progress_bar(dataloader, desc="Training")
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
        pbar.set_postfix(
            loss=f"{batch_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}"
        )

        if str(device).startswith("cuda") and step_idx % 20 == 0:
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
    bg_sisnr_vals: list[float] = []

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
            if str(device).startswith("cuda") and val_step % 20 == 0:
                torch.cuda.empty_cache()

    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

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


def create_model(
    config: Config, 
    resume_ckpt_path: str | None = None
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
        resume_ckpt = torch.load(resume_ckpt_path, map_location="cpu", weights_only=False)
        resume_state_dict = resume_ckpt.get("model_state_dict", {})
        
        # Extract existing prompt names from checkpoint
        for key in resume_state_dict.keys():
            if key.startswith("separator.prompts."):
                prompt_name = key.replace("separator.prompts.", "", 1)
                existing_prompts_in_ckpt.add(prompt_name)
        
        if existing_prompts_in_ckpt:
            print(f"  Found {len(existing_prompts_in_ckpt)} existing prompts in checkpoint: {sorted(existing_prompts_in_ckpt)}")

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
            # Small noise so each new COI prompt starts from a slightly
            # different position even if they all init from "sfx"
            noise = torch.randn_like(init_val) * 0.001
            prompts_dict[prompt_name] = torch.nn.Parameter(init_val + noise)
            newly_injected.append(prompt_name)
            print(f"  Injected NEW prompt '{prompt_name}' (init from '{init_from}')")
        else:
            print(f"  Prompt '{prompt_name}' already exists – keeping pretrained value")
    
    # Now load the resume checkpoint weights (if provided)
    # This will load existing prompts but leave newly injected ones untouched
    if resume_state_dict is not None:
        print(f"Loading checkpoint weights (newly injected prompts will be preserved)...")
        missing, unexpected = model.load_state_dict(resume_state_dict, strict=False)
        if newly_injected:
            print(f"  ✓ {len(newly_injected)} new prompt(s) preserved: {newly_injected}")
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
        'new_prompts': [],
        'continuing_prompts': [],
        'frozen_prompts': [],
        'backbone': []
    }
    
    if resume_ckpt_path:
        checkpoint_prompts = get_prompts_from_checkpoint(resume_ckpt_path)
        new_p, continuing_p, frozen_p = classify_prompts_by_training_strategy(
            config.model.coi_prompts,
            config.model.bg_prompt,
            checkpoint_prompts
        )
        
        # Collect parameters for each prompt category
        prompts_dict = model.separator.prompts
        
        for prompt_name in new_p:
            if prompt_name in prompts_dict:
                param_groups['new_prompts'].append(prompts_dict[prompt_name])
        
        for prompt_name in continuing_p:
            if prompt_name in prompts_dict:
                param_groups['continuing_prompts'].append(prompts_dict[prompt_name])
        
        for prompt_name in frozen_p:
            if prompt_name in prompts_dict:
                prompts_dict[prompt_name].requires_grad_(False)
                param_groups['frozen_prompts'].append(prompts_dict[prompt_name])
        
        # Collect backbone parameters (everything except prompts)
        for name, param in model.named_parameters():
            if 'prompts' not in name and param.requires_grad:
                param_groups['backbone'].append(param)
    else:
        # No checkpoint: all config prompts are "new", no frozen prompts
        prompts_dict = model.separator.prompts
        for prompt_name in config.model.coi_prompts + [config.model.bg_prompt]:
            if prompt_name in prompts_dict:
                param_groups['new_prompts'].append(prompts_dict[prompt_name])
        
        for name, param in model.named_parameters():
            if 'prompts' not in name and param.requires_grad:
                param_groups['backbone'].append(param)

    return model, param_groups


# =============================================================================
# Checkpoint validation utilities
# =============================================================================


def get_prompts_from_checkpoint(checkpoint_path: str | Path) -> set[str]:
    """Extract prompt names from a checkpoint file.
    
    Returns:
        Set of prompt names found in the checkpoint's model state.
    """
    if not checkpoint_path or not Path(checkpoint_path).is_file():
        return set()
    
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_state = ckpt.get("model_state_dict", {})
        
        prompts = set()
        prompt_prefix = "separator.prompts."
        for key in model_state.keys():
            if key.startswith(prompt_prefix):
                # Extract just the prompt name (use replace with count=1 to be explicit)
                prompt_name = key.replace(prompt_prefix, "", 1)
                prompts.add(prompt_name)
        
        return prompts
    except Exception as e:
        print(f"⚠ Warning: Could not read prompts from checkpoint: {e}")
        return set()


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
    
    checkpoint_prompts = get_prompts_from_checkpoint(checkpoint_path)
    
    if not checkpoint_prompts:
        # Checkpoint doesn't have prompts or couldn't be read
        return [], config_prompts + [bg_prompt]
    
    all_config_prompts = set(config_prompts + [bg_prompt])
    existing = sorted(all_config_prompts & checkpoint_prompts)
    new = sorted(all_config_prompts - checkpoint_prompts)
    frozen = sorted(checkpoint_prompts - all_config_prompts)
    
    print("\n" + "="*70)
    print("PROMPT VALIDATION")
    print("="*70)
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
        print(f"   All prompts exist in checkpoint - will continue training with more data")
    elif new and existing:
        print(f"\n📌 Training mode: EXTEND (add new classes)")
        print(f"   Existing prompts will continue fine-tuning")
        print(f"   New prompts will learn from scratch")
    elif new and not existing:
        print(f"\n📌 Training mode: FRESH START")
        print(f"   All prompts are new - training from base model")
    
    print("="*70 + "\n")
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
            print(f"ℹ️  Extending model: keeping {len(existing)} existing + adding {len(new)} new prompts")
        elif new and not existing:
            print("ℹ️  Starting fresh with new prompts")

    print("\nCreating model …")
    model, param_groups = create_model(config, resume_ckpt_path=resume_path if resume_path else None)

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
    if param_groups['new_prompts'] or param_groups['continuing_prompts'] or param_groups['frozen_prompts']:
        print("\n" + "="*70)
        print("PARAMETER GROUPS")
        print("="*70)
        
        if param_groups['new_prompts']:
            names = [name for name, p in model.separator.prompts.items() 
                    if any(p is param for param in param_groups['new_prompts'])]
            n_params = sum(p.numel() for p in param_groups['new_prompts'])
            print(f"\n🆕 New prompts (full LR: {config.training.lr:.1e}):")
            print(f"   {', '.join(names)}")
            print(f"   Total: {n_params:,} parameters")
        
        if param_groups['continuing_prompts']:
            names = [name for name, p in model.separator.prompts.items() 
                    if any(p is param for param in param_groups['continuing_prompts'])]
            n_params = sum(p.numel() for p in param_groups['continuing_prompts'])
            reduced_lr = config.training.lr * config.training.existing_prompt_lr_multiplier
            print(f"\n🔄 Continuing prompts (reduced LR: {reduced_lr:.1e}):")
            print(f"   {', '.join(names)}")
            print(f"   Total: {n_params:,} parameters")
        
        if param_groups['frozen_prompts']:
            names = [name for name, p in model.separator.prompts.items() 
                    if any(p is param for param in param_groups['frozen_prompts'])]
            n_params = sum(p.numel() for p in param_groups['frozen_prompts'])
            print(f"\n❄️  Frozen prompts (no training):")
            print(f"   {', '.join(names)}")
            print(f"   Total: {n_params:,} parameters")
        
        if param_groups['backbone']:
            n_params = sum(p.numel() for p in param_groups['backbone'])
            print(f"\n🏗️  Backbone (LR: {config.training.lr:.1e}):")
            print(f"   Total: {n_params/1e6:.1f}M parameters")
        
        print("="*70 + "\n")

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

    if param_groups['new_prompts']:
        optimizer_param_groups.append({
            'params': param_groups['new_prompts'],
            'lr': base_lr,
            'name': 'new_prompts'
        })

    if param_groups['continuing_prompts']:
        optimizer_param_groups.append({
            'params': param_groups['continuing_prompts'],
            'lr': base_lr * existing_lr_mult,
            'name': 'continuing_prompts'
        })

    if param_groups['backbone']:
        if config.model.freeze_backbone:
            # Backbone is frozen, don't add to optimizer
            pass
        else:
            optimizer_param_groups.append({
                'params': param_groups['backbone'],
                'lr': base_lr,
                'name': 'backbone'
            })

    # If no parameter groups (shouldn't happen), fall back to old behavior
    if not optimizer_param_groups:
        optimizer_param_groups = [{'params': filter(lambda p: p.requires_grad, model.parameters())}]

    optimizer = optim.AdamW(optimizer_param_groups, weight_decay=weight_decay)

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
    # GradScaler is only needed for fp16; bf16 has sufficient dynamic range.
    _amp_backend = str(config.training.device).split(":")[0]
    needs_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(_amp_backend, enabled=True) if needs_scaler else None
    
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Device: {config.training.device}")
    print(f"AMP enabled: {use_amp}")
    print(f"AMP dtype: {amp_dtype}")
    print(f"GradScaler: {scaler is not None}")
    if str(config.training.device).startswith("cuda"):
        if torch.cuda.is_available():
            device_idx = int(config.training.device.split(":")[1]) if ":" in config.training.device else 0
            print(f"GPU: {torch.cuda.get_device_name(device_idx)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(device_idx).total_memory / 1e9:.2f} GB")
        else:
            print("⚠️  WARNING: CUDA device specified but torch.cuda.is_available() = False")
    print("="*70 + "\n")

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
        ckpt = torch.load(resume_path, map_location=config.training.device, weights_only=False)
        # Model weights were already loaded in create_model(), only load optimizer/scheduler/history
        
        # Try to load optimizer state, but skip if parameter groups have changed
        # (happens when extending model with new prompts)
        has_new_prompts = bool(param_groups['new_prompts'])
        has_frozen_prompts = bool(param_groups['frozen_prompts'])
        is_extending = has_new_prompts or has_frozen_prompts
        
        if is_extending:
            print("  ⚠️  Model is being extended with new/frozen prompts")
            print("     Skipping optimizer state loading (will start fresh)")
            print("     This is expected and safe - Adam will build new momentum for all parameters")
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
        
        # Reset best_val_loss when extending with new prompts
        # Otherwise new prompts won't get a chance to save checkpoints
        if is_extending:
            best_val_loss = float("inf")
            epochs_without_improvement = 0
            print("  ⚠️  Resetting best_val_loss to inf (new prompts need to learn)")
            print(f"     Previous checkpoint had val_loss: {ckpt.get('val_loss', 'N/A')}")
        else:
            best_val_loss = float(ckpt.get("val_loss", float("inf")))
        
        history = ckpt.get("history", history)
        
        # Restore scheduler state so the LR curve continues seamlessly.
        if "scheduler_state_dict" in ckpt and not is_extending:
            # Only restore scheduler if we're continuing (not extending)
            # When extending, we want to start fresh with warmup
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            print("  ✓ Loaded scheduler state")
        else:
            if is_extending:
                print("  ⚠️  Starting fresh scheduler (will apply warmup from beginning)")
            else:
                # Fallback for checkpoints saved before scheduler state was added:
                # replay steps so the LR is consistent with where training left off.
                print("  ⚠ No scheduler state in checkpoint – replaying steps …")
                for _ in range(global_step):
                    scheduler.step()
        
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
                ckpt_payload = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
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
