"""
Dataset module for SuDORMRF audio separation training.

Provides:
    - AudioDataset: Torch Dataset for loading and mixing COI/background audio
    - _worker_init_fn: Worker init function for DataLoader reproducibility
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from common.audio_utils import ResamplerCache
from common.training_utils import get_audio_info as _get_audio_info

from .augmentations import AudioAugmentations

RESAMPLER_CACHE_MAX = 8


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
        seed: int = 42,
        multi_coi_prob: float = 0.0,
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
        self.multi_coi_prob = float(multi_coi_prob)
        self.segment_stride_samples = int(
            (segment_stride or segment_length) * sample_rate
        )

        self._seed = seed
        self._epoch: int = 0
        self._rng = np.random.default_rng(seed)
        self._resampler_cache = ResamplerCache(max_size=RESAMPLER_CACHE_MAX)
        self._file_native_info: dict[
            str, tuple[int, int]
        ] = {}  # filepath -> (orig_sr, total_frames)

        # Filter and extract file lists
        if split == "test":
            split_df = dataframe.iloc[0:0]
        else:
            split_df = dataframe[dataframe["split"] == split]

        coi_mask = split_df["label"] == 1
        self.coi_files = split_df.loc[coi_mask, "filename"].tolist()
        self.non_coi_files = split_df.loc[~coi_mask, "filename"].tolist()

        self.file_to_bounds = {}
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

        # File to class mapping for multi-class
        self.file_to_class = {}
        if n_coi_classes > 1 and "coi_class" in split_df.columns:
            self.file_to_class = dict(zip(split_df["filename"], split_df["coi_class"]))

        # Per-class file lists (needed for COI dropout and val pre-generation)
        self.class_files: dict[int, list[str]] = {}
        if n_coi_classes > 1 and "coi_class" in split_df.columns:
            for _ci in range(n_coi_classes):
                _files = split_df.loc[
                    (split_df["label"] == 1) & (split_df["coi_class"] == _ci),
                    "filename",
                ].tolist()
                if _files:
                    self.class_files[_ci] = _files

        # Precompute segments for validation/test, or for long files in train
        self.coi_segments = self._compute_segments(split_df)

        # In train, keep the full segment tuples so __getitem__ can load each
        # segment at its proper offset (with a small random jitter for
        # diversity) instead of picking a fully random crop.
        if self.split == "train":
            self.coi_segments_train = list(self.coi_segments)

        # Pre-generate val multi-class extras (deterministic, seeded at init).
        # For each val COI segment, independently decide (per other class)
        # whether to add a sample of that class to the mixture.
        self.val_multi_class: list[list[tuple[int, str, int, int]]] = []
        if (
            split == "val"
            and n_coi_classes > 1
            and self.multi_coi_prob > 0.0
            and self.class_files
        ):
            _val_rng = np.random.default_rng(seed)
            for _, _, _seg_frames, _primary_cls in self.coi_segments:
                _extras: list[tuple[int, str, int, int]] = []
                for _j in range(n_coi_classes):
                    if _j == _primary_cls or not self.class_files.get(_j):
                        continue
                    if _val_rng.random() < self.multi_coi_prob:
                        _j_files = self.class_files[_j]
                        _f = _j_files[int(_val_rng.integers(len(_j_files)))]
                        _orig_sr, _total_frames = self._file_native_info.get(
                            _f, (self.sample_rate, self.segment_samples)
                        )
                        _bounds = self.file_to_bounds.get(_f, (0.0, None))
                        _start_f = int(_bounds[0] * _orig_sr)
                        _end_f = (
                            int(_bounds[1] * _orig_sr)
                            if _bounds[1] is not None
                            else _total_frames
                        )
                        _end_f = min(_end_f, _total_frames)
                        _extra_seg = max(
                            1,
                            int(self.segment_samples * _orig_sr / self.sample_rate),
                        )
                        _max_off = max(_start_f, _end_f - _extra_seg)
                        _off = (
                            int(_val_rng.integers(_start_f, _max_off + 1))
                            if _max_off > _start_f
                            else _start_f
                        )
                        _extras.append((_j, _f, _off, _extra_seg))
                self.val_multi_class.append(_extras)

        # Background-only sample count
        self._extra_background_count = 0
        if self.coi_files and self.background_only_prob > 0.0:
            if split == "train":
                self._extra_background_count = int(
                    self.background_only_prob
                    * len(self.coi_segments_train)
                    * self.augment_multiplier
                    + 0.5
                )
            elif split == "val":
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
                f"{split} set: {len(self.coi_files)} unique COI files ({len(self.coi_segments)} segments), {len(self.non_coi_files)} non-COI files"
            )
            if split == "train" and self.augment_multiplier > 1:
                print(
                    f"  → With {self.augment_multiplier}x augmentation: {len(self.coi_segments_train) * self.augment_multiplier} effective epoch steps"
                )
            if self._extra_background_count > 0:
                print(
                    f"  → With {self._extra_background_count} background-only samples"
                )

    def set_epoch(self, epoch: int):
        """Re-seed the internal RNG for a new epoch to ensure augmentation diversity."""
        self._epoch = epoch
        self._rng = np.random.default_rng(self._seed + epoch)

    def _compute_segments(
        self, split_df: pd.DataFrame
    ) -> list[tuple[str, int, int | None, int]]:
        """Compute segments for each COI file."""
        segments = []
        info_failures = 0
        for filepath in self.coi_files:
            class_idx = int(self.file_to_class.get(filepath, 0))
            bounds = self.file_to_bounds.get(filepath, (0.0, None))
            start_sec, end_sec = bounds

            try:
                orig_sr, num_frames = _get_audio_info(filepath)
                self._file_native_info[filepath] = (orig_sr, num_frames)
            except Exception as exc:
                info_failures += 1
                if info_failures <= 5:
                    print(f"  ⚠ torchaudio.info() failed for {filepath}: {exc}")
                # Fallback: use bounds-based duration or est_segments from the
                # dataframe so we can still create a reasonable number of
                # segments even when the file header can't be read.
                orig_sr = self.sample_rate  # assume target sr as approximation
                if end_sec is not None and end_sec > start_sec:
                    num_frames = int(end_sec * orig_sr)
                else:
                    num_frames = self.segment_samples  # single segment fallback
                self._file_native_info[filepath] = (orig_sr, num_frames)

            seg_frames = max(1, int(self.segment_samples * orig_sr / self.sample_rate))
            stride_frames = max(
                1, int(self.segment_stride_samples * orig_sr / self.sample_rate)
            )

            start_frame = int(start_sec * orig_sr)
            end_frame = int(end_sec * orig_sr) if end_sec is not None else num_frames
            end_frame = min(end_frame, num_frames)

            valid_frames = max(0, end_frame - start_frame)

            n_segs = (
                1
                if valid_frames <= seg_frames
                else 1 + max(0, (valid_frames - seg_frames) // stride_frames)
            )
            for s in range(n_segs):
                offset = start_frame + s * stride_frames
                segments.append((filepath, offset, seg_frames, class_idx))

        if info_failures > 0:
            print(
                f"  ⚠ torchaudio.info() failed for {info_failures}/{len(self.coi_files)} COI files "
                f"(used bounds/fallback for segment estimation)"
            )
        return segments

    def __len__(self):
        if self.split == "train":
            if hasattr(self, "coi_segments_train") and self.coi_segments_train:
                return (
                    len(self.coi_segments_train) * self.augment_multiplier
                    + self._extra_background_count
                )
            return len(self.non_coi_files)
        return len(self.coi_segments) + self._extra_background_count

    def _load_audio(
        self,
        filepath: str,
        frame_offset: int | None = None,
        num_frames: int | None = None,
    ) -> torch.Tensor:
        """Load and preprocess audio segment.

        Never loads more than a bounded window from disk to avoid OOM on
        multi-hour files.
        """
        bounds = getattr(self, "file_to_bounds", {}).get(filepath, (0.0, None))
        start_sec, end_sec = bounds

        # --- resolve native sample-rate and file length ---
        cached = self._file_native_info.get(filepath)
        if cached is not None:
            orig_sr, total_frames = cached
        else:
            try:
                orig_sr, total_frames = _get_audio_info(filepath)
                self._file_native_info[filepath] = (orig_sr, total_frames)
            except Exception:
                # Fallback: use a bounded chunk since the header cannot be read.
                max_load = self.segment_samples * 2
                try:
                    waveform, sr = torchaudio.load(
                        filepath, frame_offset=0, num_frames=max_load
                    )
                except Exception:
                    # Return silence as a last-resort fallback.
                    return torch.zeros(self.segment_samples)
                orig_sr = sr
                total_frames = waveform.shape[-1]
                self._file_native_info[filepath] = (orig_sr, total_frames)

                seg_frames = num_frames or max(
                    1, int(self.segment_samples * orig_sr / self.sample_rate)
                )
                waveform = waveform[:, :seg_frames]
                if sr != self.sample_rate:
                    waveform = self._resampler_cache.resample(waveform, sr, self.sample_rate)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                waveform = waveform.squeeze(0)
                if waveform.shape[0] < self.segment_samples:
                    waveform = F.pad(
                        waveform, (0, self.segment_samples - waveform.shape[0])
                    )
                return waveform[: self.segment_samples]

        # --- compute the window to read ---
        seg_frames = num_frames or max(
            1, int(self.segment_samples * orig_sr / self.sample_rate)
        )

        start_frame = int(start_sec * orig_sr)
        end_frame = int(end_sec * orig_sr) if end_sec is not None else total_frames
        end_frame = min(end_frame, total_frames)

        if frame_offset is None:
            max_offset = max(start_frame, end_frame - seg_frames)
            if max_offset > start_frame:
                offset = int(self._rng.integers(start_frame, max_offset + 1))
            else:
                offset = start_frame
        else:
            offset = int(frame_offset)

        # Clamp so we never read past end of file
        offset = max(0, min(offset, max(0, total_frames - seg_frames)))

        try:
            waveform, sr = torchaudio.load(
                filepath, frame_offset=offset, num_frames=int(seg_frames)
            )
        except Exception:
            # Retry from start of file when frame_offset is unsupported
            # (some backends reject near-EOF offsets).
            try:
                waveform, sr = torchaudio.load(
                    filepath, frame_offset=0, num_frames=int(seg_frames)
                )
            except Exception:
                return torch.zeros(self.segment_samples)

        # Resample if needed using high-quality Kaiser windowed sinc
        if sr != self.sample_rate:
            waveform = self._resampler_cache.resample(waveform, sr, self.sample_rate)

        # Mono and pad/trim
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        if waveform.shape[0] < self.segment_samples:
            waveform = F.pad(
                waveform, (0, self.segment_samples - waveform.shape[0])
            )
        return waveform[: self.segment_samples]

    def _jittered_offset(
        self, base_offset: int, seg_frames: int | None, filepath: str
    ) -> int:
        """Return *base_offset* with a small random jitter.

        The jitter is at most ±half-stride (in native-sr frames) and is clamped
        so that the resulting window stays within the file's valid region.
        This gives training diversity while still systematically covering the
        whole file across an epoch.
        """
        if seg_frames is None:
            return base_offset

        cached = self._file_native_info.get(filepath)
        if cached is None:
            # Fallback: should not happen if _compute_segments ran first
            try:
                cached = _get_audio_info(filepath)
                self._file_native_info[filepath] = cached
            except Exception:
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

        # Clamp to valid region
        offset = max(start_frame, offset)
        offset = min(offset, max(start_frame, end_frame - seg_frames))
        return offset

    def __getitem__(self, idx):
        background = None

        if self.split == "train":
            coi_count = len(self.coi_segments_train)
            effective_coi_count = coi_count * self.augment_multiplier

            if coi_count > 0 and idx < effective_coi_count:
                # COI sample – use precomputed segment offset with jitter
                actual_idx = idx % coi_count
                augment_variant = idx // coi_count
                filepath, base_offset, seg_frames, class_idx = self.coi_segments_train[
                    actual_idx
                ]

                # Jitter the offset so the model sees slightly different
                # cuts each epoch while still systematically covering the
                # whole file (unlike a fully random crop).
                offset = self._jittered_offset(base_offset, seg_frames, filepath)
                coi_audio = self._load_audio(
                    filepath, frame_offset=offset, num_frames=seg_frames
                )

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

                # COI dropout: independently add each other class to the mixture
                if self.multi_coi_prob > 0.0 and self.n_coi_classes > 1:
                    for _j in range(self.n_coi_classes):
                        if _j == class_idx:
                            continue
                        if not self.class_files.get(_j):
                            continue
                        if self._rng.random() < self.multi_coi_prob:
                            _f = self.class_files[_j][
                                int(self._rng.integers(len(self.class_files[_j])))
                            ]
                            sources[_j] = self._load_audio(_f, frame_offset=None)
            else:
                # Background-only sample - create mixture of multiple background sources
                idxs = self._rng.choice(
                    len(self.non_coi_files),
                    size=max(1, self.background_mix_n),
                    replace=True,
                )
                background_sources = [
                    self._load_audio(self.non_coi_files[int(i)], frame_offset=None)
                    for i in idxs
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

                # Inject pre-generated extra COI classes for val multi-class mixtures
                if self.val_multi_class and idx < len(self.val_multi_class):
                    for _e_cls, _e_file, _e_off, _e_nf in self.val_multi_class[idx]:
                        sources[_e_cls] = self._load_audio(_e_file, _e_off, _e_nf)
            else:
                # Background-only validation sample
                idxs = self._rng.choice(
                    len(self.non_coi_files),
                    size=max(1, self.background_mix_n),
                    replace=True,
                )
                background = torch.stack(
                    [
                        self._load_audio(self.non_coi_files[int(i)], frame_offset=None)
                        for i in idxs
                    ]
                ).sum(dim=0)
                sources = [
                    torch.zeros_like(background) for _ in range(self.n_coi_classes)
                ]

        # Add background
        if background is None:
            background = self._load_audio(
                self.non_coi_files[int(self._rng.integers(len(self.non_coi_files)))],
                frame_offset=None,
            )
        sources.append(background)

        sources_tensor = torch.stack(sources, dim=0)
        return sources_tensor


def _worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker's dataset RNG deterministically per epoch and worker.

    Called by PyTorch at worker spawn time.  Because persistent_workers=False the
    worker is re-spawned every epoch, so set_epoch() has already updated
    ds._epoch before these workers are created, and each worker gets a unique
    but reproducible seed: seed + epoch * 31337 + worker_id.
    """
    info = torch.utils.data.get_worker_info()
    if info is not None:
        ds = info.dataset
        if hasattr(ds, "_seed"):
            epoch = getattr(ds, "_epoch", 0)
            ds._rng = np.random.default_rng(ds._seed + epoch * 31337 + info.id)
