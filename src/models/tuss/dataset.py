"""
Multi-class COI audio separation dataset.

Returns a (n_coi_classes + 1, T) source tensor per item:
    sources[:n_coi_classes]  – one track per COI class (most are silent)
    sources[-1]              – a single background / non-COI track

The mixture and normalisation are produced later in prepare_batch so that
they can be done efficiently on GPU.
"""

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from common.audio_utils import ResamplerCache
from common.training_utils import get_audio_info as _get_audio_info
from .augmentations import AudioAugmentations

RESAMPLER_CACHE_MAX = 8


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
        multi_coi_prob: float = 0.3,
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
        self.multi_coi_prob = multi_coi_prob if split == "train" else 0.0

        self._rng = np.random.default_rng(42)
        self._resampler_cache = ResamplerCache(max_size=RESAMPLER_CACHE_MAX)
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
        self.coi_segments_by_class: list[list[tuple[str, int, int, int]]] = [
            [] for _ in range(n_coi_classes)
        ]
        for seg in self.coi_segments:
            self.coi_segments_by_class[seg[3]].append(seg)

        # Build a class-balanced training schedule so each COI class contributes
        # the same number of samples per epoch.
        self._balanced_train_schedule: list[tuple[str, int, int, int]] = []
        if self.split == "train" and self.coi_segments_by_class:
            max_class_len = max(
                (len(cls) for cls in self.coi_segments_by_class), default=0
            )
            if max_class_len > 0:
                for class_idx, seg_list in enumerate(self.coi_segments_by_class):
                    if not seg_list:
                        continue
                    for i in range(max_class_len):
                        self._balanced_train_schedule.append(
                            seg_list[i % len(seg_list)]
                        )

                # Keep the schedule deterministic per epoch but still shuffled by DataLoader.
                self._rng.shuffle(self._balanced_train_schedule)

        if self.split == "train":
            self.coi_segments_train = list(self.coi_segments)

        self._extra_background_count = 0
        if self.coi_files and self.background_only_prob > 0.0:
            if split == "train" and self._balanced_train_schedule:
                base = len(self._balanced_train_schedule)
            else:
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
        if self._balanced_train_schedule:
            self._rng.shuffle(self._balanced_train_schedule)

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
            if self._balanced_train_schedule:
                base = len(self._balanced_train_schedule) * self.augment_multiplier
            else:
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
            waveform = self._resampler_cache.resample_padded(waveform, sr, self.sample_rate)

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
        """Return base_offset with a small random jitter, clamped to file bounds.

        The jitter is at most ±half-stride (in native-sr frames) to provide
        training diversity while still systematically covering the file."""
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
            schedule = self._balanced_train_schedule or self.coi_segments_train
            coi_count = len(schedule)
            effective_coi_count = coi_count * self.augment_multiplier

            if coi_count > 0 and idx < effective_coi_count:
                actual_idx = idx % coi_count
                augment_variant = idx // coi_count
                filepath, base_offset, seg_frames, class_idx = schedule[actual_idx]
                offset = self._jittered_offset(base_offset, seg_frames, filepath)
                coi_audio = self._load_audio(
                    filepath, frame_offset=offset, num_frames=seg_frames
                )
                if self.augment:
                    coi_audio = AudioAugmentations.random_augment(coi_audio, self._rng)

                sources = [
                    torch.zeros(self.segment_samples) for _ in range(self.n_coi_classes)
                ]
                sources[class_idx] = coi_audio

                if self.n_coi_classes > 1 and self._rng.random() < getattr(
                    self, "multi_coi_prob", 0.0
                ):
                    available_classes = [
                        c
                        for c in range(self.n_coi_classes)
                        if c != class_idx and len(self.coi_segments_by_class[c]) > 0
                    ]
                    if available_classes:
                        class_idx_2 = int(self._rng.choice(available_classes))
                        seg_list = self.coi_segments_by_class[class_idx_2]
                        seg_idx = int(self._rng.integers(len(seg_list)))
                        filepath_2, base_offset_2, seg_frames_2, _ = seg_list[seg_idx]
                        offset_2 = self._jittered_offset(
                            base_offset_2, seg_frames_2, filepath_2
                        )
                        coi_audio_2 = self._load_audio(
                            filepath_2, frame_offset=offset_2, num_frames=seg_frames_2
                        )
                        if self.augment:
                            coi_audio_2 = AudioAugmentations.random_augment(
                                coi_audio_2, self._rng
                            )
                        sources[class_idx_2] = coi_audio_2
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
