"""
WebDataset utilities for efficient compressed audio data loading.

This module provides a unified interface for loading audio data from either:
1. Regular file paths (original behavior)
2. WebDataset tar shards (compressed format)

The abstraction allows seamless switching between data sources without
modifying training/inference code.

Key components:
- AudioLoader: Abstract base class for audio loading
- FileAudioLoader: Loads audio from regular file paths
- WebDatasetAudioLoader: Loads audio from WebDataset tar shards
- create_audio_loader: Factory function to create appropriate loader

WebDataset tar shard format:
    Each sample in the tar contains:
    - {sample_id}.flac: Audio data (FLAC for lossless compression)
    - {sample_id}.json: Metadata (start_time, end_time, label, split, dataset, etc.)

Example usage:
    >>> # File-based loading (original)
    >>> loader = create_audio_loader(mode="file")
    >>> waveform, sr = loader.load_audio("/path/to/audio.wav", start=0.0, end=5.0)
    
    >>> # WebDataset loading
    >>> loader = create_audio_loader(
    ...     mode="webdataset",
    ...     tar_paths=["/data/shards/train-{000000..000099}.tar"],
    ...     cache_dir="/tmp/wds_cache"
    ... )
    >>> dataset = loader.create_dataset(target_sr=16000, segment_length=5.0)

References:
    - WebDataset: https://github.com/webdataset/webdataset
    - torchaudio: https://pytorch.org/audio/stable/index.html
"""

import io
import json
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torchaudio

from .audio_utils import ResamplerCache

# Try to import webdataset - it's optional for file-based loading
try:
    import webdataset as wds
    WEBDATASET_AVAILABLE = True
except ImportError:
    WEBDATASET_AVAILABLE = False
    wds = None


# =============================================================================
# Audio Loading Interface
# =============================================================================


class AudioLoader(ABC):
    """Abstract base class for audio loading strategies."""

    @abstractmethod
    def load_audio(
        self,
        identifier: Any,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        target_sr: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Load audio from the data source.

        Args:
            identifier: Source-specific identifier (filepath for files, key for webdataset)
            start_time: Start time in seconds (None for full file)
            end_time: End time in seconds (None for full file)
            target_sr: Target sample rate (None to keep original)

        Returns:
            Tuple of (waveform, sample_rate)
        """
        pass


class FileAudioLoader(AudioLoader):
    """
    Load audio from regular file paths.

    This is the original loading strategy used in the codebase.
    Supports partial loading via frame_offset and num_frames for efficiency.
    """

    def __init__(self, resampler_cache: Optional[ResamplerCache] = None):
        """
        Args:
            resampler_cache: Optional shared resampler cache for efficiency
        """
        self._resampler_cache = resampler_cache or ResamplerCache(max_size=8)

    def load_audio(
        self,
        identifier: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        target_sr: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Load audio segment from file.

        Args:
            identifier: Path to audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            target_sr: Target sample rate for resampling

        Returns:
            Tuple of (waveform, sample_rate)
        """
        filepath = str(identifier)

        try:
            # Get file info for efficient partial loading
            info = torchaudio.info(filepath)
            orig_sr = info.sample_rate

            # Calculate frame offsets for partial loading
            frame_offset = 0
            num_frames = -1  # Load all frames by default

            if start_time is not None and end_time is not None:
                frame_offset = int(start_time * orig_sr)
                num_frames = int((end_time - start_time) * orig_sr)
                # Clamp to valid range
                frame_offset = max(0, min(frame_offset, info.num_frames - 1))
                num_frames = min(num_frames, info.num_frames - frame_offset)

            waveform, sr = torchaudio.load(
                filepath, frame_offset=frame_offset, num_frames=num_frames
            )

        except Exception as e:
            warnings.warn(f"Failed to load {filepath}: {e}")
            raise

        # Resample if needed
        if target_sr is not None and sr != target_sr:
            waveform = self._resampler_cache.resample(waveform, sr, target_sr)
            sr = target_sr

        return waveform, sr


class WebDatasetAudioLoader(AudioLoader):
    """
    Load audio from WebDataset tar shards.

    Expects tar shards with structure:
        {sample_id}.flac - Audio data
        {sample_id}.json - Metadata (start_time, end_time, label, etc.)

    Note: This loader creates iterators over the dataset. For random access,
    use the in-memory cache or convert to a standard dataset first.
    """

    def __init__(
        self,
        tar_paths: Union[str, List[str]],
        cache_dir: Optional[str] = None,
        shuffle: bool = False,
        resampler_cache: Optional[ResamplerCache] = None,
    ):
        """
        Args:
            tar_paths: Path pattern or list of paths to tar shards
                      e.g., "/data/shards/train-{000000..000099}.tar"
            cache_dir: Directory to cache extracted samples (optional)
            shuffle: Whether to shuffle shards
            resampler_cache: Optional shared resampler cache
        """
        if not WEBDATASET_AVAILABLE:
            raise ImportError(
                "webdataset is required for WebDatasetAudioLoader. "
                "Install it with: pip install webdataset"
            )

        self._tar_paths = tar_paths if isinstance(tar_paths, list) else [tar_paths]
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._shuffle = shuffle
        self._resampler_cache = resampler_cache or ResamplerCache(max_size=8)

        # In-memory sample cache for random access (limited size)
        self._sample_cache: Dict[str, Tuple[torch.Tensor, int, Dict]] = {}
        self._max_cache_size = 1000

    def load_audio(
        self,
        identifier: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        target_sr: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Load audio from cache (WebDataset samples are pre-loaded in batch).

        For WebDataset, this method expects the sample to be in the cache.
        Use create_iterator() to iterate through the dataset instead.

        Args:
            identifier: Sample key in the cache
            start_time: Start time in seconds (not used - samples are pre-segmented)
            end_time: End time in seconds (not used - samples are pre-segmented)
            target_sr: Target sample rate

        Returns:
            Tuple of (waveform, sample_rate)
        """
        if identifier not in self._sample_cache:
            raise KeyError(
                f"Sample {identifier} not in cache. "
                "Use create_iterator() to iterate through WebDataset samples."
            )

        waveform, sr, _metadata = self._sample_cache[identifier]

        if target_sr is not None and sr != target_sr:
            waveform = self._resampler_cache.resample(waveform, sr, target_sr)
            sr = target_sr

        return waveform, sr

    def _decode_audio(self, audio_bytes: bytes) -> Tuple[torch.Tensor, int]:
        """Decode audio from bytes (FLAC format expected)."""
        buffer = io.BytesIO(audio_bytes)
        waveform, sr = torchaudio.load(buffer, format="flac")
        return waveform, sr

    def _decode_json(self, json_bytes: bytes) -> Dict:
        """Decode JSON metadata from bytes."""
        return json.loads(json_bytes.decode("utf-8"))

    def create_raw_iterator(self) -> Iterator[Tuple[str, torch.Tensor, int, Dict]]:
        """
        Create an iterator over raw WebDataset samples.

        Yields:
            Tuples of (sample_key, waveform, sample_rate, metadata)
        """
        # Expand brace patterns in paths
        url_pattern = " ".join(self._tar_paths)

        # Create WebDataset pipeline
        dataset = wds.WebDataset(url_pattern)

        if self._shuffle:
            dataset = dataset.shuffle(1000)

        for sample in dataset:
            key = sample["__key__"]

            # Find audio file (try common extensions)
            audio_data = None
            for ext in ["flac", "wav", "mp3", "ogg"]:
                if ext in sample:
                    audio_data = sample[ext]
                    break

            if audio_data is None:
                warnings.warn(f"No audio found for sample {key}")
                continue

            # Decode audio
            try:
                waveform, sr = self._decode_audio(audio_data)
            except Exception as e:
                warnings.warn(f"Failed to decode audio for {key}: {e}")
                continue

            # Decode metadata
            metadata = {}
            if "json" in sample:
                try:
                    metadata = self._decode_json(sample["json"])
                except Exception as e:
                    warnings.warn(f"Failed to decode metadata for {key}: {e}")

            yield key, waveform, sr, metadata

    def create_dataset(
        self,
        target_sr: int = 16000,
        segment_length: float = 5.0,
        transform: Optional[Callable] = None,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
    ) -> "wds.WebDataset":
        """
        Create a WebDataset with standard processing pipeline.

        Args:
            target_sr: Target sample rate
            segment_length: Target segment length in seconds
            transform: Optional transform to apply to (waveform, metadata) tuples
            filter_fn: Optional function to filter samples by metadata

        Returns:
            Configured WebDataset object
        """
        target_samples = int(target_sr * segment_length)
        resampler_cache = self._resampler_cache

        def decode_sample(sample):
            """Decode and process a single sample."""
            key = sample["__key__"]

            # Find and decode audio
            audio_data = None
            for ext in ["flac", "wav", "mp3", "ogg"]:
                if ext in sample:
                    audio_data = sample[ext]
                    break

            if audio_data is None:
                return None

            try:
                buffer = io.BytesIO(audio_data)
                waveform, sr = torchaudio.load(buffer, format="flac")
            except Exception:
                return None

            # Decode metadata
            metadata = {"__key__": key}
            if "json" in sample:
                try:
                    metadata.update(json.loads(sample["json"].decode("utf-8")))
                except Exception:
                    pass

            # Filter if needed
            if filter_fn is not None and not filter_fn(metadata):
                return None

            # Resample if needed
            if sr != target_sr:
                waveform = resampler_cache.resample(waveform, sr, target_sr)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.squeeze(0)

            # Normalize length
            if len(waveform) < target_samples:
                waveform = torch.nn.functional.pad(
                    waveform, (0, target_samples - len(waveform))
                )
            elif len(waveform) > target_samples:
                # Random crop for training diversity
                start = torch.randint(0, len(waveform) - target_samples + 1, (1,)).item()
                waveform = waveform[start : start + target_samples]

            # Apply transform
            if transform is not None:
                waveform, metadata = transform(waveform, metadata)

            return {"waveform": waveform, "metadata": metadata}

        # Build pipeline
        url_pattern = " ".join(self._tar_paths)
        dataset = (
            wds.WebDataset(url_pattern)
            .map(decode_sample)
            .select(lambda x: x is not None)
        )

        if self._shuffle:
            dataset = dataset.shuffle(1000)

        return dataset


# =============================================================================
# Factory Functions
# =============================================================================


def create_audio_loader(
    mode: str = "file",
    tar_paths: Optional[Union[str, List[str]]] = None,
    cache_dir: Optional[str] = None,
    shuffle: bool = False,
    resampler_cache: Optional[ResamplerCache] = None,
) -> AudioLoader:
    """
    Factory function to create an appropriate audio loader.

    Args:
        mode: Loading mode - "file" or "webdataset"
        tar_paths: Paths to tar shards (required for webdataset mode)
        cache_dir: Cache directory for webdataset
        shuffle: Whether to shuffle (webdataset only)
        resampler_cache: Shared resampler cache

    Returns:
        AudioLoader instance

    Example:
        >>> # File mode (default)
        >>> loader = create_audio_loader(mode="file")
        
        >>> # WebDataset mode
        >>> loader = create_audio_loader(
        ...     mode="webdataset",
        ...     tar_paths="/data/shards/train-{000000..000099}.tar"
        ... )
    """
    if mode == "file":
        return FileAudioLoader(resampler_cache=resampler_cache)
    elif mode == "webdataset":
        if tar_paths is None:
            raise ValueError("tar_paths is required for webdataset mode")
        return WebDatasetAudioLoader(
            tar_paths=tar_paths,
            cache_dir=cache_dir,
            shuffle=shuffle,
            resampler_cache=resampler_cache,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'file' or 'webdataset'.")


# =============================================================================
# WebDataset Creation Utilities
# =============================================================================


def get_shard_pattern(base_path: str, split: str, num_shards: int) -> str:
    """
    Generate a brace-expansion pattern for shard paths.

    Args:
        base_path: Base directory containing shards
        split: Data split (train, val, test)
        num_shards: Number of shards

    Returns:
        Brace-expansion pattern string

    Example:
        >>> get_shard_pattern("/data/shards", "train", 100)
        '/data/shards/train-{000000..000099}.tar'
    """
    return f"{base_path}/{split}-{{000000..{num_shards-1:06d}}}.tar"


def estimate_shard_count(
    total_samples: int,
    samples_per_shard: int = 1000,
) -> int:
    """
    Estimate the number of shards needed.

    Args:
        total_samples: Total number of samples
        samples_per_shard: Target samples per shard

    Returns:
        Number of shards
    """
    return max(1, (total_samples + samples_per_shard - 1) // samples_per_shard)


# =============================================================================
# Dataset Wrapper for PyTorch DataLoader Compatibility
# =============================================================================


class WebDatasetWrapper(torch.utils.data.IterableDataset):
    """
    Wrapper to make WebDataset compatible with standard PyTorch DataLoader.

    Handles worker distribution and provides a standard (waveform, label) interface.
    """

    def __init__(
        self,
        tar_paths: Union[str, List[str]],
        target_sr: int = 16000,
        segment_length: float = 5.0,
        label_col: str = "label",
        shuffle: bool = True,
        augment: bool = False,
        augmentation_fn: Optional[Callable] = None,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
    ):
        """
        Args:
            tar_paths: Path pattern(s) to tar shards
            target_sr: Target sample rate
            segment_length: Target segment length in seconds
            label_col: Column name for labels in metadata
            shuffle: Whether to shuffle samples
            augment: Whether to apply augmentations
            augmentation_fn: Optional augmentation function (waveform -> waveform)
            filter_fn: Optional filter function (metadata -> bool)
        """
        if not WEBDATASET_AVAILABLE:
            raise ImportError("webdataset is required. Install with: pip install webdataset")

        self.tar_paths = tar_paths if isinstance(tar_paths, list) else [tar_paths]
        self.target_sr = target_sr
        self.target_samples = int(target_sr * segment_length)
        self.label_col = label_col
        self.shuffle = shuffle
        self.augment = augment
        self.augmentation_fn = augmentation_fn
        self.filter_fn = filter_fn

        self._resampler_cache = ResamplerCache(max_size=8)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int]]:
        """Iterate over samples yielding (waveform, label) tuples."""
        url_pattern = " ".join(self.tar_paths)

        # Create pipeline with worker-aware sharding
        dataset = wds.WebDataset(url_pattern, shardshuffle=self.shuffle)

        if self.shuffle:
            dataset = dataset.shuffle(1000)

        for sample in dataset:
            result = self._process_sample(sample)
            if result is not None:
                yield result

    def _process_sample(
        self, sample: Dict
    ) -> Optional[Tuple[torch.Tensor, int]]:
        """Process a single WebDataset sample."""
        # Find and decode audio
        audio_data = None
        for ext in ["flac", "wav", "mp3", "ogg"]:
            if ext in sample:
                audio_data = sample[ext]
                break

        if audio_data is None:
            return None

        try:
            buffer = io.BytesIO(audio_data)
            waveform, sr = torchaudio.load(buffer, format="flac")
        except Exception:
            return None

        # Decode metadata
        metadata = {}
        if "json" in sample:
            try:
                metadata = json.loads(sample["json"].decode("utf-8"))
            except Exception:
                pass

        # Apply filter
        if self.filter_fn is not None and not self.filter_fn(metadata):
            return None

        # Get label
        label = metadata.get(self.label_col, 0)
        if isinstance(label, str):
            label = int(label)

        # Resample
        if sr != self.target_sr:
            waveform = self._resampler_cache.resample(waveform, sr, self.target_sr)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        # Normalize length
        if len(waveform) < self.target_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.target_samples - len(waveform))
            )
        elif len(waveform) > self.target_samples:
            start = torch.randint(0, len(waveform) - self.target_samples + 1, (1,)).item()
            waveform = waveform[start : start + self.target_samples]

        # Apply augmentation
        if self.augment and self.augmentation_fn is not None:
            waveform = self.augmentation_fn(waveform)

        return waveform, label


class COIWebDatasetWrapper(torch.utils.data.IterableDataset):
    """
    WebDataset wrapper for COI (Class of Interest) separation training.

    Creates mixtures from COI and background samples for training separation models.
    Compatible with COIAudioDataset interface.
    """

    def __init__(
        self,
        tar_paths: Union[str, List[str]],
        split: str = "train",
        target_sr: int = 16000,
        segment_length: float = 5.0,
        snr_range: Tuple[float, float] = (-5, 5),
        n_coi_classes: int = 1,
        shuffle: bool = True,
        augment: bool = True,
        stereo: bool = False,
        background_only_prob: float = 0.0,
    ):
        """
        Args:
            tar_paths: Path pattern(s) to tar shards
            split: Data split (train, val, test)
            target_sr: Target sample rate
            segment_length: Segment length in seconds
            snr_range: SNR range for mixing (min, max) in dB
            n_coi_classes: Number of COI classes
            shuffle: Whether to shuffle
            augment: Whether to augment (training only)
            stereo: Whether to output stereo
            background_only_prob: Probability of background-only samples
        """
        if not WEBDATASET_AVAILABLE:
            raise ImportError("webdataset is required. Install with: pip install webdataset")

        self.tar_paths = tar_paths if isinstance(tar_paths, list) else [tar_paths]
        self.split = split
        self.target_sr = target_sr
        self.segment_samples = int(target_sr * segment_length)
        self.snr_range = snr_range
        self.n_coi_classes = n_coi_classes
        self.shuffle = shuffle
        self.augment = augment and split == "train"
        self.stereo = stereo
        self.background_only_prob = background_only_prob

        self._resampler_cache = ResamplerCache(max_size=8)
        self._rng = np.random.default_rng(42)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate over samples yielding (mixture, sources) tuples.

        Returns:
            Tuples of (mixture, sources_tensor) where:
                - mixture: (T,) or (C, T) if stereo
                - sources_tensor: (n_src, T) or (n_src, C, T) with COI followed by background
        """
        url_pattern = " ".join(self.tar_paths)

        # Create pipeline
        dataset = wds.WebDataset(url_pattern, shardshuffle=self.shuffle)
        if self.shuffle:
            dataset = dataset.shuffle(1000)

        # Collect COI and background samples separately
        coi_buffer = []
        bg_buffer = []
        buffer_size = 100

        for sample in dataset:
            result = self._decode_sample(sample)
            if result is None:
                continue

            waveform, metadata = result
            label = metadata.get("label", 0)

            if label == 1:  # COI
                coi_buffer.append((waveform, metadata))
                if len(coi_buffer) >= buffer_size and len(bg_buffer) > 0:
                    # Yield a mixed sample
                    yield self._create_mixture(coi_buffer.pop(0), bg_buffer)
            else:  # Background
                bg_buffer.append((waveform, metadata))
                if len(bg_buffer) > buffer_size:
                    bg_buffer.pop(0)  # Keep buffer bounded

                # Occasionally yield background-only sample
                if (
                    self.background_only_prob > 0
                    and self._rng.random() < self.background_only_prob
                    and len(bg_buffer) > 0
                ):
                    yield self._create_background_only(bg_buffer)

    def _decode_sample(
        self, sample: Dict
    ) -> Optional[Tuple[torch.Tensor, Dict]]:
        """Decode a WebDataset sample to waveform and metadata."""
        audio_data = None
        for ext in ["flac", "wav", "mp3", "ogg"]:
            if ext in sample:
                audio_data = sample[ext]
                break

        if audio_data is None:
            return None

        try:
            buffer = io.BytesIO(audio_data)
            waveform, sr = torchaudio.load(buffer, format="flac")
        except Exception:
            return None

        # Decode metadata
        metadata = {}
        if "json" in sample:
            try:
                metadata = json.loads(sample["json"].decode("utf-8"))
            except Exception:
                pass

        # Filter by split
        if metadata.get("split") != self.split:
            return None

        # Resample
        if sr != self.target_sr:
            waveform = self._resampler_cache.resample(waveform, sr, self.target_sr)

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

        # Normalize length
        if waveform.shape[-1] < self.segment_samples:
            pad = self.segment_samples - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        waveform = waveform[..., : self.segment_samples]

        return waveform, metadata

    def _create_empty_source(self) -> torch.Tensor:
        """Create an empty (silent) source tensor."""
        if self.stereo:
            return torch.zeros(2, self.segment_samples)
        return torch.zeros(self.segment_samples)

    def _create_mixture(
        self,
        coi_sample: Tuple[torch.Tensor, Dict],
        bg_buffer: List[Tuple[torch.Tensor, Dict]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create a mixture from COI and background samples."""
        coi_waveform, coi_metadata = coi_sample

        # Get random background
        bg_idx = self._rng.integers(len(bg_buffer))
        bg_waveform, _ = bg_buffer[bg_idx]

        # Create sources tensor
        sources = []

        if self.n_coi_classes > 1:
            # Multi-class COI
            for i in range(self.n_coi_classes):
                if coi_metadata.get("coi_class", 0) == i:
                    sources.append(coi_waveform)
                else:
                    sources.append(self._create_empty_source())
        else:
            sources.append(coi_waveform)

        sources.append(bg_waveform)

        sources_tensor = torch.stack(sources, dim=0)
        mixture = sources_tensor.sum(dim=0)

        return mixture, sources_tensor

    def _create_background_only(
        self, bg_buffer: List[Tuple[torch.Tensor, Dict]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create a background-only sample."""
        # Mix multiple backgrounds
        n_mix = min(2, len(bg_buffer))
        indices = self._rng.choice(len(bg_buffer), size=n_mix, replace=False)
        bg_waveform = sum(bg_buffer[i][0] for i in indices)

        # Create sources with silent COI
        sources = [self._create_empty_source() for _ in range(self.n_coi_classes)]
        sources.append(bg_waveform)

        sources_tensor = torch.stack(sources, dim=0)
        mixture = sources_tensor.sum(dim=0)

        return mixture, sources_tensor
