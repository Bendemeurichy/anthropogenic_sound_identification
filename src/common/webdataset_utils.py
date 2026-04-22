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
import tempfile
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import soundfile as sf
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
# Audio Augmentations
# =============================================================================


class AudioAugmentations:
    """Audio augmentation utilities for training."""
    
    @staticmethod
    def time_stretch(waveform: torch.Tensor, rate: float) -> torch.Tensor:
        """Time stretch waveform by given rate."""
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
        """Add gaussian noise to waveform."""
        return waveform + torch.randn_like(waveform) * noise_level

    @staticmethod
    def gain(waveform: torch.Tensor, gain_db: float) -> torch.Tensor:
        """Apply gain in dB to waveform."""
        return waveform * (10 ** (gain_db / 20.0))

    @staticmethod
    def time_shift(waveform: torch.Tensor, shift_samples: int) -> torch.Tensor:
        """Circularly shift waveform by given number of samples."""
        return torch.roll(waveform, shifts=shift_samples, dims=-1)

    @staticmethod
    def low_pass_filter(
        waveform: torch.Tensor, cutoff_ratio: float = 0.8
    ) -> torch.Tensor:
        """Apply low-pass filter with given cutoff ratio."""
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
        waveform: torch.Tensor, rng: np.random.Generator
    ) -> torch.Tensor:
        """Apply random augmentations to waveform (matches train.py behavior)."""
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
        self._decode_warning_count = 0

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
        """Decode audio from bytes with robust fallbacks for HPC environments."""
        if isinstance(audio_bytes, memoryview):
            audio_bytes = audio_bytes.tobytes()

        last_error = None

        try:
            audio_np, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
            waveform = torch.from_numpy(audio_np.T.copy())
            return waveform, sr
        except Exception as e:
            last_error = e

        try:
            buffer = io.BytesIO(audio_bytes)
            waveform, sr = torchaudio.load(buffer)
            return waveform, sr
        except Exception as e:
            last_error = e

        suffix = ".flac"
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
                tmp.write(audio_bytes)
                tmp.flush()
                waveform, sr = torchaudio.load(tmp.name)
                return waveform, sr
        except Exception as e:
            last_error = e

        raise RuntimeError(f"Failed to decode audio bytes: {last_error}")

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
                # Use robust decoding with fallbacks
                if isinstance(audio_data, memoryview):
                    audio_data = audio_data.tobytes()
                
                # Try soundfile first
                try:
                    audio_np, sr = sf.read(io.BytesIO(audio_data), dtype="float32", always_2d=True)
                    waveform = torch.from_numpy(audio_np.T.copy())
                except Exception:
                    # Fallback to torchaudio
                    buffer = io.BytesIO(audio_data)
                    waveform, sr = torchaudio.load(buffer)
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
            # Use robust decoding with fallbacks
            if isinstance(audio_data, memoryview):
                audio_data = audio_data.tobytes()
            
            # Try soundfile first
            try:
                audio_np, sr = sf.read(io.BytesIO(audio_data), dtype="float32", always_2d=True)
                waveform = torch.from_numpy(audio_np.T.copy())
            except Exception:
                # Fallback to torchaudio
                buffer = io.BytesIO(audio_data)
                waveform, sr = torchaudio.load(buffer)
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
        target_classes: Optional[List[str]] = None,
        dataset_filter: Optional[str] = None,
        coi_ratio: float = 0.25,
        seed: int = 42,
        multi_coi_prob: float = 0.0,
        balance_classes: bool = True,
        coi_class_multipliers: Optional[List[int]] = None,
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
            target_classes: List of COI class labels (e.g., ["airplane", "plane"])
                           If None, treats all label==1 as COI
            dataset_filter: Optional dataset name filter (e.g., "aerosonicdb")
                          Only samples from matching datasets will be used as COI
            coi_ratio: Target ratio of COI samples in yielded data (default 0.25)
            seed: Random seed for reproducibility (default 42)
            multi_coi_prob: Probability of mixing multiple COI classes in same sample (default 0.0)
            balance_classes: If True, sample from each COI class with equal probability
                           to prevent class imbalance (default True for training splits)
            coi_class_multipliers: List of integers for per-class augmentation multipliers.
                                 E.g., [16, 1] will duplicate class-0 samples 16x to balance
                                 a 1:16 imbalance. Only applied during training. If None,
                                 all classes use multiplier 1.
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
        self.target_classes = target_classes or []
        self.dataset_filter = dataset_filter
        self.coi_ratio = coi_ratio
        self.seed = seed
        self.multi_coi_prob = multi_coi_prob if split == "train" else 0.0
        self.balance_classes = balance_classes and split == "train"  # Only balance during training
        
        # Per-class augmentation multipliers (only applied during training)
        if coi_class_multipliers is None:
            self.coi_class_multipliers = [1] * n_coi_classes
        else:
            if len(coi_class_multipliers) != n_coi_classes:
                raise ValueError(
                    f"coi_class_multipliers length ({len(coi_class_multipliers)}) "
                    f"must match n_coi_classes ({n_coi_classes})"
                )
            # Only apply multipliers during training
            self.coi_class_multipliers = (
                list(coi_class_multipliers) if split == "train" else [1] * n_coi_classes
            )

        self._resampler_cache = ResamplerCache(max_size=8)
        self._rng = np.random.default_rng(seed)  # Use provided seed
        self._decode_warning_count = 0
        
        # Try to load dataset size from manifest for __len__
        self._dataset_size = self._load_dataset_size()

    def _load_dataset_size(self) -> Optional[int]:
        """
        Load dataset size from manifest.json if available.
        
        Returns:
            Number of samples in the dataset, or None if not available
        """
        if not self.tar_paths:
            return None
        
        # Extract webdataset directory from first tar path
        # Format: /path/to/shards/train-{000000..000099}.tar -> /path/to/shards
        first_path = self.tar_paths[0]
        if '{' in first_path:
            # Brace expansion pattern - extract directory
            webdataset_dir = Path(first_path.split('{')[0]).parent
        else:
            # Single tar file - extract directory
            webdataset_dir = Path(first_path).parent
        
        manifest_path = webdataset_dir / "manifest.json"
        if not manifest_path.exists():
            return None
        
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            
            split_info = manifest.get("splits", {}).get(self.split, {})
            num_samples = split_info.get("num_samples")
            
            if num_samples:
                # Estimate based on COI ratio (typically 25% COI in the dataset)
                # Since we create mixtures, the effective dataset size is approximately
                # the number of COI samples (each mixed with background)
                coi_ratio = split_info.get("coi_ratio", 0.25)
                estimated_size = int(num_samples * coi_ratio)
                return estimated_size
            
            return None
        except (json.JSONDecodeError, KeyError, IOError):
            return None

    def __len__(self) -> int:
        """
        Return an estimate of dataset length.
        
        For IterableDatasets, this is an approximation used for progress bars
        and learning rate scheduling. The actual number of yielded samples may differ.
        """
        if self._dataset_size is not None:
            return self._dataset_size
        
        # Fallback: return a reasonable default (1000 samples per shard)
        return len(self.tar_paths) * 1000

    def __iter__(self) -> Iterator[torch.Tensor]:
        """
        Iterate over samples yielding source tensors.

        This mirrors the file-based training dataset interface used by the
        training loop, which expects only the stacked source tensor and builds
        the mixture later via prepare_batch().

        Returns:
            Tensors of shape (n_src, T) or (n_src, C, T) with COI source(s)
            followed by the background channel.
        """
        urls = list(self.tar_paths)

        # Create pipeline. Pass the shard URLs as a list so WebDataset opens
        # each tar individually instead of treating a space-joined string as
        # one invalid filename.
        dataset = wds.WebDataset(urls, shardshuffle=(100 if self.shuffle else False))
        if self.shuffle:
            dataset = dataset.shuffle(1000)

        # Collect COI and background samples separately with filtering
        # For multi-class, maintain separate buffers per COI class
        coi_buffers_by_class: List[List[Tuple[torch.Tensor, Dict]]] = [
            [] for _ in range(self.n_coi_classes)
        ]
        bg_buffer = []
        buffer_size = 100
        # Use smaller min buffer for validation (may have fewer samples per split)
        min_buffer_size = 5 if self.split in ["val", "test"] else 10
        
        coi_counts = [0 for _ in range(self.n_coi_classes)]
        bg_count = 0
        yielded_count = 0
        decoded_count = 0
        
        # For balanced sampling: track how many samples we need to yield from each class
        # to achieve balance
        yielded_per_class = [0 for _ in range(self.n_coi_classes)]

        for sample in dataset:
            result = self._decode_sample(sample)
            if result is None:
                continue

            decoded_count += 1

            waveform, metadata = result
            
            # Classify sample as COI or valid background
            is_coi = self._is_coi_sample(metadata)
            is_valid_bg = self._is_valid_background(metadata)
            
            if is_coi:  # Pure COI
                # Determine which COI class this belongs to
                coi_class_idx = self._infer_coi_class_from_label(metadata)
                
                # Apply per-class augmentation multiplier
                # Duplicate samples from minority classes to balance dataset
                multiplier = self.coi_class_multipliers[coi_class_idx]
                for _ in range(multiplier):
                    coi_buffers_by_class[coi_class_idx].append((waveform, metadata))
                
                coi_counts[coi_class_idx] += 1  # Count unique samples, not duplicates
                
                # Keep COI buffer bounded
                if len(coi_buffers_by_class[coi_class_idx]) > buffer_size * multiplier:
                    coi_buffers_by_class[coi_class_idx].pop(0)
                
                # Yield strategy depends on balance_classes setting
                if self.balance_classes:
                    # Balanced sampling: only yield if this class is underrepresented
                    # Wait until we have samples in all class buffers
                    all_classes_ready = all(
                        len(buf) >= min_buffer_size for buf in coi_buffers_by_class
                    ) and len(bg_buffer) >= min_buffer_size
                    
                    if all_classes_ready:
                        # Find the class that has been yielded the least
                        min_yielded = min(yielded_per_class)
                        underrepresented_classes = [
                            i for i, count in enumerate(yielded_per_class)
                            if count == min_yielded and len(coi_buffers_by_class[i]) > 0
                        ]
                        
                        if underrepresented_classes:
                            # Sample from underrepresented classes
                            selected_class = self._rng.choice(underrepresented_classes)
                            sources = self._create_mixture_multiclass(
                                coi_buffers_by_class, selected_class, bg_buffer
                            )
                            yielded_per_class[selected_class] += 1
                            yielded_count += 1
                            yield sources
                else:
                    # Unbalanced sampling: yield immediately when COI arrives
                    if len(bg_buffer) >= min_buffer_size:
                        sources = self._create_mixture_multiclass(
                            coi_buffers_by_class, coi_class_idx, bg_buffer
                        )
                        yielded_per_class[coi_class_idx] += 1
                        yielded_count += 1
                        yield sources
                    
            elif is_valid_bg:  # Valid background (no COI labels, not empty)
                bg_buffer.append((waveform, metadata))
                bg_count += 1
                
                if len(bg_buffer) > buffer_size:
                    bg_buffer.pop(0)  # Keep buffer bounded

                # Occasionally yield background-only sample
                if (
                    self.background_only_prob > 0
                    and self._rng.random() < self.background_only_prob
                    and len(bg_buffer) >= min_buffer_size
                ):
                    bg_only_sources = self._create_background_only(bg_buffer)
                    yielded_count += 1
                    yield bg_only_sources
            # else: Mixed sample or invalid - skip completely
        
        # IMPORTANT: Yield remaining COI samples in buffers after streaming completes
        # This ensures we don't drop COI samples at the end due to insufficient backgrounds
        if len(bg_buffer) > 0:
            if self.balance_classes:
                # Balanced mode: yield from classes in order of least represented
                while any(len(buf) > 0 for buf in coi_buffers_by_class):
                    # Find class with lowest yield count that still has samples
                    available_classes = [
                        i for i in range(self.n_coi_classes)
                        if len(coi_buffers_by_class[i]) > 0
                    ]
                    if not available_classes:
                        break
                    
                    selected_class = min(
                        available_classes,
                        key=lambda i: yielded_per_class[i]
                    )
                    
                    sources = self._create_mixture_multiclass(
                        coi_buffers_by_class, selected_class, bg_buffer
                    )
                    yielded_per_class[selected_class] += 1
                    yielded_count += 1
                    yield sources
            else:
                # Unbalanced mode: yield all remaining samples
                for class_idx in range(self.n_coi_classes):
                    while len(coi_buffers_by_class[class_idx]) > 0:
                        sources = self._create_mixture_multiclass(
                            coi_buffers_by_class, class_idx, bg_buffer
                        )
                        yielded_per_class[class_idx] += 1
                        yielded_count += 1
                        yield sources

        # Log class distribution for debugging
        if yielded_count > 0 and self.n_coi_classes > 1:
            class_dist = " ".join(
                f"class{i}={yielded_per_class[i]}" for i in range(self.n_coi_classes)
            )
            multipliers_str = (
                f", multipliers={self.coi_class_multipliers}"
                if any(m > 1 for m in self.coi_class_multipliers)
                else ""
            )
            print(
                f"[WebDataset {self.split}] Yielded {yielded_count} samples: {class_dist} "
                f"(balanced={self.balance_classes}, decoded={decoded_count}, "
                f"coi_counts={coi_counts}, bg_count={bg_count}{multipliers_str})"
            )

        if yielded_count == 0:
            raise RuntimeError(
                "WebDataset yielded 0 training samples. "
                f"split={self.split!r}, decoded={decoded_count}, "
                f"coi_matches={coi_counts}, background_matches={bg_count}, "
                f"target_classes={self.target_classes}, "
                f"dataset_filter={self.dataset_filter!r}. "
                "Check the resolved shard path and label names in the shard metadata."
            )

    def _is_coi_sample(self, metadata: Dict) -> bool:
        """
        Determine if a sample is pure COI based on target_classes.
        
        Follows the logic from sampler.py:get_coi():
        - If target_classes is empty, use label==1
        - Otherwise, check if ALL labels in the sample are in target_classes (pure COI)
        - Mixed samples (COI + other) are rejected as background
        
        Args:
            metadata: Sample metadata dict
            
        Returns:
            True if sample is pure COI, False otherwise
        """
        # Apply dataset filter if specified
        if self.dataset_filter:
            sample_dataset = metadata.get("dataset", "")
            if not sample_dataset or self.dataset_filter.lower() not in sample_dataset.lower():
                return False
        
        # If no target_classes specified, use binary label
        if not self.target_classes:
            label_value = metadata.get("label")
            # Handle both numeric labels (1/0) and string labels
            if isinstance(label_value, (int, float)):
                return label_value == 1
            return False
        
        # Check if sample labels match target_classes (pure COI only)
        sample_labels = metadata.get("label", [])
        
        # Handle different label formats
        if isinstance(sample_labels, str):
            sample_labels = [sample_labels]
        elif isinstance(sample_labels, (int, float)):
            # Numeric label - cannot match string target_classes
            return False
        elif not isinstance(sample_labels, list):
            return False
        
        if len(sample_labels) == 0:
            return False
        
        # Flatten target_classes to get all COI labels across all classes
        # target_classes is list of lists: [[airplane labels], [bird labels], ...]
        all_coi_labels = []
        if self.target_classes and isinstance(self.target_classes[0], list):
            # List of lists format
            for class_group in self.target_classes:
                all_coi_labels.extend(class_group)
        else:
            # Flat list format (single class)
            all_coi_labels = self.target_classes
        
        # Convert to lowercase for case-insensitive matching
        all_coi_labels_lower = [lbl.lower() for lbl in all_coi_labels]
        sample_labels_lower = [lbl.lower() for lbl in sample_labels]
        
        # Pure COI: ALL labels must be in the flattened target_classes
        return all(label in all_coi_labels_lower for label in sample_labels_lower)
    
    def _is_valid_background(self, metadata: Dict) -> bool:
        """
        Determine if a sample is valid for background pool.
        
        Follows the logic from sampler.py:sample_non_coi():
        - Exclude recordings with ANY COI label (including mixed)
        - Exclude recordings with no labels (None/empty)
        
        Args:
            metadata: Sample metadata dict
            
        Returns:
            True if sample is valid background, False otherwise
        """
        if not self.target_classes:
            # No target classes: accept non-COI samples (label != 1)
            label_value = metadata.get("label")
            if isinstance(label_value, (int, float)):
                return label_value != 1
            # String labels with no target_classes - cannot determine, reject
            return False
        
        sample_labels = metadata.get("label", [])
        
        # Exclude None/empty labels (unknown content)
        if sample_labels is None:
            return False
        
        # Handle different label formats
        if isinstance(sample_labels, str):
            sample_labels = [sample_labels]
        elif isinstance(sample_labels, (int, float)):
            # Numeric label - cannot match string target_classes
            # Treat as non-COI if it's not 1
            return sample_labels != 1
        elif not isinstance(sample_labels, list):
            return False
        
        if len(sample_labels) == 0:
            return False
        
        # Flatten target_classes to get all COI labels
        all_coi_labels = []
        if self.target_classes and isinstance(self.target_classes[0], list):
            # List of lists format
            for class_group in self.target_classes:
                all_coi_labels.extend(class_group)
        else:
            # Flat list format
            all_coi_labels = self.target_classes
        
        # Convert to lowercase for case-insensitive matching
        all_coi_labels_lower = [lbl.lower() for lbl in all_coi_labels]
        sample_labels_lower = [lbl.lower() for lbl in sample_labels]
        
        # Exclude if ANY label is in target_classes (prevents mixed samples)
        return not any(label in all_coi_labels_lower for label in sample_labels_lower)

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
            waveform, sr = self._decode_audio(audio_data)
        except Exception as e:
            key = sample.get("__key__", "unknown")
            if self._decode_warning_count < 3:
                warnings.warn(f"Failed to decode WebDataset sample {key}: {e}")
                self._decode_warning_count += 1
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

    def _apply_augmentation(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to waveform (training only)."""
        return AudioAugmentations.random_augment(waveform, self._rng)

    def _decode_audio(self, audio_bytes: bytes) -> Tuple[torch.Tensor, int]:
        """Decode audio bytes with soundfile first and torchaudio fallbacks."""
        if isinstance(audio_bytes, memoryview):
            audio_bytes = audio_bytes.tobytes()

        last_error = None

        try:
            audio_np, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
            waveform = torch.from_numpy(audio_np.T.copy())
            return waveform, sr
        except Exception as e:
            last_error = e

        try:
            buffer = io.BytesIO(audio_bytes)
            waveform, sr = torchaudio.load(buffer)
            return waveform, sr
        except Exception as e:
            last_error = e

        try:
            with tempfile.NamedTemporaryFile(suffix=".flac") as tmp:
                tmp.write(audio_bytes)
                tmp.flush()
                waveform, sr = torchaudio.load(tmp.name)
                return waveform, sr
        except Exception as e:
            last_error = e

        raise RuntimeError(f"Failed to decode audio bytes: {last_error}")

    def _create_empty_source(self) -> torch.Tensor:
        """Create an empty (silent) source tensor."""
        if self.stereo:
            return torch.zeros(2, self.segment_samples)
        return torch.zeros(self.segment_samples)

    def _infer_coi_class_from_label(self, metadata: Dict) -> int:
        """
        Infer the COI class index from the label field using target_classes mapping.
        
        This is critical when webdataset metadata doesn't have coi_class field set.
        The webdataset typically has string labels (e.g., "chirping_birds", "airplane"),
        and we need to map these to COI class indices using target_classes.
        
        Args:
            metadata: Sample metadata dict with 'label' field
            
        Returns:
            COI class index (0 to n_coi_classes-1), or 0 if cannot infer
        """
        # Check if coi_class is already set and valid
        existing_class = metadata.get("coi_class")
        if existing_class is not None and 0 <= existing_class < self.n_coi_classes:
            return existing_class
        
        # Need target_classes to infer from labels
        if not self.target_classes:
            # No mapping available - default to class 0
            return 0
        
        # Infer from label field
        sample_labels = metadata.get("label", [])
        
        # Handle different label formats (string, int, or list)
        if isinstance(sample_labels, str):
            sample_labels = [sample_labels]
        elif isinstance(sample_labels, (int, float)):
            # Numeric label - cannot infer class from string matching
            return 0
        elif not isinstance(sample_labels, list):
            return 0  # Unknown format - fallback to class 0
        
        if len(sample_labels) == 0:
            return 0  # Empty labels - fallback to class 0
        
        # target_classes is a list of lists: [[airplane labels], [bird labels], ...]
        # Each inner list contains string labels that belong to that COI class
        target_classes_groups = self.target_classes
        
        # Handle case where target_classes might be a flat list (single class)
        # Check if first element is a list to determine structure
        if target_classes_groups and not isinstance(target_classes_groups[0], list):
            target_classes_groups = [target_classes_groups]
        
        # Match sample labels against each target class group
        # Return the index of the FIRST matching class
        for class_idx, class_labels in enumerate(target_classes_groups):
            # Convert all class labels to lowercase for case-insensitive matching
            class_labels_lower = [lbl.lower() for lbl in class_labels]
            sample_labels_lower = [lbl.lower() for lbl in sample_labels]
            
            # Check if ANY sample label matches ANY label in this class group
            if any(sample_lbl in class_labels_lower for sample_lbl in sample_labels_lower):
                return class_idx
        
        # No match found - this sample doesn't belong to any configured COI class
        # This should not happen for true COI samples (caught by _is_coi_sample earlier)
        # But as a safety fallback, return class 0
        return 0
    
    def _create_mixture(
        self,
        coi_sample: Tuple[torch.Tensor, Dict],
        bg_buffer: List[Tuple[torch.Tensor, Dict]],
    ) -> torch.Tensor:
        """Create stacked source tensors from COI and background samples.
        
        DEPRECATED: Use _create_mixture_multiclass for multi-class support.
        Kept for backwards compatibility with single-class scenarios.
        """
        coi_waveform, coi_metadata = coi_sample
        
        # Apply augmentation if enabled (training only)
        if self.augment:
            coi_waveform = self._apply_augmentation(coi_waveform)

        # Get random background
        bg_idx = self._rng.integers(len(bg_buffer))
        bg_waveform, _ = bg_buffer[bg_idx]

        # Create sources tensor
        sources = []

        if self.n_coi_classes > 1:
            # Multi-class COI - infer the correct class from label if coi_class is missing
            coi_class_idx = self._infer_coi_class_from_label(coi_metadata)
            
            for i in range(self.n_coi_classes):
                if i == coi_class_idx:
                    sources.append(coi_waveform)
                else:
                    sources.append(self._create_empty_source())
        else:
            sources.append(coi_waveform)

        sources.append(bg_waveform)

        sources_tensor = torch.stack(sources, dim=0)
        return sources_tensor

    def _create_mixture_multiclass(
        self,
        coi_buffers_by_class: List[List[Tuple[torch.Tensor, Dict]]],
        primary_class_idx: int,
        bg_buffer: List[Tuple[torch.Tensor, Dict]],
    ) -> torch.Tensor:
        """Create stacked source tensors with optional multi-COI mixing.
        
        Args:
            coi_buffers_by_class: List of buffers, one per COI class
            primary_class_idx: Index of the primary COI class to use
            bg_buffer: List of background samples
            
        Returns:
            Source tensor of shape (n_coi_classes + 1, T)
        """
        # Pop primary COI sample
        if len(coi_buffers_by_class[primary_class_idx]) == 0:
            # Fallback: create empty sources (shouldn't happen if called correctly)
            sources = [self._create_empty_source() for _ in range(self.n_coi_classes)]
            bg_idx = self._rng.integers(len(bg_buffer))
            sources.append(bg_buffer[bg_idx][0])
            return torch.stack(sources, dim=0)
        
        primary_waveform, primary_metadata = coi_buffers_by_class[primary_class_idx].pop(0)
        
        # Apply augmentation if enabled (training only)
        if self.augment:
            primary_waveform = self._apply_augmentation(primary_waveform)
        
        # Get random background
        bg_idx = self._rng.integers(len(bg_buffer))
        bg_waveform, _ = bg_buffer[bg_idx]

        # Initialize sources - all empty except primary class
        sources = [self._create_empty_source() for _ in range(self.n_coi_classes)]
        sources[primary_class_idx] = primary_waveform
        
        # Multi-COI mixing: add a second COI class with probability multi_coi_prob
        if (
            self.n_coi_classes > 1
            and self.multi_coi_prob > 0
            and self._rng.random() < self.multi_coi_prob
        ):
            # Find other classes that have available samples
            available_classes = [
                i for i in range(self.n_coi_classes)
                if i != primary_class_idx and len(coi_buffers_by_class[i]) > 0
            ]
            
            if available_classes:
                # Pick a random secondary class
                secondary_class_idx = self._rng.choice(available_classes)
                # Pick a random sample from that class's buffer (don't pop - reuse)
                secondary_idx = self._rng.integers(len(coi_buffers_by_class[secondary_class_idx]))
                secondary_waveform, _ = coi_buffers_by_class[secondary_class_idx][secondary_idx]
                
                # Apply augmentation if enabled
                if self.augment:
                    secondary_waveform = self._apply_augmentation(secondary_waveform)
                
                sources[secondary_class_idx] = secondary_waveform

        sources.append(bg_waveform)
        sources_tensor = torch.stack(sources, dim=0)
        return sources_tensor

    def _create_background_only(
        self, bg_buffer: List[Tuple[torch.Tensor, Dict]]
    ) -> torch.Tensor:
        """Create stacked source tensors for a background-only sample."""
        # Mix multiple backgrounds
        n_mix = min(2, len(bg_buffer))
        indices = self._rng.choice(len(bg_buffer), size=n_mix, replace=False)
        bg_waveform = sum(bg_buffer[i][0] for i in indices)

        # Create sources with silent COI
        sources = [self._create_empty_source() for _ in range(self.n_coi_classes)]
        sources.append(bg_waveform)

        sources_tensor = torch.stack(sources, dim=0)
        return sources_tensor
