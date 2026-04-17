"""
Generic PyTorch dataset for audio classification tasks.

This module provides a reusable dataset implementation that works with any
PyTorch-based audio classification model (PANN, AST, BirdNET, etc.).

Supports both:
1. File-based loading (original): Load audio from disk using file paths
2. WebDataset loading: Load audio from compressed tar shards

Key features:
- Load audio from disk with variable-length annotations
- High-quality resampling using audio_utils.ResamplerCache
- Configurable augmentation pipeline
- Handles NaN timestamps (full file loading)
- Memory-efficient lazy loading

Example:
    >>> # File-based loading (original)
    >>> from src.common.audio_dataset import AudioClassificationDataset
    >>> dataset = AudioClassificationDataset(
    ...     df=train_df,
    ...     target_sample_rate=32000,
    ...     audio_duration=10.0,
    ...     augment=True
    ... )
    >>> waveform, label = dataset[0]
    
    >>> # WebDataset loading
    >>> from src.common.audio_dataset import create_classification_dataloader
    >>> loader = create_classification_dataloader(
    ...     df_path="metadata.csv",
    ...     split="train",
    ...     batch_size=32,
    ...     use_webdataset=True,
    ...     webdataset_path="/data/shards"
    ... )
"""

import pandas as pd
import numpy as np
import torch
import torchaudio
import warnings
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Dict, Tuple, Callable

from .audio_utils import ResamplerCache


class AudioClassificationDataset(Dataset):
    """
    Generic PyTorch dataset for audio classification from file paths.
    
    Loads audio files, handles variable-length annotations (start_time, end_time),
    resamples to target sample rate, normalizes duration, and applies augmentations.
    
    Args:
        df: DataFrame with audio metadata. Required columns depend on configuration.
        target_sample_rate: Resample all audio to this rate (Hz)
        audio_duration: Target duration in seconds (pad/crop to this length)
        augment: Whether to apply augmentations (training only)
        augmentation_config: Dict with augmentation parameters. Keys:
            - time_stretch_prob: Probability of time stretching
            - time_stretch_range: (min, max) stretch factors
            - noise_prob: Probability of adding noise
            - noise_stddev: Standard deviation of Gaussian noise
            - gain_prob: Probability of random gain
            - gain_range: (min, max) gain factors
        filename_col: Column name for file paths
        start_time_col: Column name for start times (can be NaN for full file)
        end_time_col: Column name for end times (can be NaN for full file)
        label_col: Column name for labels
        split_long: Whether to split long annotations into multiple clips
        min_clip_length: Minimum clip length when splitting (seconds)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        target_sample_rate: int = 32000,
        audio_duration: float = 10.0,
        augment: bool = False,
        augmentation_config: Optional[Dict] = None,
        filename_col: str = "filename",
        start_time_col: str = "start_time",
        end_time_col: str = "end_time",
        label_col: str = "label",
        split_long: bool = True,
        min_clip_length: float = 0.5,
    ):
        self.target_sample_rate = target_sample_rate
        self.audio_duration = audio_duration
        self.target_samples = int(target_sample_rate * audio_duration)
        self.augment = augment
        self.aug_config = augmentation_config or {}
        
        # Column names
        self.filename_col = filename_col
        self.start_time_col = start_time_col
        self.end_time_col = end_time_col
        self.label_col = label_col
        
        # Validate required columns
        required_cols = [filename_col, start_time_col, end_time_col, label_col]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")
        
        # Optionally split long annotations into multiple clips
        if split_long:
            self.df = self._split_long_annotations(df, min_clip_length)
        else:
            self.df = df.copy().reset_index(drop=True)
            # Ensure time columns are numeric even when not splitting
            self.df[start_time_col] = pd.to_numeric(self.df[start_time_col], errors='coerce')
            self.df[end_time_col] = pd.to_numeric(self.df[end_time_col], errors='coerce')
        
        # Initialize resampler cache for efficient resampling
        self.resampler_cache = ResamplerCache(max_size=8)
        
    def _split_long_annotations(
        self, df: pd.DataFrame, min_clip_length: float
    ) -> pd.DataFrame:
        """
        Split long annotated segments into fixed-duration clips.
        
        Args:
            df: Input DataFrame
            min_clip_length: Minimum clip length in seconds
            
        Returns:
            DataFrame with expanded rows for long annotations
        """
        expanded_rows = []
        
        for _, row in df.iterrows():
            start = row[self.start_time_col]
            end = row[self.end_time_col]
            
            # If start/end are NaN, keep as-is (full file)
            if pd.isna(start) or pd.isna(end):
                expanded_rows.append(row.to_dict())
                continue
            
            # Try to convert to float, handling non-numeric values like "unknown"
            try:
                start_float = float(start)
            except (ValueError, TypeError):
                # Invalid start time, keep row as-is (will use full file)
                expanded_rows.append(row.to_dict())
                continue
            
            try:
                end_float = float(end)
            except (ValueError, TypeError):
                # Invalid end time, keep row as-is (will use full file)
                expanded_rows.append(row.to_dict())
                continue
            
            duration = end_float - start_float
            
            # If shorter than or equal to target, keep as-is
            if duration <= self.audio_duration:
                expanded_rows.append(row.to_dict())
            else:
                # Split into multiple clips
                n_clips = int(np.ceil(duration / self.audio_duration))
                for i in range(n_clips):
                    new_start = start_float + i * self.audio_duration
                    new_end = min(start_float + (i + 1) * self.audio_duration, end_float)
                    
                    # Include if meets minimum length
                    if new_end - new_start >= min_clip_length:
                        new_row = row.to_dict()
                        new_row[self.start_time_col] = new_start
                        new_row[self.end_time_col] = new_end
                        expanded_rows.append(new_row)
        
        # Create DataFrame and explicitly ensure correct dtypes
        result_df = pd.DataFrame(expanded_rows).reset_index(drop=True)
        
        # Convert time columns to float, handling any remaining non-numeric values
        result_df[self.start_time_col] = pd.to_numeric(result_df[self.start_time_col], errors='coerce')
        result_df[self.end_time_col] = pd.to_numeric(result_df[self.end_time_col], errors='coerce')
        
        return result_df
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load and preprocess a single audio sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (waveform, label) where:
                - waveform: 1D tensor of shape (target_samples,)
                - label: Integer label
        """
        row = self.df.iloc[idx]
        
        # Load audio file
        filepath = row[self.filename_col]
        start_time_raw = row[self.start_time_col]
        end_time_raw = row[self.end_time_col]
        label = int(row[self.label_col])
        
        # Convert times to float, handling any type issues
        # This is a safety check in case dtype conversion failed earlier
        if pd.notna(start_time_raw):
            try:
                start_time = float(start_time_raw)
            except (ValueError, TypeError):
                start_time = float('nan')
        else:
            start_time = float('nan')
        
        if pd.notna(end_time_raw):
            try:
                end_time = float(end_time_raw)
            except (ValueError, TypeError):
                end_time = float('nan')
        else:
            end_time = float('nan')
        
        # Load waveform
        waveform = self._load_audio_segment(filepath, start_time, end_time)
        
        # Apply augmentation if enabled
        if self.augment:
            waveform = self._augment_waveform(waveform)
        
        return waveform, label
    
    def _load_audio_segment(
        self, filepath: str, start_time: float, end_time: float
    ) -> torch.Tensor:
        """
        Load audio segment from file.
        
        Args:
            filepath: Path to audio file
            start_time: Start time in seconds (can be NaN)
            end_time: End time in seconds (can be NaN)
            
        Returns:
            Waveform tensor of shape (target_samples,)
        """
        import math
        
        # Load audio file
        try:
            waveform, sample_rate = torchaudio.load(filepath)
        except Exception as e:
            warnings.warn(f"Failed to load {filepath}: {e}. Returning silence.")
            return torch.zeros(self.target_samples)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze(0)  # Remove channel dimension
        
        # Extract segment
        if not math.isnan(start_time) and not math.isnan(end_time):
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Clamp to valid range
            start_sample = max(0, start_sample)
            end_sample = min(len(waveform), end_sample)
            
            if start_sample < end_sample:
                waveform = waveform[start_sample:end_sample]
            else:
                warnings.warn(f"Invalid segment [{start_time}, {end_time}] for {filepath}")
                waveform = torch.zeros(int(sample_rate * 0.1))  # 0.1s of silence
        # else: use full file
        
        # Resample if needed
        if sample_rate != self.target_sample_rate:
            waveform = self.resampler_cache.resample(
                waveform, sample_rate, self.target_sample_rate
            )
        
        # Normalize length (pad or crop)
        waveform = self._normalize_length(waveform)
        
        return waveform
    
    def _normalize_length(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Normalize waveform to target length (pad or center crop).
        
        Args:
            waveform: Input waveform
            
        Returns:
            Waveform of shape (target_samples,)
        """
        current_len = len(waveform)
        
        if current_len < self.target_samples:
            # Pad with zeros
            padding = self.target_samples - current_len
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_len > self.target_samples:
            # Center crop
            start = (current_len - self.target_samples) // 2
            waveform = waveform[start : start + self.target_samples]
        
        return waveform
    
    def _augment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply data augmentation to waveform.
        
        Args:
            waveform: Input waveform
            
        Returns:
            Augmented waveform
        """
        # Time stretching
        if (
            self.aug_config.get('time_stretch_prob', 0) > 0
            and torch.rand(1).item() < self.aug_config['time_stretch_prob']
        ):
            waveform = self._time_stretch(waveform)
        
        # Additive noise
        if (
            self.aug_config.get('noise_prob', 0) > 0
            and torch.rand(1).item() < self.aug_config['noise_prob']
        ):
            noise_stddev = self.aug_config.get('noise_stddev', 0.005)
            noise = torch.randn_like(waveform) * noise_stddev
            waveform = waveform + noise
        
        # Random gain
        if (
            self.aug_config.get('gain_prob', 0) > 0
            and torch.rand(1).item() < self.aug_config['gain_prob']
        ):
            gain_range = self.aug_config.get('gain_range', (0.7, 1.3))
            gain = torch.FloatTensor(1).uniform_(*gain_range).item()
            waveform = waveform * gain
        
        # Clip to valid range
        waveform = torch.clamp(waveform, -1.0, 1.0)
        
        return waveform
    
    def _time_stretch(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply time stretching via resampling.
        
        Args:
            waveform: Input waveform
            
        Returns:
            Time-stretched waveform
        """
        stretch_range = self.aug_config.get('time_stretch_range', (0.8, 1.2))
        stretch_factor = torch.FloatTensor(1).uniform_(*stretch_range).item()
        
        # Calculate stretched length
        stretched_len = int(len(waveform) * stretch_factor)
        
        # Use resampling for time stretching
        # Note: This is an approximation; for better quality, use librosa
        if stretched_len > 0:
            stretched = self.resampler_cache.resample(
                waveform, self.target_sample_rate, 
                int(self.target_sample_rate / stretch_factor)
            )
            
            # Normalize back to target length
            stretched = self._normalize_length(stretched)
            return stretched
        
        return waveform


# =============================================================================
# Factory Functions for DataLoader Creation
# =============================================================================


def create_classification_dataloader(
    df_path: str,
    split: str,
    batch_size: int,
    target_sample_rate: int = 32000,
    audio_duration: float = 10.0,
    augment: bool = True,
    augmentation_config: Optional[Dict] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle: Optional[bool] = None,
    # WebDataset options
    use_webdataset: bool = False,
    webdataset_path: Optional[str] = None,
) -> Tuple[DataLoader, Dataset]:
    """
    Create dataloader for audio classification.
    
    Supports both file-based and WebDataset loading modes.
    
    Args:
        df_path: Path to CSV with audio metadata (ignored if use_webdataset=True)
        split: Data split ('train', 'val', 'test')
        batch_size: Batch size
        target_sample_rate: Target sample rate
        audio_duration: Target audio duration in seconds
        augment: Whether to apply augmentations (only for training)
        augmentation_config: Augmentation parameters
        num_workers: DataLoader workers
        pin_memory: Whether to pin memory
        shuffle: Whether to shuffle (default: True for train, False otherwise)
        use_webdataset: If True, load from WebDataset shards
        webdataset_path: Path to WebDataset directory (required if use_webdataset=True)
        
    Returns:
        Tuple of (DataLoader, Dataset)
        
    Example:
        >>> # File-based loading
        >>> loader, dataset = create_classification_dataloader(
        ...     df_path="train.csv",
        ...     split="train",
        ...     batch_size=32,
        ... )
        
        >>> # WebDataset loading
        >>> loader, dataset = create_classification_dataloader(
        ...     df_path="",  # Not used
        ...     split="train",
        ...     batch_size=32,
        ...     use_webdataset=True,
        ...     webdataset_path="/data/shards",
        ... )
    """
    import warnings
    
    if shuffle is None:
        shuffle = (split == "train")
    
    augment = augment and (split == "train")
    
    if use_webdataset:
        # WebDataset mode
        if webdataset_path is None:
            raise ValueError("webdataset_path is required when use_webdataset=True")
        
        from .webdataset_utils import WebDatasetWrapper
        from src.label_loading.metadata_loader import get_webdataset_paths
        
        tar_paths = get_webdataset_paths(webdataset_path, split)
        
        # Create augmentation function if needed
        augmentation_fn = None
        if augment and augmentation_config:
            def augmentation_fn(waveform):
                # Apply simple augmentations
                if torch.rand(1).item() < augmentation_config.get('noise_prob', 0):
                    noise_stddev = augmentation_config.get('noise_stddev', 0.005)
                    waveform = waveform + torch.randn_like(waveform) * noise_stddev
                if torch.rand(1).item() < augmentation_config.get('gain_prob', 0):
                    gain_range = augmentation_config.get('gain_range', (0.7, 1.3))
                    gain = torch.FloatTensor(1).uniform_(*gain_range).item()
                    waveform = waveform * gain
                return torch.clamp(waveform, -1.0, 1.0)
        
        dataset = WebDatasetWrapper(
            tar_paths=tar_paths,
            target_sr=target_sample_rate,
            segment_length=audio_duration,
            label_col="label",
            shuffle=shuffle,
            augment=augment,
            augmentation_fn=augmentation_fn,
            filter_fn=lambda m: m.get("split") == split,
        )
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
        )
        
        return loader, dataset
    
    # File-based mode (original behavior)
    df = pd.read_csv(df_path)
    
    # Filter by split
    if "split" in df.columns:
        df = df[df["split"] == split]
    
    if len(df) == 0:
        warnings.warn(f"No samples found for split '{split}'")
    
    dataset = AudioClassificationDataset(
        df=df,
        target_sample_rate=target_sample_rate,
        audio_duration=audio_duration,
        augment=augment,
        augmentation_config=augmentation_config,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=(num_workers > 0 and split == "train"),
    )
    
    return loader, dataset
