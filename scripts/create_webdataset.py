#!/usr/bin/env python3
"""
Convert audio datasets to WebDataset tar shards for efficient storage and loading.

This script reads a metadata CSV file and creates WebDataset tar shards containing:
- Audio files compressed as FLAC (lossless compression)
- JSON metadata for each sample

Usage:
    python create_webdataset.py \
        --metadata_csv /path/to/metadata.csv \
        --output_dir /path/to/output/shards \
        --samples_per_shard 1000 \
        --split train

    # Or process all splits:
    python create_webdataset.py \
        --metadata_csv /path/to/metadata.csv \
        --output_dir /path/to/output/shards \
        --samples_per_shard 1000

Features:
- Parallel audio processing with multiprocessing
- FLAC compression for lossless, efficient storage
- Progress tracking with tqdm
- Resume support (skips existing shards)
- Handles variable sample rates and formats

Expected CSV columns:
    - filename: Path to audio file
    - start_time: Start time in seconds (can be NaN for full file)
    - end_time: End time in seconds (can be NaN for full file)
    - label: Integer label
    - split: train/val/test
    - dataset: Dataset name
    - coi_class: (optional) COI class index for multi-class

Output structure:
    output_dir/
        train-000000.tar
        train-000001.tar
        ...
        val-000000.tar
        ...
        manifest.json  # Contains shard info for easy loading
"""

import argparse
import io
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torchaudio
import webdataset as wds
from tqdm import tqdm


def load_and_encode_audio(
    filepath: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    target_sr: Optional[int] = None,
) -> Tuple[bytes, int, int]:
    """
    Load audio file, optionally extract segment, and encode as FLAC.

    Args:
        filepath: Path to audio file
        start_time: Start time in seconds (None for full file)
        end_time: End time in seconds (None for full file)
        target_sr: Target sample rate (None to keep original)

    Returns:
        Tuple of (flac_bytes, sample_rate, num_samples)

    Raises:
        Exception: If audio cannot be loaded or encoded
    """
    # Load audio using soundfile (more compatible across torchaudio versions)
    try:
        # Get audio info
        info = sf.info(filepath)
        orig_sr = info.samplerate
        total_frames = info.frames

        # Calculate frame offsets for efficient partial loading
        start_frame = 0
        frames_to_read = -1

        if start_time is not None and end_time is not None:
            if not (pd.isna(start_time) or pd.isna(end_time)):
                start_frame = int(float(start_time) * orig_sr)
                frames_to_read = int((float(end_time) - float(start_time)) * orig_sr)
                start_frame = max(0, min(start_frame, total_frames - 1))
                frames_to_read = min(frames_to_read, total_frames - start_frame)

        # Load audio
        audio_np, sr = sf.read(
            filepath,
            start=start_frame,
            frames=frames_to_read if frames_to_read > 0 else None,
            dtype='float32',
            always_2d=False
        )

    except Exception as e:
        raise RuntimeError(f"Failed to load audio from {filepath}: {e}")

    # Resample if needed
    if target_sr is not None and sr != target_sr:
        import torch
        # Convert to torch tensor for resampling
        if audio_np.ndim == 1:
            waveform = torch.from_numpy(audio_np).unsqueeze(0)  # (samples,) -> (1, samples)
        else:
            waveform = torch.from_numpy(audio_np.T)  # (samples, channels) -> (channels, samples)
        
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=target_sr,
        )
        waveform = resampler(waveform)
        sr = target_sr
        
        # Convert back to numpy
        audio_np = waveform.numpy()
        if audio_np.ndim == 2:
            audio_np = audio_np.T  # (channels, samples) -> (samples, channels)
    
    # Ensure correct shape for soundfile
    if audio_np.ndim == 1:
        num_samples = len(audio_np)
    else:
        num_samples = audio_np.shape[0]

    # Encode to FLAC
    buffer = io.BytesIO()
    sf.write(buffer, audio_np, sr, format="FLAC", subtype="PCM_16")
    flac_bytes = buffer.getvalue()

    return flac_bytes, sr, num_samples


def process_sample(
    row: pd.Series,
    sample_idx: int,
    target_sr: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Process a single sample from the metadata DataFrame.

    Args:
        row: DataFrame row containing sample metadata
        sample_idx: Unique sample index
        target_sr: Target sample rate (None to keep original)

    Returns:
        Dictionary containing processed sample data, or None if processing failed
    """
    try:
        # Load and encode audio
        flac_bytes, sr, num_samples = load_and_encode_audio(
            filepath=row["filename"],
            start_time=row.get("start_time"),
            end_time=row.get("end_time"),
            target_sr=target_sr,
        )

        # Build metadata
        # Handle label: can be string, int, or list (or string representation of list)
        label_value = row.get("label")
        if pd.notna(label_value):
            # If it's already an int, use it
            if isinstance(label_value, (int, np.integer)):
                label = int(label_value)
            # If it's a list, keep it as list
            elif isinstance(label_value, list):
                label = label_value
            # If it's a string, check if it's a string representation of a list
            elif isinstance(label_value, str):
                # Try to parse as list (e.g., "['Rain', 'Thunder']")
                if label_value.startswith('[') and label_value.endswith(']'):
                    try:
                        import ast
                        label = ast.literal_eval(label_value)
                    except (ValueError, SyntaxError):
                        label = label_value
                else:
                    label = label_value
            else:
                # Try to convert to int, fallback to string
                try:
                    label = int(label_value)
                except (ValueError, TypeError):
                    label = str(label_value)
        else:
            label = None
        
        metadata = {
            "original_filename": str(row["filename"]),
            "sample_rate": sr,
            "num_samples": num_samples,
            "duration": num_samples / sr,
            "label": label,
            "split": str(row.get("split", "train")),
            "dataset": str(row.get("dataset", "unknown")),
        }

        # Add optional fields
        if "start_time" in row and pd.notna(row["start_time"]):
            metadata["original_start_time"] = float(row["start_time"])
        if "end_time" in row and pd.notna(row["end_time"]):
            metadata["original_end_time"] = float(row["end_time"])
        if "coi_class" in row and pd.notna(row["coi_class"]):
            metadata["coi_class"] = int(row["coi_class"])

        # Create unique sample key
        sample_key = f"{sample_idx:08d}"

        return {
            "key": sample_key,
            "flac": flac_bytes,
            "metadata": metadata,
        }

    except Exception as e:
        warnings.warn(f"Failed to process sample {sample_idx} ({row.get('filename', 'unknown')}): {e}")
        return None


def process_samples_batch(
    rows: List[Tuple[int, pd.Series]],
    target_sr: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Process a batch of samples (for multiprocessing).

    Args:
        rows: List of (index, row) tuples
        target_sr: Target sample rate

    Returns:
        List of processed sample dictionaries
    """
    results = []
    for idx, row in rows:
        result = process_sample(row, idx, target_sr)
        if result is not None:
            results.append(result)
    return results


def create_shards(
    df: pd.DataFrame,
    output_dir: Path,
    split: str,
    samples_per_shard: int = 1000,
    target_sr: Optional[int] = None,
    num_workers: int = 4,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    Create WebDataset tar shards from a DataFrame.

    Args:
        df: DataFrame with audio metadata
        output_dir: Directory to write shards
        split: Data split name (train, val, test)
        samples_per_shard: Number of samples per shard
        target_sr: Target sample rate (None to keep original)
        num_workers: Number of parallel workers for audio processing
        resume: Whether to skip existing shards

    Returns:
        Dictionary with shard creation statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to split
    split_df = df[df["split"] == split].reset_index(drop=True)

    if len(split_df) == 0:
        print(f"No samples found for split '{split}'")
        return {"split": split, "num_samples": 0, "num_shards": 0}

    print(f"\nProcessing {split} split: {len(split_df)} samples")

    # Calculate number of shards
    num_shards = (len(split_df) + samples_per_shard - 1) // samples_per_shard

    # Check for existing shards
    existing_shards = set()
    if resume:
        for i in range(num_shards):
            shard_path = output_dir / f"{split}-{i:06d}.tar"
            if shard_path.exists():
                existing_shards.add(i)
        if existing_shards:
            print(f"Found {len(existing_shards)} existing shards, will skip them")

    # Process samples in batches by shard
    total_samples = 0
    failed_samples = 0

    for shard_idx in tqdm(range(num_shards), desc=f"Creating {split} shards"):
        if shard_idx in existing_shards:
            # Estimate samples in existing shard
            total_samples += samples_per_shard
            continue

        # Get samples for this shard
        start_idx = shard_idx * samples_per_shard
        end_idx = min((shard_idx + 1) * samples_per_shard, len(split_df))
        shard_df = split_df.iloc[start_idx:end_idx]

        # Prepare rows for processing
        rows = [(start_idx + i, row) for i, (_, row) in enumerate(shard_df.iterrows())]

        # Process samples (with optional parallelism)
        samples = []
        if num_workers > 1:
            # Split into batches for workers
            batch_size = max(1, len(rows) // num_workers)
            batches = [
                rows[i : i + batch_size]
                for i in range(0, len(rows), batch_size)
            ]

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(process_samples_batch, batch, target_sr)
                    for batch in batches
                ]
                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                        samples.extend(batch_results)
                    except Exception as e:
                        warnings.warn(f"Batch processing failed: {e}")
        else:
            # Single-threaded processing
            for idx, row in rows:
                result = process_sample(row, idx, target_sr)
                if result is not None:
                    samples.append(result)

        # Count failures
        failed_samples += len(rows) - len(samples)
        total_samples += len(samples)

        # Write shard
        shard_path = output_dir / f"{split}-{shard_idx:06d}.tar"
        with wds.TarWriter(str(shard_path)) as sink:
            for sample in samples:
                sink.write({
                    "__key__": sample["key"],
                    "flac": sample["flac"],
                    "json": json.dumps(sample["metadata"]).encode("utf-8"),
                })

    return {
        "split": split,
        "num_samples": total_samples,
        "num_shards": num_shards,
        "num_failed": failed_samples,
        "shard_pattern": f"{split}-{{000000..{num_shards-1:06d}}}.tar",
    }


def create_manifest(
    output_dir: Path,
    shard_stats: List[Dict[str, Any]],
    target_sr: Optional[int],
    source_csv: str,
) -> None:
    """
    Create a manifest file with shard information.

    Args:
        output_dir: Output directory
        shard_stats: Statistics from shard creation
        target_sr: Target sample rate used
        source_csv: Path to source metadata CSV
    """
    manifest = {
        "source_csv": str(source_csv),
        "target_sample_rate": target_sr,
        "splits": {stats["split"]: stats for stats in shard_stats},
        "total_samples": sum(s["num_samples"] for s in shard_stats),
        "total_shards": sum(s["num_shards"] for s in shard_stats),
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert audio dataset to WebDataset tar shards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--metadata_csv",
        type=str,
        required=True,
        help="Path to metadata CSV file with audio file paths and labels",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write tar shards",
    )
    parser.add_argument(
        "--samples_per_shard",
        type=int,
        default=1000,
        help="Number of samples per tar shard (default: 1000)",
    )
    parser.add_argument(
        "--target_sr",
        type=int,
        default=None,
        help="Target sample rate for resampling (default: keep original)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Process only this split (default: process all splits)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Don't skip existing shards (recreate all)",
    )

    args = parser.parse_args()

    # Load metadata
    print(f"Loading metadata from: {args.metadata_csv}")
    df = pd.read_csv(args.metadata_csv)
    print(f"Total samples: {len(df)}")

    # Validate required columns
    required_cols = ["filename", "split", "label"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Show split distribution
    print("\nSplit distribution:")
    for split, count in df["split"].value_counts().items():
        print(f"  {split}: {count}")

    # Determine splits to process
    if args.split:
        splits = [args.split]
    else:
        splits = df["split"].unique().tolist()

    # Create shards
    output_dir = Path(args.output_dir)
    shard_stats = []

    for split in splits:
        stats = create_shards(
            df=df,
            output_dir=output_dir,
            split=split,
            samples_per_shard=args.samples_per_shard,
            target_sr=args.target_sr,
            num_workers=args.num_workers,
            resume=not args.no_resume,
        )
        shard_stats.append(stats)
        print(f"  Created {stats['num_shards']} shards with {stats['num_samples']} samples")
        if stats["num_failed"] > 0:
            print(f"  Warning: {stats['num_failed']} samples failed to process")

    # Create manifest
    create_manifest(
        output_dir=output_dir,
        shard_stats=shard_stats,
        target_sr=args.target_sr,
        source_csv=args.metadata_csv,
    )

    print("\nDone!")
    print(f"Output directory: {output_dir}")
    print(f"Total shards: {sum(s['num_shards'] for s in shard_stats)}")
    print(f"Total samples: {sum(s['num_samples'] for s in shard_stats)}")

    # Print loading example
    print("\n" + "=" * 60)
    print("Example usage:")
    print("=" * 60)
    print(f"""
from src.common.webdataset_utils import WebDatasetWrapper

# For classification training:
dataset = WebDatasetWrapper(
    tar_paths="{output_dir}/train-{{000000..{shard_stats[0]['num_shards']-1:06d}}}.tar",
    target_sr={args.target_sr or 16000},
    segment_length=5.0,
    shuffle=True,
)

# For COI separation training:
from src.common.webdataset_utils import COIWebDatasetWrapper

dataset = COIWebDatasetWrapper(
    tar_paths="{output_dir}/train-{{000000..{shard_stats[0]['num_shards']-1:06d}}}.tar",
    split="train",
    target_sr={args.target_sr or 16000},
    segment_length=5.0,
)
""")


if __name__ == "__main__":
    main()
