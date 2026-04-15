#!/usr/bin/env python3
"""
Create a combined metadata CSV file from all datasets.

This script loads metadata from all configured datasets (ESC-50, AerosonicDB, 
Risoux, Freesound, Birdset) and combines them into a single CSV file with 
full audio file paths.

Usage:
    python scripts/create_combined_metadata.py
    
    # Or specify custom paths:
    python scripts/create_combined_metadata.py \
        --metadata_dir /path/to/metadata \
        --audio_base_dir /path/to/audio \
        --output combined_metadata.csv
"""

import argparse
import sys
from pathlib import Path

# Add src to path so we can import from label_loading
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.label_loading.metadata_loader import load_metadata_datasets


def main():
    parser = argparse.ArgumentParser(
        description="Create combined metadata CSV from all datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--metadata_dir",
        type=str,
        default=None,
        help="Directory containing dataset metadata files (default: project_root/data/metadata)",
    )
    parser.add_argument(
        "--audio_base_dir",
        type=str,
        default=None,
        help="Base directory containing audio files (default: project_root/../datasets)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/combined_metadata.csv",
        help="Output CSV file path (default: data/combined_metadata.csv)",
    )
    
    args = parser.parse_args()
    
    # Determine paths
    project_root = Path(__file__).parent.parent
    metadata_dir = args.metadata_dir or str(project_root / "data" / "metadata")
    audio_base_dir = args.audio_base_dir or str(project_root.parent / "datasets")
    output_file = Path(args.output)
    
    # Validate paths exist
    if not Path(metadata_dir).exists():
        print(f"Error: Metadata directory not found: {metadata_dir}")
        sys.exit(1)
    
    if not Path(audio_base_dir).exists():
        print(f"Error: Audio base directory not found: {audio_base_dir}")
        sys.exit(1)
    
    print("=" * 70)
    print("Creating combined metadata CSV")
    print("=" * 70)
    print(f"Metadata directory: {metadata_dir}")
    print(f"Audio base directory: {audio_base_dir}")
    print(f"Output file: {output_file}")
    print()
    
    # Load all datasets
    print("Loading datasets...")
    try:
        df = load_metadata_datasets(metadata_dir, audio_base_dir)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        sys.exit(1)
    
    print()
    print("=" * 70)
    print("Dataset Summary")
    print("=" * 70)
    print(f"Total samples: {len(df)}")
    print(f"\nDatasets included:")
    for dataset, count in df["dataset"].value_counts().items():
        print(f"  {dataset}: {count} samples")
    
    print(f"\nSplit distribution:")
    for split, count in df["split"].value_counts().items():
        print(f"  {split}: {count} samples")
    
    print(f"\nColumns: {list(df.columns)}")
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    print()
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print(f"✓ Successfully saved {len(df)} samples to {output_file}")
    print()
    print("=" * 70)
    print("Next steps:")
    print("=" * 70)
    print(f"""
Run create_webdataset.py to convert to WebDataset format:

    python scripts/create_webdataset.py \\
        --metadata_csv {output_file} \\
        --output_dir data/webdatasets/combined \\
        --samples_per_shard 1000 \\
        --target_sr 16000 \\
        --num_workers 8
""")


if __name__ == "__main__":
    main()
