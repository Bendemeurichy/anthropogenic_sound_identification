#!/usr/bin/env python3
"""
Utility script to validate audio files in a dataset.
This helps identify corrupted or invalid WAV files before training.
"""

import sys
from pathlib import Path
import pandas as pd
import argparse

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent))

from helpers import validate_audio_file


def validate_files_from_csv(
    csv_path: str, filename_column: str = "filename", max_files: int = None
):
    """Validate audio files listed in a CSV file.

    Args:
        csv_path: Path to CSV file containing filenames
        filename_column: Name of column with file paths
        max_files: Maximum number of files to check (None = all)
    """
    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    if filename_column not in df.columns:
        print(f"Error: Column '{filename_column}' not found in CSV")
        print(f"Available columns: {list(df.columns)}")
        return

    files_to_check = df[filename_column].unique()

    if max_files:
        files_to_check = files_to_check[:max_files]
        print(f"Checking first {max_files} unique files...")
    else:
        print(f"Checking all {len(files_to_check)} unique files...")

    invalid_files = []
    valid_count = 0

    for i, file_path in enumerate(files_to_check, 1):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(files_to_check)} files checked...")

        is_valid, error_msg = validate_audio_file(str(file_path))

        if not is_valid:
            invalid_files.append((file_path, error_msg))
            print(f"\n❌ Invalid: {file_path}")
            print(f"   Error: {error_msg}")
        else:
            valid_count += 1

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Total files checked: {len(files_to_check)}")
    print(f"Valid files: {valid_count}")
    print(f"Invalid files: {len(invalid_files)}")

    if invalid_files:
        print("\n" + "=" * 70)
        print("INVALID FILES DETAILS")
        print("=" * 70)
        for file_path, error_msg in invalid_files:
            print(f"\n{file_path}")
            print(f"  → {error_msg}")

        # Save invalid files to a text file
        output_file = Path(csv_path).parent / "invalid_audio_files.txt"
        with open(output_file, "w") as f:
            f.write("Invalid Audio Files\n")
            f.write("=" * 70 + "\n\n")
            for file_path, error_msg in invalid_files:
                f.write(f"{file_path}\n")
                f.write(f"  Error: {error_msg}\n\n")

        print(f"\n💾 Invalid files list saved to: {output_file}")


def validate_single_file(file_path: str):
    """Validate a single audio file."""
    print(f"Validating: {file_path}")
    is_valid, error_msg = validate_audio_file(file_path)

    if is_valid:
        print("✅ File is valid")
    else:
        print(f"❌ File is invalid: {error_msg}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate audio files for WAV format compatibility"
    )
    parser.add_argument(
        "input", help="Path to audio file or CSV file containing file paths"
    )
    parser.add_argument(
        "--column",
        default="filename",
        help="Column name in CSV containing file paths (default: filename)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to check (default: all)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    if input_path.suffix == ".csv":
        validate_files_from_csv(str(input_path), args.column, args.max_files)
    else:
        validate_single_file(str(input_path))


if __name__ == "__main__":
    main()
