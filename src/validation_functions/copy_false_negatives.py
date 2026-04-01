#!/usr/bin/env python3
"""
Script to copy false negative samples from validation results to a review directory.

This script reads the JSON results file produced by test_pipeline.py (when run with
save_false_negatives=True) and copies all misclassified COI samples to a separate
directory for manual review.

Usage:
    python copy_false_negatives.py --results results_test_20260401_123456.json --output ./review_samples
    
Or with custom base path:
    python copy_false_negatives.py --results results.json --output ./review --base-path /datasets
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional


def _convert_path(filepath: str, base_path: Optional[str] = None) -> str:
    """Convert Windows paths to Linux paths and apply base_path if needed.
    
    This mirrors the logic from ValidationPipeline._convert_path().
    """
    if filepath.startswith("D:\\") or filepath.startswith("C:\\"):
        filepath = filepath.replace("\\", "/")
        filepath = "/" + filepath[3:]

    if base_path:
        for marker in ["/datasets/", "/masterproef/datasets/"]:
            if marker in filepath:
                rel_path = filepath.split(marker)[-1]
                return os.path.join(base_path, rel_path)
    return filepath


def load_false_negatives(results_file: str) -> List[Dict[str, Any]]:
    """Load false negative samples from a JSON results file.
    
    Args:
        results_file: Path to the JSON results file from test_pipeline.py
        
    Returns:
        List of false negative sample dictionaries
    """
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # The results file has multiple test stages (e.g., as_is_cls, as_is_sep_cls)
    # We need to collect false negatives from all stages
    all_false_negatives = []
    
    for test_name, test_results in data.items():
        # Skip non-test metadata fields
        if not isinstance(test_results, dict):
            continue
        
        if "false_negative_samples" in test_results:
            fn_samples = test_results["false_negative_samples"]
            print(f"  Found {len(fn_samples)} false negatives in '{test_name}'")
            
            # Add test name to each sample for tracking
            for sample in fn_samples:
                sample["test_stage"] = test_name
                all_false_negatives.append(sample)
    
    return all_false_negatives


def copy_false_negatives(
    results_file: str,
    output_dir: str,
    base_path: Optional[str] = None,
    create_manifest: bool = True,
) -> None:
    """Copy false negative audio samples to a review directory.
    
    Args:
        results_file: Path to JSON results file from test_pipeline.py
        output_dir: Directory where samples will be copied
        base_path: Optional base path for converting dataset paths
        create_manifest: If True, create a JSON manifest with sample metadata
    """
    print(f"Loading false negatives from: {results_file}")
    false_negatives = load_false_negatives(results_file)
    
    if not false_negatives:
        print("No false negative samples found in results file.")
        return
    
    print(f"\nTotal false negatives found: {len(false_negatives)}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path.absolute()}")
    
    # Track statistics
    copied = 0
    missing = 0
    errors = 0
    missing_files = []
    
    # Copy each false negative sample
    for idx, sample in enumerate(false_negatives, 1):
        filename = sample.get("filename", "")
        if not filename:
            print(f"  [{idx}/{len(false_negatives)}] Skipping: no filename in sample")
            errors += 1
            continue
        
        # Convert path if needed
        src_path = _convert_path(filename, base_path)
        
        if not os.path.exists(src_path):
            print(f"  [{idx}/{len(false_negatives)}] Missing: {src_path}")
            missing += 1
            missing_files.append({
                "original_path": filename,
                "converted_path": src_path,
                "sample_metadata": sample
            })
            continue
        
        # Create a meaningful filename for the copy
        # Include index, original label, and confidence in the filename
        orig_label = str(sample.get("orig_label", "unknown")).replace("/", "-")[:50]
        confidence = sample.get("confidence", 0.0)
        test_stage = sample.get("test_stage", "unknown")
        
        # Get original filename without path
        orig_name = Path(src_path).stem
        orig_ext = Path(src_path).suffix
        
        # Create descriptive filename
        new_filename = f"{idx:03d}_{test_stage}_{orig_label}_conf{confidence:.3f}_{orig_name}{orig_ext}"
        # Clean up any invalid characters
        new_filename = new_filename.replace(" ", "_").replace('"', "")
        
        dst_path = output_path / new_filename
        
        try:
            shutil.copy2(src_path, dst_path)
            print(f"  [{idx}/{len(false_negatives)}] Copied: {orig_name}{orig_ext} -> {new_filename}")
            copied += 1
        except Exception as e:
            print(f"  [{idx}/{len(false_negatives)}] Error copying {src_path}: {e}")
            errors += 1
    
    # Create manifest file with all metadata
    if create_manifest:
        manifest_path = output_path / "false_negatives_manifest.json"
        manifest = {
            "source_results_file": str(Path(results_file).absolute()),
            "total_false_negatives": len(false_negatives),
            "copied": copied,
            "missing": missing,
            "errors": errors,
            "samples": false_negatives,
            "missing_files": missing_files
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"\nManifest saved to: {manifest_path}")
    
    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  Total false negatives: {len(false_negatives)}")
    print(f"  Successfully copied:   {copied}")
    print(f"  Missing files:         {missing}")
    print(f"  Errors:                {errors}")
    print(f"{'=' * 60}")
    
    if missing_files:
        print(f"\nWarning: {len(missing_files)} files could not be found.")
        print(f"See {output_path / 'false_negatives_manifest.json'} for details.")


def main():
    parser = argparse.ArgumentParser(
        description="Copy false negative samples from validation results for manual review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - copy false negatives to review directory
  python copy_false_negatives.py --results validation_results/cnn/results_test_risoux_test_20260401_123456.json --output ./review_samples
  
  # With custom base path for dataset conversion
  python copy_false_negatives.py --results results.json --output ./review --base-path /mnt/datasets
  
  # Without creating manifest file
  python copy_false_negatives.py --results results.json --output ./review --no-manifest
        """
    )
    
    parser.add_argument(
        "--results", "-r",
        required=True,
        help="Path to JSON results file from test_pipeline.py"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory where samples will be copied"
    )
    
    parser.add_argument(
        "--base-path", "-b",
        default=None,
        help="Base path for dataset files (for converting Windows/relative paths)"
    )
    
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Don't create a JSON manifest file with sample metadata"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.results):
        print(f"Error: Results file not found: {args.results}")
        return 1
    
    try:
        copy_false_negatives(
            results_file=args.results,
            output_dir=args.output,
            base_path=args.base_path,
            create_manifest=not args.no_manifest,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
