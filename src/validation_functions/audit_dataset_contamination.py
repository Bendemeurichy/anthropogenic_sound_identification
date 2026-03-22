#!/usr/bin/env python3
"""
Audit script to check contamination in an actual separation dataset CSV.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from src.validation_functions.test_pipeline import _filter_contaminated_backgrounds


def audit_dataset(csv_path: str):
    """Audit a dataset CSV for contaminated background samples."""
    print("=" * 70)
    print(f"AUDITING DATASET: {csv_path}")
    print("=" * 70)

    # Load dataset
    df = pd.read_csv(csv_path)
    print(f"\nTotal samples: {len(df)}")

    # Separate COI and background
    df_coi = df[df["label"] == 1].copy()
    df_bg = df[df["label"] == 0].copy()

    print(f"  COI samples: {len(df_coi)}")
    print(f"  Background samples: {len(df_bg)}")

    # Check for orig_label column
    if "orig_label" not in df_bg.columns:
        print("\n⚠️  No 'orig_label' column found - cannot detect contamination")
        return

    # Apply filter with verbose output
    print("\n" + "=" * 70)
    print("CONTAMINATION ANALYSIS")
    print("=" * 70)

    filtered_df_bg, n_contaminated = _filter_contaminated_backgrounds(
        df_bg, verbose=True
    )

    # Summary
    print("\nSUMMARY:")
    print(f"  Original background samples: {len(df_bg)}")
    print(f"  Contaminated samples: {n_contaminated}")
    print(f"  Clean background samples: {len(filtered_df_bg)}")
    print(
        f"  Contamination rate: {100 * n_contaminated / len(df_bg) if len(df_bg) > 0 else 0:.2f}%"
    )

    # Show detailed breakdown by split
    print("\nDETAILED BREAKDOWN BY SPLIT:")
    for split in ["train", "val", "test"]:
        split_bg = df_bg[df_bg["split"] == split]
        if len(split_bg) == 0:
            continue

        split_filtered, split_contam = _filter_contaminated_backgrounds(
            split_bg, verbose=False
        )

        print(f"\n  {split.upper()}:")
        print(f"    Total background: {len(split_bg)}")
        print(f"    Contaminated: {split_contam}")
        print(f"    Clean: {len(split_filtered)}")
        print(
            f"    Contamination rate: {100 * split_contam / len(split_bg):.2f}%"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Default to the checkpoint CSV if no argument provided
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "/home/bendm/Thesis/project/code/src/models/sudormrf/checkpoints/20260316_191707/separation_dataset.csv"

    if not Path(csv_path).exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    audit_dataset(csv_path)
