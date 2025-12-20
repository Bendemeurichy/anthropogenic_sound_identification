"""
Validate training data quality - check if mixtures and sources are properly aligned.
"""

import pandas as pd
import numpy as np
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm


def validate_training_pair(mixture_path, source_path, sample_rate=16000):
    """Validate a single mixture-source pair."""
    try:
        # Load mixture
        mix, sr1 = torchaudio.load(mixture_path)
        if sr1 != sample_rate:
            mix = torchaudio.transforms.Resample(sr1, sample_rate)(mix)
        if mix.shape[0] > 1:
            mix = mix.mean(dim=0)
        else:
            mix = mix.squeeze(0)

        # Load source
        src, sr2 = torchaudio.load(source_path)
        if sr2 != sample_rate:
            src = torchaudio.transforms.Resample(sr2, sample_rate)(src)
        if src.shape[0] > 1:
            src = src.mean(dim=0)
        else:
            src = src.squeeze(0)

        # Basic checks
        issues = []

        # 1. Length mismatch
        if abs(mix.shape[0] - src.shape[0]) > sample_rate * 0.1:  # >100ms difference
            issues.append(f"Length mismatch: mix={mix.shape[0]}, src={src.shape[0]}")

        # 2. Silent mixture
        if mix.abs().max() < 0.001:
            issues.append("Mixture is essentially silent")

        # 3. Silent source
        if src.abs().max() < 0.001:
            issues.append("Source is essentially silent")

        # 4. Source louder than mixture (impossible)
        if src.abs().max() > mix.abs().max() * 1.5:
            issues.append(
                f"Source louder than mixture: src_max={src.abs().max():.4f}, mix_max={mix.abs().max():.4f}"
            )

        # 5. Very low energy ratio
        L = min(mix.shape[0], src.shape[0])
        mix_energy = mix[:L].pow(2).sum()
        src_energy = src[:L].pow(2).sum()
        ratio = src_energy / (mix_energy + 1e-8)
        if ratio < 0.001:
            issues.append(f"Source has very low energy ratio: {ratio:.6f}")

        # 6. Check if source is contained in mixture (correlation)
        if L > 0:
            corr = torch.corrcoef(torch.stack([mix[:L], src[:L]]))[0, 1]
            if abs(corr) < 0.05:  # Very low correlation suggests misalignment
                issues.append(f"Very low correlation: {corr:.4f}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "mix_energy": float(mix_energy),
            "src_energy": float(src_energy),
            "ratio": float(ratio),
            "correlation": float(corr) if L > 0 else 0.0,
        }

    except Exception as e:
        return {
            "valid": False,
            "issues": [f"Error loading: {str(e)}"],
            "mix_energy": 0.0,
            "src_energy": 0.0,
            "ratio": 0.0,
            "correlation": 0.0,
        }


def validate_dataset(csv_path, base_path=None, max_samples=100):
    """Validate training dataset."""
    print("Loading dataset CSV...")
    df = pd.read_csv(csv_path)

    if "split" in df.columns:
        train_df = df[df["split"] == "train"]
    else:
        train_df = df

    print(f"Found {len(train_df)} training samples")

    # Sample randomly if too many
    if len(train_df) > max_samples:
        train_df = train_df.sample(n=max_samples, random_state=42)
        print(f"Sampling {max_samples} for validation")

    results = []
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Validating"):
        # Assuming columns: mixture_path, source_path
        mix_path = row.get("mixture_path") or row.get("filename")
        src_path = row.get("source_path")

        if pd.isna(src_path):
            results.append(
                {
                    "valid": False,
                    "issues": ["Missing source_path"],
                    "mix_energy": 0.0,
                    "src_energy": 0.0,
                    "ratio": 0.0,
                    "correlation": 0.0,
                }
            )
            continue

        # Convert paths if needed
        if base_path:
            mix_path = str(Path(base_path) / Path(mix_path).name)
            src_path = str(Path(base_path) / Path(src_path).name)

        result = validate_training_pair(mix_path, src_path)
        results.append(result)

    # Summary
    valid_count = sum(1 for r in results if r["valid"])
    print(f"\n{'='*80}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total samples checked: {len(results)}")
    print(f"Valid samples: {valid_count} ({valid_count/len(results)*100:.1f}%)")
    print(f"Invalid samples: {len(results) - valid_count}")

    # Common issues
    all_issues = []
    for r in results:
        all_issues.extend(r["issues"])

    if all_issues:
        print(f"\nMost common issues:")
        from collections import Counter

        issue_counts = Counter(all_issues)
        for issue, count in issue_counts.most_common(10):
            print(f"  - {issue}: {count} samples")

    # Energy statistics
    valid_results = [r for r in results if r["valid"]]
    if valid_results:
        ratios = [r["ratio"] for r in valid_results]
        corrs = [r["correlation"] for r in valid_results]
        print(f"\nEnergy ratio statistics (source/mixture):")
        print(f"  Mean: {np.mean(ratios):.4f}")
        print(f"  Median: {np.median(ratios):.4f}")
        print(f"  Min: {np.min(ratios):.4f}, Max: {np.max(ratios):.4f}")

        print(f"\nCorrelation statistics:")
        print(f"  Mean: {np.mean(corrs):.4f}")
        print(f"  Median: {np.median(corrs):.4f}")
        print(f"  Min: {np.min(corrs):.4f}, Max: {np.max(corrs):.4f}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="/home/bendm/Thesis/project/code/src/models/sudormrf/checkpoints/20251217_120142/separation_dataset.csv",
    )
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()

    validate_dataset(args.csv, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
