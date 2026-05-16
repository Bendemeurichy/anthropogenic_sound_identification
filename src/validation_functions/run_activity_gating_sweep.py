"""
Activity gating sweep — TUSS multi-class, airplane head.

Sweeps the mask recycler similarity_threshold over a range of values and records:
- Cache hit rate
- Mean SI-SNR of separated output
- Mean SI-SNRi (improvement over mixture)
- Mean RMS error
- Classification F1 score
- Mean SEL error

Results are saved to activity_gating_results/sweep_results.json so they can be
loaded by the plotting scripts without re-running.

Design:
  For each threshold value the full validation pipeline is re-initialised with
  mask recycling enabled at that threshold.  The test split of the airplane
  dataset is used (synthetic mixtures only, Risoux excluded).  Results are
  accumulated across all recordings and written to JSON at the end.
"""

import io
import json
import sys
from pathlib import Path

if sys.stdout is not None and hasattr(sys.stdout, "buffer"):
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != "utf-8":
        try:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", line_buffering=True
            )
        except Exception:
            pass

if sys.stderr is not None and hasattr(sys.stderr, "buffer"):
    if not isinstance(sys.stderr, io.TextIOWrapper) or sys.stderr.encoding != "utf-8":
        try:
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, encoding="utf-8", line_buffering=True
            )
        except Exception:
            pass

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation_functions.test_pipeline import ValidationPipeline

# ================== CONFIGURATION ==================
BASE_PATH = str(PROJECT_ROOT.parent / "datasets")

SEP_CHECKPOINT = str(
    PROJECT_ROOT / "src/models/tuss/checkpoints/multi_coi_29_04/best_model.pt"
)

DATA_CSV = str(
    PROJECT_ROOT / "src/models/tuss/checkpoints/20260328_150704/separation_dataset.csv"
)

PRIMARY_CLASSIFIER = "plane"

# Thresholds to sweep (higher = more conservative, fewer cache hits)
THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

SPLIT = "test"
SNR_RANGE = (-5, 5)
SEED = 42
OUTPUT_DIR = Path("./activity_gating_results")
CACHE_SIZE = 10
# ===================================================


def run_at_threshold(threshold: float, device: str) -> dict:
    """Run validation at a single similarity_threshold and return result dict."""
    print(f"\n{'='*60}")
    print(f"  Threshold: {threshold:.2f}")
    print(f"{'='*60}")

    pipeline = ValidationPipeline(base_path=BASE_PATH, device=device)
    pipeline.load_models(
        sep_checkpoint=SEP_CHECKPOINT,
        cls_weights=None,
        classifier_type=PRIMARY_CLASSIFIER,
        use_tuss=True,
        tuss_coi_prompt="airplane",
        tuss_bg_prompt="background",
        tuss_enable_mask_recycling=True,
        tuss_cache_size=CACHE_SIZE,
        tuss_similarity_threshold=threshold,
        use_clapsep=False,
        use_ast_finetuned=False,  # Use only primary classifier for speed
        use_bird_mae=False,
        use_audioprotopnet=False,
    )

    results = pipeline.run(
        split=SPLIT,
        snr_range=SNR_RANGE,
        data_csv=DATA_CSV,
        output_dir=None,  # Don't write per-threshold JSON files
        seed=SEED,
        save_examples_dir=None,
        save_n_examples=0,
        exclude_datasets=["risoux_test"],
        skip_clean_tests=True,   # Only mixtures for speed
        save_false_negatives=False,
        balance_classes=True,
    )

    # Extract metrics from primary classifier
    # results is {classifier_name: {test_name: ClassificationMetrics}}
    primary_key = list(results.keys())[0]
    test_metrics = results[primary_key]

    # We want the separation+classification mixture results, not classification-only
    if "mix_sep_cls" in test_metrics:
        metrics = test_metrics["mix_sep_cls"]
    elif "mix_cls" in test_metrics:
        metrics = test_metrics["mix_cls"]
    else:
        metrics = test_metrics[list(test_metrics.keys())[0]]

    # Get cache stats from the separator
    cache_stats = None
    if hasattr(pipeline, "separator") and hasattr(pipeline.separator, "get_stats"):
        cache_stats = pipeline.separator.get_stats()

    result = {
        "threshold": threshold,
        "f1_score": float(metrics.f1_score),
        "recall": float(metrics.recall),
        "precision": float(metrics.precision),
        "balanced_accuracy": float(metrics.balanced_accuracy),
    }

    if metrics.mean_si_snr is not None:
        result["mean_si_snr_db"] = float(metrics.mean_si_snr)
    if metrics.mean_si_snri is not None:
        result["mean_si_snri_db"] = float(metrics.mean_si_snri)
    if metrics.mean_rms_error_db is not None:
        result["mean_rms_error_db"] = float(metrics.mean_rms_error_db)
    if metrics.mean_sel_error_db is not None:
        result["mean_sel_error_db"] = float(metrics.mean_sel_error_db)
    if metrics.mean_sdr is not None:
        result["mean_sdr_db"] = float(metrics.mean_sdr)

    if cache_stats:
        result["cache_hit_rate"] = float(cache_stats["hit_rate"])
        result["cache_hits"] = int(cache_stats["hits"])
        result["cache_misses"] = int(cache_stats["misses"])
        result["cache_total"] = int(cache_stats["total_requests"])

    print(f"  F1={metrics.f1_score:.4f}  "
          f"SI-SNR={metrics.mean_si_snr:+.2f} dB  " if metrics.mean_si_snr else "",
          end="")
    if cache_stats:
        print(f"  hit_rate={cache_stats['hit_rate']:.1%}")
    else:
        print()

    return result


def main():
    print("=" * 70)
    print("ACTIVITY GATING SWEEP — TUSS multi_coi_29_04 (airplane head)")
    print("=" * 70)
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Cache size: {CACHE_SIZE}")
    print(f"Dataset: {DATA_CSV}")
    print("=" * 70 + "\n")

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = "cuda:1"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    print(f"Using device: {device}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sweep_results = []
    for threshold in THRESHOLDS:
        result = run_at_threshold(threshold, device)
        sweep_results.append(result)
        print(f"  -> threshold={threshold:.2f}: {result}")

    # Also run with mask recycling disabled (baseline, hit_rate=0)
    print("\n" + "=" * 60)
    print("  Baseline: mask recycling DISABLED (threshold=N/A)")
    print("=" * 60)
    pipeline_base = ValidationPipeline(base_path=BASE_PATH, device=device)
    pipeline_base.load_models(
        sep_checkpoint=SEP_CHECKPOINT,
        cls_weights=None,
        classifier_type=PRIMARY_CLASSIFIER,
        use_tuss=True,
        tuss_coi_prompt="airplane",
        tuss_bg_prompt="background",
        tuss_enable_mask_recycling=False,
        use_clapsep=False,
        use_ast_finetuned=False,
        use_bird_mae=False,
        use_audioprotopnet=False,
    )
    base_results = pipeline_base.run(
        split=SPLIT,
        snr_range=SNR_RANGE,
        data_csv=DATA_CSV,
        output_dir=None,
        seed=SEED,
        save_examples_dir=None,
        save_n_examples=0,
        exclude_datasets=["risoux_test"],
        skip_clean_tests=True,
        save_false_negatives=False,
        balance_classes=True,
    )
    primary_key = list(base_results.keys())[0]
    test_metrics = base_results[primary_key]
    if "mix_sep_cls" in test_metrics:
        base_metrics = test_metrics["mix_sep_cls"]
    elif "mix_cls" in test_metrics:
        base_metrics = test_metrics["mix_cls"]
    else:
        base_metrics = test_metrics[list(test_metrics.keys())[0]]

    baseline_entry = {
        "threshold": None,
        "mask_recycling": False,
        "f1_score": float(base_metrics.f1_score),
        "recall": float(base_metrics.recall),
        "precision": float(base_metrics.precision),
        "balanced_accuracy": float(base_metrics.balanced_accuracy),
        "cache_hit_rate": 0.0,
        "cache_hits": 0,
        "cache_misses": 0,
        "cache_total": 0,
    }
    if base_metrics.mean_si_snr is not None:
        baseline_entry["mean_si_snr_db"] = float(base_metrics.mean_si_snr)
    if base_metrics.mean_si_snri is not None:
        baseline_entry["mean_si_snri_db"] = float(base_metrics.mean_si_snri)
    if base_metrics.mean_rms_error_db is not None:
        baseline_entry["mean_rms_error_db"] = float(base_metrics.mean_rms_error_db)
    if base_metrics.mean_sel_error_db is not None:
        baseline_entry["mean_sel_error_db"] = float(base_metrics.mean_sel_error_db)

    output = {
        "config": {
            "sep_checkpoint": SEP_CHECKPOINT,
            "data_csv": DATA_CSV,
            "thresholds": THRESHOLDS,
            "cache_size": CACHE_SIZE,
            "snr_range": list(SNR_RANGE),
            "split": SPLIT,
            "seed": SEED,
            "classifier": PRIMARY_CLASSIFIER,
        },
        "baseline_no_recycling": baseline_entry,
        "sweep": sweep_results,
    }

    out_path = OUTPUT_DIR / "sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSweep complete. Results saved to: {out_path}")
    print("\nSummary:")
    print(f"  {'Threshold':>10}  {'Hit rate':>9}  {'F1':>7}  {'SI-SNR':>8}  {'SI-SNRi':>9}")
    print(f"  {'baseline':>10}  {'0.0%':>9}  {baseline_entry['f1_score']:>7.4f}  "
          f"{baseline_entry.get('mean_si_snr_db', float('nan')):>+8.2f}  "
          f"{baseline_entry.get('mean_si_snri_db', float('nan')):>+9.2f}")
    for r in sweep_results:
        hit = r.get("cache_hit_rate", float("nan"))
        print(f"  {r['threshold']:>10.2f}  {hit:>9.1%}  {r['f1_score']:>7.4f}  "
              f"{r.get('mean_si_snr_db', float('nan')):>+8.2f}  "
              f"{r.get('mean_si_snri_db', float('nan')):>+9.2f}")


if __name__ == "__main__":
    main()
