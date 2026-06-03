#!/usr/bin/env python3
"""
Unified validation script for separation + classification pipeline.

Runs airplane or bird detection validation across any combination of:
  - Separators: TUSS, SuDoRM-RF, CLAPSep
  - Classifiers: plane (CNN), pann_finetuned, ast_finetuned, bird_mae, audioprotopnet

The *first* classifier in --classifier becomes the primary (sets COI synonyms and
TUSS/CLAPSep prompt text).  Any additional classifiers are loaded as secondary
classifiers and produce independent results columns.

Pipeline stages on synthetic-mixture data (training-domain test sets):
  1. Clean audio — classification only
  2. Clean audio — separation + classification
  3. Synthetic mixtures — classification only
  4. Synthetic mixtures — separation + classification

When ``--with-risoux`` is given an extra run evaluates the Risoux dataset
*as-is* (only cls and sep+cls on the original recordings).

Examples
--------
.. code-block:: bash

    # TUSS airplane — single plane CNN classifier
    python run_validation.py \\
        --tuss-checkpoint path/to/best_model.pt \\
        --data-csv path/to/dataset.csv \\
        --classifier plane

    # TUSS airplane — plane + AST (both classifiers)
    python run_validation.py \\
        --tuss-checkpoint path/to/best_model.pt \\
        --data-csv path/to/dataset.csv \\
        --classifier plane ast_finetuned

    # CLAPSep airplane — PANN + AST
    python run_validation.py \\
        --separator clapsep \\
        --clapsep-checkpoint path/to/best_model.ckpt \\
        --data-csv path/to/dataset.csv \\
        --classifier pann_finetuned ast_finetuned

    # SuDoRM-RF airplane — PANN + AST
    python run_validation.py \\
        --separator sudormrf \\
        --sudormrf-checkpoint path/to/best_model.pt \\
        --data-csv path/to/dataset.csv \\
        --classifier pann_finetuned ast_finetuned

    # TUSS birds — bird_mae + audioprotopnet, with Risoux eval
    python run_validation.py \\
        --tuss-checkpoint path/to/best_model.pt \\
        --tuss-coi-prompt birds \\
        --data-csv path/to/dataset.csv \\
        --classifier bird_mae audioprotopnet \\
        --with-risoux

    # Single-class TUSS airplane with PANN
    python run_validation.py \\
        --tuss-checkpoint path/to/best_model.pt \\
        --data-csv path/to/dataset.csv \\
        --classifier pann_finetuned
"""

import argparse
import io
import sys

# ---------------------------------------------------------------------------
# Fix stdout/stderr for pythonw / Windows non-console environments.
# Under pythonw there is no console and sys.stdout/stderr are None.
# This ensures proper UTF-8 encoding in those contexts.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Path setup — make both "code/" and "code/src/" importable
# ---------------------------------------------------------------------------
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import torch

from src.validation_functions.test_pipeline import ValidationPipeline
from common.training_utils import _is_tty  # noqa: F401  (used by progress helpers)

# ---------------------------------------------------------------------------
# Default paths (matching the most common configuration: TUSS airplane)
# ---------------------------------------------------------------------------
_DEFAULT_BASE_PATH = str(_PROJECT_ROOT.parent / "datasets")

_DEFAULT_TUSS_CHECKPOINT = str(
    _PROJECT_ROOT / "src/models/tuss/checkpoints/multi_coi_29_04/best_model.pt"
)
_DEFAULT_SUDORMRF_CHECKPOINT = str(
    _PROJECT_ROOT / "src/models/sudormrf/checkpoints/20260316_191707/best_model.pt"
)
_DEFAULT_CLAPSEP_CHECKPOINT = str(
    _PROJECT_ROOT / "src/models/clapsep/checkpoint/CLAPSep/model/best_model.ckpt"
)
_DEFAULT_DATA_CSV = str(
    _PROJECT_ROOT
    / "src/models/tuss/checkpoints/20260328_150704/separation_dataset.csv"
)

# ---------------------------------------------------------------------------
# Classifier helpers
# ---------------------------------------------------------------------------
_BIRD_CLASSIFIERS = frozenset({"bird_mae", "audioprotopnet"})


def _is_bird_classifier(classifier_type: str) -> bool:
    """Return True when *classifier_type* is a bird-detection classifier."""
    return classifier_type in _BIRD_CLASSIFIERS


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified separation+classification validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Separator ---
    sep = p.add_argument_group("Separator")
    sep.add_argument(
        "--separator",
        choices=["tuss", "sudormrf", "clapsep"],
        default="tuss",
        help="Separator model type  [%(default)s]",
    )
    sep.add_argument(
        "--tuss-checkpoint",
        default=_DEFAULT_TUSS_CHECKPOINT,
        help="Path to TUSS checkpoint (.pt)  [built-in default]",
    )
    sep.add_argument(
        "--tuss-coi-prompt",
        default=None,
        help="TUSS COI prompt name (auto-detected from primary classifier if omitted)",
    )
    sep.add_argument(
        "--tuss-bg-prompt",
        default="background",
        help="TUSS background prompt name  [%(default)s]",
    )
    sep.add_argument(
        "--sudormrf-checkpoint",
        default=_DEFAULT_SUDORMRF_CHECKPOINT,
        help="Path to SuDoRM-RF checkpoint (.pt)  [built-in default]",
    )
    sep.add_argument(
        "--clapsep-checkpoint",
        default=_DEFAULT_CLAPSEP_CHECKPOINT,
        help="Path to CLAPSep checkpoint (.ckpt)  [built-in default]",
    )
    sep.add_argument(
        "--clapsep-text-pos",
        default=None,
        help="CLAPSep positive text prompt (auto-detected if omitted)",
    )
    sep.add_argument(
        "--clapsep-text-neg",
        default="",
        help="CLAPSep negative text prompt  [%(default)s]",
    )

    # --- Classifiers ---
    cls_grp = p.add_argument_group("Classifiers")
    cls_grp.add_argument(
        "--classifier",
        nargs="+",
        choices=[
            "plane",
            "pann_finetuned",
            "ast_finetuned",
            "bird_mae",
            "audioprotopnet",
        ],
        default=["plane"],
        help="Classifiers to load.  First = primary (sets COI synonyms).  "
        "Any extras are loaded as secondary classifiers.  [%(default)s]",
    )
    cls_grp.add_argument(
        "--cls-weights",
        default=None,
        help="Path to CNN plane classifier weights (.h5).  "
        "Only needed when 'plane' is the primary classifier.",
    )

    # --- Data ---
    data = p.add_argument_group("Data")
    data.add_argument(
        "--data-csv",
        default=_DEFAULT_DATA_CSV,
        help="Path to dataset CSV (must contain orig_label column)  [built-in default]",
    )
    data.add_argument(
        "--base-path",
        default=_DEFAULT_BASE_PATH,
        help="Base path for audio files (for converting Windows paths)  [built-in default]",
    )
    data.add_argument(
        "--split",
        default="test",
        help="Dataset split to evaluate  [%(default)s]",
    )
    data.add_argument(
        "--exclude-datasets",
        default="risoux_test",
        help="Comma-separated datasets to drop from synthetic mixtures.  "
        "Pass an empty string to exclude nothing.  [%(default)s]",
    )

    # --- Output ---
    out = p.add_argument_group("Output")
    out.add_argument(
        "--output-dir",
        default="./validation_results",
        help="Root directory for JSON results  [%(default)s]",
    )
    out.add_argument(
        "--save-examples-dir",
        default="./validation_examples",
        help="Root directory for saved audio examples  [%(default)s]",
    )
    out.add_argument(
        "--save-n-examples",
        type=int,
        default=5,
        help="Number of random examples to save per test stage  [%(default)s]",
    )

    # --- Evaluation ---
    ev = p.add_argument_group("Evaluation")
    ev.add_argument(
        "--snr-min",
        type=float,
        default=-5,
        help="Minimum SNR (dB) for synthetic mixtures  [%(default)s]",
    )
    ev.add_argument(
        "--snr-max",
        type=float,
        default=5,
        help="Maximum SNR (dB) for synthetic mixtures  [%(default)s]",
    )
    ev.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility  [%(default)s]",
    )
    ev.add_argument(
        "--skip-clean-tests",
        action="store_true",
        help="Skip clean-audio test stages (1 and 2)",
    )
    ev.add_argument(
        "--no-balance",
        action="store_true",
        help="Disable class balancing (by default classes are balanced)",
    )
    ev.add_argument(
        "--with-risoux",
        action="store_true",
        help="Run a separate independent evaluation on the Risoux dataset as-is",
    )

    # --- Hardware ---
    hw = p.add_argument_group("Hardware")
    hw.add_argument(
        "--device",
        default=None,
        help="Device for inference (e.g. 'cuda:0', 'cuda:1', 'cpu').  "
        "Auto-detected when omitted.",
    )

    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_results(results, label: str) -> None:
    """Print validation results dict to stdout."""
    print(f"\n{'=' * 70}")
    print(f"RESULTS — {label}")
    print(f"{'=' * 70}")
    for cls_name, cls_results in results.items():
        print(f"\nClassifier: {cls_name}")
        for test_name, metrics in cls_results.items():
            print(
                f"  {test_name}: "
                f"F1={metrics.f1_score:.4f}  "
                f"Recall={metrics.recall:.4f}  "
                f"Precision={metrics.precision:.4f}"
            )
            if hasattr(metrics, "mean_si_snr") and metrics.mean_si_snr is not None:
                snri = (
                    f"{metrics.mean_si_snri:+.2f}"
                    if metrics.mean_si_snri is not None
                    else "n/a"
                )
                print(f"    SI-SNR={metrics.mean_si_snr:+.2f} dB  SI-SNRi={snri} dB")
            if (
                hasattr(metrics, "contaminated_backgrounds_removed")
                and metrics.contaminated_backgrounds_removed > 0
            ):
                print(
                    f"    Removed {metrics.contaminated_backgrounds_removed} "
                    f"contaminated backgrounds"
                )
            if hasattr(metrics, "classes_balanced") and metrics.classes_balanced:
                print(
                    f"    Classes balanced "
                    f"(from {metrics.original_coi_count}:{metrics.original_background_count})"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = build_parser().parse_args()

    # ---- Classifier configuration ------------------------------------------
    primary_cls = args.classifier[0]
    secondary_cls = set(args.classifier[1:])

    is_bird = _is_bird_classifier(primary_cls)
    coi_label = "bird" if is_bird else "airplane"

    # ---- Separator selection ------------------------------------------------
    use_tuss = args.separator == "tuss"
    use_clapsep = args.separator == "clapsep"

    if use_tuss:
        sep_checkpoint = args.tuss_checkpoint
    elif use_clapsep:
        sep_checkpoint = args.clapsep_checkpoint
    else:
        sep_checkpoint = args.sudormrf_checkpoint

    # ---- Prompt auto-detection ----------------------------------------------
    tuss_coi_prompt = args.tuss_coi_prompt or ("bird" if is_bird else "airplane")
    clapsep_text_pos = args.clapsep_text_pos or ("bird" if is_bird else "airplane")

    # ---- Secondary classifier flags -----------------------------------------
    use_ast = "ast_finetuned" in secondary_cls
    use_audioprotopnet = "audioprotopnet" in secondary_cls
    use_bird_mae = (
        primary_cls != "bird_mae" and "bird_mae" in secondary_cls
    )

    # ---- Excluded datasets --------------------------------------------------
    exclude_datasets: list[str] | None = None
    if args.exclude_datasets.strip():
        exclude_datasets = [
            d.strip() for d in args.exclude_datasets.split(",") if d.strip()
        ]
    if not exclude_datasets:
        exclude_datasets = None

    # ---- Banner ------------------------------------------------------------
    print("=" * 70)
    print(f"{coi_label.upper()} CLASSIFICATION VALIDATION")
    print("=" * 70)
    print(f"Separator:        {args.separator}  ({sep_checkpoint})")
    print(f"Primary classifier:  {primary_cls}")
    if secondary_cls:
        print(f"Secondary classifier(s): {', '.join(sorted(secondary_cls))}")
    print(f"COI:               {coi_label}")
    print(f"Dataset CSV:       {args.data_csv}")
    print(f"Output dir:        {args.output_dir}")
    if exclude_datasets:
        print(f"Excluded from mixtures:  {exclude_datasets}")
    print(f"Risoux eval:       {'yes' if args.with_risoux else 'no'}")
    if args.device:
        print(f"Device:            {args.device}")
    print("=" * 70 + "\n")

    # ---- Pipeline ----------------------------------------------------------
    pipeline = ValidationPipeline(
        base_path=args.base_path,
        device=args.device,
    )

    pipeline.load_models(
        sep_checkpoint=sep_checkpoint,
        cls_weights=args.cls_weights,
        classifier_type=primary_cls,
        use_tuss=use_tuss,
        tuss_coi_prompt=tuss_coi_prompt,
        tuss_bg_prompt=args.tuss_bg_prompt,
        use_clapsep=use_clapsep,
        clapsep_text_pos=clapsep_text_pos,
        clapsep_text_neg=args.clapsep_text_neg,
        use_ast_finetuned=use_ast,
        use_bird_mae=use_bird_mae,
        use_audioprotopnet=use_audioprotopnet,
    )

    print(f"COI synonyms: {pipeline.coi_synonyms}\n")

    # ---- Shared run parameters --------------------------------------------
    snr_range = (args.snr_min, args.snr_max)
    balance = not args.no_balance

    # ---- Main evaluation --------------------------------------------------
    results = pipeline.run(
        split=args.split,
        snr_range=snr_range,
        data_csv=args.data_csv,
        output_dir=args.output_dir,
        seed=args.seed,
        save_examples_dir=args.save_examples_dir,
        save_n_examples=args.save_n_examples,
        exclude_datasets=exclude_datasets,
        skip_clean_tests=args.skip_clean_tests,
        save_false_negatives=True,
        balance_classes=balance,
    )

    _print_results(results, "MAIN EVALUATION")
    print(f"\nResults saved to: {args.output_dir}")

    # ---- Risoux independent evaluation (optional) --------------------------
    if args.with_risoux:
        print("\n" + "=" * 70)
        print("RISOUX INDEPENDENT TEST SET (as-is evaluation)")
        print("=" * 70)

        risoux_out = args.output_dir.rstrip("/") + "_risoux"
        risoux_examples = args.save_examples_dir.rstrip("/") + "_risoux"

        try:
            risoux_results = pipeline.run(
                split=args.split,
                only_dataset="risoux_test",
                snr_range=snr_range,
                data_csv=args.data_csv,
                output_dir=risoux_out,
                seed=args.seed,
                save_examples_dir=risoux_examples,
                save_n_examples=args.save_n_examples,
                skip_clean_tests=True,
                save_false_negatives=True,
                balance_classes=False,
            )
            _print_results(risoux_results, "RISOUX")
        except Exception as e:
            print(f"No Risoux data or error: {e}")

    print("\n" + "=" * 70)
    print(f"{coi_label.upper()} VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
