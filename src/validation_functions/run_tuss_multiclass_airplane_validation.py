"""
Airplane validation — TUSS multi-class separator (airplane head), both airplane classifiers.

Tests the multi_coi_29_04 checkpoint with tuss_coi_prompt="airplane" and:
- plane (primary classifier)
- ast_finetuned (secondary classifier)

This is used to compare the multi-class TUSS head against the single-class baseline.

COI configuration:
- Airplane samples (orig_label) → label=1 (COI)
- Bird samples (orig_label) → label=0 (background)
- Other samples → label=0 (background)
"""

import io
import sys

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

from pathlib import Path
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

PRIMARY_CLASSIFIER = "pann_finetuned"
SPLIT = "test"
SNR_RANGE = (-5, 5)
SEED = 42
OUTPUT_DIR = "./tuss_multiclass_airplane_results"
SAVE_EXAMPLES_DIR = "./tuss_multiclass_airplane_examples"
EXCLUDE_DATASETS = ["risoux_test"]
# ===================================================


def main():
    print("=" * 70)
    print("AIRPLANE VALIDATION — TUSS MULTI-CLASS (airplane head), BOTH CLASSIFIERS")
    print("=" * 70)
    print(f"Separator: TUSS multi_coi_29_04 — airplane head")
    print(f"Primary Classifier: {PRIMARY_CLASSIFIER}")
    print(f"Secondary Classifier: ast_finetuned")
    print(f"Dataset: {DATA_CSV}")
    print("=" * 70 + "\n")

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = "cuda:1"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    print(f"Using device: {device}\n")

    pipeline = ValidationPipeline(base_path=BASE_PATH, device=device)

    pipeline.load_models(
        sep_checkpoint=SEP_CHECKPOINT,
        cls_weights=None,
        classifier_type=PRIMARY_CLASSIFIER,
        use_tuss=True,
        tuss_coi_prompt="airplane",
        tuss_bg_prompt="background",
        use_clapsep=False,
        use_ast_finetuned=True,
        use_bird_mae=False,
        use_audioprotopnet=False,
    )

    print(f"COI synonyms: {pipeline.coi_synonyms}\n")

    results = pipeline.run(
        split=SPLIT,
        snr_range=SNR_RANGE,
        data_csv=DATA_CSV,
        output_dir=OUTPUT_DIR,
        seed=SEED,
        save_examples_dir=SAVE_EXAMPLES_DIR,
        save_n_examples=5,
        exclude_datasets=EXCLUDE_DATASETS,
        skip_clean_tests=False,
        save_false_negatives=True,
        balance_classes=True,
    )

    print("\n" + "=" * 70)
    print("RESULTS — TUSS MULTI-CLASS (airplane head), BOTH CLASSIFIERS")
    print("=" * 70)
    for cls_name, cls_results in results.items():
        print(f"\nClassifier: {cls_name}")
        for test_name, metrics in cls_results.items():
            print(
                f"  {test_name}: F1={metrics.f1_score:.4f}  Recall={metrics.recall:.4f}  Precision={metrics.precision:.4f}"
            )
            if metrics.mean_si_snr is not None:
                snri = (
                    f"{metrics.mean_si_snri:+.2f}"
                    if metrics.mean_si_snri is not None
                    else "n/a"
                )
                print(f"    SI-SNR={metrics.mean_si_snr:+.2f} dB  SI-SNRi={snri} dB")

    # Risoux independent test set
    print("\n" + "=" * 70)
    print("RISOUX INDEPENDENT TEST SET")
    print("=" * 70)
    try:
        risoux_results = pipeline.run(
            split="test",
            only_dataset="risoux_test",
            snr_range=SNR_RANGE,
            data_csv=DATA_CSV,
            output_dir=OUTPUT_DIR + "_risoux",
            seed=SEED,
            save_examples_dir=SAVE_EXAMPLES_DIR + "_risoux",
            save_n_examples=5,
            skip_clean_tests=True,
            save_false_negatives=True,
            balance_classes=False,
        )
        for cls_name, cls_results in risoux_results.items():
            print(f"\nClassifier: {cls_name}")
            for test_name, metrics in cls_results.items():
                print(f"  {test_name}: F1={metrics.f1_score:.4f}")
    except Exception as e:
        print(f"No Risoux data or error: {e}")

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
