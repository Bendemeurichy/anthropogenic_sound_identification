"""
Airplane classification validation - Tests BOTH airplane classifiers.

This script tests the multi_coi_29_04 separator with:
- plane (primary classifier)
- ast_finetuned (secondary classifier)

The validation pipeline will test BOTH classifiers and report results separately.

COI configuration:
- Airplane samples (orig_label) → label=1 (COI)
- Bird samples (orig_label) → label=0 (background)
- Other samples → label=0 (background)
"""

import io
import sys

# Under pythonw there is no console and sys.stdout/stderr are None.
# Wrap only when the underlying buffer actually exists and is not already wrapped properly.
# This ensures proper UTF-8 encoding for Windows Start-Process / pythonw environments.
if sys.stdout is not None and hasattr(sys.stdout, "buffer"):
    # Only wrap if not already a TextIOWrapper with the correct encoding
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != "utf-8":
        try:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", line_buffering=True
            )
        except Exception:
            pass  # Keep original stdout if wrapping fails
            
if sys.stderr is not None and hasattr(sys.stderr, "buffer"):
    if not isinstance(sys.stderr, io.TextIOWrapper) or sys.stderr.encoding != "utf-8":
        try:
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, encoding="utf-8", line_buffering=True
            )
        except Exception:
            pass  # Keep original stderr if wrapping fails

from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation_functions.test_pipeline import ValidationPipeline

# ================== CONFIGURATION ==================
BASE_PATH = str(PROJECT_ROOT / "datasets")

# Multi-COI separator checkpoint (trained on both birds and airplanes)
SEP_CHECKPOINT = str(
    PROJECT_ROOT / "src/models/tuss/checkpoints/multi_coi_29_04/best_model.pt"
)

# Dataset CSV (using bird model's CSV with orig_label for re-binarization)
DATA_CSV = str(
    PROJECT_ROOT / "src/models/tuss/checkpoints/20260328_150704/separation_dataset.csv"
)

# Primary classifier for COI synonym detection
PRIMARY_CLASSIFIER = "plane"

# Evaluation settings
SPLIT = "test"
SNR_RANGE = (-5, 5)
SEED = 42
OUTPUT_DIR = "./airplane_validation_both_classifiers_results"
SAVE_EXAMPLES_DIR = "./airplane_validation_both_classifiers_examples"

# Exclude Risoux from synthetic mixtures (evaluated separately)
EXCLUDE_DATASETS = ["risoux_test"]

# ===================================================

def main():
    print("=" * 70)
    print("AIRPLANE CLASSIFICATION VALIDATION - BOTH CLASSIFIERS")
    print("=" * 70)
    print(f"Separator: multi_coi_29_04 (multi-COI model)")
    print(f"Primary Classifier: {PRIMARY_CLASSIFIER}")
    print(f"Secondary Classifier: ast_finetuned")
    print(f"Dataset: {DATA_CSV}")
    print(f"Expected test samples: 2,451 (from 20260328_150704 dataset)")
    print("=" * 70 + "\n")
    
    # Device selection: prefer cuda:1 when multiple GPUs present
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = "cuda:1"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPUs available: {torch.cuda.device_count()}\n")
    
    # Initialize pipeline
    pipeline = ValidationPipeline(base_path=BASE_PATH, device=device)
    
    # Load models - will test BOTH airplane classifiers
    # classifier_type sets COI synonyms to AIRPLANE_SYNONYMS
    # use_ast_finetuned=True loads ast_finetuned as secondary classifier
    pipeline.load_models(
        sep_checkpoint=SEP_CHECKPOINT,
        cls_weights=None,  # Uses default plane classifier weights
        classifier_type=PRIMARY_CLASSIFIER,  # Sets COI=AIRPLANE_SYNONYMS
        use_tuss=True,
        tuss_coi_prompt="airplane",  # Multi-COI model has "airplane" prompt
        tuss_bg_prompt="background",
        use_clapsep=False,
        use_ast_finetuned=True,  # Load as secondary classifier
        use_bird_mae=False,  # Bird classifier
        use_audioprotopnet=False,  # Bird classifier
    )
    
    print("\n" + "=" * 70)
    print("COI SYNONYM CONFIGURATION")
    print("=" * 70)
    print(f"Using COI synonyms: {pipeline.coi_synonyms}")
    print("\nLabel re-binarization (applies to BOTH classifiers):")
    print("  - Airplane samples (in orig_label) → label=1 (COI)")
    print("  - Bird samples (in orig_label) → label=0 (background)")
    print("  - Other samples → label=0 (background)")
    print("=" * 70 + "\n")
    
    # Run validation - will test BOTH plane AND ast_finetuned
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
        balance_classes=True,  # Balance for fair confusion matrix
    )
    
    # Print summary for BOTH classifiers
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE - RESULTS FOR BOTH CLASSIFIERS")
    print("=" * 70)
    for cls_name, cls_results in results.items():
        print(f"\n{'='*70}")
        print(f"Classifier: {cls_name}")
        print(f"{'='*70}")
        for test_name, metrics in cls_results.items():
            print(f"\n  {test_name}:")
            print(f"    F1: {metrics.f1_score:.4f}")
            print(f"    Recall: {metrics.recall:.4f}")
            print(f"    Precision: {metrics.precision:.4f}")
            print(f"    Accuracy: {metrics.accuracy:.4f}")
            print(f"    COI samples: {metrics.final_coi_count}")
            print(f"    BG samples: {metrics.final_background_count}")
            if metrics.contaminated_backgrounds_removed > 0:
                print(f"    ⚠️  Removed {metrics.contaminated_backgrounds_removed} contaminated backgrounds")
            if hasattr(metrics, 'classes_balanced') and metrics.classes_balanced:
                print(f"    ⚖️  Classes balanced (from {metrics.original_coi_count}:{metrics.original_background_count})")
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Examples saved to: {SAVE_EXAMPLES_DIR}")
    print("=" * 70)
    
    # Risoux evaluation (if present in CSV)
    print("\n" + "=" * 70)
    print("RISOUX INDEPENDENT TEST SET (if present)")
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
        )
        
        print("\nRisoux Results:")
        for cls_name, cls_results in risoux_results.items():
            print(f"\nClassifier: {cls_name}")
            for test_name, metrics in cls_results.items():
                print(f"  {test_name}: F1={metrics.f1_score:.4f}")
    except Exception as e:
        print(f"No Risoux data in this CSV or error: {e}")
    
    print("\n" + "=" * 70)
    print("ALL AIRPLANE VALIDATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
