"""
Example script for running airplane classification validation.

This script demonstrates how to run validation for AIRPLANE detection using:
- An airplane-trained separator (TUSS with airplane prompt, or SudoRM-RF)
- An airplane classifier (plane, ast_finetuned, or pann_finetuned)
- Automatic COI synonym detection (AIRPLANE_SYNONYMS)

The pipeline automatically:
1. Re-binarizes labels: airplane samples → label=1, bird samples → label=0 (background)
2. Excludes Risoux from synthetic mixtures (evaluated separately as independent test set)
3. Creates synthetic mixtures with airplanes + background at various SNRs
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation_functions.test_pipeline import ValidationPipeline

# ================== CONFIGURATION ==================
BASE_PATH = str(PROJECT_ROOT.parent / "datasets")

# Airplane TUSS model checkpoint (update with your trained checkpoint)
SEP_CHECKPOINT = str(
    PROJECT_ROOT / "src/models/tuss/checkpoints/YOUR_AIRPLANE_CHECKPOINT/best_model.pt"
)

# Dataset CSV from TUSS training (contains both birds and airplanes with orig_label preserved)
DATA_CSV = str(
    PROJECT_ROOT / "src/models/tuss/checkpoints/YOUR_AIRPLANE_CHECKPOINT/separation_dataset.csv"
)

# Classifier type: "plane", "ast_finetuned", or "pann_finetuned"
CLASSIFIER_TYPE = "plane"

# Evaluation settings
SPLIT = "test"
SNR_RANGE = (-5, 5)
SEED = 42
OUTPUT_DIR = "./airplane_validation_results"
SAVE_EXAMPLES_DIR = "./airplane_validation_examples"

# Exclude Risoux from synthetic mixtures (it's evaluated separately)
EXCLUDE_DATASETS = ["risoux_test"]

# ===================================================

def main():
    print("=" * 70)
    print("AIRPLANE CLASSIFICATION VALIDATION")
    print("=" * 70)
    print(f"Classifier: {CLASSIFIER_TYPE}")
    print(f"Separator: TUSS (airplane prompt)")
    print(f"Dataset: {DATA_CSV}")
    print(f"Excluding: {EXCLUDE_DATASETS}")
    print("=" * 70 + "\n")
    
    # Initialize pipeline
    pipeline = ValidationPipeline(base_path=BASE_PATH, device=None)
    
    # Load models
    # - classifier_type="plane" automatically sets coi_synonyms=AIRPLANE_SYNONYMS
    # - tuss_coi_prompt="airplane" is auto-detected from classifier_type
    pipeline.load_models(
        sep_checkpoint=SEP_CHECKPOINT,
        cls_weights=None,  # Uses default plane classifier weights
        classifier_type=CLASSIFIER_TYPE,
        use_tuss=True,
        tuss_coi_prompt="airplane",  # Or let it auto-detect from classifier_type
        tuss_bg_prompt="background",
        use_clapsep=False,
        use_ast_finetuned=False,  # Set to True if using as secondary classifier
        use_bird_mae=False,  # Bird classifier, not relevant for airplanes
        use_audioprotopnet=False,
    )
    
    print("\n" + "=" * 70)
    print("COI SYNONYM CONFIGURATION")
    print("=" * 70)
    print(f"Using COI synonyms: {pipeline.coi_synonyms}")
    print("\nLabel re-binarization:")
    print("  - Airplane samples (in orig_label) → label=1 (COI)")
    print("  - Bird samples (in orig_label) → label=0 (background)")
    print("  - Other samples → label=0 (background)")
    print("=" * 70 + "\n")
    
    # Run validation
    # The pipeline will:
    # 1. Load CSV and filter by split="test"
    # 2. Exclude datasets in EXCLUDE_DATASETS (risoux_test)
    # 3. Re-binarize labels using AIRPLANE_SYNONYMS
    # 4. Balance classes by downsampling majority class (default: balance_classes=True)
    # 5. Run 4 test stages: clean cls, clean sep+cls, mixture cls, mixture sep+cls
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
        balance_classes=True,  # Balance for fair confusion matrix visualization
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    for cls_name, cls_results in results.items():
        print(f"\nClassifier: {cls_name}")
        for test_name, metrics in cls_results.items():
            print(f"  {test_name}:")
            print(f"    F1: {metrics.f1_score:.4f}, Recall: {metrics.recall:.4f}, Precision: {metrics.precision:.4f}")
            print(f"    COI samples: {metrics.final_coi_count}, BG samples: {metrics.final_background_count}")
            if metrics.contaminated_backgrounds_removed > 0:
                print(f"    ⚠️  Removed {metrics.contaminated_backgrounds_removed} contaminated backgrounds")
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Examples saved to: {SAVE_EXAMPLES_DIR}")
    print("=" * 70)
    
    # Now run Risoux evaluation (independent test set, as-is)
    print("\n" + "=" * 70)
    print("RISOUX INDEPENDENT TEST SET (as-is evaluation)")
    print("=" * 70)
    risoux_results = pipeline.run(
        split="test",
        only_dataset="risoux_test",  # Evaluate Risoux separately
        snr_range=SNR_RANGE,
        data_csv=DATA_CSV,
        output_dir=OUTPUT_DIR,
        seed=SEED,
        save_examples_dir=SAVE_EXAMPLES_DIR,
        save_n_examples=5,
        skip_clean_tests=True,  # Automatically forced for independent datasets
        save_false_negatives=True,
    )
    
    print("\nRisoux Results:")
    for cls_name, cls_results in risoux_results.items():
        for test_name, metrics in cls_results.items():
            print(f"  {test_name}: F1={metrics.f1_score:.4f}")
    
    print("\n" + "=" * 70)
    print("ALL AIRPLANE VALIDATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
