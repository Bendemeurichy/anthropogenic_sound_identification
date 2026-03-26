"""
Validation module for separation + classification pipeline.
Computes confusion matrix, precision, recall, F1-score, and other metrics.
Includes signal-level separation quality metrics (SI-SNR, SDR).

Also computes simple absolute-energy metrics before/after separation:
- MSR: mean-square-root (i.e. RMS) of the waveform
- SEL: Sound Exposure Level proxy computed from total energy over the segment
       (reported as 10*log10(sum(x^2) + eps)). Note: this is a relative SEL-like
       value unless you calibrate to physical units (Pa).
"""

import io
import sys

# Under pythonw there is no console and sys.stdout/stderr are None.
# Wrap only when the underlying buffer actually exists.
if sys.stdout is not None and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", line_buffering=True
    )
if sys.stderr is not None and hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", line_buffering=True
    )

import importlib
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from torchmetrics.audio import (
    ScaleInvariantSignalDistortionRatio,
    ScaleInvariantSignalNoiseRatio,
    SignalDistortionRatio,
)
from tqdm import tqdm

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(
    0, str(Path(__file__).parent / "classification_models" / "plane_clasifier")
)

from src.label_loading.coi_labels import (
    COI_SYNONYMS,
    _extract_label_atoms as _extract_label_atoms,
    is_coi_label as _is_coi_label,
    normalize_label as _norm_label,
)
from src.models.clapsep.inference import COI_HEAD_INDEX as CLAPSEP_COI_HEAD
from src.models.clapsep.inference import CLAPSepInference
from src.models.sudormrf.inference import COI_HEAD_INDEX as SUDORMRF_COI_HEAD
from src.models.sudormrf.inference import SeparationInference
from src.models.tuss.inference import COI_HEAD_INDEX as TUSS_COI_HEAD
from src.models.tuss.inference import TUSSInference
from src.validation_functions.classification_models.plane_clasifier.config import (
    TrainingConfig,
)
from src.validation_functions.classification_models.plane_clasifier.inference import (
    PlaneClassifierInference,
)


# ---------------------------------------------------------------------------
# Optional auxiliary classifiers — probed lazily; actual imports happen inside
# load_models / _classify_pann / _classify_ast to avoid hard top-level deps.
# ---------------------------------------------------------------------------
def _probe_pann() -> bool:
    """Return True if the panns_inference package is importable."""
    try:
        import panns_inference  # noqa: F401

        return True
    except Exception as _err:
        print(f"[Warning] PANN classifier unavailable: {_err}", file=sys.stderr)
        return False


def _probe_ast() -> bool:
    """Return True if the transformers package (for AST) is importable."""
    try:
        import transformers  # noqa: F401

        return True
    except Exception as _err:
        print(f"[Warning] AST classifier unavailable: {_err}", file=sys.stderr)
        return False


# Temporarily turn of pann and ast for testing
# _pann_available: bool = _probe_pann()
# _ast_available: bool = _probe_ast()
_pann_available: bool = False
_ast_available: bool = False


try:
    import src.models.base.sudo_rm_rf
except Exception as e:
    try:
        real_mod = importlib.import_module("models.sudormrf.base.sudo_rm_rf")
        sys.modules["src.models.base.sudo_rm_rf"] = real_mod
        sys.modules["sudo_rm_rf"] = real_mod
    except Exception as e2:
        # best-effort only; if mapping fails, fallback to normal error
        import traceback

        print(f"[Warning] Failed to map sudormrf modules: {e2}", file=sys.stderr)
        traceback.print_exc()

# Project root (code directory)
PROJECT_ROOT = Path(__file__).parent.parent.parent


# ============================================================================
# CLASSIFIER LABEL CONFIGURATION
# ============================================================================
# Edit these lists to control which AudioSet / model labels are treated as the
# positive class (i.e. "plane present") for each classifier.

# The string that PlaneClassifierInference returns for a positive (plane) sample.
CNN_POSITIVE_CLASS: str = "plane"

# ---------------------------------------------------------------------------
# COI (class-of-interest) label normalization
# ---------------------------------------------------------------------------
# COI_SYNONYMS, normalize_label, and is_coi_label are now imported from
# src.label_loading.coi_labels to ensure consistency between dataset creation
# (sampler.py) and evaluation (test_pipeline.py).
#
# See src/label_loading/coi_labels.py for the canonical definitions.


def _filter_contaminated_backgrounds(
    df_bg: pd.DataFrame, verbose: bool = True
) -> Tuple[pd.DataFrame, int]:
    """Filter out background samples that have COI synonyms in orig_label.

    This prevents contamination from samples that were incorrectly included
    in the background pool during dataset creation due to incomplete synonym
    matching in the sampler.

    Args:
        df_bg: DataFrame of background samples (label=0)
        verbose: If True, print detailed filtering report

    Returns:
        Tuple of (filtered DataFrame, number of contaminated samples removed)
    """
    if "orig_label" not in df_bg.columns:
        if verbose:
            print("[Info] No 'orig_label' column found - skipping contamination filter")
        return df_bg, 0

    # Check each background sample for COI synonyms in orig_label
    contaminated_mask = df_bg["orig_label"].apply(_is_coi_label)

    n_contaminated = int(contaminated_mask.sum())
    if n_contaminated == 0:
        if verbose:
            print("✓ No contaminated background samples detected")
        return df_bg, 0

    # Report contamination
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"⚠️  CONTAMINATION DETECTED: {n_contaminated} background samples")
        print(f"   contain COI synonyms in orig_label and will be EXCLUDED")
        print(f"{'=' * 60}")

        # Breakdown by split
        for split in ["train", "val", "test"]:
            split_mask = df_bg["split"] == split
            split_contam = int((split_mask & contaminated_mask).sum())
            split_total = int(split_mask.sum())
            if split_total > 0:
                pct = 100 * split_contam / split_total
                print(
                    f"  {split:5s}: {split_contam:3d}/{split_total:4d} ({pct:4.1f}%) contaminated"
                )

        # Show some examples
        print(f"\n  Example contaminated orig_labels:")
        contaminated_labels = df_bg[contaminated_mask]["orig_label"].unique()[:5]
        for lbl in contaminated_labels:
            lbl_str = str(lbl)[:80]
            print(f"    - {lbl_str}")
        print(f"{'=' * 60}\n")

    # Return filtered dataframe
    return df_bg[~contaminated_mask].reset_index(drop=True), n_contaminated


# AudioSet label names that PANN should treat as the positive class.
# Unrecognised names are silently ignored at inference time.
PANN_POSITIVE_LABELS: List[str] = [
    "Fixed-wing aircraft, airplane",
    "Aircraft",
    "Aircraft engine",
    "Jet engine",
    "Propeller, airscrew",
]

# AudioSet label names that AST should treat as the positive class.
# Unrecognised names are silently ignored at inference time.
AST_POSITIVE_LABELS: List[str] = [
    "Fixed-wing aircraft, airplane",
    "Aircraft",
    "Aircraft engine",
    "Jet engine",
    "Propeller, airscrew",
]


# ============================================================================
# METRICS
# ============================================================================


@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""

    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    specificity: float = 0.0
    balanced_accuracy: float = 0.0
    matthews_corrcoef: float = 0.0

    n_samples: int = 0
    labels: List[int] = field(default_factory=list)
    predictions: List[int] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    # Keep track of how many times each *transition* (true → predicted) occurred.
    # this is a dictionary mapping strings like "0->1" or "3->7" to counts.
    misclassified_transitions: Dict[str, int] = field(default_factory=dict)
    # count how often each true label ends up being mis‑classified, regardless of
    # the incorrect target; useful for seeing which classes the model struggles
    # with overall.
    misclassified_per_label: Dict[int, int] = field(default_factory=dict)
    # Per-raw-label false positive counts: ground truth = background (0), predicted
    # positive (1).  Keyed by the original string label from the CSV.
    fp_raw_counts: Dict[str, int] = field(default_factory=dict)
    # Per-raw-label false negative counts: ground truth = COI (1), predicted
    # negative (0).  Keyed by the original string label from the CSV.
    fn_raw_counts: Dict[str, int] = field(default_factory=dict)
    # Total misclassification count keyed by the original (multi-class) raw label.
    misclassified_raw_counts: Dict[str, int] = field(default_factory=dict)
    # Misclassification counts keyed by atomic raw labels split out from any
    # multi-label background/sample annotations.
    misclassified_raw_atomic_counts: Dict[str, int] = field(default_factory=dict)
    # Atomic false positive counts, split from any multi-label raw annotations.
    fp_raw_atomic_counts: Dict[str, int] = field(default_factory=dict)
    # Atomic false negative counts, split from any multi-label raw annotations.
    fn_raw_atomic_counts: Dict[str, int] = field(default_factory=dict)

    # Signal-level separation quality metrics (populated when reference is available)
    si_snr_scores: List[float] = field(default_factory=list)
    sdr_scores: List[float] = field(default_factory=list)
    si_sdr_scores: List[float] = field(default_factory=list)
    mean_si_snr: Optional[float] = None
    mean_sdr: Optional[float] = None
    mean_si_sdr: Optional[float] = None

    # Actual SNR values achieved after clamping (for mixture experiments)
    actual_snrs: List[float] = field(default_factory=list)

    # Dataset filtering statistics
    contaminated_backgrounds_removed: int = 0
    final_background_count: int = 0
    final_coi_count: int = 0

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray = None,
        raw_labels: Sequence[Any] = None,
    ):
        """Compute all metrics from predictions.

        ``raw_labels`` is an optional parallel sequence containing the original
        (multi‑class) label for each sample; it is used only for bookkeeping and
        does not affect the numeric metrics.
        """
        self.n_samples = len(y_true)
        self.labels = y_true.tolist()
        self.predictions = y_pred.tolist()
        if raw_labels is not None:
            self.raw_labels = list(raw_labels)
        # reset misclassified info in case the same object is reused
        self.misclassified_transitions = {}
        self.misclassified_per_label = {}
        self.misclassified_raw_counts = {}
        self.misclassified_raw_atomic_counts = {}
        self.fp_raw_counts = {}
        self.fp_raw_atomic_counts = {}
        self.fn_raw_counts = {}
        self.fn_raw_atomic_counts = {}
        if y_scores is not None:
            self.confidences = y_scores.tolist()

        self.true_positives = int(np.sum((y_true == 1) & (y_pred == 1)))
        self.true_negatives = int(np.sum((y_true == 0) & (y_pred == 0)))
        self.false_positives = int(np.sum((y_true == 0) & (y_pred == 1)))
        self.false_negatives = int(np.sum((y_true == 1) & (y_pred == 0)))

        tp, tn, fp, fn = (
            self.true_positives,
            self.true_negatives,
            self.false_positives,
            self.false_negatives,
        )

        self.accuracy = (tp + tn) / max(self.n_samples, 1)
        self.precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        self.recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        self.specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        self.f1_score = (
            2 * self.precision * self.recall / (self.precision + self.recall)
            if (self.precision + self.recall) > 0
            else 0.0
        )
        self.balanced_accuracy = (self.recall + self.specificity) / 2

        mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        self.matthews_corrcoef = (
            (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0.0
        )

        # count up misclassification transitions (and per‑label/raw summaries)
        for idx, (true_val, pred_val) in enumerate(zip(y_true, y_pred)):
            if true_val != pred_val:
                key = f"{true_val}->{pred_val}"
                self.misclassified_transitions[key] = (
                    self.misclassified_transitions.get(key, 0) + 1
                )
                self.misclassified_per_label[true_val] = (
                    self.misclassified_per_label.get(true_val, 0) + 1
                )
                if raw_labels is not None:
                    raw_key = str(raw_labels[idx])
                    self.misclassified_raw_counts[raw_key] = (
                        self.misclassified_raw_counts.get(raw_key, 0) + 1
                    )
                    atomic_labels = [
                        _norm_label(x) for x in _extract_label_atoms(raw_labels[idx])
                    ]
                    for atomic in atomic_labels:
                        self.misclassified_raw_atomic_counts[atomic] = (
                            self.misclassified_raw_atomic_counts.get(atomic, 0) + 1
                        )
                    # Split into FP (bg predicted as COI) and FN (COI missed).
                    if true_val == 0 and pred_val == 1:
                        self.fp_raw_counts[raw_key] = (
                            self.fp_raw_counts.get(raw_key, 0) + 1
                        )
                        for atomic in atomic_labels:
                            self.fp_raw_atomic_counts[atomic] = (
                                self.fp_raw_atomic_counts.get(atomic, 0) + 1
                            )
                    elif true_val == 1 and pred_val == 0:
                        self.fn_raw_counts[raw_key] = (
                            self.fn_raw_counts.get(raw_key, 0) + 1
                        )
                        for atomic in atomic_labels:
                            self.fn_raw_atomic_counts[atomic] = (
                                self.fn_raw_atomic_counts.get(atomic, 0) + 1
                            )

        # Aggregate signal metrics if populated
        if self.si_snr_scores:
            self.mean_si_snr = float(np.mean(self.si_snr_scores))
        if self.sdr_scores:
            self.mean_sdr = float(np.mean(self.sdr_scores))
        if self.si_sdr_scores:
            self.mean_si_sdr = float(np.mean(self.si_sdr_scores))

    def __str__(self):
        s = f"""
{"=" * 50}
Confusion Matrix:  TN={self.true_negatives}  FP={self.false_positives}
                   FN={self.false_negatives}  TP={self.true_positives}

Accuracy:  {self.accuracy:.4f}    Precision: {self.precision:.4f}
Recall:    {self.recall:.4f}    F1-Score:  {self.f1_score:.4f}
Specificity: {self.specificity:.4f}  Balanced Acc: {self.balanced_accuracy:.4f}
MCC: {self.matthews_corrcoef:.4f}"""

        # provide a quick summary of which class-to-class transitions were mis-predicted
        s += (
            f"\n\nMisclassified transition counts:\n"
            f"  Actual 0 → Pred 1: {self.misclassified_transitions.get('0->1', 0)}\n"
            f"  Actual 1 → Pred 0: {self.misclassified_transitions.get('1->0', 0)}"
        )
        if self.misclassified_per_label:
            s += "\n\nMisclassified by binary true label:\n"
            for cls, cnt in sorted(self.misclassified_per_label.items()):
                s += f"  Class {cls}: {cnt}\n"
        if self.fp_raw_counts or self.fn_raw_counts:
            s += "\nMisclassified by original label:\n"
            all_raw_keys = sorted(
                set(self.fp_raw_counts) | set(self.fn_raw_counts), key=str
            )
            for raw in all_raw_keys:
                fp = self.fp_raw_counts.get(raw, 0)
                fn = self.fn_raw_counts.get(raw, 0)
                s += f"  {raw}: FP={fp}  FN={fn}\n"

        if self.fp_raw_atomic_counts or self.fn_raw_atomic_counts:
            s += "\nMisclassified by atomic raw label:\n"
            all_atomic_keys = sorted(
                set(self.fp_raw_atomic_counts) | set(self.fn_raw_atomic_counts), key=str
            )
            for raw in all_atomic_keys:
                fp = self.fp_raw_atomic_counts.get(raw, 0)
                fn = self.fn_raw_atomic_counts.get(raw, 0)
                s += f"  {raw}: FP={fp}  FN={fn}\n"

        if self.mean_si_snr is not None:
            sdr_str = f"{self.mean_sdr:+.2f} dB" if self.mean_sdr is not None else "n/a"
            si_sdr_str = (
                f"{self.mean_si_sdr:+.2f} dB" if self.mean_si_sdr is not None else "n/a"
            )
            s += f"""

Signal-Level Metrics (COI samples, n={len(self.si_snr_scores)}):
  SI-SNR: {self.mean_si_snr:+.2f} dB    SDR: {sdr_str}    SI-SDR: {si_sdr_str}"""

        if self.actual_snrs:
            s += f"""
  Actual SNR range: [{min(self.actual_snrs):.1f}, {max(self.actual_snrs):.1f}] dB  (mean: {np.mean(self.actual_snrs):.1f} dB)"""

        s += f"\n{'=' * 50}"
        return s

    def to_dict(self):
        """Return a JSON-serializable dictionary representation of the metrics.

        This ensures any numpy / torch scalar/array types are converted to native
        Python `int`/`float`/`list` types so json.dump won't fail or produce
        non-serializable objects.
        """

        def _sanitize(obj):
            """Recursively convert numpy/torch types to native Python types."""
            # Local imports so module-level imports remain unchanged and to keep
            # this function self-contained.
            import numpy as _np

            # torch may not be available in all contexts where this method is used
            _torch = None
            try:
                import torch as _t

                _torch = _t
            except Exception:
                _torch = None

            if isinstance(obj, dict):
                return {str(k): _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_sanitize(v) for v in obj]
            if _torch is not None and isinstance(obj, _torch.Tensor):
                # Convert tensors to numpy first, then sanitize
                return _sanitize(obj.detach().cpu().numpy())
            if isinstance(obj, _np.ndarray):
                return _sanitize(obj.tolist())
            # numpy scalar types
            if isinstance(obj, _np.integer):
                return int(obj)
            if isinstance(obj, _np.floating):
                return float(obj)
            # Fallback: plain Python ints/floats/strs remain unchanged
            return obj

        d = {
            "confusion_matrix": {
                "tp": int(self.true_positives),
                "tn": int(self.true_negatives),
                "fp": int(self.false_positives),
                "fn": int(self.false_negatives),
            },
            "accuracy": float(self.accuracy),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1_score": float(self.f1_score),
            "specificity": float(self.specificity),
            "balanced_accuracy": float(self.balanced_accuracy),
            "mcc": float(self.matthews_corrcoef),
        }
        if self.mean_si_snr is not None:
            d["signal_metrics"] = {
                "mean_si_snr_db": float(self.mean_si_snr),
                "mean_sdr_db": float(self.mean_sdr),
                "mean_si_sdr_db": float(self.mean_si_sdr),
                "n_signal_samples": int(len(self.si_snr_scores)),
            }
        if self.actual_snrs:
            d["actual_snr_stats"] = {
                "min": float(min(self.actual_snrs)),
                "max": float(max(self.actual_snrs)),
                "mean": float(np.mean(self.actual_snrs)),
            }
        if self.misclassified_transitions:
            # Ensure counts are native ints
            d["misclassified_transitions"] = {
                str(k): int(v) for k, v in self.misclassified_transitions.items()
            }
        if self.misclassified_per_label:
            d["misclassified_per_label"] = {
                str(k): int(v) for k, v in self.misclassified_per_label.items()
            }
        if self.misclassified_raw_counts:
            d["misclassified_raw_counts"] = {
                str(k): int(v) for k, v in self.misclassified_raw_counts.items()
            }
        if self.misclassified_raw_atomic_counts:
            d["misclassified_raw_atomic_counts"] = {
                str(k): int(v) for k, v in self.misclassified_raw_atomic_counts.items()
            }
        if self.fp_raw_counts:
            d["fp_raw_counts"] = {str(k): int(v) for k, v in self.fp_raw_counts.items()}
        if self.fp_raw_atomic_counts:
            d["fp_raw_atomic_counts"] = {
                str(k): int(v) for k, v in self.fp_raw_atomic_counts.items()
            }
        if self.fn_raw_counts:
            d["fn_raw_counts"] = {str(k): int(v) for k, v in self.fn_raw_counts.items()}
        if self.fn_raw_atomic_counts:
            d["fn_raw_atomic_counts"] = {
                str(k): int(v) for k, v in self.fn_raw_atomic_counts.items()
            }
        if getattr(self, "raw_labels", None):
            # preserve the sequence of original labels for downstream inspection
            d["raw_labels"] = [(_sanitize(x)) for x in list(self.raw_labels)]
        # Add dataset filtering stats if any contamination was detected
        if self.contaminated_backgrounds_removed > 0 or self.final_background_count > 0:
            d["dataset_filtering"] = {
                "contaminated_backgrounds_removed": int(
                    self.contaminated_backgrounds_removed
                ),
                "final_background_count": int(self.final_background_count),
                "final_coi_count": int(self.final_coi_count),
            }
        # Sanitize any remaining numpy/torch types recursively before returning
        return _sanitize(d)


# ============================================================================
# VALIDATION PIPELINE
# ============================================================================


class ValidationPipeline:
    """Pipeline for validating separation + classification."""

    # Default paths (relative to project root)
    SEP_CHECKPOINT = PROJECT_ROOT / "src/models/sudormrf/checkpoints/best_model.pt"
    CLS_WEIGHTS = (
        PROJECT_ROOT
        / "src/validation_functions/classification_models/plane_clasifier/results/checkpoints/final_model.weights.h5"
    )
    DATA_CSV = PROJECT_ROOT / "src/models/sudormrf/checkpoints/separation_dataset.csv"

    def __init__(self, base_path: str = None):
        """
        Args:
            base_path: Base path for audio files (to convert Windows paths in CSV)
        """
        self.base_path = base_path
        self.sample_rate = 16000
        self.segment_length = 5.0
        self.classifier_sample_rate = self.sample_rate
        self.classifier_segment_samples = int(
            self.classifier_sample_rate * self.segment_length
        )
        self.segment_samples = int(self.sample_rate * self.segment_length)
        # Prefer cuda:1 when multiple GPUs are present; fall back to cuda:0 on
        # a single-GPU machine; use CPU if no CUDA is available.
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.device = "cuda:1"
        elif torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"
            print(f"Cuda available: {torch.cuda.is_available()}")
            print(f"Gpus available: {torch.cuda.device_count()}")
        # report the choice so users can immediately see which hardware will be used
        print(f"ValidationPipeline using device: {self.device}")
        # will be populated when a separation checkpoint is loaded
        self.target_classes = None

        self.separator = None
        self.classifier = None
        self._resamplers = {}

        # Optional auxiliary AudioSet classifiers
        self.pann_model = None
        self.ast_extractor = None
        self.ast_model = None

        # Store checkpoint paths for logging
        self.sep_checkpoint_path = None
        self.cls_checkpoint_path = None

        # Signal-level metric functions (torchmetrics)
        self._si_snr = ScaleInvariantSignalNoiseRatio()
        self._sdr = SignalDistortionRatio()
        self._si_sdr = ScaleInvariantSignalDistortionRatio()

    def load_models(
        self,
        sep_checkpoint: str = None,
        cls_weights: str = None,
        use_clapsep: bool = False,
        use_tuss: bool = False,
        tuss_coi_prompt: str = "airplane",
        tuss_bg_prompt: str = "background",
        clapsep_text_pos: str = "train passing",
        clapsep_text_neg: str = "",
        use_pann: bool = True,
        use_ast: bool = True,
    ):
        """Load separation and classification models.

        Args:
            sep_checkpoint: Path to the separation model checkpoint.
            cls_weights: Path to the CNN classifier weights.
            use_clapsep: If True, load the CLAPSep separator instead of SudoRM-RF.
            use_tuss: If True, load the TUSS separator instead of SudoRM-RF.
            tuss_coi_prompt: Prompt name for TUSS Class of Interest (default: "airplane").
            tuss_bg_prompt: Prompt name for TUSS background (default: "background").
            clapsep_text_pos: Positive text prompt for CLAPSep.
            clapsep_text_neg: Negative text prompt for CLAPSep.
            use_pann: If True (default), load the PANN AudioTagging model as an
                additional classifier.  Requires the ``panns_inference`` package.
            use_ast: If True (default), load the AST model from HuggingFace as an
                additional classifier.  Requires the ``transformers`` package.
        """
        sep_path = sep_checkpoint or self.SEP_CHECKPOINT
        cls_path = cls_weights or self.CLS_WEIGHTS

        # Store checkpoint paths
        self.sep_checkpoint_path = sep_path
        self.cls_checkpoint_path = cls_path

        if use_tuss:
            print(f"Loading TUSS model from {sep_path}")
            print(
                f"  COI prompt: '{tuss_coi_prompt}', Background prompt: '{tuss_bg_prompt}'"
            )
            self.separator = TUSSInference.from_checkpoint(
                sep_path,
                device=self.device,
                coi_prompt=tuss_coi_prompt,
                bg_prompt=tuss_bg_prompt,
            )
            # Propagate sample_rate and segment_samples from TUSS model
            self.sample_rate = self.separator.sample_rate
            self.segment_samples = self.separator.segment_samples
            self.segment_length = self.segment_samples / self.sample_rate
            print(
                f"TUSS config: sample_rate={self.sample_rate} Hz, "
                f"segment_length={self.segment_length:.2f} s "
                f"({self.segment_samples} samples)"
            )
            # Re-derive classifier segment size using the (now updated) segment_length
            # so the classifier window stays consistent with the separator window.
            self.classifier_segment_samples = int(
                self.classifier_sample_rate * self.segment_length
            )
        elif use_clapsep:
            ckpt_label = sep_path if sep_checkpoint else "default CLAPSep checkpoint"
            print(
                f"Loading CLAPSep model from {ckpt_label} "
                f"(text_pos='{clapsep_text_pos}', text_neg='{clapsep_text_neg}')"
            )
            self.separator = CLAPSepInference.from_pretrained(
                model_ckpt_path=sep_path if sep_checkpoint else None,
                device=self.device,
                text_pos=clapsep_text_pos,
                text_neg=clapsep_text_neg,
            )
            self.sample_rate = self.separator.sample_rate
            self.segment_samples = int(self.sample_rate * self.segment_length)
        else:
            print(f"Loading separation model from {sep_path}")
            self.separator = SeparationInference.from_checkpoint(
                sep_path, device=self.device
            )
            # Propagate sample_rate and segment_length from the checkpoint config,
            # mirroring what is done for CLAPSep above. Without this the pipeline
            # would keep its hardcoded defaults (16 kHz / 5 s) while the model
            # was trained at a different rate / window length (e.g. 32 kHz / 4 s).
            self.sample_rate = self.separator.sample_rate
            self.segment_samples = self.separator.segment_samples
            self.segment_length = self.segment_samples / self.sample_rate
            print(
                f"Separator config: sample_rate={self.sample_rate} Hz, "
                f"segment_length={self.segment_length:.2f} s "
                f"({self.segment_samples} samples)"
            )
            # Re-derive classifier segment size using the (now updated) segment_length
            # so the classifier window stays consistent with the separator window.
            self.classifier_segment_samples = int(
                self.classifier_sample_rate * self.segment_length
            )

        # recover and log the target class list if the checkpoint included a
        # config (it should, since the training routine saves the YAML)
        if hasattr(self.separator, "config") and self.separator.config:
            # Try multiple paths depending on model type
            tc = None
            if hasattr(self.separator.config, "data"):
                # SudoRM-RF / CLAPSep format
                tc = getattr(self.separator.config.data, "target_classes", None)
            elif isinstance(self.separator.config, dict):
                # TUSS format (config is a dict from hparams.yaml)
                data_section = self.separator.config.get("data", {})
                tc = data_section.get("target_classes", None)

            if tc:
                self.target_classes = tc
                print(f"Target classes from separation checkpoint: {tc}")

        # (The target_classes list has already been recovered and printed above;
        # no need to repeat it.)

        print(f"Loading classification model from {cls_path}")
        config = TrainingConfig(
            sample_rate=self.classifier_sample_rate, audio_duration=self.segment_length
        )
        self.classifier = PlaneClassifierInference(cls_path, config)
        self.classifier_segment_samples = int(
            self.classifier_sample_rate * self.segment_length
        )

        # ------------------------------------------------------------------
        # Optional: PANN AudioTagging classifier
        # ------------------------------------------------------------------
        if use_pann:
            if _pann_available:
                try:
                    from src.validation_functions.classification_models.pann_inference import (
                        load_pann_model as _load_pann,
                    )

                    self.pann_model = _load_pann(device=self.device)
                except Exception as e:
                    print(
                        f"[Warning] Failed to load PANN model: {e}",
                        file=sys.stderr,
                    )
                    self.pann_model = None
            else:
                print(
                    "[Warning] PANN classifier requested but panns_inference is not installed – skipping.",
                    file=sys.stderr,
                )

        # ------------------------------------------------------------------
        # Optional: AST (Audio Spectrogram Transformer) classifier
        # ------------------------------------------------------------------
        if use_ast:
            if _ast_available:
                try:
                    from src.validation_functions.classification_models.ast_inference import (
                        load_ast_model as _load_ast,
                    )

                    self.ast_extractor, self.ast_model = _load_ast(device=self.device)
                except Exception as e:
                    print(
                        f"[Warning] Failed to load AST model: {e}",
                        file=sys.stderr,
                    )
                    self.ast_extractor = None
                    self.ast_model = None
            else:
                print(
                    "[Warning] AST classifier requested but transformers is not installed – skipping.",
                    file=sys.stderr,
                )

    def _convert_path(self, filepath: str) -> str:
        """Convert Windows paths to Linux paths."""
        if filepath.startswith("D:\\") or filepath.startswith("C:\\"):
            filepath = filepath.replace("\\", "/")
            filepath = "/" + filepath[3:]

        if self.base_path:
            for marker in ["/datasets/", "/masterproef/datasets/"]:
                if marker in filepath:
                    rel_path = filepath.split(marker)[-1]
                    return os.path.join(self.base_path, rel_path)
        return filepath

    def _load_audio(self, filepath: str) -> torch.Tensor:
        """Load audio using torchaudio, resampled and truncated/padded to segment_samples."""
        filepath = self._convert_path(filepath)
        waveform, sr = torchaudio.load(filepath)

        if sr != self.sample_rate:
            key = (sr, self.sample_rate)
            if key not in self._resamplers:
                self._resamplers[key] = torchaudio.transforms.Resample(
                    sr, self.sample_rate
                )
            waveform = self._resamplers[key](waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        if waveform.shape[0] < self.segment_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.segment_samples - waveform.shape[0])
            )
        else:
            waveform = waveform[: self.segment_samples]

        return waveform

    def _load_labeled_audio(
        self,
        filepath: str,
        start_time=None,
        end_time=None,
    ) -> torch.Tensor:
        """Load the labeled portion of a recording as a variable-length tensor.

        Unlike _load_audio this method does NOT truncate to segment_samples — it
        returns the full labeled window so callers can split it into however many
        model-sized segments are needed.

        start_time / end_time semantics:
          - None / NaN / "unknown" / "null" → treat as absent (use whole file).
          - A value that exceeds the file duration is assumed to be an external
            timestamp (e.g. a YouTube offset stored in AudioSet TSVs) and is
            silently ignored so the whole file is used instead.
        """
        filepath = self._convert_path(filepath)

        _INVALID = {"", "nan", "none", "unknown", "null"}

        t_start = 0.0
        try:
            if (
                start_time is not None
                and str(start_time).strip().lower() not in _INVALID
            ):
                t_start = max(0.0, float(start_time))
        except (ValueError, TypeError):
            t_start = 0.0

        t_end: Optional[float] = None
        try:
            if end_time is not None and str(end_time).strip().lower() not in _INVALID:
                t_end = float(end_time)
        except (ValueError, TypeError):
            t_end = None

        waveform, sr = torchaudio.load(filepath)

        if sr != self.sample_rate:
            key = (sr, self.sample_rate)
            if key not in self._resamplers:
                self._resamplers[key] = torchaudio.transforms.Resample(
                    sr, self.sample_rate
                )
            waveform = self._resamplers[key](waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        total_samples = waveform.shape[0]

        sample_start = int(t_start * self.sample_rate)

        # If start_time is beyond end of file it is an external timestamp — ignore.
        if sample_start >= total_samples:
            sample_start = 0
            t_end = None

        sample_end = (
            min(int(t_end * self.sample_rate), total_samples)
            if t_end is not None
            else total_samples
        )

        # Guard degenerate windows
        if sample_end <= sample_start:
            sample_start = 0
            sample_end = total_samples

        return waveform[sample_start:sample_end]

    def _load_full_audio(self, filepath: str) -> torch.Tensor:
        """Load the entire audio file as a variable-length tensor at self.sample_rate.

        Unlike _load_labeled_audio, this method ignores start_time/end_time entirely
        and always returns the complete file.  Use this for recording-level evaluation
        where the ground-truth label applies to the whole recording and the classifier
        should be given all segments to find the event of interest in any of them.
        """
        filepath = self._convert_path(filepath)
        waveform, sr = torchaudio.load(filepath)

        if sr != self.sample_rate:
            key = (sr, self.sample_rate)
            if key not in self._resamplers:
                self._resamplers[key] = torchaudio.transforms.Resample(
                    sr, self.sample_rate
                )
            waveform = self._resamplers[key](waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        return waveform

    def _split_into_segments(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """Split a variable-length waveform into non-overlapping segment_samples chunks.

        Each chunk is exactly segment_samples long; the final chunk is zero-padded
        when necessary.  Always returns at least one segment.
        """
        n = waveform.shape[0]
        if n == 0:
            return [torch.zeros(self.segment_samples)]

        segments = []
        for start in range(0, max(n, 1), self.segment_samples):
            chunk = waveform[start : start + self.segment_samples]
            if chunk.shape[0] < self.segment_samples:
                chunk = torch.nn.functional.pad(
                    chunk, (0, self.segment_samples - chunk.shape[0])
                )
            segments.append(chunk)
        return segments

    def _separate(self, waveform: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """Run separation model. Returns all sources.

        All separator types (TUSS, CLAPSep, SudoRM-RF) implement separate_waveform()
        which handles normalization, windowing, device placement, and model-specific logic.
        """
        if debug:
            input_rms = torch.sqrt(torch.mean(waveform**2)).item()
            print(
                f"  [DEBUG _separate] Input: shape={waveform.shape}, device={waveform.device}, RMS={input_rms:.6f}",
                file=sys.stderr,
            )

        result = self.separator.separate_waveform(waveform)

        if debug:
            output_rms = torch.sqrt(torch.mean(result**2)).item()
            print(
                f"  [DEBUG _separate] Output: shape={result.shape}, device={result.device}, RMS={output_rms:.6f}",
                file=sys.stderr,
            )
            if output_rms < 1e-6:
                print(
                    f"  [WARNING _separate] Output is SILENT!",
                    file=sys.stderr,
                )

        return result

    def _classify(self, waveform: torch.Tensor) -> Tuple[int, float]:
        """Run classification. Returns (prediction, confidence)."""
        wav = waveform.detach().cpu()

        if self.classifier_sample_rate != self.sample_rate:
            key = (self.sample_rate, self.classifier_sample_rate)
            if key not in self._resamplers:
                self._resamplers[key] = torchaudio.transforms.Resample(
                    self.sample_rate, self.classifier_sample_rate
                )
            wav = self._resamplers[key](wav.unsqueeze(0)).squeeze(0)

        if wav.shape[0] < self.classifier_segment_samples:
            wav = torch.nn.functional.pad(
                wav, (0, self.classifier_segment_samples - wav.shape[0])
            )
        else:
            wav = wav[: self.classifier_segment_samples]

        result = self.classifier.predict_waveform(wav.numpy())
        pred = 1 if result["prediction"] == CNN_POSITIVE_CLASS else 0
        return pred, result["confidence"]

    def _create_mixture(
        self, source: torch.Tensor, noise: torch.Tensor, snr_db: float,
        disable_clamping: bool = False
    ) -> Tuple[torch.Tensor, float]:
        """Mix source and noise at given SNR.

        SNR (dB) = 10 * log10(source_power / noise_power)
        Scales noise to achieve target SNR with optional clamping to prevent extreme levels.

        Args:
            source: Source signal (COI)
            noise: Noise/background signal
            snr_db: Target SNR in decibels
            disable_clamping: If True, allows extreme noise levels for robustness testing.
                             If False (default), clamps scale to [0.1, 3.0] for consistency
                             with separation model training.

        Returns:
            Tuple of (mixture, actual_snr_db) where actual_snr_db reflects the
            SNR after clamping the noise scale factor (if enabled).
        """
        eps = 1e-8
        source_power = torch.mean(source**2) + eps
        noise_power = torch.mean(noise**2) + eps
        scale = torch.sqrt(source_power / (10 ** (snr_db / 10) * noise_power))
        scale_unclamped = scale.item()
        
        # Optionally clamp scaling to prevent extreme noise (consistent with training)
        if not disable_clamping:
            scale = torch.clamp(scale, min=0.1, max=3.0)

        # Compute actual SNR achieved after clamping
        scaled_noise_power = torch.mean((noise * scale) ** 2) + eps
        actual_snr_db = 10 * torch.log10(source_power / scaled_noise_power).item()

        if abs(scale.item() - scale_unclamped) > 1e-6:
            clamp_status = "DISABLED" if disable_clamping else "ACTIVE"
            print(
                f"  [SNR {clamp_status}] requested={snr_db:.1f}dB -> "
                f"actual={actual_snr_db:.1f}dB (scale {scale_unclamped:.3f} -> {scale.item():.3f})"
            )

        return source + noise * scale, actual_snr_db

    def _create_mixture_rms(
        self, source: torch.Tensor, noise: torch.Tensor, snr_db: float
    ) -> Tuple[torch.Tensor, float]:
        """Mix source and noise at given SNR using RMS-based normalization.
        
        This is the CORRECT way to create SNR mixtures for classifier evaluation:
        1. Normalize both signal and noise to same RMS level
        2. Scale noise to achieve target SNR
        3. Uniformly scale mixture to fit within [-1, 1] (preserves SNR)
        4. Optional safety clipping (should rarely trigger)
        
        This approach:
        - Preserves SNR relationship accurately
        - Keeps signals within valid audio range [-1, 1]
        - No harmonic distortion from clipping (at reasonable SNRs)
        - Matches realistic audio mixing scenarios
        
        Args:
            source: Source signal (COI)
            noise: Noise/background signal
            snr_db: Target SNR in decibels
            
        Returns:
            Tuple of (mixture, actual_snr_db)
        """
        eps = 1e-8
        
        # Step 1: Normalize both to same RMS level (0.1 gives headroom)
        target_rms = 0.1
        
        source_rms = torch.sqrt(torch.mean(source**2)) + eps
        noise_rms = torch.sqrt(torch.mean(noise**2)) + eps
        
        source_normalized = source * (target_rms / source_rms)
        noise_normalized = noise * (target_rms / noise_rms)
        
        # Step 2: Scale noise to achieve target SNR
        # SNR (dB) = 10 * log10(signal_power / noise_power)
        # => noise_power_target = signal_power / 10^(SNR/10)
        signal_power = torch.mean(source_normalized**2) + eps
        target_noise_power = signal_power / (10 ** (snr_db / 10))
        
        current_noise_power = torch.mean(noise_normalized**2) + eps
        noise_scale = torch.sqrt(target_noise_power / current_noise_power)
        
        noise_scaled = noise_normalized * noise_scale
        
        # Step 3: Create mixture
        mixture = source_normalized + noise_scaled
        
        # Step 4: Uniformly scale to fit within [-1, 1]
        # This preserves SNR because both signal and noise are scaled equally
        mixture_peak = mixture.abs().max()
        scale_factor = 1.0
        
        if mixture_peak > 0.95:  # Leave small headroom
            scale_factor = 0.95 / (mixture_peak + eps)
            mixture = mixture * scale_factor
        
        # Step 5: Safety clip (should rarely trigger with proper scaling)
        mixture = torch.clamp(mixture, -1.0, 1.0)
        
        # Compute actual SNR achieved
        final_signal_power = torch.mean((source_normalized * scale_factor)**2) + eps
        final_noise_power = torch.mean((noise_scaled * scale_factor)**2) + eps
        actual_snr_db = 10 * torch.log10(final_signal_power / final_noise_power).item()
        
        return mixture, actual_snr_db

    def _normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        """Peak-normalize a waveform to [-1, 1].

        Peak normalization (dividing by the maximum absolute value) keeps the
        signal within the valid PCM range, which is required both for correct
        SNR mixing math and for artifact-free saved WAV files.  The previous
        z-score normalization had unit variance but an unbounded peak, which
        caused hard clipping when the mixed signal was written to disk.
        """
        peak = waveform.abs().max()
        if peak < 1e-8:
            return waveform
        return waveform / peak

    def _get_coi_head_index(self) -> int:
        """Return the COI head index for the current separator model."""
        if isinstance(self.separator, TUSSInference):
            return TUSS_COI_HEAD
        if isinstance(self.separator, CLAPSepInference):
            return CLAPSEP_COI_HEAD
        return SUDORMRF_COI_HEAD

    def _classify_pann(self, waveform: torch.Tensor) -> Tuple[int, float]:
        """Classify using the PANN AudioTagging model.

        Resamples *waveform* from ``self.sample_rate`` to the 32 kHz required
        by PANN internally via :func:`run_pann_inference`.

        Returns:
            ``(prediction, confidence)`` – 1/0 and the max sigmoid score across
            all :data:`PANN_POSITIVE_LABELS`.
        """
        if self.pann_model is None:
            raise RuntimeError(
                "PANN model is not loaded. Call load_models(use_pann=True)."
            )
        from src.validation_functions.classification_models.pann_inference import (
            run_pann_inference as _run_pann,
        )

        return _run_pann(
            self.pann_model,
            waveform,
            self.sample_rate,
            PANN_POSITIVE_LABELS,
        )

    def _classify_ast(self, waveform: torch.Tensor) -> Tuple[int, float]:
        """Classify using the AST (Audio Spectrogram Transformer) model.

        Resamples *waveform* from ``self.sample_rate`` to the 16 kHz required
        by AST internally via :func:`run_ast_inference`.

        Returns:
            ``(prediction, confidence)`` – 1/0 and the max sigmoid score across
            all :data:`AST_POSITIVE_LABELS`.
        """
        if self.ast_model is None:
            raise RuntimeError(
                "AST model is not loaded. Call load_models(use_ast=True)."
            )
        assert (
            self.ast_extractor is not None
        ), "ast_extractor should be set whenever ast_model is set"
        from src.validation_functions.classification_models.ast_inference import (
            run_ast_inference as _run_ast,
        )

        return _run_ast(
            self.ast_extractor,
            self.ast_model,
            waveform,
            self.sample_rate,
            AST_POSITIVE_LABELS,
            device=self.device,
        )

    def _classify_separated(
        self,
        separated: torch.Tensor,
        classify_fn: Optional[Callable[[torch.Tensor], Tuple[int, float]]] = None,
    ) -> Tuple[int, float]:
        """Classify the COI output from a separated signal.

        Uses the designated COI head index rather than cherry-picking
        the source with the highest confidence, so the evaluation
        matches real-world deployment behaviour.

        Args:
            separated: Either a 1-D waveform or a (n_sources, T) tensor.
            classify_fn: Classification function to use.  Defaults to
                ``self._classify`` (the custom CNN) when *None*.
        """
        fn = classify_fn if classify_fn is not None else self._classify
        if separated.dim() == 1:
            return fn(separated)
        coi_source = separated[self._get_coi_head_index()]
        return fn(coi_source)

    def _compute_signal_metrics(
        self, reference: torch.Tensor, estimate: torch.Tensor
    ) -> Tuple[float, float, float]:
        """Compute signal-level separation quality metrics.

        Args:
            reference: Clean reference signal (T,)
            estimate:  Separated estimate signal (T,)

        Returns:
            (si_snr, sdr, si_sdr) in dB
        """
        ref = reference.detach().cpu().float()
        est = estimate.detach().cpu().float()
        # Align lengths
        min_len = min(ref.shape[0], est.shape[0])
        ref = ref[:min_len]
        est = est[:min_len]

        si_snr = self._si_snr(est.unsqueeze(0), ref.unsqueeze(0)).item()
        sdr = self._sdr(est.unsqueeze(0), ref.unsqueeze(0)).item()
        si_sdr = self._si_sdr(est.unsqueeze(0), ref.unsqueeze(0)).item()
        return si_snr, sdr, si_sdr

    def validate_clean(
        self,
        df_coi: pd.DataFrame,
        df_bg: pd.DataFrame,
        use_separation: bool,
        save_examples_dir: str = None,
        save_n_examples: int = 1,
        classify_fn: Optional[Callable[[torch.Tensor], Tuple[int, float]]] = None,
    ) -> ClassificationMetrics:
        """Validate on clean (unmixed) audio - both COI and background.

        If `save_examples_dir` is provided, pick up to `save_n_examples` random COI
        samples and save:
          - the original clean COI input(s)
          - the separated outputs (either single COI head or all sources)
        This helps inspect a small number of separation examples for the clean condition.

        Args:
            classify_fn: Classification callable ``(waveform) -> (pred, conf)``.
                Defaults to ``self._classify`` (the custom CNN) when *None*.
        """
        # Default to the CNN classifier when no override is supplied.
        if classify_fn is None:
            classify_fn = self._classify

        y_true, y_pred, y_scores = [], [], []
        raw_labels = []
        si_snr_scores, sdr_scores, si_sdr_scores = [], [], []
        desc = "Clean (sep+cls)" if use_separation else "Clean (cls only)"

        # Prepare optional example saving: choose up to save_n_examples distinct indices
        save_dir = Path(save_examples_dir) if save_examples_dir else None
        sample_choices = []
        if save_dir is not None and len(df_coi) > 0:
            save_dir.mkdir(parents=True, exist_ok=True)
            n = min(save_n_examples, len(df_coi))
            sample_choices = list(
                np.random.choice(len(df_coi), size=n, replace=False).tolist()
            )

        # Process COI samples (label=1)
        for idx, row in enumerate(
            tqdm(df_coi.itertuples(), total=len(df_coi), desc=f"{desc} - COI")
        ):
            try:
                # Load the labeled window and slice into model-sized segments.
                # For strongly labeled recordings start_time/end_time is a tight window
                # containing the COI; for weakly labeled recordings it spans the full
                # file.  Either way the recording-level prediction is aggregated below
                # with any(), so a positive on any segment counts as a positive for the
                # whole recording.
                waveform_full = self._load_labeled_audio(
                    row.filename,
                    getattr(row, "start_time", None),
                    getattr(row, "end_time", None),
                )
                segments = self._split_into_segments(waveform_full)

                # Optionally save the first segment of selected examples.
                if save_dir is not None and idx in sample_choices:
                    try:
                        k = list(sample_choices).index(idx)
                        torchaudio.save(
                            str(save_dir / f"clean_coi_{k}.wav"),
                            segments[0].unsqueeze(0).cpu(),
                            self.sample_rate,
                        )
                    except Exception:
                        print(
                            f"Warning: failed to save clean_coi for {row.filename}",
                            file=sys.stderr,
                        )

                if use_separation:
                    seg_preds: List[int] = []
                    seg_confs: List[float] = []
                    seg_si_snr: List[float] = []
                    seg_sdr: List[float] = []
                    seg_si_sdr: List[float] = []

                    for seg_idx, seg in enumerate(segments):
                        separated = self._separate(seg)
                        seg_pred, seg_conf = self._classify_separated(
                            separated, classify_fn
                        )
                        seg_preds.append(seg_pred)
                        seg_confs.append(seg_conf)

                        coi_est = (
                            separated
                            if separated.dim() == 1
                            else separated[self._get_coi_head_index()]
                        )

                        # Signal metrics (note: here reference is the input segment)
                        si_snr, sdr, si_sdr = self._compute_signal_metrics(seg, coi_est)
                        seg_si_snr.append(si_snr)
                        seg_sdr.append(sdr)
                        seg_si_sdr.append(si_sdr)

                        # Save separated outputs for the first segment of chosen samples.
                        if (
                            save_dir is not None
                            and idx in sample_choices
                            and seg_idx == 0
                        ):
                            try:
                                k = list(sample_choices).index(idx)
                                # Debug: print separation stats before saving
                                sep_rms = torch.sqrt(torch.mean(separated**2)).item()
                                sep_max = separated.abs().max().item()
                                print(
                                    f"  [DEBUG] Saving separated sample {k}: "
                                    f"shape={separated.shape}, device={separated.device}, "
                                    f"RMS={sep_rms:.6f}, max={sep_max:.6f}",
                                    file=sys.stderr,
                                )
                                if sep_rms < 1e-6:
                                    print(
                                        f"  [WARNING] Separated output is SILENT!",
                                        file=sys.stderr,
                                    )
                                if separated.dim() == 1:
                                    torchaudio.save(
                                        str(save_dir / f"separated_coi_est_{k}.wav"),
                                        separated.unsqueeze(0).cpu(),
                                        self.sample_rate,
                                    )
                                else:
                                    for s in range(separated.shape[0]):
                                        src_rms = torch.sqrt(
                                            torch.mean(separated[s] ** 2)
                                        ).item()
                                        print(
                                            f"    [DEBUG] Source {s}: RMS={src_rms:.6f}",
                                            file=sys.stderr,
                                        )
                                        torchaudio.save(
                                            str(save_dir / f"separated_src{s}_{k}.wav"),
                                            separated[s].unsqueeze(0).cpu(),
                                            self.sample_rate,
                                        )
                                    coi_head = separated[self._get_coi_head_index()]
                                    torchaudio.save(
                                        str(save_dir / f"separated_coi_head_{k}.wav"),
                                        coi_head.unsqueeze(0).cpu(),
                                        self.sample_rate,
                                    )
                            except Exception as e:
                                import traceback

                                print(
                                    f"Warning: failed to save separated outputs for {row.filename}: {e}",
                                    file=sys.stderr,
                                )
                                traceback.print_exc()

                    # A recording is classified as COI if ANY segment triggers a
                    # positive prediction.  Confidence is the maximum over all segments.
                    pred = 1 if any(p == 1 for p in seg_preds) else 0
                    conf = max(seg_confs)
                    si_snr_scores.append(float(np.mean(seg_si_snr)))
                    sdr_scores.append(float(np.mean(seg_sdr)))
                    si_sdr_scores.append(float(np.mean(seg_si_sdr)))
                else:
                    seg_preds = []
                    seg_confs = []
                    for seg in segments:
                        seg_pred, seg_conf = classify_fn(seg)
                        seg_preds.append(seg_pred)
                        seg_confs.append(seg_conf)
                    pred = 1 if any(p == 1 for p in seg_preds) else 0
                    conf = max(seg_confs)

                y_true.append(1)
                y_pred.append(pred)
                y_scores.append(conf)
                raw_labels.append(getattr(row, "orig_label", row.label))
            except Exception as e:
                import traceback

                print(f"Error: {row.filename}: {e}", file=sys.stderr)
                traceback.print_exc()

        # Process background samples (label=0)
        for row in tqdm(df_bg.itertuples(), total=len(df_bg), desc=f"{desc} - BG"):
            try:
                waveform_full = self._load_labeled_audio(
                    row.filename,
                    getattr(row, "start_time", None),
                    getattr(row, "end_time", None),
                )
                segments = self._split_into_segments(waveform_full)

                seg_preds = []
                seg_confs = []
                for seg in segments:
                    if use_separation:
                        separated = self._separate(seg)
                        seg_pred, seg_conf = self._classify_separated(
                            separated, classify_fn
                        )
                    else:
                        seg_pred, seg_conf = classify_fn(seg)
                    seg_preds.append(seg_pred)
                    seg_confs.append(seg_conf)

                pred = 1 if any(p == 1 for p in seg_preds) else 0
                conf = max(seg_confs)
                y_true.append(0)
                y_pred.append(pred)
                y_scores.append(conf)
                raw_labels.append(getattr(row, "orig_label", row.label))
            except Exception as e:
                import traceback

                print(f"Error: {row.filename}: {e}", file=sys.stderr)
                traceback.print_exc()

        metrics = ClassificationMetrics()
        metrics.si_snr_scores = si_snr_scores
        metrics.sdr_scores = sdr_scores
        metrics.si_sdr_scores = si_sdr_scores
        metrics.final_coi_count = len(df_coi)
        metrics.final_background_count = len(df_bg)
        metrics.compute(
            np.array(y_true),
            np.array(y_pred),
            np.array(y_scores),
            raw_labels=raw_labels,
        )
        return metrics

    def validate_mixtures(
        self,
        df_coi: pd.DataFrame,
        df_bg: pd.DataFrame,
        snr_range: Tuple[float, float],
        use_separation: bool,
        save_examples_dir: str = None,
        save_n_examples: int = 1,
        classify_fn: Optional[Callable[[torch.Tensor], Tuple[int, float]]] = None,
    ) -> ClassificationMetrics:
        """Validate on mixtures at random SNR (COI+BG) and clean background (BG only).

        If `save_examples_dir` is provided, pick up to `save_n_examples` random COI
        mixtures and save:
          - the original clean COI and BG inputs used to create those mixtures
          - the created mixture file(s)
          - the separated outputs (either single COI head or all sources)

        Args:
            classify_fn: Classification callable ``(waveform) -> (pred, conf)``.
                Defaults to ``self._classify`` (the custom CNN) when *None*.
        """
        # Default to the CNN classifier when no override is supplied.
        if classify_fn is None:
            classify_fn = self._classify

        y_true, y_pred, y_scores = [], [], []
        raw_labels = []
        si_snr_scores, sdr_scores, si_sdr_scores = [], [], []
        actual_snrs: List[float] = []
        desc = "Mixtures (sep+cls)" if use_separation else "Mixtures (cls only)"
        # Keep per-row timing info so background clips are also sliced from
        # their labeled window when splitting into segments.
        bg_records = df_bg[
            ["filename"] + [c for c in ["start_time", "end_time"] if c in df_bg.columns]
        ].to_dict("records")

        # Prepare optional example saving: choose up to save_n_examples distinct indices
        save_dir = Path(save_examples_dir) if save_examples_dir else None
        sample_choices = []
        if save_dir is not None and len(df_coi) > 0:
            save_dir.mkdir(parents=True, exist_ok=True)
            n = min(save_n_examples, len(df_coi))
            sample_choices = list(
                np.random.choice(len(df_coi), size=n, replace=False).tolist()
            )

        # Process COI + background mixtures (label=1)
        for idx, row in enumerate(
            tqdm(df_coi.itertuples(), total=len(df_coi), desc=f"{desc} - COI+BG")
        ):
            try:
                # Load full labeled windows for both COI and a random background clip,
                # then split into matching model-sized segment pairs for mixing.
                coi_full = self._load_labeled_audio(
                    row.filename,
                    getattr(row, "start_time", None),
                    getattr(row, "end_time", None),
                )
                bg_idx = np.random.randint(len(bg_records))
                bg_rec = bg_records[bg_idx]
                bg_full = self._load_labeled_audio(
                    bg_rec["filename"],
                    bg_rec.get("start_time"),
                    bg_rec.get("end_time"),
                )

                coi_segments = self._split_into_segments(coi_full)
                bg_segments = self._split_into_segments(bg_full)

                seg_preds: List[int] = []
                seg_confs: List[float] = []
                seg_snr_vals: List[float] = []
                seg_si_snr: List[float] = []
                seg_sdr: List[float] = []
                seg_si_sdr: List[float] = []

                for seg_idx, coi_seg in enumerate(coi_segments):
                    # Cycle through BG segments if fewer than COI segments.
                    bg_seg = bg_segments[seg_idx % len(bg_segments)]
                    coi_n = self._normalize(coi_seg)
                    bg_n = self._normalize(bg_seg)
                    snr = np.random.uniform(*snr_range)
                    mixture, actual_snr = self._create_mixture(coi_n, bg_n, snr)
                    seg_snr_vals.append(actual_snr)

                    # Save first segment of chosen examples.
                    if save_dir is not None and idx in sample_choices and seg_idx == 0:
                        try:
                            k = list(sample_choices).index(idx)
                            # coi_n and bg_n are already normalized (lines 1495-1496)
                            # so we don't need to normalize them again.
                            torchaudio.save(
                                str(save_dir / f"mixture_coi_clean_{k}.wav"),
                                coi_n.unsqueeze(0).cpu(),
                                self.sample_rate,
                            )
                            torchaudio.save(
                                str(save_dir / f"mixture_bg_clean_{k}.wav"),
                                bg_n.unsqueeze(0).cpu(),
                                self.sample_rate,
                            )
                            # Peak-normalize the mixture independently so its
                            # loudest sample sits at ±1 (the mix may sum above
                            # the individual peaks).
                            torchaudio.save(
                                str(save_dir / f"mixture_created_{k}.wav"),
                                self._normalize(mixture).unsqueeze(0).cpu(),
                                self.sample_rate,
                            )
                        except Exception:
                            print(
                                f"Warning: failed to save mixture example for {row.filename}",
                                file=sys.stderr,
                            )

                    if use_separation:
                        separated = self._separate(mixture)
                        seg_pred, seg_conf = self._classify_separated(
                            separated, classify_fn
                        )
                        seg_preds.append(seg_pred)
                        seg_confs.append(seg_conf)

                        coi_est = (
                            separated
                            if separated.dim() == 1
                            else separated[self._get_coi_head_index()]
                        )
                        si_snr_val, sdr_val, si_sdr_val = self._compute_signal_metrics(
                            coi_n, coi_est
                        )
                        seg_si_snr.append(si_snr_val)
                        seg_sdr.append(sdr_val)
                        seg_si_sdr.append(si_sdr_val)

                        # Save separated outputs for first segment of chosen examples.
                        if (
                            save_dir is not None
                            and idx in sample_choices
                            and seg_idx == 0
                        ):
                            try:
                                k = list(sample_choices).index(idx)
                                if separated.dim() == 1:
                                    torchaudio.save(
                                        str(
                                            save_dir
                                            / f"mixture_separated_coi_est_{k}.wav"
                                        ),
                                        separated.unsqueeze(0).cpu(),
                                        self.sample_rate,
                                    )
                                else:
                                    for s in range(separated.shape[0]):
                                        torchaudio.save(
                                            str(
                                                save_dir
                                                / f"mixture_separated_src{s}_{k}.wav"
                                            ),
                                            separated[s].unsqueeze(0).cpu(),
                                            self.sample_rate,
                                        )
                                    coi_head = separated[self._get_coi_head_index()]
                                    torchaudio.save(
                                        str(
                                            save_dir
                                            / f"mixture_separated_coi_head_{k}.wav"
                                        ),
                                        coi_head.unsqueeze(0).cpu(),
                                        self.sample_rate,
                                    )
                            except Exception:
                                print(
                                    f"Warning: failed to save separated outputs for mixture {row.filename}",
                                    file=sys.stderr,
                                )
                    else:
                        seg_pred, seg_conf = classify_fn(mixture)
                        seg_preds.append(seg_pred)
                        seg_confs.append(seg_conf)

                # Aggregate across segments: positive if any segment detects COI.
                pred = 1 if any(p == 1 for p in seg_preds) else 0
                conf = max(seg_confs)
                actual_snrs.append(float(np.mean(seg_snr_vals)))
                if use_separation:
                    si_snr_scores.append(float(np.mean(seg_si_snr)))
                    sdr_scores.append(float(np.mean(seg_sdr)))
                    si_sdr_scores.append(float(np.mean(seg_si_sdr)))

                y_true.append(1)
                y_pred.append(pred)
                y_scores.append(conf)
                raw_labels.append(getattr(row, "orig_label", row.label))
            except Exception as e:
                import traceback

                print(f"Error: {e}", file=sys.stderr)
                traceback.print_exc()

        # Process background-only samples (label=0)
        for row in tqdm(df_bg.itertuples(), total=len(df_bg), desc=f"{desc} - BG only"):
            try:
                waveform_full = self._load_labeled_audio(
                    row.filename,
                    getattr(row, "start_time", None),
                    getattr(row, "end_time", None),
                )
                segments = self._split_into_segments(waveform_full)

                seg_preds = []
                seg_confs = []
                for seg in segments:
                    if use_separation:
                        separated = self._separate(seg)
                        seg_pred, seg_conf = self._classify_separated(
                            separated, classify_fn
                        )
                    else:
                        seg_pred, seg_conf = classify_fn(seg)
                    seg_preds.append(seg_pred)
                    seg_confs.append(seg_conf)

                pred = 1 if any(p == 1 for p in seg_preds) else 0
                conf = max(seg_confs)
                y_true.append(0)
                y_pred.append(pred)
                y_scores.append(conf)
                raw_labels.append(getattr(row, "orig_label", row.label))
            except Exception as e:
                import traceback

                print(f"Error: {e}", file=sys.stderr)
                traceback.print_exc()

        metrics = ClassificationMetrics()
        metrics.si_snr_scores = si_snr_scores
        metrics.sdr_scores = sdr_scores
        metrics.si_sdr_scores = si_sdr_scores
        metrics.actual_snrs = actual_snrs
        metrics.final_coi_count = len(df_coi)
        metrics.final_background_count = len(df_bg)
        # pass raw_labels so that misclassification counts by the original
        # multi‑class label (and the per‑label counter) work correctly for
        # mixture evaluations, matching the clean path above.
        metrics.compute(
            np.array(y_true),
            np.array(y_pred),
            np.array(y_scores),
            raw_labels=raw_labels,
        )
        return metrics

    def run(
        self,
        split: str = "test",
        snr_range: Tuple[float, float] = (-5, 5),
        data_csv: str = None,
        output_dir: str = None,
        seed: int = 42,
        save_examples_dir: str = None,
        save_n_examples: int = 1,
        only_dataset: Optional[str] = None,
        exclude_datasets: Optional[List[str]] = None,
        skip_clean_tests: bool = False,
    ) -> Dict[str, Dict[str, ClassificationMetrics]]:
        """Run full validation suite for every loaded classifier.

        Default behaviour (training-domain test sets):
          1. Clean audio — classification only
          2. Clean audio — separation + classification
          3. Synthetic mixtures — classification only
          4. Synthetic mixtures — separation + classification

        Independent datasets (``only_dataset=...``, e.g. risoux_test):
        To avoid extra compute and to prevent "mixture-of-mixtures" evaluation,
        we run ONLY TWO steps on the recordings **as-is**:
          1. As-is audio — classification only
          2. As-is audio — separation + classification

        Results and saved audio examples are placed in per-classifier
        subdirectories so they never overwrite each other:

        - ``<output_dir>/cnn/``       — custom PlaneClassifier CNN
        - ``<output_dir>/pann/``      — PANN AudioTagging (if loaded)
        - ``<output_dir>/ast/``       — AST from HuggingFace (if loaded)

        Args:
            split: Dataset split to evaluate on (e.g. ``"test"``).  When
                ``only_dataset`` is set the split filter is skipped and all
                rows belonging to that dataset are used directly.
            snr_range: (min, max) SNR in dB for mixture creation (synthetic-mixture tests only).
            data_csv: Path to dataset CSV. Falls back to self.DATA_CSV.
            output_dir: Root directory for JSON results.  Each classifier writes
                into its own subdirectory so runs never overwrite each other.
            seed: Random seed for reproducibility of mixture creation.
            save_examples_dir: Root directory for saved audio examples.  Each
                classifier writes into its own subdirectory.
            save_n_examples: Number of random examples to save per test stage.
                Defaults to 1.
            only_dataset: If provided, evaluate this dataset in isolation. When set,
                the pipeline runs only two "as-is" evaluation steps (cls only, sep+cls).
            exclude_datasets: Optional list of dataset names to drop before
                applying the split filter.
            skip_clean_tests: When ``True`` on normal test sets, steps 1 and 2 are skipped.

        Returns:
            Nested dict ``{classifier_name: {test_name: ClassificationMetrics}}``.
        """
        # Set seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"Random seed set to {seed}")

        csv_path = data_csv or self.DATA_CSV
        df = pd.read_csv(csv_path)

        # ------------------------------------------------------------------
        # Dataset-level filtering
        # ------------------------------------------------------------------
        if only_dataset is not None:
            # Evaluate a single independent dataset regardless of split value.
            if "dataset" not in df.columns:
                print(
                    f"[Warning] 'dataset' column not found in CSV; "
                    f"'only_dataset={only_dataset}' filter cannot be applied.",
                    file=sys.stderr,
                )
                df_split = df[df["split"] == split]
            else:
                df_split = df[df["dataset"] == only_dataset].reset_index(drop=True)
        else:
            # Normal path: filter by split, optionally excluding certain datasets.
            if exclude_datasets and "dataset" in df.columns:
                df = df[~df["dataset"].isin(exclude_datasets)]
            elif exclude_datasets:
                print(
                    "[Warning] 'dataset' column not found in CSV; "
                    "'exclude_datasets' filter cannot be applied.",
                    file=sys.stderr,
                )
            df_split = df[df["split"] == split]

        # ------------------------------------------------------------------
        # Label normalization / binarization for evaluation
        # ------------------------------------------------------------------
        # We prefer to use `orig_label` (preserved raw label) when available,
        # because some pipelines overwrite `label` with a numeric/binary value.
        #
        # Policy:
        # - If `label` is already numeric 0/1, keep it.
        # - Otherwise, derive a binary `label` from (`orig_label` if present else `label`)
        #   using COI_SYNONYMS so "plane" and "airplane" map consistently.
        #
        # Note: rows with missing labels are still treated as background (0) as before.
        df_split = df_split.copy()
        df_split["label"] = df_split["label"].fillna(0)

        # If label column isn't already binary numeric, binarize from raw strings.
        if not pd.api.types.is_numeric_dtype(df_split["label"]):
            raw_series = (
                df_split["orig_label"]
                if "orig_label" in df_split.columns
                else df_split["label"]
            )
            df_split["label"] = raw_series.apply(lambda x: 1 if _is_coi_label(x) else 0)

        df_coi = df_split[df_split["label"] == 1].reset_index(drop=True)
        df_bg = df_split[df_split["label"] == 0].reset_index(drop=True)

        # Filter contaminated backgrounds (only when orig_label exists)
        df_bg, n_contaminated = _filter_contaminated_backgrounds(df_bg, verbose=True)

        # Human-readable label for logging and output filenames.
        run_label = only_dataset if only_dataset is not None else split

        print(f"\n{'=' * 60}")
        print(
            f"Validation on {run_label.upper()} set: {len(df_coi)} COI, {len(df_bg)} background"
        )
        print(f"SNR range: {snr_range} dB")
        print(f"{'=' * 60}\n")

        if getattr(self, "target_classes", None):
            print(f"Target classes: {self.target_classes}")
        if only_dataset:
            print(f"[Independent test set] only_dataset={only_dataset!r}")
        if exclude_datasets:
            print(f"[Excluded from this run] exclude_datasets={exclude_datasets!r}")

        # ------------------------------------------------------------------
        # Build the list of (name, classify_fn) pairs to iterate over.
        # ------------------------------------------------------------------
        classifiers = []
        if self.classifier is not None:
            classifiers.append(("cnn", self._classify))
        if self.pann_model is not None:
            classifiers.append(("pann", self._classify_pann))
        if self.ast_model is not None:
            classifiers.append(("ast", self._classify_ast))

        if not classifiers:
            print(
                "[Warning] No classifiers are loaded — nothing to evaluate.",
                file=sys.stderr,
            )
            return {}

        # ------------------------------------------------------------------
        # Decide which evaluation regime to use.
        #
        # - Normal regime (training-domain test split):
        #     up to 4 steps (clean cls, clean sep+cls, synthetic mix cls, synthetic mix sep+cls)
        #
        # - Independent dataset regime (only_dataset is set, e.g. risoux_test):
        #     run ONLY 2 steps on recordings AS-IS to avoid extra compute and
        #     to prevent "mixture-of-mixtures" evaluation.
        # ------------------------------------------------------------------
        independent_as_is = only_dataset is not None

        # Independent dataset evaluation always skips the clean-audio steps:
        # recordings are evaluated as-is rather than creating new mixtures, so
        # synthesising separate clean tests would just duplicate the as-is run.
        if independent_as_is and not skip_clean_tests:
            print(
                "[Info] independent dataset mode: skip_clean_tests forced to True "
                "(recordings are evaluated as-is — 2 steps only)."
            )
            skip_clean_tests = True

        if independent_as_is:
            total_steps = 2
            mix_cls_step = 1
            mix_sep_step = 2
        else:
            total_steps = 2 if skip_clean_tests else 4
            # Step numbers for the mixture tests depend on whether clean tests run.
            mix_cls_step = 1 if skip_clean_tests else 3
            mix_sep_step = 2 if skip_clean_tests else 4

            if skip_clean_tests:
                print(
                    "[Info] skip_clean_tests=True: steps 1 (clean cls) and 2 "
                    "(clean sep+cls) will be skipped for this dataset."
                )

        all_results: Dict[str, Dict[str, ClassificationMetrics]] = {}

        for cls_name, classify_fn in classifiers:
            print(f"\n{'#' * 60}")
            print(f"#  Classifier: {cls_name.upper()}")
            print(f"{'#' * 60}")

            # Per-classifier output and example directories.
            cls_output_dir = str(Path(output_dir) / cls_name) if output_dir else None
            cls_examples_dir = (
                str(Path(save_examples_dir) / cls_name) if save_examples_dir else None
            )

            results: Dict[str, ClassificationMetrics] = {}

            if not skip_clean_tests:
                # 1. Clean - classification only
                print(
                    f"\n[{cls_name}][1/{total_steps}] Clean audio - classification only"
                )
                results["clean_cls"] = self.validate_clean(
                    df_coi,
                    df_bg,
                    use_separation=False,
                    classify_fn=classify_fn,
                )
                print(results["clean_cls"])

                # 2. Clean - separation + classification
                print(
                    f"\n[{cls_name}][2/{total_steps}] Clean audio - separation + classification"
                )
                clean_save_dir = (
                    str(Path(cls_examples_dir) / "clean_sep")
                    if cls_examples_dir
                    else None
                )
                results["clean_sep_cls"] = self.validate_clean(
                    df_coi,
                    df_bg,
                    use_separation=True,
                    save_examples_dir=clean_save_dir,
                    save_n_examples=save_n_examples,
                    classify_fn=classify_fn,
                )
                print(results["clean_sep_cls"])

            # 3/4. Mixture or as-is tests (depending on whether this is an
            # independent dataset or the normal training-domain test split).
            if only_dataset is not None:
                # Independent dataset: evaluate recordings as-is (no synthetic
                # mixing) to avoid a mixture-of-mixtures situation.
                print(
                    f"\n[{cls_name}][{mix_cls_step}/{total_steps}] Independent dataset - as-is audio (cls only)"
                )
                results["as_is_cls"] = self.validate_clean(
                    df_coi,
                    df_bg,
                    use_separation=False,
                    classify_fn=classify_fn,
                )
                print(results["as_is_cls"])

                print(
                    f"\n[{cls_name}][{mix_sep_step}/{total_steps}] Independent dataset - as-is audio (sep+cls)"
                )
                as_is_save_dir = (
                    str(Path(cls_examples_dir) / "as_is_sep")
                    if cls_examples_dir
                    else None
                )
                results["as_is_sep_cls"] = self.validate_clean(
                    df_coi,
                    df_bg,
                    use_separation=True,
                    save_examples_dir=as_is_save_dir,
                    save_n_examples=save_n_examples,
                    classify_fn=classify_fn,
                )
                print(results["as_is_sep_cls"])
            else:
                # Normal synthetic mixture tests (COI+BG @ random SNR).
                if len(df_bg) > 0:
                    print(
                        f"\n[{cls_name}][{mix_cls_step}/{total_steps}] Mixtures ({snr_range}dB) - classification only"
                    )
                    results["mix_cls"] = self.validate_mixtures(
                        df_coi,
                        df_bg,
                        snr_range,
                        use_separation=False,
                        classify_fn=classify_fn,
                    )
                    print(results["mix_cls"])

                    # Synthetic mixtures - separation + classification
                    print(
                        f"\n[{cls_name}][{mix_sep_step}/{total_steps}] Mixtures ({snr_range}dB) - separation + classification"
                    )
                    mix_save_dir = (
                        str(Path(cls_examples_dir) / "mixture_sep")
                        if cls_examples_dir
                        else None
                    )
                    results["mix_sep_cls"] = self.validate_mixtures(
                        df_coi,
                        df_bg,
                        snr_range,
                        use_separation=True,
                        save_examples_dir=mix_save_dir,
                        save_n_examples=save_n_examples,
                        classify_fn=classify_fn,
                    )
                    print(results["mix_sep_cls"])
                else:
                    print(
                        f"\n[{cls_name}] No background samples found — mixture tests skipped.",
                        file=sys.stderr,
                    )

            # Set contamination stats on all metrics objects
            for metrics_obj in results.values():
                metrics_obj.contaminated_backgrounds_removed = n_contaminated

            # Save per-classifier JSON results.
            if cls_output_dir:
                Path(cls_output_dir).mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_dict = {k: v.to_dict() for k, v in results.items()}
                results_dict["split"] = split
                results_dict["skip_clean_tests"] = skip_clean_tests
                results_dict["classifier"] = cls_name
                if only_dataset:
                    results_dict["only_dataset"] = only_dataset
                results_dict["checkpoint_paths"] = {
                    "separator": str(self.sep_checkpoint_path),
                    "classifier": str(self.cls_checkpoint_path),
                }
                # Include positive-label config for AudioSet classifiers.
                if cls_name == "pann":
                    results_dict["positive_labels"] = PANN_POSITIVE_LABELS
                elif cls_name == "ast":
                    results_dict["positive_labels"] = AST_POSITIVE_LABELS
                elif cls_name == "cnn":
                    results_dict["positive_class"] = CNN_POSITIVE_CLASS
                tag = f"{split}_{only_dataset}" if only_dataset else split
                out_file = Path(cls_output_dir) / f"results_{tag}_{ts}.json"
                with open(out_file, "w") as f:
                    json.dump(results_dict, f, indent=2)
                print(f"\n[{cls_name}] Results saved to {out_file}")

            all_results[cls_name] = results

        return all_results


def demo_two_wav_separation(
    src_path: str,
    noise_path: str,
    snr_db: float = 0.0,
    sep_checkpoint: str = None,
    out_dir: str = None,
) -> Dict[str, str]:
    """Create a mixture from two WAV paths, run the separation model on the
    mixture, and save the results.

    Args:
        src_path: Path to the source (COI) WAV file.
        noise_path: Path to the noise/background WAV file.
        snr_db: Desired SNR for the mixture.
        sep_checkpoint: Optional checkpoint path to load separator from.
        out_dir: Directory to save outputs. Defaults to ./separation_output.

    Returns:
        Dict with saved file paths.
    """
    src_path = Path(src_path)
    noise_path = Path(noise_path)

    if not src_path.exists() or not noise_path.exists():
        raise ValueError(
            "Both src_path and noise_path must exist and point to .wav files"
        )

    pipeline = ValidationPipeline()

    # Load separator (and classifier) if a checkpoint is provided or if not loaded
    if sep_checkpoint is not None or pipeline.separator is None:
        pipeline.load_models(sep_checkpoint=sep_checkpoint)

    # Load full audio without truncation (demo should preserve full 10s clips)
    def _load_full(path: Path) -> torch.Tensor:
        wav, sr = torchaudio.load(str(path))
        if sr != pipeline.sample_rate:
            key = (sr, pipeline.sample_rate)
            resampler = torchaudio.transforms.Resample(sr, pipeline.sample_rate)
            wav = resampler(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0)
        else:
            wav = wav.squeeze(0)
        return wav

    print(f"Loading source (full): {src_path}")
    src = _load_full(src_path)
    print(f"Loading noise (full): {noise_path}")
    noise = _load_full(noise_path)

    # Align lengths: extend/trim noise to match source length
    src_len = src.shape[0]
    noise_len = noise.shape[0]
    if noise_len < src_len:
        repeats = int((src_len + noise_len - 1) // noise_len)
        noise = noise.repeat(repeats)[:src_len]
    elif noise_len > src_len:
        noise = noise[:src_len]

    # Normalize before mixing
    src_n = pipeline._normalize(src)
    noise_n = pipeline._normalize(noise)

    mixture, actual_snr = pipeline._create_mixture(src_n, noise_n, snr_db)
    print(f"Requested SNR: {snr_db:.1f} dB -> Actual SNR: {actual_snr:.1f} dB")

    print("Running separation on the mixture...")
    separated = pipeline._separate(mixture)

    out_dir = Path(out_dir or Path.cwd() / "separation_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prepare tensors for saving: (channels, samples)
    def _prep(t: torch.Tensor) -> torch.Tensor:
        t = t.detach().cpu()
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t

    mix_t = _prep(mixture)
    sep_t = _prep(separated)
    src_t = _prep(src)
    noise_t = _prep(noise)

    mix_path = out_dir / f"mixture_{ts}.wav"
    sep_path = out_dir / f"separated_{ts}.wav"
    src_out = out_dir / f"source_{ts}.wav"
    noise_out = out_dir / f"noise_{ts}.wav"

    torchaudio.save(str(mix_path), mix_t, pipeline.sample_rate)
    # Print shape for debugging and save each separated source individually
    print(f"Separated tensor shape: {tuple(sep_t.shape)}")
    separated_files = []
    if sep_t.dim() == 1:
        # single channel
        single_path = out_dir / f"separated_{ts}_src0.wav"
        torchaudio.save(str(single_path), sep_t.unsqueeze(0), pipeline.sample_rate)
        separated_files.append(str(single_path))
    elif sep_t.dim() == 2:
        for i in range(sep_t.shape[0]):
            single_path = out_dir / f"separated_{ts}_src{i}.wav"
            torchaudio.save(str(single_path), sep_t[i : i + 1, :], pipeline.sample_rate)
            separated_files.append(str(single_path))
    else:
        # Unexpected shape: save as single file
        single_path = out_dir / f"separated_{ts}.wav"
        torchaudio.save(str(single_path), sep_t, pipeline.sample_rate)
        separated_files.append(str(single_path))
    torchaudio.save(str(src_out), src_t, pipeline.sample_rate)
    torchaudio.save(str(noise_out), noise_t, pipeline.sample_rate)

    print(f"Saved mixture -> {mix_path}")
    print("Saved separated source files:")
    for p in separated_files:
        print(" ", p)

    return {
        "mixture": str(mix_path),
        "separated_files": separated_files,
        "source": str(src_out),
        "noise": str(noise_out),
    }


def main():
    # ============ CONFIGURE PATHS HERE ============
    SEP_CHECKPOINT = (
        PROJECT_ROOT / "src/models/sudormrf/checkpoints/20260316_191707/best_model.pt"
    )
    CLS_WEIGHTS = (
        PROJECT_ROOT
        / "src/validation_functions/classification_models/plane_clasifier/results/checkpoints/final_model.weights.h5"
    )
    # Trains
    DATA_CSV = (
        PROJECT_ROOT
        / "src/models/sudormrf/checkpoints/20260316_191707/separation_dataset.csv"
    )

    # planes/ "src/models/sudormrf/checkpoints/20260129_113352/separation_dataset.csv"

    BASE_PATH = PROJECT_ROOT.parent / "datasets"  # For converting Windows paths in CSV
    # ==============================================

    pipeline = ValidationPipeline(base_path=BASE_PATH)
    pipeline.load_models(
        sep_checkpoint=SEP_CHECKPOINT,
        cls_weights=CLS_WEIGHTS,
        use_clapsep=False,
        use_tuss=False,
    )

    # Pass 1: standard held-out test split (esc50, aerosonicdb, freesound)
    pipeline.run(
        split="test",
        snr_range=(-5, 5),
        data_csv=DATA_CSV,
        output_dir="./validation_results",
        save_examples_dir="./validation_examples_test",
        save_n_examples=2,
        exclude_datasets=["risoux_test"],
    )

    # Pass 2: independent Risoux test set — kept separate because
    # load_risoux_test() assigns split="test" to all its rows, so it
    # must be filtered by dataset name rather than split name.
    # Clean-audio tests (steps 1 & 2) are skipped because risoux_test is
    # weakly labelled: recordings are not tightly trimmed to isolated events,
    # so a clean-audio baseline is not meaningful for this set.
    pipeline.run(
        split="test",
        only_dataset="risoux_test",
        snr_range=(-5, 5),
        data_csv=DATA_CSV,
        output_dir="./validation_results",
        save_examples_dir="./validation_examples_risoux_test",
        save_n_examples=2,
        skip_clean_tests=True,
    )


if __name__ == "__main__":
    main()
    # import random

    # example_src = Path("./ho6sg-47RD0.wav")
    # example_noise = Path("./LEEC02__0__20161128_183900_ma.wav")

    # if not example_src.exists() or not example_noise.exists():
    #     missing = []
    #     if not example_src.exists():
    #         missing.append(str(example_src))
    #     if not example_noise.exists():
    #         missing.append(str(example_noise))
    #     print("Example WAV files not found; skipping demo. Missing:", *missing)
    # else:
    #     demo_two_wav_separation(
    #         src_path=str(example_src),
    #         noise_path=str(example_noise),
    #         snr_db=random.uniform(-5, 5),
    #         sep_checkpoint=PROJECT_ROOT
    #         / "src/models/sudormrf/checkpoints/20251226_170458/best_model.pt",
    #         out_dir="./separation_output_demo",
    #     )
