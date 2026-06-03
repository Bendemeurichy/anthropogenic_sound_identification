"""
Validation metrics: contamination filtering and classification metrics container.

Extracted from test_pipeline.py to reduce file size and improve modularity.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.label_loading.coi_labels import (
    COI_SYNONYMS,
    _extract_label_atoms,
    is_coi_label as _is_coi_label,
    normalize_label as _norm_label,
)


def _filter_contaminated_backgrounds(
    df_bg: pd.DataFrame, coi_synonyms: set = None, verbose: bool = True
) -> Tuple[pd.DataFrame, int]:
    if coi_synonyms is None:
        coi_synonyms = COI_SYNONYMS

    if "orig_label" not in df_bg.columns:
        if verbose:
            print("[Info] No 'orig_label' column found - skipping contamination filter")
        return df_bg, 0

    contaminated_mask = df_bg["orig_label"].apply(
        lambda x: _is_coi_label(x, coi_synonyms)
    )

    n_contaminated = int(contaminated_mask.sum())
    if n_contaminated == 0:
        if verbose:
            print("✓ No contaminated background samples detected")
        return df_bg, 0

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"⚠️  CONTAMINATION DETECTED: {n_contaminated} background samples")
        print(f"   contain COI synonyms in orig_label and will be EXCLUDED")
        print(f"{'=' * 60}")

        for split in ["train", "val", "test"]:
            split_mask = df_bg["split"] == split
            split_contam = int((split_mask & contaminated_mask).sum())
            split_total = int(split_mask.sum())
            if split_total > 0:
                pct = 100 * split_contam / split_total
                print(
                    f"  {split:5s}: {split_contam:3d}/{split_total:4d} ({pct:4.1f}%) contaminated"
                )

        print(f"\n  Example contaminated orig_labels:")
        contaminated_labels = df_bg[contaminated_mask]["orig_label"].unique()[:5]
        for lbl in contaminated_labels:
            lbl_str = str(lbl)[:80]
            print(f"    - {lbl_str}")
        print(f"{'=' * 60}\n")

    return df_bg[~contaminated_mask].reset_index(drop=True), n_contaminated


@dataclass
class ClassificationMetrics:
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
    misclassified_transitions: Dict[str, int] = field(default_factory=dict)
    misclassified_per_label: Dict[int, int] = field(default_factory=dict)
    fp_raw_counts: Dict[str, int] = field(default_factory=dict)
    fn_raw_counts: Dict[str, int] = field(default_factory=dict)
    misclassified_raw_counts: Dict[str, int] = field(default_factory=dict)
    misclassified_raw_atomic_counts: Dict[str, int] = field(default_factory=dict)
    fp_raw_atomic_counts: Dict[str, int] = field(default_factory=dict)
    fn_raw_atomic_counts: Dict[str, int] = field(default_factory=dict)

    false_negative_samples: List[Dict[str, Any]] = field(default_factory=list)

    si_snr_scores: List[float] = field(default_factory=list)
    sdr_scores: List[float] = field(default_factory=list)
    si_sdr_scores: List[float] = field(default_factory=list)
    mean_si_snr: Optional[float] = None
    mean_sdr: Optional[float] = None
    mean_si_sdr: Optional[float] = None

    si_snri_scores: List[float] = field(default_factory=list)
    mean_si_snri: Optional[float] = None

    rms_error_scores: List[float] = field(default_factory=list)
    mean_rms_error_db: Optional[float] = None
    sel_error_scores: List[float] = field(default_factory=list)
    mean_sel_error_db: Optional[float] = None

    actual_snrs: List[float] = field(default_factory=list)

    contaminated_backgrounds_removed: int = 0
    final_background_count: int = 0
    final_coi_count: int = 0

    classes_balanced: bool = False
    original_coi_count: int = 0
    original_background_count: int = 0

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray = None,
        raw_labels: Sequence[Any] = None,
        sample_info: List[Dict[str, Any]] = None,
    ):
        self.n_samples = len(y_true)
        self.labels = y_true.tolist()
        self.predictions = y_pred.tolist()
        if raw_labels is not None:
            self.raw_labels = list(raw_labels)
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
                        if sample_info is not None and idx < len(sample_info):
                            fn_entry = {
                                "filename": sample_info[idx].get("filename", ""),
                                "orig_label": raw_key,
                                "confidence": (
                                    y_scores[idx] if y_scores is not None else 0.0
                                ),
                            }
                            if "start_time" in sample_info[idx]:
                                fn_entry["start_time"] = sample_info[idx]["start_time"]
                            if "end_time" in sample_info[idx]:
                                fn_entry["end_time"] = sample_info[idx]["end_time"]
                            self.false_negative_samples.append(fn_entry)

        if self.si_snr_scores:
            self.mean_si_snr = float(np.mean(self.si_snr_scores))
        if self.sdr_scores:
            self.mean_sdr = float(np.mean(self.sdr_scores))
        if self.si_sdr_scores:
            self.mean_si_sdr = float(np.mean(self.si_sdr_scores))
        if self.si_snri_scores:
            self.mean_si_snri = float(np.mean(self.si_snri_scores))
        if self.rms_error_scores:
            self.mean_rms_error_db = float(np.mean(self.rms_error_scores))
        if self.sel_error_scores:
            self.mean_sel_error_db = float(np.mean(self.sel_error_scores))

    def __str__(self):
        s = f"""
{"=" * 50}
Confusion Matrix:  TN={self.true_negatives}  FP={self.false_positives}
                   FN={self.false_negatives}  TP={self.true_positives}

Accuracy:  {self.accuracy:.4f}    Precision: {self.precision:.4f}
Recall:    {self.recall:.4f}    F1-Score:  {self.f1_score:.4f}
Specificity: {self.specificity:.4f}  Balanced Acc: {self.balanced_accuracy:.4f}
MCC: {self.matthews_corrcoef:.4f}"""

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
            si_snri_str = (
                f"{self.mean_si_snri:+.2f} dB"
                if self.mean_si_snri is not None
                else "n/a"
            )
            rms_err_str = (
                f"{self.mean_rms_error_db:+.2f} dB"
                if self.mean_rms_error_db is not None
                else "n/a"
            )
            sel_err_str = (
                f"{self.mean_sel_error_db:+.2f} dB"
                if self.mean_sel_error_db is not None
                else "n/a"
            )
            s += f"""

Signal-Level Metrics (COI samples, n={len(self.si_snr_scores)}):
  SI-SNR: {self.mean_si_snr:+.2f} dB    SDR: {sdr_str}    SI-SDR: {si_sdr_str}
  SI-SNRi (improvement): {si_snri_str}
  RMS error: {rms_err_str}    SEL error: {sel_err_str}"""

        if self.actual_snrs:
            s += f"""
  Actual SNR range: [{min(self.actual_snrs):.1f}, {max(self.actual_snrs):.1f}] dB  (mean: {np.mean(self.actual_snrs):.1f} dB)"""

        s += f"\n{'=' * 50}"
        return s

    def to_dict(self):
        def _sanitize(obj):
            import numpy as _np

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
                return _sanitize(obj.detach().cpu().numpy())
            if isinstance(obj, _np.ndarray):
                return _sanitize(obj.tolist())
            if isinstance(obj, _np.integer):
                return int(obj)
            if isinstance(obj, _np.floating):
                return float(obj)
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
            if self.mean_si_snri is not None:
                d["signal_metrics"]["mean_si_snri_db"] = float(self.mean_si_snri)
            if self.mean_rms_error_db is not None:
                d["signal_metrics"]["mean_rms_error_db"] = float(self.mean_rms_error_db)
            if self.mean_sel_error_db is not None:
                d["signal_metrics"]["mean_sel_error_db"] = float(self.mean_sel_error_db)
        if self.actual_snrs:
            d["actual_snr_stats"] = {
                "min": float(min(self.actual_snrs)),
                "max": float(max(self.actual_snrs)),
                "mean": float(np.mean(self.actual_snrs)),
            }
        if self.misclassified_transitions:
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
        if self.false_negative_samples:
            d["false_negative_samples"] = _sanitize(self.false_negative_samples)
        if getattr(self, "raw_labels", None):
            d["raw_labels"] = [(_sanitize(x)) for x in list(self.raw_labels)]
        if self.contaminated_backgrounds_removed > 0 or self.final_background_count > 0:
            d["dataset_filtering"] = {
                "contaminated_backgrounds_removed": int(
                    self.contaminated_backgrounds_removed
                ),
                "final_background_count": int(self.final_background_count),
                "final_coi_count": int(self.final_coi_count),
            }
        if (
            self.classes_balanced
            or self.original_coi_count > 0
            or self.original_background_count > 0
        ):
            d["class_balancing"] = {
                "balanced": bool(self.classes_balanced),
                "original_coi_count": int(self.original_coi_count),
                "original_background_count": int(self.original_background_count),
                "final_coi_count": int(self.final_coi_count),
                "final_background_count": int(self.final_background_count),
            }
        return _sanitize(d)
