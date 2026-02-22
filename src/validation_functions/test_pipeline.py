"""
Validation module for separation + classification pipeline.
Computes confusion matrix, precision, recall, F1-score, and other metrics.
Includes signal-level separation quality metrics (SI-SNR, SDR).
"""

import importlib
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
sys.path.insert(0, str(Path(__file__).parent / "plane_clasifier"))

from src.models.clapsep.inference import COI_HEAD_INDEX as CLAPSEP_COI_HEAD
from src.models.clapsep.inference import CLAPSepInference
from src.models.sudormrf.inference import COI_HEAD_INDEX as SUDORMRF_COI_HEAD
from src.models.sudormrf.inference import SeparationInference
from src.validation_functions.plane_clasifier.config import TrainingConfig
from src.validation_functions.plane_clasifier.inference import PlaneClassifierInference

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

    # Signal-level separation quality metrics (populated when reference is available)
    si_snr_scores: List[float] = field(default_factory=list)
    sdr_scores: List[float] = field(default_factory=list)
    si_sdr_scores: List[float] = field(default_factory=list)
    mean_si_snr: Optional[float] = None
    mean_sdr: Optional[float] = None
    mean_si_sdr: Optional[float] = None

    # Actual SNR values achieved after clamping (for mixture experiments)
    actual_snrs: List[float] = field(default_factory=list)

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

        # provide a quick summary of which class‑to‑class *transitions* were wrong
        # provide a quick summary of which class transitions were mis‑predicted
        s += (
            f"\n\nMisclassified transition counts:\n"
            f"  Actual 0 → Pred 1: {self.misclassified_transitions.get('0->1', 0)}\n"
            f"  Actual 1 → Pred 0: {self.misclassified_transitions.get('1->0', 0)}"
        )
        if self.misclassified_per_label:
            s += "\n\nMisclassified by binary true label:\n"
            for cls, cnt in sorted(self.misclassified_per_label.items()):
                s += f"  Class {cls}: {cnt}\n"
        if self.misclassified_raw_counts:
            s += "\nMisclassified by original label:\n"
            for raw, cnt in sorted(
                self.misclassified_raw_counts.items(), key=lambda x: str(x[0])
            ):
                s += f"  {raw}: {cnt}\n"

        if self.mean_si_snr is not None:
            s += f"""

Signal-Level Metrics (COI samples, n={len(self.si_snr_scores)}):
  SI-SNR: {self.mean_si_snr:+.2f} dB    SDR: {self.mean_sdr:+.2f} dB    SI-SDR: {self.mean_si_sdr:+.2f} dB"""

        if self.actual_snrs:
            s += f"""
  Actual SNR range: [{min(self.actual_snrs):.1f}, {max(self.actual_snrs):.1f}] dB  (mean: {np.mean(self.actual_snrs):.1f} dB)"""

        s += f"\n{'=' * 50}"
        return s

    def to_dict(self):
        d = {
            "confusion_matrix": {
                "tp": self.true_positives,
                "tn": self.true_negatives,
                "fp": self.false_positives,
                "fn": self.false_negatives,
            },
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "specificity": self.specificity,
            "balanced_accuracy": self.balanced_accuracy,
            "mcc": self.matthews_corrcoef,
        }
        if self.mean_si_snr is not None:
            d["signal_metrics"] = {
                "mean_si_snr_db": self.mean_si_snr,
                "mean_sdr_db": self.mean_sdr,
                "mean_si_sdr_db": self.mean_si_sdr,
                "n_signal_samples": len(self.si_snr_scores),
            }
        if self.actual_snrs:
            d["actual_snr_stats"] = {
                "min": float(min(self.actual_snrs)),
                "max": float(max(self.actual_snrs)),
                "mean": float(np.mean(self.actual_snrs)),
            }
        if self.misclassified_transitions:
            d["misclassified_transitions"] = self.misclassified_transitions
        if self.misclassified_per_label:
            d["misclassified_per_label"] = self.misclassified_per_label
        if self.misclassified_raw_counts:
            d["misclassified_raw_counts"] = {
                str(k): v for k, v in self.misclassified_raw_counts.items()
            }
        if self.raw_labels:
            # preserve the sequence of original labels for downstream inspection
            d["raw_labels"] = list(self.raw_labels)
        return d


# ============================================================================
# VALIDATION PIPELINE
# ============================================================================


class ValidationPipeline:
    """Pipeline for validating separation + classification."""

    # Default paths (relative to project root)
    SEP_CHECKPOINT = PROJECT_ROOT / "src/models/sudormrf/checkpoints/best_model.pt"
    CLS_WEIGHTS = (
        PROJECT_ROOT
        / "src/validation_functions/plane_clasifier/results/checkpoints/final_model.weights.h5"
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
        # Prefer the second CUDA device (index 1) when more than one GPU is present.
        # If a second GPU isn't available we fall back to the CPU rather than using
        # the first GPU; this matches the requirement to "use the second cuda gpu
        # if available and else the cpu".
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.device = "cuda:1"
        else:
            self.device = "cpu"
        # report the choice so users can immediately see which hardware will be used
        print(f"ValidationPipeline using device: {self.device}")
        # will be populated when a separation checkpoint is loaded
        self.target_classes = None

        self.separator = None
        self.classifier = None
        self._resamplers = {}

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
        clapsep_text_pos: str = "train passing",
        clapsep_text_neg: str = "",
    ):
        """Load separation and classification models."""
        sep_path = sep_checkpoint or self.SEP_CHECKPOINT
        cls_path = cls_weights or self.CLS_WEIGHTS

        # Store checkpoint paths
        self.sep_checkpoint_path = sep_path
        self.cls_checkpoint_path = cls_path

        if use_clapsep:
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

        # recover and log the target class list if the checkpoint included a
        # config (it should, since the training routine saves the YAML)
        if hasattr(self.separator, "config") and self.separator.config:
            tc = getattr(self.separator.config.data, "target_classes", None)
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
        """Load audio using torchaudio."""
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

    def _separate(self, waveform: torch.Tensor) -> torch.Tensor:
        """Run separation model. Returns all sources.

        The model was trained with independently normalized sources.
        Outputs are scaled by mixture std to restore reasonable amplitude.
        """
        orig_len = waveform.shape[0]

        if isinstance(self.separator, CLAPSepInference):
            return self.separator.separate_waveform(waveform)

        # If waveform is the expected segment length, run as before
        if orig_len <= self.segment_samples:
            mean = waveform.mean()
            std = waveform.std() + 1e-8
            x = ((waveform - mean) / std).unsqueeze(0).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                estimates = self.separator.model(x)

            # estimates shape: (1, n_sources, T) - normalized sources
            # Scale by std only (model outputs zero-mean signals)
            sources = estimates[0].cpu() * std
            # Return all sources (n_sources, T) trimmed to original length
            return sources[:, :orig_len]

        # For longer waveforms: split into non-overlapping segment windows,
        # run separation per-window, then concatenate and trim to original length.
        n_chunks = int(np.ceil(orig_len / self.segment_samples))
        # determine number of sources from model
        with torch.inference_mode():
            # try to get num_sources attribute, fallback to a dummy forward
            n_sources = getattr(self.separator.model, "num_sources", None)
            if n_sources is None:
                # run a dummy pass to get shape
                dummy = torch.zeros(1, 1, self.segment_samples).to(self.device)
                est = self.separator.model(dummy)
                n_sources = est.shape[1]

        # Prepare lists to collect chunks per source
        outputs_per_source: List[List[torch.Tensor]] = [[] for _ in range(n_sources)]

        for i in range(n_chunks):
            start = i * self.segment_samples
            end = min((i + 1) * self.segment_samples, orig_len)
            chunk = waveform[start:end]

            # If last chunk is shorter, pad to segment size
            if chunk.shape[0] < self.segment_samples:
                chunk = torch.nn.functional.pad(
                    chunk, (0, self.segment_samples - chunk.shape[0])
                )

            mean = chunk.mean()
            std = chunk.std() + 1e-8
            x = ((chunk - mean) / std).unsqueeze(0).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                estimates = self.separator.model(x)

            # Scale by std only (model outputs zero-mean signals)
            src_chunk_all = estimates[0].cpu() * std

            # Trim if last chunk shorter
            length = end - start
            if length < self.segment_samples:
                src_chunk_all = src_chunk_all[:, :length]

            for s in range(n_sources):
                outputs_per_source[s].append(src_chunk_all[s])

        # Concatenate per-source and stack
        concatenated = [
            torch.cat(chunks, dim=0)[:orig_len] for chunks in outputs_per_source
        ]
        return torch.stack(concatenated, dim=0)

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
        pred = 1 if result["prediction"] == "plane" else 0
        return pred, result["confidence"]

    def _create_mixture(
        self, source: torch.Tensor, noise: torch.Tensor, snr_db: float
    ) -> Tuple[torch.Tensor, float]:
        """Mix source and noise at given SNR.

        SNR (dB) = 10 * log10(source_power / noise_power)
        Scales noise to achieve target SNR with clamping to prevent extreme levels.

        Returns:
            Tuple of (mixture, actual_snr_db) where actual_snr_db reflects the
            SNR after clamping the noise scale factor.
        """
        eps = 1e-8
        source_power = torch.mean(source**2) + eps
        noise_power = torch.mean(noise**2) + eps
        scale = torch.sqrt(source_power / (10 ** (snr_db / 10) * noise_power))
        scale_unclamped = scale.item()
        # Clamp scaling to prevent extreme noise (consistent with training)
        scale = torch.clamp(scale, min=0.1, max=3.0)

        # Compute actual SNR achieved after clamping
        scaled_noise_power = torch.mean((noise * scale) ** 2) + eps
        actual_snr_db = 10 * torch.log10(source_power / scaled_noise_power).item()

        if abs(scale.item() - scale_unclamped) > 1e-6:
            print(
                f"  [SNR clamping] requested={snr_db:.1f}dB -> "
                f"actual={actual_snr_db:.1f}dB (scale {scale_unclamped:.3f} -> {scale.item():.3f})"
            )

        return source + noise * scale, actual_snr_db

    def _normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        return (waveform - waveform.mean()) / (waveform.std() + 1e-8)

    def _get_coi_head_index(self) -> int:
        """Return the COI head index for the current separator model."""
        if isinstance(self.separator, CLAPSepInference):
            return CLAPSEP_COI_HEAD
        return SUDORMRF_COI_HEAD

    def _classify_separated(self, separated: torch.Tensor) -> Tuple[int, float]:
        """Classify the COI output from a separated signal.

        Uses the designated COI head index rather than cherry-picking
        the source with the highest confidence, so the evaluation
        matches real-world deployment behaviour.
        """
        if separated.dim() == 1:
            return self._classify(separated)
        coi_source = separated[self._get_coi_head_index()]
        return self._classify(coi_source)

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
    ) -> ClassificationMetrics:
        """Validate on clean (unmixed) audio - both COI and background.

        If `save_examples_dir` is provided, pick up to `save_n_examples` random COI
        samples and save:
          - the original clean COI input(s)
          - the separated outputs (either single COI head or all sources)
        This helps inspect a small number of separation examples for the clean condition.
        """
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
                waveform = self._load_audio(row.filename)
                if save_dir is not None and idx in sample_choices:
                    # save original clean COI (with suffix indicating selection index)
                    try:
                        k = list(sample_choices).index(idx)
                        torchaudio.save(
                            str(save_dir / f"clean_coi_{k}.wav"),
                            waveform.unsqueeze(0).cpu(),
                            self.sample_rate,
                        )
                    except Exception:
                        # don't fail validation on save errors
                        print(
                            f"Warning: failed to save clean_coi for {row.filename}",
                            file=sys.stderr,
                        )

                if use_separation:
                    separated = self._separate(waveform)
                    pred, conf = self._classify_separated(separated)
                    # Compute signal metrics: compare COI head output to original
                    coi_est = (
                        separated
                        if separated.dim() == 1
                        else separated[self._get_coi_head_index()]
                    )
                    si_snr, sdr, si_sdr = self._compute_signal_metrics(
                        waveform, coi_est
                    )
                    si_snr_scores.append(si_snr)
                    sdr_scores.append(sdr)
                    si_sdr_scores.append(si_sdr)

                    # Save separated outputs for the chosen sample(s)
                    if save_dir is not None and idx in sample_choices:
                        try:
                            k = list(sample_choices).index(idx)
                            if separated.dim() == 1:
                                torchaudio.save(
                                    str(save_dir / f"separated_coi_est_{k}.wav"),
                                    separated.unsqueeze(0).cpu(),
                                    self.sample_rate,
                                )
                            else:
                                # save all sources and the COI head separately with index suffixes
                                for s in range(separated.shape[0]):
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
                        except Exception:
                            print(
                                f"Warning: failed to save separated outputs for {row.filename}",
                                file=sys.stderr,
                            )
                else:
                    pred, conf = self._classify(waveform)
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
                waveform = self._load_audio(row.filename)
                if use_separation:
                    separated = self._separate(waveform)
                    pred, conf = self._classify_separated(separated)
                else:
                    pred, conf = self._classify(waveform)
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
    ) -> ClassificationMetrics:
        """Validate on mixtures at random SNR (COI+BG) and clean background (BG only).

        If `save_examples_dir` is provided, pick up to `save_n_examples` random COI
        mixtures and save:
          - the original clean COI and BG inputs used to create those mixtures
          - the created mixture file(s)
          - the separated outputs (either single COI head or all sources)
        """
        y_true, y_pred, y_scores = [], [], []
        raw_labels = []
        si_snr_scores, sdr_scores, si_sdr_scores = [], [], []
        actual_snrs: List[float] = []
        desc = "Mixtures (sep+cls)" if use_separation else "Mixtures (cls only)"
        bg_files = df_bg["filename"].tolist()

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
                coi = self._normalize(self._load_audio(row.filename))
                # pick a random background file for this mixture
                bg_idx = np.random.randint(len(bg_files))
                bg_file = bg_files[bg_idx]
                bg = self._normalize(self._load_audio(bg_file))
                snr = np.random.uniform(*snr_range)
                mixture, actual_snr = self._create_mixture(coi, bg, snr)
                actual_snrs.append(actual_snr)

                # Save chosen example inputs/mixture (for any selected indices)
                if save_dir is not None and idx in sample_choices:
                    try:
                        k = list(sample_choices).index(idx)
                        torchaudio.save(
                            str(save_dir / f"mixture_coi_clean_{k}.wav"),
                            coi.unsqueeze(0).cpu(),
                            self.sample_rate,
                        )
                        torchaudio.save(
                            str(save_dir / f"mixture_bg_clean_{k}.wav"),
                            bg.unsqueeze(0).cpu(),
                            self.sample_rate,
                        )
                        torchaudio.save(
                            str(save_dir / f"mixture_created_{k}.wav"),
                            mixture.unsqueeze(0).cpu(),
                            self.sample_rate,
                        )
                    except Exception:
                        print(
                            f"Warning: failed to save mixture example for {row.filename}",
                            file=sys.stderr,
                        )

                if use_separation:
                    separated = self._separate(mixture)
                    pred, conf = self._classify_separated(separated)
                    # Compute signal metrics: compare COI head output to clean COI
                    coi_est = (
                        separated
                        if separated.dim() == 1
                        else separated[self._get_coi_head_index()]
                    )
                    si_snr_val, sdr_val, si_sdr_val = self._compute_signal_metrics(
                        coi, coi_est
                    )
                    si_snr_scores.append(si_snr_val)
                    sdr_scores.append(sdr_val)
                    si_sdr_scores.append(si_sdr_val)

                    # Save separated outputs for the chosen mixture sample(s)
                    if save_dir is not None and idx in sample_choices:
                        try:
                            k = list(sample_choices).index(idx)
                            if separated.dim() == 1:
                                torchaudio.save(
                                    str(
                                        save_dir / f"mixture_separated_coi_est_{k}.wav"
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
                                        save_dir / f"mixture_separated_coi_head_{k}.wav"
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
                    pred, conf = self._classify(mixture)
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
                waveform = self._load_audio(row.filename)
                if use_separation:
                    separated = self._separate(waveform)
                    pred, conf = self._classify_separated(separated)
                else:
                    pred, conf = self._classify(waveform)
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
        metrics.compute(np.array(y_true), np.array(y_pred), np.array(y_scores))
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
    ) -> Dict[str, ClassificationMetrics]:
        """Run full validation suite.

        Args:
            split: Dataset split to evaluate on.
            snr_range: (min, max) SNR in dB for mixture creation.
            data_csv: Path to dataset CSV. Falls back to self.DATA_CSV.
            output_dir: Directory to save JSON results.
            seed: Random seed for reproducibility of mixture creation.
            save_examples_dir: If provided, save example audio for up to `save_n_examples`
                random separations in the clean separation run and up to `save_n_examples`
                random mixtures in the mixture separation run. Separate subdirectories will be created:
                - <save_examples_dir>/clean_sep
                - <save_examples_dir>/mixture_sep
            save_n_examples: Number of random examples to save for each of clean and
                mixture tests (1..N). Defaults to 1.
        """
        # Set seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"Random seed set to {seed}")

        csv_path = data_csv or self.DATA_CSV
        df = pd.read_csv(csv_path)
        df_split = df[df["split"] == split]
        df_coi = df_split[df_split["label"] == 1].reset_index(drop=True)
        df_bg = df_split[df_split["label"] == 0].reset_index(drop=True)

        print(f"\n{'=' * 60}")
        print(
            f"Validation on {split.upper()} set: {len(df_coi)} COI, {len(df_bg)} background"
        )
        print(f"SNR range: {snr_range} dB")
        print(f"{'=' * 60}\n")
        # if the pipeline has recovered a list of target classes from the
        # separation checkpoint, surface it here so the user can verify what
        # semantic labels the model was trained to treat as COI.
        if getattr(self, "target_classes", None):
            print(f"Target classes: {self.target_classes}")

        results = {}

        # 1. Clean - classification only
        print("\n[1/4] Clean audio - classification only")
        results["clean_cls"] = self.validate_clean(df_coi, df_bg, use_separation=False)
        print(results["clean_cls"])

        # 2. Clean - separation + classification
        print("\n[2/4] Clean audio - separation + classification")
        clean_save_dir = (
            str(Path(save_examples_dir) / "clean_sep") if save_examples_dir else None
        )
        results["clean_sep_cls"] = self.validate_clean(
            df_coi,
            df_bg,
            use_separation=True,
            save_examples_dir=clean_save_dir,
            save_n_examples=save_n_examples,
        )
        print(results["clean_sep_cls"])

        # 3. Mixtures - classification only
        if len(df_bg) > 0:
            print(f"\n[3/4] Mixtures ({snr_range}dB) - classification only")
            results["mix_cls"] = self.validate_mixtures(
                df_coi, df_bg, snr_range, use_separation=False
            )
            print(results["mix_cls"])

            # 4. Mixtures - separation + classification
            print(f"\n[4/4] Mixtures ({snr_range}dB) - separation + classification")
            mix_save_dir = (
                str(Path(save_examples_dir) / "mixture_sep")
                if save_examples_dir
                else None
            )
            results["mix_sep_cls"] = self.validate_mixtures(
                df_coi,
                df_bg,
                snr_range,
                use_separation=True,
                save_examples_dir=mix_save_dir,
                save_n_examples=save_n_examples,
            )
            print(results["mix_sep_cls"])

        # Save results
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dict = {k: v.to_dict() for k, v in results.items()}
            # Add checkpoint paths to output
            results_dict["checkpoint_paths"] = {
                "separator": str(self.sep_checkpoint_path),
                "classifier": str(self.cls_checkpoint_path),
            }
            with open(Path(output_dir) / f"results_{ts}.json", "w") as f:
                json.dump(results_dict, f, indent=2)

        return results


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
        PROJECT_ROOT / "src/models/clapsep/checkpoint/CLAPSep/model/best_model.ckpt"
    )
    CLS_WEIGHTS = (
        PROJECT_ROOT
        / "src/validation_functions/plane_clasifier/checkpoints/final_model.weights.h5"
    )
    # Trains
    DATA_CSV = (
        PROJECT_ROOT
        / "src/models/sudormrf/checkpoints/20260219_124144/separation_dataset.csv"
    )

    # planes/ "src/models/sudormrf/checkpoints/20260129_113352/separation_dataset.csv"

    BASE_PATH = PROJECT_ROOT.parent / "datasets"  # For converting Windows paths in CSV
    # ==============================================

    pipeline = ValidationPipeline(base_path=BASE_PATH)
    pipeline.load_models(
        sep_checkpoint=SEP_CHECKPOINT, cls_weights=CLS_WEIGHTS, use_clapsep=True
    )
    pipeline.run(
        split="test",
        snr_range=(-5, 5),
        data_csv=DATA_CSV,
        output_dir="./validation_results",
        save_examples_dir=f"./validation_examples_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        save_n_examples=2,
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
