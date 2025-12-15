"""
Validation module for separation + classification pipeline.
Computes confusion matrix, precision, recall, F1-score, and other metrics.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent / "plane_clasifier"))

from src.models.sudormrf.inference import SeparationInference
from src.validation_functions.plane_clasifier.inference import PlaneClassifierInference
from src.validation_functions.plane_clasifier.config import TrainingConfig

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
    predictions: List[int] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)

    def compute(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray = None
    ):
        """Compute all metrics from predictions."""
        self.n_samples = len(y_true)
        self.labels = y_true.tolist()
        self.predictions = y_pred.tolist()
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

    def __str__(self):
        return f"""
{'='*50}
Confusion Matrix:  TN={self.true_negatives}  FP={self.false_positives}
                   FN={self.false_negatives}  TP={self.true_positives}

Accuracy:  {self.accuracy:.4f}    Precision: {self.precision:.4f}
Recall:    {self.recall:.4f}    F1-Score:  {self.f1_score:.4f}
Specificity: {self.specificity:.4f}  Balanced Acc: {self.balanced_accuracy:.4f}
MCC: {self.matthews_corrcoef:.4f}
{'='*50}"""

    def to_dict(self):
        return {
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
        self.segment_samples = int(self.sample_rate * self.segment_length)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.separator = None
        self.classifier = None
        self._resamplers = {}

    def load_models(self, sep_checkpoint: str = None, cls_weights: str = None):
        """Load separation and classification models."""
        sep_path = sep_checkpoint or self.SEP_CHECKPOINT
        cls_path = cls_weights or self.CLS_WEIGHTS

        print(f"Loading separation model from {sep_path}")
        self.separator = SeparationInference.from_checkpoint(
            sep_path, device=self.device
        )

        print(f"Loading classification model from {cls_path}")
        config = TrainingConfig(
            sample_rate=self.sample_rate, audio_duration=self.segment_length
        )
        self.classifier = PlaneClassifierInference(cls_path, config)

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
        """Run separation model. Returns COI source."""
        mean, std = waveform.mean(), waveform.std() + 1e-8
        x = ((waveform - mean) / std).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            estimates = self.separator.model(x)

        sources = estimates[0].cpu() * std + mean
        return sources[0]

    def _classify(self, waveform: torch.Tensor) -> Tuple[int, float]:
        """Run classification. Returns (prediction, confidence)."""
        result = self.classifier.predict_waveform(waveform.numpy())
        pred = 1 if result["prediction"] == "plane" else 0
        return pred, result["confidence"]

    def _create_mixture(
        self, source: torch.Tensor, noise: torch.Tensor, snr_db: float
    ) -> torch.Tensor:
        """Mix source and noise at given SNR."""
        source_power = torch.mean(source**2) + 1e-8
        noise_power = torch.mean(noise**2) + 1e-8
        scale = torch.sqrt(source_power / (10 ** (snr_db / 10) * noise_power))
        return source + noise * scale

    def _normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        return (waveform - waveform.mean()) / (waveform.std() + 1e-8)

    def validate_clean(
        self, df_coi: pd.DataFrame, use_separation: bool
    ) -> ClassificationMetrics:
        """Validate on clean (unmixed) COI audio."""
        y_true, y_pred, y_scores = [], [], []
        desc = "Clean (sep+cls)" if use_separation else "Clean (cls only)"

        for _, row in tqdm(df_coi.iterrows(), total=len(df_coi), desc=desc):
            try:
                waveform = self._load_audio(row["filename"])
                if use_separation:
                    waveform = self._separate(waveform)
                pred, conf = self._classify(waveform)
                y_true.append(1)
                y_pred.append(pred)
                y_scores.append(conf)
            except Exception as e:
                print(f"Error: {row['filename']}: {e}")

        metrics = ClassificationMetrics()
        metrics.compute(np.array(y_true), np.array(y_pred), np.array(y_scores))
        return metrics

    def validate_mixtures(
        self,
        df_coi: pd.DataFrame,
        df_bg: pd.DataFrame,
        snr_range: Tuple[float, float],
        use_separation: bool,
    ) -> ClassificationMetrics:
        """Validate on mixtures at random SNR."""
        y_true, y_pred, y_scores = [], [], []
        desc = "Mixtures (sep+cls)" if use_separation else "Mixtures (cls only)"
        bg_files = df_bg["filename"].tolist()

        for _, row in tqdm(df_coi.iterrows(), total=len(df_coi), desc=desc):
            try:
                coi = self._normalize(self._load_audio(row["filename"]))
                bg = self._normalize(
                    self._load_audio(bg_files[np.random.randint(len(bg_files))])
                )
                snr = np.random.uniform(*snr_range)
                mixture = self._create_mixture(coi, bg, snr)

                if use_separation:
                    mixture = self._separate(mixture)
                pred, conf = self._classify(mixture)
                y_true.append(1)
                y_pred.append(pred)
                y_scores.append(conf)
            except Exception as e:
                print(f"Error: {e}")

        metrics = ClassificationMetrics()
        metrics.compute(np.array(y_true), np.array(y_pred), np.array(y_scores))
        return metrics

    def run(
        self,
        split: str = "test",
        snr_range: Tuple[float, float] = (-5, 5),
        data_csv: str = None,
        output_dir: str = None,
    ) -> Dict[str, ClassificationMetrics]:
        """Run full validation suite."""
        csv_path = data_csv or self.DATA_CSV
        df = pd.read_csv(csv_path)
        df_split = df[df["split"] == split]
        df_coi = df_split[df_split["label"] == 1].reset_index(drop=True)
        df_bg = df_split[df_split["label"] == 0].reset_index(drop=True)

        print(f"\n{'='*60}")
        print(
            f"Validation on {split.upper()} set: {len(df_coi)} COI, {len(df_bg)} background"
        )
        print(f"SNR range: {snr_range} dB")
        print(f"{'='*60}\n")

        results = {}

        # 1. Clean - classification only
        print("\n[1/4] Clean audio - classification only")
        results["clean_cls"] = self.validate_clean(df_coi, use_separation=False)
        print(results["clean_cls"])

        # 2. Clean - separation + classification
        print("\n[2/4] Clean audio - separation + classification")
        results["clean_sep_cls"] = self.validate_clean(df_coi, use_separation=True)
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
            results["mix_sep_cls"] = self.validate_mixtures(
                df_coi, df_bg, snr_range, use_separation=True
            )
            print(results["mix_sep_cls"])

        # Save results
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(Path(output_dir) / f"results_{ts}.json", "w") as f:
                json.dump({k: v.to_dict() for k, v in results.items()}, f, indent=2)

        return results


def main():
    # ============ CONFIGURE PATHS HERE ============
    SEP_CHECKPOINT = PROJECT_ROOT / "src/models/sudormrf/checkpoints/best_model.pt"
    CLS_WEIGHTS = (
        PROJECT_ROOT
        / "src/validation_functions/plane_clasifier/results/checkpoints/final_model.weights.h5"
    )
    DATA_CSV = PROJECT_ROOT / "src/models/sudormrf/checkpoints/separation_dataset.csv"
    BASE_PATH = "/path/to/your/datasets"  # For converting Windows paths in CSV
    # ==============================================

    pipeline = ValidationPipeline(base_path=BASE_PATH)
    pipeline.load_models(sep_checkpoint=SEP_CHECKPOINT, cls_weights=CLS_WEIGHTS)
    pipeline.run(
        split="test",
        snr_range=(-5, 5),
        data_csv=DATA_CSV,
        output_dir="./validation_results",
    )


if __name__ == "__main__":
    main()
