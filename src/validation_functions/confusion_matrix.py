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
        orig_len = waveform.shape[0]

        # If waveform is the expected segment length, run as before
        if orig_len <= self.segment_samples:
            mean, std = waveform.mean(), waveform.std() + 1e-8
            x = ((waveform - mean) / std).unsqueeze(0).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                estimates = self.separator.model(x)

            # estimates shape: (1, n_sources, T)
            sources = estimates[0].cpu() * std + mean
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

            mean, std = chunk.mean(), chunk.std() + 1e-8
            x = ((chunk - mean) / std).unsqueeze(0).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                estimates = self.separator.model(x)

            # estimates[0].cpu() -> (n_sources, segment_samples)
            src_chunk_all = estimates[0].cpu() * std + mean

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
        self, df_coi: pd.DataFrame, df_bg: pd.DataFrame, use_separation: bool
    ) -> ClassificationMetrics:
        """Validate on clean (unmixed) audio - both COI and background."""
        y_true, y_pred, y_scores = [], [], []
        desc = "Clean (sep+cls)" if use_separation else "Clean (cls only)"

        # Process COI samples (label=1)
        for _, row in tqdm(df_coi.iterrows(), total=len(df_coi), desc=f"{desc} - COI"):
            try:
                waveform = self._load_audio(row["filename"])
                if use_separation:
                    separated = self._separate(waveform)
                    # If multiple sources returned, pick the source with highest
                    # plane-confidence from the classifier
                    if separated.dim() == 1:
                        pred, conf = self._classify(separated)
                    else:
                        best_conf = -1.0
                        best_pred = 0
                        for s_idx in range(separated.shape[0]):
                            p, c = self._classify(separated[s_idx])
                            if c > best_conf:
                                best_conf = c
                                best_pred = p
                        pred, conf = best_pred, best_conf
                else:
                    pred, conf = self._classify(waveform)
                y_true.append(1)
                y_pred.append(pred)
                y_scores.append(conf)
            except Exception as e:
                print(f"Error: {row['filename']}: {e}")

        # Process background samples (label=0)
        for _, row in tqdm(df_bg.iterrows(), total=len(df_bg), desc=f"{desc} - BG"):
            try:
                waveform = self._load_audio(row["filename"])
                if use_separation:
                    separated = self._separate(waveform)
                    if separated.dim() == 1:
                        pred, conf = self._classify(separated)
                    else:
                        best_conf = -1.0
                        best_pred = 0
                        for s_idx in range(separated.shape[0]):
                            p, c = self._classify(separated[s_idx])
                            if c > best_conf:
                                best_conf = c
                                best_pred = p
                        pred, conf = best_pred, best_conf
                else:
                    pred, conf = self._classify(waveform)
                y_true.append(0)
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
        """Validate on mixtures at random SNR (COI+BG) and clean background (BG only)."""
        y_true, y_pred, y_scores = [], [], []
        desc = "Mixtures (sep+cls)" if use_separation else "Mixtures (cls only)"
        bg_files = df_bg["filename"].tolist()

        # Process COI + background mixtures (label=1)
        for _, row in tqdm(
            df_coi.iterrows(), total=len(df_coi), desc=f"{desc} - COI+BG"
        ):
            try:
                coi = self._normalize(self._load_audio(row["filename"]))
                bg = self._normalize(
                    self._load_audio(bg_files[np.random.randint(len(bg_files))])
                )
                snr = np.random.uniform(*snr_range)
                mixture = self._create_mixture(coi, bg, snr)
                if use_separation:
                    separated = self._separate(mixture)
                    if separated.dim() == 1:
                        pred, conf = self._classify(separated)
                    else:
                        best_conf = -1.0
                        best_pred = 0
                        for s_idx in range(separated.shape[0]):
                            p, c = self._classify(separated[s_idx])
                            if c > best_conf:
                                best_conf = c
                                best_pred = p
                        pred, conf = best_pred, best_conf
                else:
                    pred, conf = self._classify(mixture)
                y_true.append(1)
                y_pred.append(pred)
                y_scores.append(conf)
            except Exception as e:
                print(f"Error: {e}")

        # Process background-only samples (label=0)
        for _, row in tqdm(
            df_bg.iterrows(), total=len(df_bg), desc=f"{desc} - BG only"
        ):
            try:
                waveform = self._load_audio(row["filename"])
                if use_separation:
                    separated = self._separate(waveform)
                    if separated.dim() == 1:
                        pred, conf = self._classify(separated)
                    else:
                        best_conf = -1.0
                        best_pred = 0
                        for s_idx in range(separated.shape[0]):
                            p, c = self._classify(separated[s_idx])
                            if c > best_conf:
                                best_conf = c
                                best_pred = p
                        pred, conf = best_pred, best_conf
                else:
                    pred, conf = self._classify(waveform)
                y_true.append(0)
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
        results["clean_cls"] = self.validate_clean(df_coi, df_bg, use_separation=False)
        print(results["clean_cls"])

        # 2. Clean - separation + classification
        print("\n[2/4] Clean audio - separation + classification")
        results["clean_sep_cls"] = self.validate_clean(
            df_coi, df_bg, use_separation=True
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

    mixture = pipeline._create_mixture(src_n, noise_n, snr_db)

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
    # main()
    import random

    example_src = Path("./ho6sg-47RD0.wav")
    example_noise = Path("./6PQQPzEhCjM.wav")

    if not example_src.exists() or not example_noise.exists():
        missing = []
        if not example_src.exists():
            missing.append(str(example_src))
        if not example_noise.exists():
            missing.append(str(example_noise))
        print("Example WAV files not found; skipping demo. Missing:", *missing)
    else:
        demo_two_wav_separation(
            src_path=str(example_src),
            noise_path=str(example_noise),
            snr_db=random.uniform(-5, 5),
            sep_checkpoint=PROJECT_ROOT
            / "src/models/sudormrf/checkpoints/20251215_234806/best_model.pt",
            out_dir="./separation_output_demo",
        )
