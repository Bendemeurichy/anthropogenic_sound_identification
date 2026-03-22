"""Experiment that gradually adds artificial noise to COI samples and measures how
separation affects classification robustness.

This script uses generated white noise instead of background samples from a dataset,
with a noise range that is interpolated between for each sample.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

# Ensure project root imports work when launched from different cwd's.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation_functions.test_pipeline import (  # noqa: E402
    ValidationPipeline,
    _filter_contaminated_backgrounds,
    _is_coi_label,
)


@dataclass
class SNRStats:
    snr_db: float
    n_samples: int
    cls_only_recall: float
    sep_cls_recall: float
    cls_only_mean_conf: float
    sep_cls_mean_conf: float
    cls_only_std_conf: float
    sep_cls_std_conf: float
    mean_actual_snr_db: float


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _extract_coi_df(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows considered COI in a robust way."""
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    out = df.copy()

    # Prefer numeric label if available.
    if pd.api.types.is_numeric_dtype(out["label"]):
        out["label"] = out["label"].fillna(0)
        coi = out[out["label"] == 1].copy()
        return coi.reset_index(drop=True)

    # String fallback: use the canonical COI synonym set from test_pipeline so
    # both experiments operate on the same definition of "COI sample".
    base_series = out["orig_label"] if "orig_label" in out.columns else out["label"]
    out["label_bin"] = base_series.apply(lambda x: 1 if _is_coi_label(x) else 0)
    coi = out[out["label_bin"] == 1].copy()
    return coi.reset_index(drop=True)


def _extract_bg_df(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows considered background (non-COI) in a robust way.
    
    This function mirrors _extract_coi_df but for background samples.
    Useful when the experiment needs real background samples instead of
    synthetic white noise.
    """
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    out = df.copy()

    # Prefer numeric label if available.
    if pd.api.types.is_numeric_dtype(out["label"]):
        out["label"] = out["label"].fillna(0)
        bg = out[out["label"] == 0].copy()
        return bg.reset_index(drop=True)

    # String fallback: use the canonical COI synonym set from test_pipeline so
    # both experiments operate on the same definition of "background sample".
    base_series = out["orig_label"] if "orig_label" in out.columns else out["label"]
    out["label_bin"] = base_series.apply(lambda x: 1 if _is_coi_label(x) else 0)
    bg = out[out["label_bin"] == 0].copy()
    return bg.reset_index(drop=True)



def run_noise_increase_experiment(
    pipeline: ValidationPipeline,
    df_coi: pd.DataFrame,
    snr_levels_db: List[float],
    max_samples: int | None = None,
    seed: int = 42,
) -> Dict[str, object]:
    """Run robustness sweep over SNR levels using artificial noise.

    Positive-only experiment:
      - target class is always present (y_true=1).
      - metric of interest is recall (fraction predicted positive).
      - noise is generated as white noise and scaled to match target SNR.
    """
    if len(df_coi) == 0:
        raise ValueError("No COI samples available for experiment.")

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    coi_eval = df_coi.copy()
    if max_samples is not None and max_samples > 0 and len(coi_eval) > max_samples:
        coi_eval = coi_eval.sample(n=max_samples, random_state=seed).reset_index(
            drop=True
        )

    per_snr: List[SNRStats] = []

    for snr_db in snr_levels_db:
        cls_preds: List[int] = []
        sep_preds: List[int] = []
        cls_conf: List[float] = []
        sep_conf: List[float] = []
        actual_snrs: List[float] = []

        for row in coi_eval.itertuples(index=False):
            coi_full = pipeline._load_labeled_audio(  # noqa: SLF001
                row.filename,
                getattr(row, "start_time", None),
                getattr(row, "end_time", None),
            )

            coi_segments = pipeline._split_into_segments(coi_full)  # noqa: SLF001

            seg_cls_pred, seg_sep_pred = [], []
            seg_cls_conf, seg_sep_conf = [], []
            seg_actual_snr = []

            for coi_seg in coi_segments:
                # Generate artificial white noise of the same shape
                noise_seg = torch.randn_like(coi_seg)

                coi_n = pipeline._normalize(coi_seg)  # noqa: SLF001
                noise_n = pipeline._normalize(noise_seg)  # noqa: SLF001

                # Create mixture using artificial noise
                mixture, actual_snr = pipeline._create_mixture(  # noqa: SLF001
                    coi_n, noise_n, float(snr_db)
                )
                seg_actual_snr.append(actual_snr)

                p_cls, c_cls = pipeline._classify(mixture)  # noqa: SLF001
                seg_cls_pred.append(p_cls)
                seg_cls_conf.append(_safe_float(c_cls, 0.0))

                separated = pipeline._separate(mixture)  # noqa: SLF001
                p_sep, c_sep = pipeline._classify_separated(separated)  # noqa: SLF001
                seg_sep_pred.append(p_sep)
                seg_sep_conf.append(_safe_float(c_sep, 0.0))

            # Recording-level aggregation: any() positive over segments.
            rec_cls_pred = 1 if any(p == 1 for p in seg_cls_pred) else 0
            rec_sep_pred = 1 if any(p == 1 for p in seg_sep_pred) else 0
            rec_cls_conf = float(max(seg_cls_conf)) if seg_cls_conf else 0.0
            rec_sep_conf = float(max(seg_sep_conf)) if seg_sep_conf else 0.0
            rec_snr = (
                float(np.mean(seg_actual_snr)) if seg_actual_snr else float(snr_db)
            )

            cls_preds.append(rec_cls_pred)
            sep_preds.append(rec_sep_pred)
            cls_conf.append(rec_cls_conf)
            sep_conf.append(rec_sep_conf)
            actual_snrs.append(rec_snr)

        n = len(coi_eval)
        cls_recall = float(np.mean(np.array(cls_preds) == 1)) if n else 0.0
        sep_recall = float(np.mean(np.array(sep_preds) == 1)) if n else 0.0

        per_snr.append(
            SNRStats(
                snr_db=float(snr_db),
                n_samples=n,
                cls_only_recall=cls_recall,
                sep_cls_recall=sep_recall,
                cls_only_mean_conf=float(np.mean(cls_conf)) if cls_conf else 0.0,
                sep_cls_mean_conf=float(np.mean(sep_conf)) if sep_conf else 0.0,
                cls_only_std_conf=float(np.std(cls_conf)) if cls_conf else 0.0,
                sep_cls_std_conf=float(np.std(sep_conf)) if sep_conf else 0.0,
                mean_actual_snr_db=float(np.mean(actual_snrs))
                if actual_snrs
                else float(snr_db),
            )
        )

    # Compact summary
    summary = {
        "best_sep_gain_recall": float(
            max((s.sep_cls_recall - s.cls_only_recall) for s in per_snr)
            if per_snr
            else 0.0
        ),
        "mean_sep_gain_recall": float(
            np.mean([s.sep_cls_recall - s.cls_only_recall for s in per_snr])
            if per_snr
            else 0.0
        ),
    }

    return {
        "snr_results": [s.__dict__ for s in per_snr],
        "summary": summary,
    }


def main() -> None:
    # ================== HARD-CODED CONFIG ==================
    BASE_PATH = str(PROJECT_ROOT.parent / "datasets")
    
    # ---- Model selection ----
    # Set USE_TUSS = True to test TUSS model instead of SudoRM-RF
    USE_TUSS = False
    
    if USE_TUSS:
        # TUSS model configuration
        # Update these paths to point to your trained TUSS checkpoint
        DATA_CSV = str(
            PROJECT_ROOT
            / "src/models/tuss/checkpoints/YOUR_CHECKPOINT_DIR/separation_dataset.csv"
        )
        SEP_CHECKPOINT = str(
            PROJECT_ROOT / "src/models/tuss/checkpoints/YOUR_CHECKPOINT_DIR"
        )
        # TUSS prompts (should match training config)
        TUSS_COI_PROMPT = "airplane"
        TUSS_BG_PROMPT = "background"
    else:
        # SudoRM-RF model configuration (default)
        DATA_CSV = str(
            PROJECT_ROOT
            / "src/models/sudormrf/checkpoints/20260219_124144/separation_dataset.csv"
        )
        SEP_CHECKPOINT = str(PROJECT_ROOT / "src/models/sudormrf/checkpoints/best_model.pt")

    # Classifier checkpoint (shared across all separation models)
    CLS_WEIGHTS = str(
        PROJECT_ROOT
        / "src/validation_functions/plane_clasifier/checkpoints/final_model.weights.h5"
    )

    # Dataset filtering
    SPLIT = "test"
    EXCLUDE_DATASETS = ["risoux_test"]  # keep independent set out of this experiment

    # Experiment sweep
    # Linear interpolation between 25 and -10 dB with 8 steps
    SNR_START = 25
    SNR_END = -10
    NUM_STEPS = 8
    SNR_LEVELS_DB = list(np.linspace(SNR_START, SNR_END, NUM_STEPS))

    MAX_SAMPLES = 200
    SEED = 42

    # Output
    OUTPUT_DIR = Path("./noise_increase_results")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # ======================================================

    print("Initializing pipeline...")
    pipeline = ValidationPipeline(base_path=BASE_PATH)
    
    # Load models with appropriate configuration
    if USE_TUSS:
        print(f"Using TUSS model with prompts: COI='{TUSS_COI_PROMPT}', BG='{TUSS_BG_PROMPT}'")
        pipeline.load_models(
            sep_checkpoint=SEP_CHECKPOINT,
            cls_weights=CLS_WEIGHTS,
            use_tuss=True,
            tuss_coi_prompt=TUSS_COI_PROMPT,
            tuss_bg_prompt=TUSS_BG_PROMPT,
            use_pann=False,
            use_ast=False,
        )
    else:
        print("Using SudoRM-RF model")
        pipeline.load_models(
            sep_checkpoint=SEP_CHECKPOINT,
            cls_weights=CLS_WEIGHTS,
            use_clapsep=False,
            use_tuss=False,
            use_pann=False,
            use_ast=False,
        )

    print("Loading metadata CSV...")
    df = pd.read_csv(DATA_CSV)
    if EXCLUDE_DATASETS and "dataset" in df.columns:
        df = df[~df["dataset"].isin(EXCLUDE_DATASETS)].copy()
    if "split" in df.columns:
        df = df[df["split"] == SPLIT].copy()

    # Extract COI and background DataFrames
    df_coi = _extract_coi_df(df)
    df_bg = _extract_bg_df(df)
    
    # Apply contamination filtering to background samples
    # (This ensures consistency with test_pipeline.py, even though this experiment
    # currently uses synthetic white noise instead of real background samples)
    df_bg_clean, n_contaminated = _filter_contaminated_backgrounds(df_bg, verbose=True)

    print(f"\n{'=' * 60}")
    print(f"Dataset Statistics:")
    print(f"  COI samples:        {len(df_coi)}")
    print(f"  Background samples: {len(df_bg)} total, {len(df_bg_clean)} clean")
    if n_contaminated > 0:
        print(f"  Contaminated removed: {n_contaminated}")
    print(f"  SNR sweep:          {SNR_LEVELS_DB}")
    print(f"{'=' * 60}\n")

    results = run_noise_increase_experiment(
        pipeline=pipeline,
        df_coi=df_coi,
        snr_levels_db=SNR_LEVELS_DB,
        max_samples=MAX_SAMPLES,
        seed=SEED,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "tuss" if USE_TUSS else "sudormrf"
    out_json = OUTPUT_DIR / f"noise_increase_results_artificial_{model_name}_{ts}.json"
    out_csv = OUTPUT_DIR / f"noise_increase_results_artificial_{model_name}_{ts}.csv"

    payload = {
        "config": {
            "base_path": BASE_PATH,
            "data_csv": DATA_CSV,
            "sep_checkpoint": SEP_CHECKPOINT,
            "cls_weights": CLS_WEIGHTS,
            "model_type": "TUSS" if USE_TUSS else "SudoRM-RF",
            "split": SPLIT,
            "exclude_datasets": EXCLUDE_DATASETS,
            "snr_levels_db": SNR_LEVELS_DB,
            "max_samples": MAX_SAMPLES,
            "seed": SEED,
            "noise_type": "artificial_white_noise",
        },
        "dataset_stats": {
            "n_coi_samples": len(df_coi),
            "n_background_total": len(df_bg),
            "n_background_clean": len(df_bg_clean),
            "n_contaminated_removed": n_contaminated,
        },
        **results,
    }
    
    # Add TUSS-specific config if applicable
    if USE_TUSS:
        payload["config"]["tuss_coi_prompt"] = TUSS_COI_PROMPT
        payload["config"]["tuss_bg_prompt"] = TUSS_BG_PROMPT

    with out_json.open("w") as f:
        json.dump(payload, f, indent=2)

    pd.DataFrame(results["snr_results"]).to_csv(out_csv, index=False)

    print(f"Saved JSON: {out_json}")
    print(f"Saved CSV:  {out_csv}")


if __name__ == "__main__":
    main()
