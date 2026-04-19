"""Experiment that gradually adds artificial noise to COI samples and measures
separation's effect on energy preservation (RMS and SEL).

This script measures how well separation preserves COI energy at different
SNR levels by comparing clean baseline energy metrics with noisy separation results.
Classification is not used since classifiers are not robust to the extreme noise
levels tested here.
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
from src.validation_functions.demo_separation import compute_energy_metrics  # noqa: E402


@dataclass
class SegmentEnergyMetrics:
    """Per-segment energy metrics for detailed analysis and plotting.
    
    All energy values are in dBFS (dB relative to full scale).
    Delta values show change from clean baseline (positive = more energy).
    """
    # Identifiers
    filename: str
    recording_idx: int
    segment_idx: int
    snr_db: float | None  # None for clean baseline
    
    # Clean baseline (no noise, original signal)
    original_clean_rms_db: float
    original_clean_sel_db: float
    separated_clean_rms_db: float
    separated_clean_sel_db: float
    
    # Noisy experiment (only populated for SNR sweeps)
    mixture_rms_db: float | None = None
    mixture_sel_db: float | None = None
    separated_noisy_rms_db: float | None = None
    separated_noisy_sel_db: float | None = None
    
    # Energy preservation metrics (dB differences from clean baseline)
    clean_sep_rms_delta: float = 0.0  # separated_clean - original_clean
    clean_sep_sel_delta: float = 0.0
    noisy_sep_rms_delta: float | None = None  # separated_noisy - original_clean
    noisy_sep_sel_delta: float | None = None
    
    actual_snr_db: float | None = None  # Measured SNR for verification


@dataclass
class CleanBaselineStats:
    """Aggregate statistics for clean baseline (no noise)."""
    n_samples: int
    n_segments: int
    mean_original_rms_db: float
    std_original_rms_db: float
    mean_original_sel_db: float
    std_original_sel_db: float
    mean_separated_rms_db: float
    std_separated_rms_db: float
    mean_separated_sel_db: float
    std_separated_sel_db: float
    # Energy preservation on clean signals
    mean_rms_preservation_db: float  # How much RMS changes after separation
    mean_sel_preservation_db: float  # How much SEL changes after separation


@dataclass
class SNREnergyStats:
    """Aggregate energy statistics at a specific SNR level."""
    snr_db: float
    n_segments: int
    
    # Mean energy metrics
    mean_mixture_rms_db: float
    mean_mixture_sel_db: float
    mean_separated_noisy_rms_db: float
    mean_separated_noisy_sel_db: float
    
    # Energy degradation from clean baseline
    mean_rms_degradation_db: float  # How much RMS drops from clean baseline
    mean_sel_degradation_db: float  # How much SEL drops from clean baseline
    std_rms_degradation_db: float
    std_sel_degradation_db: float
    
    # Actual measured SNR
    mean_actual_snr_db: float


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _coi_source_from_separated(
    pipeline: ValidationPipeline, separated: torch.Tensor
) -> torch.Tensor:
    """Extract the COI source from a separator output tensor."""
    if separated.dim() == 1:
        return separated
    return separated[pipeline._get_coi_head_index()]


def _extract_coi_df(df: pd.DataFrame, coi_synonyms: set = None) -> pd.DataFrame:
    """Keep rows considered COI in a robust way.
    
    Args:
        df: DataFrame with label/orig_label columns
        coi_synonyms: Set of COI synonyms to use for label matching.
            If None, uses default from test_pipeline (_is_coi_label default).
    
    Returns:
        DataFrame containing only COI samples (label=1)
    """
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
    out["label_bin"] = base_series.apply(
        lambda x: 1 if _is_coi_label(x, coi_synonyms) else 0
    )
    coi = out[out["label_bin"] == 1].copy()
    return coi.reset_index(drop=True)


def _extract_bg_df(df: pd.DataFrame, coi_synonyms: set = None) -> pd.DataFrame:
    """Keep rows considered background (non-COI) in a robust way.
    
    This function mirrors _extract_coi_df but for background samples.
    Useful when the experiment needs real background samples instead of
    synthetic white noise.
    
    Args:
        df: DataFrame with label/orig_label columns
        coi_synonyms: Set of COI synonyms to use for label matching.
            If None, uses default from test_pipeline (_is_coi_label default).
    
    Returns:
        DataFrame containing only background samples (label=0)
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
    out["label_bin"] = base_series.apply(
        lambda x: 1 if _is_coi_label(x, coi_synonyms) else 0
    )
    bg = out[out["label_bin"] == 0].copy()
    return bg.reset_index(drop=True)


def _compute_clean_baseline_energy(
    pipeline: ValidationPipeline,
    df_coi: pd.DataFrame,
    seed: int = 42,
) -> Tuple[CleanBaselineStats, Dict[Tuple[int, int], SegmentEnergyMetrics]]:
    """Compute baseline energy metrics on clean (noise-free) COI samples.
    
    This provides the reference point to measure energy degradation at different
    noise levels. Measures both original and separated clean signals.
    
    Args:
        pipeline: ValidationPipeline with loaded separator model
        df_coi: DataFrame of COI samples
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of:
            - CleanBaselineStats: Aggregated statistics
            - Dict mapping (rec_idx, seg_idx) to detailed SegmentEnergyMetrics
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    all_metrics: List[SegmentEnergyMetrics] = []
    baseline_map: Dict[Tuple[int, int], SegmentEnergyMetrics] = {}
    
    print(f"\nComputing clean baseline energy (no noise) on {len(df_coi)} samples...")
    
    for rec_idx, row in enumerate(df_coi.itertuples(index=False)):
        if rec_idx % 50 == 0:
            print(f"  Processing recording {rec_idx}/{len(df_coi)}...")
            
        coi_full = pipeline._load_labeled_audio(
            row.filename,
            getattr(row, "start_time", None),
            getattr(row, "end_time", None),
        )
        coi_segments = pipeline._split_into_segments(coi_full)
        
        for seg_idx, coi_seg in enumerate(coi_segments):
            # Use the same preprocessing as the RMS mixer in test_pipeline.py.
            coi_preprocessed = pipeline._prepare_rms_mixing_input(coi_seg)
            
            # Compute energy on original clean signal
            orig_metrics = compute_energy_metrics(coi_preprocessed, pipeline.sample_rate)
            
            # Separate clean signal (even though there's no noise to remove)
            separated = pipeline._separate(coi_preprocessed)
            
            # Compute energy on the COI output only.
            coi_est = _coi_source_from_separated(pipeline, separated)
            sep_metrics = compute_energy_metrics(coi_est, pipeline.sample_rate)
            
            # Calculate energy preservation
            rms_delta = sep_metrics["rms_db"] - orig_metrics["rms_db"]
            sel_delta = sep_metrics["sel_db"] - orig_metrics["sel_db"]
            
            # Create segment metrics
            seg_metrics = SegmentEnergyMetrics(
                filename=row.filename,
                recording_idx=rec_idx,
                segment_idx=seg_idx,
                snr_db=None,  # Clean baseline has no noise
                original_clean_rms_db=orig_metrics["rms_db"],
                original_clean_sel_db=orig_metrics["sel_db"],
                separated_clean_rms_db=sep_metrics["rms_db"],
                separated_clean_sel_db=sep_metrics["sel_db"],
                clean_sep_rms_delta=rms_delta,
                clean_sep_sel_delta=sel_delta,
            )
            
            all_metrics.append(seg_metrics)
            baseline_map[(rec_idx, seg_idx)] = seg_metrics
    
    # Compute aggregate statistics
    n_segs = len(all_metrics)
    
    orig_rms = [m.original_clean_rms_db for m in all_metrics]
    orig_sel = [m.original_clean_sel_db for m in all_metrics]
    sep_rms = [m.separated_clean_rms_db for m in all_metrics]
    sep_sel = [m.separated_clean_sel_db for m in all_metrics]
    rms_deltas = [m.clean_sep_rms_delta for m in all_metrics]
    sel_deltas = [m.clean_sep_sel_delta for m in all_metrics]
    
    baseline_stats = CleanBaselineStats(
        n_samples=len(df_coi),
        n_segments=n_segs,
        mean_original_rms_db=float(np.mean(orig_rms)),
        std_original_rms_db=float(np.std(orig_rms)),
        mean_original_sel_db=float(np.mean(orig_sel)),
        std_original_sel_db=float(np.std(orig_sel)),
        mean_separated_rms_db=float(np.mean(sep_rms)),
        std_separated_rms_db=float(np.std(sep_rms)),
        mean_separated_sel_db=float(np.mean(sep_sel)),
        std_separated_sel_db=float(np.std(sep_sel)),
        mean_rms_preservation_db=float(np.mean(rms_deltas)),
        mean_sel_preservation_db=float(np.mean(sel_deltas)),
    )
    
    print(f"\nClean Baseline Results ({n_segs} segments):")
    print(f"  Original: RMS={baseline_stats.mean_original_rms_db:.2f} dBFS, "
          f"SEL={baseline_stats.mean_original_sel_db:.2f} dBFS")
    print(f"  Separated: RMS={baseline_stats.mean_separated_rms_db:.2f} dBFS, "
          f"SEL={baseline_stats.mean_separated_sel_db:.2f} dBFS")
    print(f"  Preservation: RMS_delta={baseline_stats.mean_rms_preservation_db:+.2f} dB, "
          f"SEL_delta={baseline_stats.mean_sel_preservation_db:+.2f} dB")
    
    return baseline_stats, baseline_map


def run_noise_increase_experiment(
    pipeline: ValidationPipeline,
    df_coi: pd.DataFrame,
    baseline_map: Dict[Tuple[int, int], SegmentEnergyMetrics],
    snr_levels_db: List[float],
    seed: int = 42,
) -> Tuple[List[SegmentEnergyMetrics], List[SNREnergyStats]]:
    """Run robustness sweep over SNR levels measuring energy preservation.
    
    For each SNR level, mixes COI with white noise and measures how much energy
    is preserved after separation compared to the clean baseline.
    
    Args:
        pipeline: ValidationPipeline with loaded separator model
        df_coi: DataFrame of COI samples
        baseline_map: Dictionary mapping (rec_idx, seg_idx) to clean baseline metrics
        snr_levels_db: List of SNR levels to test (in dB)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of:
            - List of detailed SegmentEnergyMetrics for all segments and SNR levels
            - List of SNREnergyStats with aggregate statistics per SNR level
    """
    if len(df_coi) == 0:
        raise ValueError("No COI samples available for experiment.")

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Pre-generate noise for all segments to ensure consistency across SNR levels
    # Key: (recording_index, segment_index), Value: noise tensor
    print("\nPre-generating noise realizations for all segments...")
    noise_cache: Dict[Tuple[int, int], torch.Tensor] = {}
    
    for rec_idx, row in enumerate(df_coi.itertuples(index=False)):
        coi_full = pipeline._load_labeled_audio(
            row.filename,
            getattr(row, "start_time", None),
            getattr(row, "end_time", None),
        )
        coi_segments = pipeline._split_into_segments(coi_full)
        
        for seg_idx, coi_seg in enumerate(coi_segments):
            # Generate and cache noise for this segment
            noise_cache[(rec_idx, seg_idx)] = torch.randn_like(coi_seg)
    
    print(f"Generated {len(noise_cache)} noise realizations for {len(df_coi)} recordings")

    all_segment_metrics: List[SegmentEnergyMetrics] = []
    per_snr_stats: List[SNREnergyStats] = []

    for idx, snr_db in enumerate(snr_levels_db, 1):
        print(f"\n[{idx}/{len(snr_levels_db)}] Processing SNR = {snr_db:.1f} dB...")
        
        snr_segment_metrics: List[SegmentEnergyMetrics] = []

        for rec_idx, row in enumerate(df_coi.itertuples(index=False)):
            coi_full = pipeline._load_labeled_audio(
                row.filename,
                getattr(row, "start_time", None),
                getattr(row, "end_time", None),
            )

            coi_segments = pipeline._split_into_segments(coi_full)

            for seg_idx, coi_seg in enumerate(coi_segments):
                # Retrieve baseline metrics for comparison
                baseline = baseline_map[(rec_idx, seg_idx)]
                
                # Retrieve pre-generated noise for this segment
                # This ensures the SAME noise is used at all SNR levels
                noise_seg = noise_cache[(rec_idx, seg_idx)]

                # Create mixture using RMS-based mixing
                mixture, actual_snr = pipeline._create_mixture_rms(
                    coi_seg, noise_seg, float(snr_db)
                )

                # Compute energy on noisy mixture
                mixture_metrics = compute_energy_metrics(mixture, pipeline.sample_rate)
                
                # Separate noisy mixture
                separated = pipeline._separate(mixture)
                
                # Compute energy on the COI output only.
                coi_est = _coi_source_from_separated(pipeline, separated)
                sep_noisy_metrics = compute_energy_metrics(coi_est, pipeline.sample_rate)
                
                # Calculate energy degradation from clean baseline
                noisy_rms_delta = sep_noisy_metrics["rms_db"] - baseline.original_clean_rms_db
                noisy_sel_delta = sep_noisy_metrics["sel_db"] - baseline.original_clean_sel_db
                
                # Create segment metrics with all data
                seg_metrics = SegmentEnergyMetrics(
                    filename=row.filename,
                    recording_idx=rec_idx,
                    segment_idx=seg_idx,
                    snr_db=float(snr_db),
                    # Copy clean baseline values
                    original_clean_rms_db=baseline.original_clean_rms_db,
                    original_clean_sel_db=baseline.original_clean_sel_db,
                    separated_clean_rms_db=baseline.separated_clean_rms_db,
                    separated_clean_sel_db=baseline.separated_clean_sel_db,
                    clean_sep_rms_delta=baseline.clean_sep_rms_delta,
                    clean_sep_sel_delta=baseline.clean_sep_sel_delta,
                    # Add noisy experiment values
                    mixture_rms_db=mixture_metrics["rms_db"],
                    mixture_sel_db=mixture_metrics["sel_db"],
                    separated_noisy_rms_db=sep_noisy_metrics["rms_db"],
                    separated_noisy_sel_db=sep_noisy_metrics["sel_db"],
                    noisy_sep_rms_delta=noisy_rms_delta,
                    noisy_sep_sel_delta=noisy_sel_delta,
                    actual_snr_db=actual_snr,
                )
                
                snr_segment_metrics.append(seg_metrics)
                all_segment_metrics.append(seg_metrics)
        
        # Compute aggregate statistics for this SNR level
        n_segs = len(snr_segment_metrics)
        
        mixture_rms = [m.mixture_rms_db for m in snr_segment_metrics]
        mixture_sel = [m.mixture_sel_db for m in snr_segment_metrics]
        sep_noisy_rms = [m.separated_noisy_rms_db for m in snr_segment_metrics]
        sep_noisy_sel = [m.separated_noisy_sel_db for m in snr_segment_metrics]
        rms_deltas = [m.noisy_sep_rms_delta for m in snr_segment_metrics]
        sel_deltas = [m.noisy_sep_sel_delta for m in snr_segment_metrics]
        actual_snrs = [m.actual_snr_db for m in snr_segment_metrics]
        
        snr_stats = SNREnergyStats(
            snr_db=float(snr_db),
            n_segments=n_segs,
            mean_mixture_rms_db=float(np.mean(mixture_rms)),
            mean_mixture_sel_db=float(np.mean(mixture_sel)),
            mean_separated_noisy_rms_db=float(np.mean(sep_noisy_rms)),
            mean_separated_noisy_sel_db=float(np.mean(sep_noisy_sel)),
            mean_rms_degradation_db=float(np.mean(rms_deltas)),
            mean_sel_degradation_db=float(np.mean(sel_deltas)),
            std_rms_degradation_db=float(np.std(rms_deltas)),
            std_sel_degradation_db=float(np.std(sel_deltas)),
            mean_actual_snr_db=float(np.mean(actual_snrs)),
        )
        
        per_snr_stats.append(snr_stats)
        
        # Log results for this SNR level
        print(f"  Segments: {n_segs}")
        print(f"  Target SNR: {snr_db:.1f} dB, Actual: {snr_stats.mean_actual_snr_db:.1f} dB")
        print(f"  Separated RMS: {snr_stats.mean_separated_noisy_rms_db:.2f} dBFS "
              f"(degradation: {snr_stats.mean_rms_degradation_db:+.2f} dB)")
        print(f"  Separated SEL: {snr_stats.mean_separated_noisy_sel_db:.2f} dBFS "
              f"(degradation: {snr_stats.mean_sel_degradation_db:+.2f} dB)")
    
    # Print summary statistics
    print(f"\n{'=' * 60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 60}")
    print("Energy Degradation Trend (Separated Signal vs Clean Baseline):")
    for s in per_snr_stats:
        print(f"  SNR {s.snr_db:+6.1f} dB: RMS {s.mean_rms_degradation_db:+6.2f} dB, "
              f"SEL {s.mean_sel_degradation_db:+6.2f} dB")
    print(f"{'=' * 60}\n")

    return all_segment_metrics, per_snr_stats


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

    # Dataset filtering
    SPLIT = "test"
    EXCLUDE_DATASETS = ["risoux_test"]  # keep independent set out of this experiment

    # Experiment sweep
    # Extended range to -20 dB to test extreme noise conditions
    SNR_START = 25
    SNR_END = -20
    NUM_STEPS = 10
    SNR_LEVELS_DB = list(np.linspace(SNR_START, SNR_END, NUM_STEPS))

    SEED = 42

    # Output
    OUTPUT_DIR = Path("./noise_increase_results")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # ======================================================

    print("Initializing pipeline...")
    pipeline = ValidationPipeline(base_path=BASE_PATH)
    
    # Load ONLY separator model (no classifier needed for energy metrics)
    if USE_TUSS:
        print(f"Using TUSS model with prompts: COI='{TUSS_COI_PROMPT}', BG='{TUSS_BG_PROMPT}'")
        pipeline.load_models(
            sep_checkpoint=SEP_CHECKPOINT,
            cls_weights=None,  # No classifier needed
            use_tuss=True,
            tuss_coi_prompt=TUSS_COI_PROMPT,
            tuss_bg_prompt=TUSS_BG_PROMPT,
        )
    else:
        print("Using SudoRM-RF model")
        pipeline.load_models(
            sep_checkpoint=SEP_CHECKPOINT,
            cls_weights=None,  # No classifier needed
            classifier_type="plane", # Default classifier to get correct COI synonyms
            use_clapsep=False,
            use_tuss=False,
        )

    print("Loading metadata CSV...")
    df = pd.read_csv(DATA_CSV)
    if EXCLUDE_DATASETS and "dataset" in df.columns:
        df = df[~df["dataset"].isin(EXCLUDE_DATASETS)].copy()
    if "split" in df.columns:
        df = df[df["split"] == SPLIT].copy()

    # Extract COI and background DataFrames
    # Use pipeline's COI synonyms if available (set by load_models), otherwise use default
    coi_syns = getattr(pipeline, 'coi_synonyms', None)
    df_coi = _extract_coi_df(df, coi_synonyms=coi_syns)
    df_bg = _extract_bg_df(df, coi_synonyms=coi_syns)
    
    # Apply contamination filtering to background samples
    # (This ensures consistency with test_pipeline.py, even though this experiment
    # currently uses synthetic white noise instead of real background samples)
    df_bg_clean, n_contaminated = _filter_contaminated_backgrounds(
        df_bg, coi_synonyms=coi_syns, verbose=True
    )

    print(f"\n{'=' * 60}")
    print(f"Dataset Statistics:")
    print(f"  COI samples:        {len(df_coi)} (using ALL samples)")
    print(f"  Background samples: {len(df_bg)} total, {len(df_bg_clean)} clean")
    if n_contaminated > 0:
        print(f"  Contaminated removed: {n_contaminated}")
    print(f"  SNR sweep:          {SNR_LEVELS_DB}")
    print(f"{'=' * 60}\n")

    # Compute clean baseline first (no noise added)
    clean_baseline_stats, baseline_map = _compute_clean_baseline_energy(
        pipeline=pipeline,
        df_coi=df_coi,
        seed=SEED,
    )

    # Run noise increase experiment
    all_segment_metrics, per_snr_stats = run_noise_increase_experiment(
        pipeline=pipeline,
        df_coi=df_coi,
        baseline_map=baseline_map,
        snr_levels_db=SNR_LEVELS_DB,
        seed=SEED,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "tuss" if USE_TUSS else "sudormrf"
    out_json = OUTPUT_DIR / f"noise_increase_energy_{model_name}_{ts}.json"
    out_csv = OUTPUT_DIR / f"noise_increase_energy_{model_name}_{ts}.csv"

    # Prepare detailed CSV with all segment-level data
    # Include clean baseline segments (snr_db=None) and all noisy SNR levels
    clean_baseline_segments = list(baseline_map.values())
    all_segments_for_csv = clean_baseline_segments + all_segment_metrics
    
    # Convert to DataFrame for CSV export
    csv_data = []
    for seg in all_segments_for_csv:
        csv_data.append({
            "filename": seg.filename,
            "recording_idx": seg.recording_idx,
            "segment_idx": seg.segment_idx,
            "snr_db": seg.snr_db if seg.snr_db is not None else "clean",
            "original_clean_rms_db": seg.original_clean_rms_db,
            "original_clean_sel_db": seg.original_clean_sel_db,
            "separated_clean_rms_db": seg.separated_clean_rms_db,
            "separated_clean_sel_db": seg.separated_clean_sel_db,
            "mixture_rms_db": seg.mixture_rms_db,
            "mixture_sel_db": seg.mixture_sel_db,
            "separated_noisy_rms_db": seg.separated_noisy_rms_db,
            "separated_noisy_sel_db": seg.separated_noisy_sel_db,
            "clean_sep_rms_delta": seg.clean_sep_rms_delta,
            "clean_sep_sel_delta": seg.clean_sep_sel_delta,
            "noisy_sep_rms_delta": seg.noisy_sep_rms_delta,
            "noisy_sep_sel_delta": seg.noisy_sep_sel_delta,
            "actual_snr_db": seg.actual_snr_db,
        })
    
    df_results = pd.DataFrame(csv_data)
    df_results.to_csv(out_csv, index=False)

    # Prepare JSON with config, baseline, per-SNR stats, and summary
    payload = {
        "config": {
            "base_path": BASE_PATH,
            "data_csv": DATA_CSV,
            "sep_checkpoint": SEP_CHECKPOINT,
            "model_type": "TUSS" if USE_TUSS else "SudoRM-RF",
            "split": SPLIT,
            "exclude_datasets": EXCLUDE_DATASETS,
            "snr_levels_db": SNR_LEVELS_DB,
            "seed": SEED,
            "noise_type": "artificial_white_noise",
            "experiment_type": "energy_preservation",
        },
        "dataset_stats": {
            "n_coi_samples": len(df_coi),
            "n_background_total": len(df_bg),
            "n_background_clean": len(df_bg_clean),
            "n_contaminated_removed": n_contaminated,
        },
        "clean_baseline": clean_baseline_stats.__dict__,
        "snr_results": [s.__dict__ for s in per_snr_stats],
        "summary": {
            "total_segments": len(all_segments_for_csv),
            "clean_baseline_segments": len(clean_baseline_segments),
            "noisy_experiment_segments": len(all_segment_metrics),
            "max_rms_degradation_db": float(max(s.mean_rms_degradation_db for s in per_snr_stats)),
            "max_sel_degradation_db": float(max(s.mean_sel_degradation_db for s in per_snr_stats)),
            "min_rms_degradation_db": float(min(s.mean_rms_degradation_db for s in per_snr_stats)),
            "min_sel_degradation_db": float(min(s.mean_sel_degradation_db for s in per_snr_stats)),
        },
    }
    
    # Add TUSS-specific config if applicable
    if USE_TUSS:
        payload["config"]["tuss_coi_prompt"] = TUSS_COI_PROMPT
        payload["config"]["tuss_bg_prompt"] = TUSS_BG_PROMPT

    with out_json.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved JSON: {out_json}")
    print(f"Saved CSV:  {out_csv}")
    print(f"\nCSV contains {len(df_results)} rows:")
    print(f"  - {len(clean_baseline_segments)} clean baseline segments")
    print(f"  - {len(all_segment_metrics)} noisy experiment segments across {len(SNR_LEVELS_DB)} SNR levels")


if __name__ == "__main__":
    main()
