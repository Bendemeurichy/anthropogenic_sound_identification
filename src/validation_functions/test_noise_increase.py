"""Experiment that gradually adds artificial white noise to COI+BG mixtures and
measures separation's effect on energy preservation (RMS and SEL).

This script measures how well separation preserves COI energy at different
white-noise SNR levels by comparing noisy separation results against a clean
baseline that uses a realistic COI+background mixture.  Classification is not
used since classifiers are not robust to the extreme noise levels tested here.

Mixing formulae (no peak-clipping):

    Clean baseline (no added white noise):
        mixture = coi_unit_rms + bg_unit_rms * 10^(−bg_snr_db/20)

    Noisy sweep:
        mixture = coi_unit_rms + bg_unit_rms * 10^(−bg_snr_db/20)
                + noise_unit_rms * 10^(−snr_db/20)

The BG SNR is fixed at ``BASELINE_BG_SNR_DB`` (default +10 dB, top of the TUSS
training range [-10, +10] dB) so that separation is meaningful even in the
clean condition.  TUSS normalises the full mixture internally by its std and
rescales all outputs by the same std.  Any scalar applied to the mixture before
TUSS propagates to every separated output, so omitting peak-clipping ensures
that a perfect separator reports 0 dB degradation at every white-noise level.
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

    # Energy preservation metrics (dB differences)
    clean_sep_rms_delta: float = 0.0  # separated_clean - original_clean
    clean_sep_sel_delta: float = 0.0
    noisy_sep_rms_delta: float | None = None  # separated_noisy - separated_clean
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

    # Energy degradation from clean separated baseline
    mean_rms_degradation_db: float  # separated_noisy - separated_clean
    mean_sel_degradation_db: float
    std_rms_degradation_db: float
    std_sel_degradation_db: float

    # Actual measured SNR
    mean_actual_snr_db: float


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

    # Prefer numeric label if available
    if pd.api.types.is_numeric_dtype(out["label"]):
        out["label"] = out["label"].fillna(0)

        # If coi_class exists, use it for filtering (matches training behavior)
        if "coi_class" in out.columns and coi_synonyms is not None:
            from src.label_loading.coi_labels import get_coi_synonyms_for_classifier
            # Map coi_synonyms to coi_class index
            if coi_synonyms == get_coi_synonyms_for_classifier("plane"):
                target_coi_class = 0
            elif coi_synonyms == get_coi_synonyms_for_classifier("bird_mae"):
                target_coi_class = 1
            else:
                target_coi_class = None

            if target_coi_class is not None:
                coi = out[out["coi_class"] == target_coi_class].copy()
                return coi.reset_index(drop=True)

        # Fallback: filter by orig_label matching coi_synonyms
        if "orig_label" in out.columns and coi_synonyms is not None:
            base_series = out["orig_label"]
            mask = (out["label"] == 1) & base_series.apply(
                lambda x: _is_coi_label(x, coi_synonyms)
            )
            coi = out[mask].copy()
            return coi.reset_index(drop=True)

        # No filtering available, just use label=1
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


def _preload_coi_segments(
    pipeline: ValidationPipeline,
    df_coi: pd.DataFrame,
    seed: int = 42,
) -> Tuple[
    Dict[Tuple[int, int], torch.Tensor],  # segment_cache: unit-RMS COI tensors
    Dict[Tuple[int, int], torch.Tensor],  # noise_cache:   unit-RMS white noise
    Dict[int, str],                        # filename_map:  rec_idx → filename
    Dict[int, int],                        # rec_seg_count: rec_idx → n_segments
]:
    """Load all COI segments once, normalise to unit RMS, and generate noise.

    Avoids re-loading audio N_SNR_LEVELS times by caching everything up front.
    The same noise tensor is reused at every SNR level — only the amplitude
    changes via ``noise_amplitude = 10 ** (−snr_db / 20)`` — so noise shape
    is identical across all comparisons.

    Both the COI segment and the noise are normalised to unit RMS.  The mixing
    step (adding BG and noise) is performed in the downstream functions that
    consume these caches, so the formula does not appear here.

    This matches TUSS training: ``separate_batch`` normalises the raw mixture
    by ``std`` only (no mean subtraction); for zero-mean audio ``std ≈ RMS``,
    so unit-RMS inputs are consistent with the training distribution.

    Args:
        pipeline: ValidationPipeline instance.
        df_coi: DataFrame of COI samples.
        seed: Random seed for reproducibility.

    Returns:
        segment_cache: ``{(rec_idx, seg_idx): unit-RMS COI tensor}``
        noise_cache:   ``{(rec_idx, seg_idx): unit-RMS white noise tensor}``
        filename_map:  ``{rec_idx: filename}``
        rec_seg_count: ``{rec_idx: n_segments}``
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    segment_cache: Dict[Tuple[int, int], torch.Tensor] = {}
    noise_cache: Dict[Tuple[int, int], torch.Tensor] = {}
    filename_map: Dict[int, str] = {}
    rec_seg_count: Dict[int, int] = {}

    print(f"\nPre-loading {len(df_coi)} COI recordings...")

    for rec_idx, row in enumerate(df_coi.itertuples(index=False)):
        if rec_idx % 50 == 0:
            print(f"  Loading recording {rec_idx}/{len(df_coi)}...")

        coi_full = pipeline._load_labeled_audio(
            row.filename,
            getattr(row, "start_time", None),
            getattr(row, "end_time", None),
        )
        coi_segments = pipeline._split_into_segments(coi_full)

        filename_map[rec_idx] = row.filename
        rec_seg_count[rec_idx] = len(coi_segments)

        for seg_idx, seg in enumerate(coi_segments):
            # Normalise COI to unit RMS (matches TUSS training distribution)
            coi_norm = pipeline._prepare_rms_mixing_input(seg)

            # Independent unit-RMS white noise (same shape as COI segment).
            # Re-normalising the raw Gaussian with _prepare_rms_mixing_input
            # removes the variance fluctuation of the finite-length sample.
            noise_raw = torch.randn_like(coi_norm)
            noise_norm = pipeline._prepare_rms_mixing_input(noise_raw)

            segment_cache[(rec_idx, seg_idx)] = coi_norm
            noise_cache[(rec_idx, seg_idx)] = noise_norm

    total_segs = sum(rec_seg_count.values())
    print(f"Pre-loaded {total_segs} segments from {len(df_coi)} recordings.")
    return segment_cache, noise_cache, filename_map, rec_seg_count


def _assign_backgrounds(
    pipeline: ValidationPipeline,
    df_bg: pd.DataFrame,
    rec_seg_count: Dict[int, int],
    seed: int = 42,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """Load background recordings and assign one unit-RMS segment to each COI slot.

    Loads background audio from ``df_bg``, splits into model-sized segments,
    normalises each to unit RMS, and assigns one background segment to every
    ``(rec_idx, seg_idx)`` slot in *rec_seg_count*.  The same assignment is
    reused across ``_compute_clean_baseline_energy`` and every SNR level in
    ``run_noise_increase_experiment`` so the only thing that changes between
    measurements is the white-noise amplitude.

    Sampling is done with replacement when the background pool is smaller than
    the number of COI slots.  Silent background segments (RMS < 1e-6) are
    skipped to avoid normalisation artefacts.

    Args:
        pipeline: ValidationPipeline instance (used for audio loading helpers).
        df_bg: DataFrame of background (non-COI) samples from the test split.
        rec_seg_count: ``{rec_idx: n_segments}`` from ``_preload_coi_segments``.
        seed: Random seed for reproducibility (uses ``seed + 1`` to stay
            independent from the COI / noise seeds in ``_preload_coi_segments``).

    Returns:
        bg_assignment: ``{(rec_idx, seg_idx): unit-RMS BG tensor}``
    """
    rng = np.random.default_rng(seed + 1)
    total_slots = sum(rec_seg_count.values())

    bg_pool: List[torch.Tensor] = []
    print(f"\nPre-loading background pool (need {total_slots} slots)...")

    n_loaded = 0
    for row in df_bg.itertuples(index=False):
        if n_loaded % 100 == 0:
            print(f"  Loading background recording {n_loaded}/{len(df_bg)}...")

        try:
            bg_full = pipeline._load_labeled_audio(
                row.filename,
                getattr(row, "start_time", None),
                getattr(row, "end_time", None),
            )
        except Exception:
            n_loaded += 1
            continue

        for seg in pipeline._split_into_segments(bg_full):
            seg_rms = float(torch.sqrt(torch.mean(seg ** 2)))
            if seg_rms < 1e-6:
                continue  # skip silent segments to avoid normalisation artefacts
            bg_pool.append(pipeline._prepare_rms_mixing_input(seg))

        n_loaded += 1
        # Gather at least 3× the required slots so sampling is well-mixed
        if len(bg_pool) >= total_slots * 3:
            break

    if len(bg_pool) == 0:
        raise ValueError(
            "No usable background segments found — check that df_bg is non-empty "
            "and that the background recordings are not silent."
        )

    print(
        f"  Background pool: {len(bg_pool)} segments "
        f"from {n_loaded} recordings (need {total_slots})."
    )

    n_pool = len(bg_pool)
    bg_assignment: Dict[Tuple[int, int], torch.Tensor] = {}
    for rec_idx in sorted(rec_seg_count.keys()):
        for seg_idx in range(rec_seg_count[rec_idx]):
            pool_idx = int(rng.integers(0, n_pool))
            bg_assignment[(rec_idx, seg_idx)] = bg_pool[pool_idx]

    return bg_assignment


def _compute_clean_baseline_energy(
    pipeline: ValidationPipeline,
    segment_cache: Dict[Tuple[int, int], torch.Tensor],
    bg_assignment: Dict[Tuple[int, int], torch.Tensor],
    bg_snr_db: float,
    filename_map: Dict[int, str],
    rec_seg_count: Dict[int, int],
    batch_size: int = 16,
) -> Tuple[CleanBaselineStats, Dict[Tuple[int, int], SegmentEnergyMetrics]]:
    """Compute baseline energy metrics on COI+BG mixtures (no added white noise).

    Uses pre-loaded unit-RMS segments from ``_preload_coi_segments`` and the
    fixed background assignment from ``_assign_backgrounds``.  Provides the
    reference point used to measure energy degradation at each white-noise
    level.  ``original_clean_rms_db`` tracks COI-only energy (not mixture
    energy) to keep the energy-preservation metric interpretable.

    Args:
        pipeline: ValidationPipeline with loaded separator model.
        segment_cache: ``{(rec_idx, seg_idx): unit-RMS COI tensor}`` from
            ``_preload_coi_segments``.
        bg_assignment: ``{(rec_idx, seg_idx): unit-RMS BG tensor}`` from
            ``_assign_backgrounds``.
        bg_snr_db: COI:BG SNR for the baseline mixture (dB).  Positive values
            mean COI is louder than BG.
        filename_map: ``{rec_idx: filename}`` from ``_preload_coi_segments``.
        rec_seg_count: ``{rec_idx: n_segments}`` from ``_preload_coi_segments``.
        batch_size: Number of segments to separate in one GPU call.

    Returns:
        Tuple of:
            - CleanBaselineStats: Aggregated statistics.
            - Dict mapping ``(rec_idx, seg_idx)`` to
              :class:`SegmentEnergyMetrics`.
    """
    all_metrics: List[SegmentEnergyMetrics] = []
    baseline_map: Dict[Tuple[int, int], SegmentEnergyMetrics] = {}

    bg_amp = 10 ** (-bg_snr_db / 20)
    n_recordings = len(rec_seg_count)
    print(
        f"\nComputing clean baseline energy (COI+BG, BG SNR={bg_snr_db:.1f} dB) "
        f"on {n_recordings} recordings..."
    )

    for rec_idx in sorted(rec_seg_count.keys()):
        if rec_idx % 50 == 0:
            print(f"  Processing recording {rec_idx}/{n_recordings}...")

        n_segs = rec_seg_count[rec_idx]
        if n_segs == 0:
            continue
        filename = filename_map[rec_idx]

        # Build COI+BG mixtures — no added white noise for the clean baseline.
        mixture_tensors = []
        for s in range(n_segs):
            coi_norm = segment_cache[(rec_idx, s)]
            bg_norm = bg_assignment[(rec_idx, s)]
            mixture_tensors.append(coi_norm + bg_norm * bg_amp)
        mixture_stack = torch.stack(mixture_tensors)

        for i in range(0, n_segs, batch_size):
            batch = mixture_stack[i : i + batch_size]

            separated_batch = pipeline._separate_batch(batch)

            if separated_batch.dim() == 2:
                coi_est_batch = separated_batch
            else:
                coi_est_batch = separated_batch[:, pipeline._get_coi_head_index()]

            for b_idx in range(len(batch)):
                seg_idx = i + b_idx
                # Reference energy: COI-only (unit-RMS, ~0 dBFS)
                coi_only = segment_cache[(rec_idx, seg_idx)]
                coi_est = coi_est_batch[b_idx]

                orig_metrics = compute_energy_metrics(coi_only, pipeline.sample_rate)
                sep_metrics = compute_energy_metrics(coi_est, pipeline.sample_rate)

                rms_delta = sep_metrics["rms_db"] - orig_metrics["rms_db"]
                sel_delta = sep_metrics["sel_db"] - orig_metrics["sel_db"]

                seg_metrics = SegmentEnergyMetrics(
                    filename=filename,
                    recording_idx=rec_idx,
                    segment_idx=seg_idx,
                    snr_db=None,  # Clean baseline has no added white noise
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
    n_segs_total = len(all_metrics)

    orig_rms = [m.original_clean_rms_db for m in all_metrics]
    orig_sel = [m.original_clean_sel_db for m in all_metrics]
    sep_rms = [m.separated_clean_rms_db for m in all_metrics]
    sep_sel = [m.separated_clean_sel_db for m in all_metrics]
    rms_deltas = [m.clean_sep_rms_delta for m in all_metrics]
    sel_deltas = [m.clean_sep_sel_delta for m in all_metrics]

    baseline_stats = CleanBaselineStats(
        n_samples=n_recordings,
        n_segments=n_segs_total,
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

    print(f"\nClean Baseline Results ({n_segs_total} segments):")
    print(f"  Original:  RMS={baseline_stats.mean_original_rms_db:.2f} dBFS, "
          f"SEL={baseline_stats.mean_original_sel_db:.2f} dBFS")
    print(f"  Separated: RMS={baseline_stats.mean_separated_rms_db:.2f} dBFS, "
          f"SEL={baseline_stats.mean_separated_sel_db:.2f} dBFS")
    print(f"  Preservation: RMS_delta={baseline_stats.mean_rms_preservation_db:+.2f} dB, "
          f"SEL_delta={baseline_stats.mean_sel_preservation_db:+.2f} dB")

    return baseline_stats, baseline_map


def run_noise_increase_experiment(
    pipeline: ValidationPipeline,
    segment_cache: Dict[Tuple[int, int], torch.Tensor],
    noise_cache: Dict[Tuple[int, int], torch.Tensor],
    bg_assignment: Dict[Tuple[int, int], torch.Tensor],
    bg_snr_db: float,
    filename_map: Dict[int, str],
    rec_seg_count: Dict[int, int],
    baseline_map: Dict[Tuple[int, int], SegmentEnergyMetrics],
    snr_levels_db: List[float],
    batch_size: int = 16,
) -> Tuple[List[SegmentEnergyMetrics], List[SNREnergyStats]]:
    """Run robustness sweep over white-noise SNR levels measuring separation degradation.

    For each SNR level, mixes pre-loaded unit-RMS COI segments with a fixed
    unit-RMS background and unit-RMS white noise, then measures how much the
    separated COI signal degrades compared to the clean baseline (COI+BG,
    no added noise).  This isolates the effect of white noise on separation
    quality independently of the COI:BG balance.

    Mixture formula::

        bg_amplitude   = 10 ** (−bg_snr_db / 20)
        noise_amplitude = 10 ** (−snr_db / 20)
        mixture = coi_unit_rms + bg_unit_rms * bg_amplitude
                + noise_unit_rms * noise_amplitude

    ``actual_snr_db`` is measured as COI power vs. white-noise power only (BG
    is treated as the fixed acoustic context, not the swept variable).

    **Why no peak-clipping?**  TUSS ``_separate_batch`` normalises the
    mixture by its std and rescales all outputs by the same std.  A scalar
    applied to the mixture before TUSS therefore propagates to every separated
    output:  ``separated_out ∝ scale_factor × true_output``.  For a perfect
    separator the degradation metric ``separated_noisy − separated_clean``
    would then read ``20·log₁₀(scale_factor) < 0`` instead of 0 dB, a
    purely artefactual bias.

    Degradation metric: ``separated_noisy_rms − separated_clean_rms``
    (negative = noise caused the separator to suppress more energy).

    Args:
        pipeline: ValidationPipeline with loaded separator model.
        segment_cache: ``{(rec_idx, seg_idx): unit-RMS COI tensor}`` from
            ``_preload_coi_segments``.
        noise_cache: ``{(rec_idx, seg_idx): unit-RMS noise tensor}`` from
            ``_preload_coi_segments``.
        bg_assignment: ``{(rec_idx, seg_idx): unit-RMS BG tensor}`` from
            ``_assign_backgrounds``.
        bg_snr_db: COI:BG SNR used in every mixture (same value as the clean
            baseline so the BG component is unchanged across measurements).
        filename_map: ``{rec_idx: filename}`` from ``_preload_coi_segments``.
        rec_seg_count: ``{rec_idx: n_segments}`` from ``_preload_coi_segments``.
        baseline_map: ``{(rec_idx, seg_idx): SegmentEnergyMetrics}`` from
            ``_compute_clean_baseline_energy``.
        snr_levels_db: List of white-noise SNR levels to test (in dB).
        batch_size: Number of segments to separate in one GPU call.

    Returns:
        Tuple of:
            - List of :class:`SegmentEnergyMetrics` for all segments and SNR
              levels.
            - List of :class:`SNREnergyStats` with aggregate statistics per
              SNR level.
    """
    if not segment_cache:
        raise ValueError("No COI segments available for experiment.")

    all_segment_metrics: List[SegmentEnergyMetrics] = []
    per_snr_stats: List[SNREnergyStats] = []

    # BG amplitude is constant across all SNR levels
    bg_amp = 10 ** (-bg_snr_db / 20)

    for idx, snr_db in enumerate(snr_levels_db, 1):
        print(f"\n[{idx}/{len(snr_levels_db)}] Processing SNR = {snr_db:.1f} dB...")

        # Amplitude scale for white noise: COI RMS = 1.0, so noise_amplitude
        # gives SNR(COI vs. white noise) = snr_db exactly.
        noise_amplitude = 10 ** (-snr_db / 20)

        snr_segment_metrics: List[SegmentEnergyMetrics] = []

        for rec_idx in sorted(rec_seg_count.keys()):
            n_segs = rec_seg_count[rec_idx]
            if n_segs == 0:
                continue
            filename = filename_map[rec_idx]

            # Build COI+BG+noise mixtures (no peak-clipping — see docstring)
            mixtures: List[torch.Tensor] = []
            mixture_metrics_list: List[dict] = []
            actual_snrs: List[float] = []

            for seg_idx in range(n_segs):
                coi_norm = segment_cache[(rec_idx, seg_idx)]
                noise_norm = noise_cache[(rec_idx, seg_idx)]
                bg_norm = bg_assignment[(rec_idx, seg_idx)]

                mixture = coi_norm + bg_norm * bg_amp + noise_norm * noise_amplitude

                # Verify actual SNR: COI power vs. added white-noise power
                sig_power = float(torch.mean(coi_norm ** 2))
                noise_power = float(torch.mean((noise_norm * noise_amplitude) ** 2))
                actual_snr = 10.0 * np.log10(sig_power / (noise_power + 1e-8))

                mixture_metrics_list.append(
                    compute_energy_metrics(mixture, pipeline.sample_rate)
                )
                mixtures.append(mixture)
                actual_snrs.append(actual_snr)

            mixtures_tensor = torch.stack(mixtures)

            for i in range(0, n_segs, batch_size):
                batch_mix = mixtures_tensor[i : i + batch_size]

                separated_batch = pipeline._separate_batch(batch_mix)

                if separated_batch.dim() == 2:
                    coi_est_batch = separated_batch
                else:
                    coi_est_batch = separated_batch[:, pipeline._get_coi_head_index()]

                for b_idx in range(len(batch_mix)):
                    seg_idx = i + b_idx
                    baseline = baseline_map[(rec_idx, seg_idx)]
                    mixture_metrics = mixture_metrics_list[seg_idx]
                    coi_est = coi_est_batch[b_idx]

                    sep_noisy_metrics = compute_energy_metrics(coi_est, pipeline.sample_rate)

                    # Degradation: how much does noise degrade separated energy?
                    noisy_rms_delta = sep_noisy_metrics["rms_db"] - baseline.separated_clean_rms_db
                    noisy_sel_delta = sep_noisy_metrics["sel_db"] - baseline.separated_clean_sel_db

                    seg_metrics = SegmentEnergyMetrics(
                        filename=filename,
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
                        # Noisy experiment values
                        mixture_rms_db=mixture_metrics["rms_db"],
                        mixture_sel_db=mixture_metrics["sel_db"],
                        separated_noisy_rms_db=sep_noisy_metrics["rms_db"],
                        separated_noisy_sel_db=sep_noisy_metrics["sel_db"],
                        noisy_sep_rms_delta=noisy_rms_delta,
                        noisy_sep_sel_delta=noisy_sel_delta,
                        actual_snr_db=actual_snrs[seg_idx],
                    )

                    snr_segment_metrics.append(seg_metrics)
                    all_segment_metrics.append(seg_metrics)

        # Compute aggregate statistics for this SNR level
        n_segs_total = len(snr_segment_metrics)

        mixture_rms = [m.mixture_rms_db for m in snr_segment_metrics]
        mixture_sel = [m.mixture_sel_db for m in snr_segment_metrics]
        sep_noisy_rms = [m.separated_noisy_rms_db for m in snr_segment_metrics]
        sep_noisy_sel = [m.separated_noisy_sel_db for m in snr_segment_metrics]
        rms_deltas = [m.noisy_sep_rms_delta for m in snr_segment_metrics]
        sel_deltas = [m.noisy_sep_sel_delta for m in snr_segment_metrics]
        act_snrs = [m.actual_snr_db for m in snr_segment_metrics]

        snr_stats = SNREnergyStats(
            snr_db=float(snr_db),
            n_segments=n_segs_total,
            mean_mixture_rms_db=float(np.mean(mixture_rms)),
            mean_mixture_sel_db=float(np.mean(mixture_sel)),
            mean_separated_noisy_rms_db=float(np.mean(sep_noisy_rms)),
            mean_separated_noisy_sel_db=float(np.mean(sep_noisy_sel)),
            mean_rms_degradation_db=float(np.mean(rms_deltas)),
            mean_sel_degradation_db=float(np.mean(sel_deltas)),
            std_rms_degradation_db=float(np.std(rms_deltas)),
            std_sel_degradation_db=float(np.std(sel_deltas)),
            mean_actual_snr_db=float(np.mean(act_snrs)),
        )

        per_snr_stats.append(snr_stats)

        print(f"  Segments: {n_segs_total}")
        print(f"  Target SNR: {snr_db:.1f} dB, Actual: {snr_stats.mean_actual_snr_db:.1f} dB")
        print(f"  Separated RMS: {snr_stats.mean_separated_noisy_rms_db:.2f} dBFS "
              f"(degradation: {snr_stats.mean_rms_degradation_db:+.2f} dB)")
        print(f"  Separated SEL: {snr_stats.mean_separated_noisy_sel_db:.2f} dBFS "
              f"(degradation: {snr_stats.mean_sel_degradation_db:+.2f} dB)")

    # Print summary
    print(f"\n{'=' * 60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 60}")
    print("Separation Quality Degradation vs Clean (separated_noisy - separated_clean):")
    print("  Negative values = signal gets worse with noise")
    print("  Positive values = noise leaking into COI output")
    for s in per_snr_stats:
        print(f"  SNR {s.snr_db:+6.1f} dB: RMS {s.mean_rms_degradation_db:+6.2f} dB, "
              f"SEL {s.mean_sel_degradation_db:+6.2f} dB")
    print(f"{'=' * 60}\n")

    return all_segment_metrics, per_snr_stats


def main() -> None:
    # ================== HARD-CODED CONFIG ==================
    BASE_PATH = str(PROJECT_ROOT.parent / "datasets")

    # ---- Device selection ----
    # Set to "cuda:0", "cuda:1", "cpu", etc. or None for auto-detection
    DEVICE = None  # None = auto-select (prefers cuda:1, falls back to cuda:0, then cpu)

    # ---- COI type ----
    # "plane" → airplane synonyms, TUSS prompt "airplane"
    # "bird"  → bird synonyms,     TUSS prompt "birds"
    COI_TYPE = "plane"

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
        # TUSS prompts — must match training config (multi_coi_29_04: "airplane", "birds")
        TUSS_COI_PROMPT = "birds" if COI_TYPE == "bird" else "airplane"
        TUSS_BG_PROMPT = "background"
    else:
        # SudoRM-RF model configuration (default)
        DATA_CSV = str(
            PROJECT_ROOT
            / "src/models/sudormrf/checkpoints/20260219_124144/separation_dataset.csv"
        )
        SEP_CHECKPOINT = str(PROJECT_ROOT / "src/models/sudormrf/checkpoints/best_model.pt")

    # Batch size for separation inference
    BATCH_SIZE = 16

    # Dataset filtering
    SPLIT = "test"
    EXCLUDE_DATASETS = ["risoux_test"]  # keep independent set out of this experiment

    # Background SNR for the clean baseline mixture (COI:BG ratio in dB).
    # +10 dB sits at the top of the TUSS training range [-10, +10] dB so
    # separation is realistic without being pathologically easy.
    BASELINE_BG_SNR_DB = 10.0

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

    # Guard: catch un-edited placeholder paths early
    if USE_TUSS and "YOUR_CHECKPOINT_DIR" in DATA_CSV:
        raise RuntimeError(
            "USE_TUSS=True but DATA_CSV/SEP_CHECKPOINT still contain "
            "'YOUR_CHECKPOINT_DIR'.  Update the paths in the CONFIG section "
            "before running."
        )

    print("Initializing pipeline...")
    pipeline = ValidationPipeline(base_path=BASE_PATH, device=DEVICE)

    # classifier_type drives COI synonym selection only (classifier not loaded)
    classifier_type = "bird_mae" if COI_TYPE == "bird" else "plane"

    if USE_TUSS:
        print(f"Using TUSS model with prompts: COI='{TUSS_COI_PROMPT}', BG='{TUSS_BG_PROMPT}'")
        pipeline.load_models(
            sep_checkpoint=SEP_CHECKPOINT,
            classifier_type=classifier_type,
            use_tuss=True,
            tuss_coi_prompt=TUSS_COI_PROMPT,
            tuss_bg_prompt=TUSS_BG_PROMPT,
            skip_classifier=True,  # Energy-only experiment — no classifier needed
        )
    else:
        print("Using SudoRM-RF model")
        pipeline.load_models(
            sep_checkpoint=SEP_CHECKPOINT,
            classifier_type=classifier_type,
            use_clapsep=False,
            use_tuss=False,
            skip_classifier=True,  # Energy-only experiment — no classifier needed
        )

    print("Loading metadata CSV...")
    df = pd.read_csv(DATA_CSV, low_memory=False)
    if EXCLUDE_DATASETS and "dataset" in df.columns:
        df = df[~df["dataset"].isin(EXCLUDE_DATASETS)].copy()
    if "split" in df.columns:
        df = df[df["split"] == SPLIT].copy()

    coi_syns = getattr(pipeline, "coi_synonyms", None)
    df_coi = _extract_coi_df(df, coi_synonyms=coi_syns)
    df_bg = _extract_bg_df(df, coi_synonyms=coi_syns)

    print(f"\n{'=' * 60}")
    print("Dataset Statistics:")
    print(f"  COI samples:  {len(df_coi)} (split={SPLIT!r})")
    print(f"  BG samples:   {len(df_bg)} (split={SPLIT!r})")
    print(f"  BG SNR (baseline): {BASELINE_BG_SNR_DB:.1f} dB")
    print(f"  SNR sweep:    {[round(s, 1) for s in SNR_LEVELS_DB]}")
    print(f"{'=' * 60}\n")

    # Pre-load all COI segments and noise once.
    # Avoids reloading audio N_SNR_LEVELS × N_COI times.
    segment_cache, noise_cache, filename_map, rec_seg_count = _preload_coi_segments(
        pipeline=pipeline,
        df_coi=df_coi,
        seed=SEED,
    )

    # Assign one fixed unit-RMS background segment to every (rec_idx, seg_idx) slot.
    # The same BG is reused for the clean baseline and every noisy SNR level so
    # the only thing that varies between measurements is white-noise amplitude.
    bg_assignment = _assign_backgrounds(
        pipeline=pipeline,
        df_bg=df_bg,
        rec_seg_count=rec_seg_count,
        seed=SEED,
    )

    # Compute clean baseline (separator run on COI+BG mixture, no white noise)
    clean_baseline_stats, baseline_map = _compute_clean_baseline_energy(
        pipeline=pipeline,
        segment_cache=segment_cache,
        bg_assignment=bg_assignment,
        bg_snr_db=BASELINE_BG_SNR_DB,
        filename_map=filename_map,
        rec_seg_count=rec_seg_count,
        batch_size=BATCH_SIZE,
    )

    # SNR sweep — no peak-clipping, mixture built directly from caches
    all_segment_metrics, per_snr_stats = run_noise_increase_experiment(
        pipeline=pipeline,
        segment_cache=segment_cache,
        noise_cache=noise_cache,
        bg_assignment=bg_assignment,
        bg_snr_db=BASELINE_BG_SNR_DB,
        filename_map=filename_map,
        rec_seg_count=rec_seg_count,
        baseline_map=baseline_map,
        snr_levels_db=SNR_LEVELS_DB,
        batch_size=BATCH_SIZE,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "tuss" if USE_TUSS else "sudormrf"
    out_json = OUTPUT_DIR / f"noise_increase_energy_{model_name}_{ts}.json"
    out_csv = OUTPUT_DIR / f"noise_increase_energy_{model_name}_{ts}.csv"

    # Detailed CSV — clean baseline rows (snr_db=None) + all noisy rows
    clean_baseline_segments = list(baseline_map.values())
    all_segments_for_csv = clean_baseline_segments + all_segment_metrics

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

    payload = {
        "config": {
            "base_path": BASE_PATH,
            "data_csv": DATA_CSV,
            "sep_checkpoint": SEP_CHECKPOINT,
            "model_type": "TUSS" if USE_TUSS else "SudoRM-RF",
            "coi_type": COI_TYPE,
            "split": SPLIT,
            "exclude_datasets": EXCLUDE_DATASETS,
            "baseline_bg_snr_db": BASELINE_BG_SNR_DB,
            "snr_levels_db": SNR_LEVELS_DB,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "noise_type": "artificial_white_noise",
            "experiment_type": "energy_preservation",
            "mixing": "no_peak_clipping",
        },
        "dataset_stats": {
            "n_coi_samples": len(df_coi),
            "n_bg_samples": len(df_bg),
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

    if USE_TUSS:
        payload["config"]["tuss_coi_prompt"] = TUSS_COI_PROMPT
        payload["config"]["tuss_bg_prompt"] = TUSS_BG_PROMPT

    with out_json.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved JSON: {out_json}")
    print(f"Saved CSV:  {out_csv}")
    print(f"\nCSV contains {len(df_results)} rows:")
    print(f"  - {len(clean_baseline_segments)} clean baseline segments")
    print(f"  - {len(all_segment_metrics)} noisy experiment segments "
          f"across {len(SNR_LEVELS_DB)} SNR levels")


if __name__ == "__main__":
    main()
