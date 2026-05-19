#!/usr/bin/env python3
"""Run all three separation models on the same airplane-COI input and plot a
side-by-side spectrogram comparison.

Two modes
--------
Single-WAV  (default)    python run_demo_compare_separators.py [mixture.wav]
3-WAV       (--coi ...)  python run_demo_compare_separators.py --coi plane.wav --bg bg.wav --mixture mix.wav

SuDoRM-RF  → airplane checkpoint
CLAPSep    → pretrained with text_pos="airplane engine"
TUSS       → multi-COI checkpoint (airplane+birds), extracting airplane
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml

PREFIX = "/home/bendm/Thesis/project/code/src"
sys.path.insert(0, PREFIX)

from models.clapsep.inference import CLAPSepInference, COI_HEAD_INDEX as CLAPSEP_COI
from models.sudormrf.inference import COI_HEAD_INDEX as SUDORMRF_COI
from models.sudormrf.inference import SeparationInference
from models.tuss.inference import TUSSInference
from validation_functions.demo_separation import (
    compute_energy_metrics,
    compute_spectrogram,
    load_wav,
    plot_combined_spectrograms,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MIXTURE = (
    "/home/bendm/Thesis/project/data/misclassifications/"
    "239_as_is_sep_cls_['plane',_'wind',_'biophony']_conf0.456_S4A04430_20180716_113000.wav"
)

SUDORMRF_CKPT = (
    "/home/bendm/Thesis/project/code/src/models/sudormrf/checkpoints/"
    "sudormrf_planes_10_5/best_model.pt"
)

TUSS_CKPT = (
    "/home/bendm/Thesis/project/code/src/models/tuss/checkpoints/multi_coi_11_05"
)

OUT_DIR = Path("/home/bendm/Thesis/project/code/src/validation_functions/demo_output_compare")
COMMON_SR = 44_100  # resample all outputs to this rate for a shared spectrogram axis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def peak_normalize(waveform: torch.Tensor, target_peak: float = 0.95) -> torch.Tensor:
    peak = waveform.abs().max()
    if peak < 1e-8:
        return waveform
    return waveform * (target_peak / peak)


def make_mono(wav: torch.Tensor) -> torch.Tensor:
    if wav.ndim == 0:
        return wav.unsqueeze(0)
    if wav.ndim == 1:
        return wav
    return wav.mean(dim=0)


def resample_if_needed(wav: torch.Tensor, src_sr: int, tgt_sr: int) -> torch.Tensor:
    if src_sr == tgt_sr:
        return wav
    w = wav.unsqueeze(0) if wav.ndim == 1 else wav
    return torchaudio.transforms.Resample(src_sr, tgt_sr)(w).squeeze(0)


def save_wav(path: Path, wav: torch.Tensor, sr: int) -> None:
    wav = make_mono(wav)
    wav = peak_normalize(wav)
    sf.write(str(path), wav.numpy(), sr)


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    """Load a WAV, convert to mono, return (waveform, sr)."""
    wav_np, sr = sf.read(str(path))
    wav = torch.from_numpy(wav_np.astype(np.float32))
    if wav.ndim == 2:
        wav = wav.T.mean(dim=0)
    else:
        wav = wav.squeeze()
    return wav, sr


def get_first_coi(sources: torch.Tensor, coi_idx: int = 0) -> torch.Tensor:
    """Extract the first COI source (index 0 for single-COI / first multi-COI)."""
    return make_mono(sources[coi_idx])


def _run_models(wav_path: str, common_sr: int, out_dir: Path,
                device: str) -> list[tuple[Path, str]]:
    """Run all 3 separation models on *wav_path*.  Returns list of
    (saved_wav_path, display_label) in the order they should appear in the plot.
    """
    results: list[tuple[Path, str]] = []

    # -- 1. SuDoRM-RF -------------------------------------------------------
    print(f"\n--- SuDoRM-RF ---")
    print(f"  Checkpoint: {SUDORMRF_CKPT}")
    sudormrf = SeparationInference.from_checkpoint(SUDORMRF_CKPT, device=device)
    print(f"  Sample rate: {sudormrf.sample_rate} Hz")
    with torch.inference_mode():
        sources = sudormrf.separate(wav_path)
    plane = make_mono(sources[SUDORMRF_COI])
    plane = resample_if_needed(plane, sudormrf.sample_rate, common_sr)
    p = out_dir / "separated_plane_sudormrf.wav"
    save_wav(p, plane, common_sr)
    results.append((p, "SuDoRM-RF — Separated Plane"))
    print(f"  Saved -> {p}")

    # -- 2. CLAPSep ---------------------------------------------------------
    print(f"\n--- CLAPSep ---")
    clapsep = CLAPSepInference.from_pretrained(
        device=device, text_pos="airplane engine", text_neg="",
    )
    print(f"  Sample rate: {clapsep.sample_rate} Hz")
    with torch.inference_mode():
        sources = clapsep.separate(wav_path)
    plane = make_mono(sources[CLAPSEP_COI])
    plane = resample_if_needed(plane, clapsep.sample_rate, common_sr)
    p = out_dir / "separated_plane_clapsep.wav"
    save_wav(p, plane, common_sr)
    results.append((p, "CLAPSep — Separated Plane"))
    print(f"  Saved -> {p}")

    # -- 3. TUSS ------------------------------------------------------------
    print(f"\n--- TUSS ---")
    config_path = Path(TUSS_CKPT) / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        coi_prompts = cfg.get("coi_prompts", ["airplane"])
        bg_prompt = cfg.get("bg_prompt", "background")
    else:
        coi_prompts, bg_prompt = ["airplane", "birds"], "background"
    print(f"  Checkpoint: {TUSS_CKPT}")
    print(f"  COI: {coi_prompts}, BG: {bg_prompt}")
    tuss = TUSSInference.from_checkpoint(
        TUSS_CKPT, device=device, coi_prompt=coi_prompts, bg_prompt=bg_prompt,
    )
    print(f"  Sample rate: {tuss.sample_rate} Hz")
    with torch.inference_mode():
        sources = tuss.separate(wav_path)
    plane = get_first_coi(sources, 0)
    plane = resample_if_needed(plane, tuss.sample_rate, common_sr)
    p = out_dir / "separated_plane_tuss.wav"
    save_wav(p, plane, common_sr)
    results.append((p, "TUSS — Separated Plane"))
    print(f"  Saved -> {p}")

    return results


# ---------------------------------------------------------------------------
# Single-WAV mode (mixture only)
# ---------------------------------------------------------------------------

def run_single_wav(wav_path: str, out_dir: Path, common_sr: int,
                   device: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save resampled mixture
    mixture, mix_sr = load_audio(wav_path)
    mixture = make_mono(mixture)
    mixture = resample_if_needed(mixture, mix_sr, common_sr)
    mix_path = out_dir / "mixture.wav"
    save_wav(mix_path, mixture, common_sr)
    print(f"Mixture saved -> {mix_path}  (sr={mix_sr} Hz)")

    # Run separators
    sep_results = _run_models(wav_path, common_sr, out_dir, device)

    # Plot: [mixture, sudo, clap, tuss]
    paths = [mix_path] + [p for p, _ in sep_results]
    titles = ["Original Mixture"] + [t for _, t in sep_results]

    print("\nPlotting comparison ...")
    specs, metrics = _compute_specs_and_metrics(paths, common_sr)
    save_png = out_dir / "spectrogram_comparison_all_separators.png"
    plot_combined_spectrograms(
        specs, titles, save_png, sr=common_sr,
        metrics=metrics, ref_idx=0, delta_indices=[1, 2, 3],
    )
    print(f"Done. Figure -> {save_png}")


# ---------------------------------------------------------------------------
# 3-WAV mode (COI + BG + Mixture as ground truth)
# ---------------------------------------------------------------------------

def run_three_wav(coi_path: str, bg_path: str, mixture_path: str,
                  out_dir: Path, common_sr: int, device: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load & resample ground-truth sources
    print(f"\nLoading ground truth:")
    coi, coi_sr = load_audio(coi_path)
    coi = make_mono(coi)
    coi = resample_if_needed(coi, coi_sr, common_sr)
    coi_out = out_dir / "coi_ground_truth.wav"
    save_wav(coi_out, coi, common_sr)
    print(f"  COI  -> {coi_out}")

    bg, bg_sr = load_audio(bg_path)
    bg = make_mono(bg)
    bg = resample_if_needed(bg, bg_sr, common_sr)
    bg_out = out_dir / "bg_ground_truth.wav"
    save_wav(bg_out, bg, common_sr)
    print(f"  BG   -> {bg_out}")

    # Save resampled mixture
    mixture, mix_sr = load_audio(mixture_path)
    mixture = make_mono(mixture)
    mixture = resample_if_needed(mixture, mix_sr, common_sr)
    mix_out = out_dir / "mixture.wav"
    save_wav(mix_out, mixture, common_sr)
    print(f"  Mixture -> {mix_out}  (sr={mix_sr} Hz)")

    # Run separators on the mixture
    sep_results = _run_models(mixture_path, common_sr, out_dir, device)

    # Plot: [COI, BG, Mixture, sudo, clap, tuss]
    paths = [coi_out, bg_out, mix_out] + [p for p, _ in sep_results]
    titles = [
        "COI (Ground Truth)",
        "Background (Ground Truth)",
        "Mixture (Model Input)",
    ] + [t for _, t in sep_results]

    print("\nPlotting comparison ...")
    specs, metrics = _compute_specs_and_metrics(paths, common_sr)
    save_png = out_dir / "spectrogram_comparison_three_wav.png"
    plot_combined_spectrograms(
        specs, titles, save_png, sr=common_sr,
        metrics=metrics, ref_idx=0, delta_indices=[3, 4, 5],
    )
    print(f"Done. Figure -> {save_png}")


def _compute_specs_and_metrics(
    paths: list[Path], common_sr: int,
) -> tuple[list[np.ndarray], list[dict]]:
    specs, metrics_list = [], []
    for p in paths:
        wav, sr = load_wav(p)
        spec, _ = compute_spectrogram(wav, sr=sr, display_sr=common_sr)
        specs.append(spec)
        metrics_list.append(compute_energy_metrics(wav, sr))
    return specs, metrics_list


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare 3 separation models on an airplane-COI recording",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Real-world mixture only
  %(prog)s recording.wav

  # Ground-truth sources + mixture
  %(prog)s --coi plane.wav --bg bg.wav --mixture mix.wav
""",
    )
    parser.add_argument(
        "wav", nargs="?", default=DEFAULT_MIXTURE,
        help="Mixture WAV (single-file mode; default: airplane recording)")
    parser.add_argument("--coi", help="Ground-truth COI WAV (enables 3-file mode)")
    parser.add_argument("--bg", help="Ground-truth background WAV (requires --coi)")
    parser.add_argument("--mixture", help="Mixture WAV (requires --coi)")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--common-sr", type=int, default=COMMON_SR)

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.coi:
        if not args.bg or not args.mixture:
            parser.error("--coi requires --bg and --mixture")
        run_three_wav(
            coi_path=args.coi, bg_path=args.bg, mixture_path=args.mixture,
            out_dir=args.output_dir, common_sr=args.common_sr, device=device,
        )
    else:
        run_single_wav(
            wav_path=args.wav, out_dir=args.output_dir,
            common_sr=args.common_sr, device=device,
        )
