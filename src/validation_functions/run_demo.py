#!/usr/bin/env python3
"""Run TUSS separation on audio with bird extraction.

Two modes
--------
Single-WAV  (default)    python run_demo.py [mixture.wav]
3-WAV       (--coi ...)  python run_demo.py --coi birds.wav --bg bg.wav [--mixture mix.wav]

TUSS -> multi-COI checkpoint (airplane+birds), extracting the bird source.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml

from src.common.paths import get_data_dir, get_project_root, get_output_dir, setup_python_path
setup_python_path()

_torchaudio_load_original = torchaudio.load


def _sf_load(path, *args, **kwargs):
    data, _sr = sf.read(str(path), always_2d=True, dtype="float32")
    wav = torch.from_numpy(data.T.copy())
    return wav, _sr


torchaudio.load = _sf_load

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
DEFAULT_MIXTURE = str(
    get_data_dir() / "misclassifications/"
    "239_as_is_sep_cls_['plane',_'wind',_'biophony']_conf0.456_S4A04430_20180716_113000.wav"
)

TUSS_CKPT = str(get_project_root() / "src/models/tuss/checkpoints/multi_coi_11_05")

OUT_DIR = get_output_dir() / "demo"


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
    was_1d = wav.ndim == 1
    if was_1d:
        wav = wav.unsqueeze(0)
    resampler = torchaudio.transforms.Resample(src_sr, tgt_sr)
    result = resampler(wav)
    if was_1d:
        result = result.squeeze(0)
    return result


def save_wav(path: Path, wav: torch.Tensor, sr: int) -> None:
    wav = make_mono(wav)
    wav = peak_normalize(wav)
    sf.write(str(path), wav.numpy(), sr)


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    wav_np, sr = sf.read(str(path))
    wav = torch.from_numpy(wav_np.astype(np.float32))
    if wav.ndim == 2:
        wav = wav.T.mean(dim=0)
    else:
        wav = wav.squeeze()
    return wav, sr


def mix_coi_bg(coi: torch.Tensor, bg: torch.Tensor, snr_db: float) -> torch.Tensor:
    import random

    min_len = min(coi.shape[-1], bg.shape[-1])
    coi = coi[..., :min_len]
    if bg.shape[-1] > min_len:
        start = random.randint(0, bg.shape[-1] - min_len)
        bg = bg[..., start : start + min_len]
    else:
        bg = bg[..., :min_len]
    coi_power = coi.pow(2).mean()
    bg_power = bg.pow(2).mean()
    if bg_power < 1e-12:
        return coi
    alpha = torch.sqrt(coi_power / (bg_power * (10 ** (snr_db / 10))))
    return coi + alpha * bg


def get_tuss_prompts(ckpt_path: str | Path) -> tuple[list[str], str]:
    config_path = Path(ckpt_path) / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        coi_prompts = cfg.get("coi_prompts", ["airplane", "birds"])
        bg_prompt = cfg.get("bg_prompt", "background")
    else:
        coi_prompts, bg_prompt = ["airplane", "birds"], "background"
    return coi_prompts, bg_prompt


# ---------------------------------------------------------------------------
# Single-WAV mode (mixture only)
# ---------------------------------------------------------------------------


def run_single_wav(
    wav_path: str,
    out_dir: Path,
    device: str,
    coi_prompts: list[str],
    bg_prompt: str,
    tuss_ckpt: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading TUSS from {tuss_ckpt}")
    print(f"  COI prompts: {coi_prompts}, BG prompt: {bg_prompt}")
    inferencer = TUSSInference.from_checkpoint(
        tuss_ckpt,
        device=device,
        coi_prompt=coi_prompts,
        bg_prompt=bg_prompt,
    )
    model_sr = inferencer.sample_rate
    print(f"  Sample rate: {model_sr} Hz")

    print(f"\nSeparating: {wav_path}")
    with torch.inference_mode():
        sources = inferencer.separate(wav_path)

    num_cois = len(coi_prompts)
    coi_sources = [make_mono(sources[i]) for i in range(num_cois)]
    bg_source = make_mono(sources[num_cois])

    # Save mixture
    mixture, mix_sr = load_audio(wav_path)
    mixture = make_mono(mixture)
    mixture = resample_if_needed(mixture, mix_sr, model_sr)
    mix_path = out_dir / "mixture.wav"
    save_wav(mix_path, mixture, model_sr)
    print(f"  Mixture -> {mix_path}")

    # Save separated sources
    sep_paths: list[Path] = []
    for i, (label, src) in enumerate(
        [(p, coi_sources[i]) for i, p in enumerate(coi_prompts)]
        + [(bg_prompt, bg_source)]
    ):
        p = out_dir / f"separated_{label.replace(' ', '_')}.wav"
        save_wav(p, src, model_sr)
        sep_paths.append(p)
        print(f"  Separated {label} -> {p}")

    # Plot: mixture + all separated sources
    paths = [mix_path] + sep_paths
    titles = ["Original Mixture"] + [
        f"Separated {lbl}" for lbl in coi_prompts + [bg_prompt]
    ]

    specs, metrics = _compute_specs_and_metrics(paths, model_sr)
    save_png = out_dir / "spectrogram_tuss_separation.png"
    plot_combined_spectrograms(
        specs,
        titles,
        save_png,
        sr=model_sr,
        metrics=metrics,
        ref_idx=0,
        delta_indices=list(range(1, len(paths))),
    )
    print(f"Figure -> {save_png}")


# ---------------------------------------------------------------------------
# 3-WAV mode (COI + BG + optional mixture)
# ---------------------------------------------------------------------------


def run_three_wav(
    coi_path: str,
    bg_path: str,
    mixture_path: Optional[str],
    out_dir: Path,
    device: str,
    coi_prompts: list[str],
    bg_prompt: str,
    tuss_ckpt: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading TUSS from {tuss_ckpt}")
    print(f"  COI prompts: {coi_prompts}, BG prompt: {bg_prompt}")
    inferencer = TUSSInference.from_checkpoint(
        tuss_ckpt,
        device=device,
        coi_prompt=coi_prompts,
        bg_prompt=bg_prompt,
    )
    model_sr = inferencer.sample_rate
    print(f"  Sample rate: {model_sr} Hz")

    # Load ground-truth sources
    print(f"\nLoading ground truth:")
    coi, coi_sr = load_audio(coi_path)
    coi = make_mono(coi)
    coi = resample_if_needed(coi, coi_sr, model_sr)
    coi_out = out_dir / "coi_ground_truth.wav"
    save_wav(coi_out, coi, model_sr)
    print(f"  COI  -> {coi_out}")

    bg, bg_sr = load_audio(bg_path)
    bg = make_mono(bg)
    bg = resample_if_needed(bg, bg_sr, model_sr)
    bg_out = out_dir / "bg_ground_truth.wav"
    save_wav(bg_out, bg, model_sr)
    print(f"  BG   -> {bg_out}")

    # Load or generate mixture
    snr_info = None
    if mixture_path is not None:
        print(f"\nLoading mixture from file: {mixture_path}")
        mixture, mix_sr = load_audio(mixture_path)
        mixture = make_mono(mixture)
        mixture = resample_if_needed(mixture, mix_sr, model_sr)
    else:
        import random

        snr_db = random.uniform(-3.0, 3.0)
        snr_info = f" (SNR = {snr_db:.1f} dB)"
        print(f"\nGenerating mixture at {snr_db:.1f} dB SNR:")
        coi_mix = coi.clone()
        bg_mix = bg.clone()
        mixture = mix_coi_bg(coi_mix, bg_mix, snr_db)

    mix_out = out_dir / "mixture.wav"
    save_wav(mix_out, mixture, model_sr)
    print(f"  Mixture -> {mix_out}{snr_info or ''}")

    # Write temp mixture for separation
    mix_temp = out_dir / "_mix_for_sep.wav"
    save_wav(mix_temp, mixture, model_sr)

    # Run TUSS separation
    print(f"\nSeparating mixture ...")
    with torch.inference_mode():
        sources = inferencer.separate(str(mix_temp))

    num_cois = len(coi_prompts)
    coi_sources = [make_mono(sources[i]) for i in range(num_cois)]
    bg_source = make_mono(sources[num_cois])

    sep_paths: list[Path] = []
    for i, label in enumerate(coi_prompts):
        p = out_dir / f"separated_{label.replace(' ', '_')}.wav"
        save_wav(p, coi_sources[i], model_sr)
        sep_paths.append(p)
        print(f"  Separated {label} -> {p}")

    p = out_dir / f"separated_{bg_prompt.replace(' ', '_')}.wav"
    save_wav(p, bg_source, model_sr)
    sep_paths.append(p)
    print(f"  Separated {bg_prompt} -> {p}")

    # Plot: COI GT, BG GT, Mixture, then separated sources
    mix_title = "Mixture (COI+BG)" if snr_info else "Mixture (Model Input)"
    paths = [coi_out, bg_out, mix_out] + sep_paths
    titles = [
        "COI (Ground Truth)",
        "Background (Ground Truth)",
        mix_title,
    ] + [f"Separated {lbl}" for lbl in coi_prompts + [bg_prompt]]

    specs, metrics = _compute_specs_and_metrics(paths, model_sr)
    sep_start_idx = 3  # first separated source is at index 3
    save_png = out_dir / "spectrogram_tuss_separation_three_wav.png"
    plot_combined_spectrograms(
        specs,
        titles,
        save_png,
        sr=model_sr,
        metrics=metrics,
        ref_idx=0,  # COI ground truth as reference
        delta_indices=list(range(sep_start_idx, len(paths))),
    )
    print(f"Figure -> {save_png}")


def _compute_specs_and_metrics(
    paths: list[Path],
    sr: int,
) -> tuple[list[np.ndarray], list[dict]]:
    specs, metrics_list = [], []
    for p in paths:
        wav, _ = load_wav(p)
        spec, _ = compute_spectrogram(wav, sr=sr)
        specs.append(spec)
        metrics_list.append(compute_energy_metrics(wav, sr))
    return specs, metrics_list


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run TUSS separation and plot spectrograms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Real-world mixture only
  %(prog)s recording.wav

  # Ground-truth sources (auto-mix at random SNR in [-3, 3] dB)
  %(prog)s --coi birds.wav --bg bg.wav

  # Ground-truth sources + pre-made mixture
  %(prog)s --coi birds.wav --bg bg.wav --mixture mix.wav
""",
    )
    parser.add_argument(
        "wav",
        nargs="?",
        default=DEFAULT_MIXTURE,
        help="Mixture WAV (single-file mode; default: airplane recording)",
    )
    parser.add_argument("--coi", help="Ground-truth COI WAV (enables 3-file mode)")
    parser.add_argument("--bg", help="Ground-truth background WAV (requires --coi)")
    parser.add_argument(
        "--mixture",
        help="Mixture WAV (optional; if omitted, COI+BG are mixed at a random SNR in [-3, 3] dB)",
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tuss-ckpt", default=TUSS_CKPT)

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    coi_prompts, bg_prompt = get_tuss_prompts(args.tuss_ckpt)

    if args.coi:
        if not args.bg:
            parser.error("--coi requires --bg")
        run_three_wav(
            coi_path=args.coi,
            bg_path=args.bg,
            mixture_path=args.mixture,
            out_dir=args.output_dir,
            device=device,
            coi_prompts=coi_prompts,
            bg_prompt=bg_prompt,
            tuss_ckpt=args.tuss_ckpt,
        )
    else:
        run_single_wav(
            wav_path=args.wav,
            out_dir=args.output_dir,
            device=device,
            coi_prompts=coi_prompts,
            bg_prompt=bg_prompt,
            tuss_ckpt=args.tuss_ckpt,
        )
