#!/usr/bin/env python3
"""
Generate paper-friendly 2-column spectrogram comparison figures (EN + NL).

Uses pre-separated audio files from the dissertation figures directory.
Produces narrower, higher-contrast spectrograms with larger fonts.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

from src.common.paths import get_project_root, get_output_dir, setup_python_path
setup_python_path()

from validation_functions.demo_separation import (
    compute_spectrogram,
    compute_energy_metrics,
    load_wav,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AUDIO_DIR = (
    get_project_root()
    / "src/validation_functions/final_results/dissertation_figures/qualitative/audio"
)
OUTPUT_DIR = get_output_dir() / "paper/spectrograms"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# The Risoux plane scenario: 4 files to compare
RISOUX_FILES = {
    "mixture": AUDIO_DIR / "mixture_risoux_plane.wav",
    "sudormrf": AUDIO_DIR / "separated_plane_sudormrf_risoux_plane.wav",
    "clapsep": AUDIO_DIR / "separated_plane_clapsep_risoux_plane.wav",
    "tuss": AUDIO_DIR / "separated_plane_tuss_risoux_plane.wav",
}

SPECTROGRAM_SR = 44100  # display sample rate

# ---------------------------------------------------------------------------
# Language labels
# ---------------------------------------------------------------------------
LABELS = {
    "en": {
        "rows": {
            "mixture": "Original Mixture",
            "sudormrf": "SuDoRM-RF",
            "clapsep": "CLAPSep",
            "tuss": "TUSS",
        },
        "ylab": "Freq (Hz)",
        "xlab": "Time (s)",
        "cbar": "dB",
        "suptitle": "Spectrogram comparison — Risoux field recording (airplane COI)",
        "figlabel": "spectrogram_risoux_plane_en.png",
    },
    "nl": {
        "rows": {
            "mixture": "Origineel mengsel",
            "sudormrf": "SuDoRM-RF",
            "clapsep": "CLAPSep",
            "tuss": "TUSS",
        },
        "ylab": "Freq (Hz)",
        "xlab": "Tijd (s)",
        "cbar": "dB",
        "suptitle": "Spectrogramvergelijking — Risoux-veldopname (vliegtuig-doelgeluid)",
        "figlabel": "spectrogram_risoux_plane_nl.png",
    },
}

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
HUMAN_HEARING_MAX = 20000
N_MELS = 128


def mel_to_hz(mel_val: float) -> float:
    return 700.0 * (10.0 ** (mel_val / 2595.0) - 1.0)


def fmt_hz(hz: float) -> str:
    return f"{hz / 1000:.1f}k" if hz >= 1000 else f"{int(hz)}"


def make_paper_spectrogram(lang: str) -> None:
    t = LABELS[lang]
    keys = ["mixture", "sudormrf", "clapsep", "tuss"]

    # Load audio and compute spectrograms
    specs, metrics_list = [], []
    for key in keys:
        wav, sr = load_wav(RISOUX_FILES[key])
        spec, used_sr = compute_spectrogram(
            wav, sr=sr, display_sr=SPECTROGRAM_SR, n_mels=N_MELS
        )
        specs.append(spec)
        metrics_list.append(compute_energy_metrics(wav, sr=used_sr))

    n_plots = len(specs)

    # Paper-optimised dimensions: narrow but readable
    fig = plt.figure(figsize=(7.0, 2.2 * n_plots))

    gs = gridspec.GridSpec(n_plots, 1, figure=fig, hspace=0.55)

    nyquist = SPECTROGRAM_SR / 2.0
    display_max_hz = min(HUMAN_HEARING_MAX, nyquist)
    mel_max = 2595.0 * np.log10(1.0 + nyquist / 700.0)
    mel_display_max = 2595.0 * np.log10(1.0 + display_max_hz / 700.0)

    # Global normalisation
    global_vmin = min(s.min() for s in specs)
    global_vmax = max(s.max() for s in specs)
    normalized = [s - global_vmin for s in specs]
    norm_vmax = global_vmax - global_vmin

    # Use perceptually uniform grayscale-colormap hybrid for better print visibility
    cmap = plt.cm.magma

    for row, (key, spec) in enumerate(zip(keys, normalized)):
        ax = fig.add_subplot(gs[row, 0])

        time_sec = spec.shape[1] * HOP_LENGTH / SPECTROGRAM_SR
        n_mels_count = spec.shape[0]
        display_max_bin = (mel_display_max / mel_max) * (n_mels_count - 1)

        im = ax.imshow(
            spec,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            extent=(0, time_sec, 0, n_mels_count),
            vmin=0,
            vmax=norm_vmax,
        )

        # Styling
        ax.set_title(
            t["rows"][key], fontsize=13, fontweight="bold", pad=4, fontfamily="serif"
        )
        ax.set_ylabel(t["ylab"], fontsize=11, fontfamily="serif")
        ax.set_ylim(0, display_max_bin)

        if row == n_plots - 1:
            ax.set_xlabel(t["xlab"], fontsize=11, fontfamily="serif")
        else:
            ax.set_xticklabels([])

        # Y-axis Hz ticks
        num_ticks = 5
        mel_ticks = np.linspace(0.0, mel_display_max, num_ticks)
        mel_bin_indices = (mel_ticks / mel_max) * (n_mels_count - 1)
        hz_labels = [fmt_hz(mel_to_hz(m)) for m in mel_ticks]
        ax.set_yticks(mel_bin_indices)
        ax.set_yticklabels(hz_labels, fontsize=9, fontfamily="serif")
        ax.tick_params(axis="x", labelsize=9)

        # RMS annotation (compact)
        m = metrics_list[row]
        ax.text(
            0.99,
            0.96,
            f"RMS: {m['rms_db']:.1f} dBFS",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            fontfamily="monospace",
            color="#f0f0f0",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="#1a1a2e",
                alpha=0.55,
                edgecolor="none",
            ),
        )

        # Thin border
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    # Single compact colorbar at bottom
    cbar_ax = fig.add_axes([0.15, -0.02, 0.7, 0.018])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(t["cbar"], fontsize=10, fontfamily="serif")
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        t["suptitle"], fontsize=14, fontweight="bold", fontfamily="serif", y=1.01
    )

    out_path = OUTPUT_DIR / t["figlabel"]
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    for lang in ["en", "nl"]:
        print(f"Generating {lang.upper()} spectrogram...")
        make_paper_spectrogram(lang)
    print("Done.")
