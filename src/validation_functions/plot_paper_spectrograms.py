#!/usr/bin/env python3
"""
Generate paper-friendly spectrogram comparison figures.

Uses pre-separated audio files from the qualitative dissertation figures
directory. Produces narrower, higher-contrast spectrograms with larger fonts.
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from validation_functions.demo_separation import (
    compute_spectrogram,
    compute_energy_metrics,
    load_wav,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AUDIO_DIR = (
    PROJECT_ROOT
    / "src/validation_functions/final_results/dissertation_figures/qualitative/audio"
)
OUTPUT_DIR = AUDIO_DIR.parent / "paper"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPECTROGRAM_SR = 44100  # display sample rate

# ---------------------------------------------------------------------------
# Figure definitions
# ---------------------------------------------------------------------------
ROW_LABELS = {
    "mixture": "Original Mixture",
    "mixture_model_input": "Mixture (Model Input)",
    "coi_gt": "COI (Ground Truth)",
    "bg_gt": "Background (Ground Truth)",
    "sudormrf": "SuDoRM-RF",
    "clapsep": "CLAPSep",
    "tuss": "TUSS",
    "separated_airplane": "Separated Airplane",
    "separated_birds": "Separated Birds",
    "separated_background": "Separated Background",
}

SCENARIOS = [
    {
        "stem": "spectrogram_comparison_all_separators_clean",
        "title": "Spectrogram comparison - clean recording (airplane COI)",
        "files": [
            ("mixture", "mixture_clean.wav"),
            ("sudormrf", "separated_plane_sudormrf_clean.wav"),
            ("clapsep", "separated_plane_clapsep_clean.wav"),
            ("tuss", "separated_plane_tuss_clean.wav"),
        ],
    },
    {
        "stem": "spectrogram_comparison_all_separators_risoux_plane",
        "title": "Spectrogram comparison - Risoux field recording (airplane COI)",
        "files": [
            ("mixture", "mixture_risoux_plane.wav"),
            ("sudormrf", "separated_plane_sudormrf_risoux_plane.wav"),
            ("clapsep", "separated_plane_clapsep_risoux_plane.wav"),
            ("tuss", "separated_plane_tuss_risoux_plane.wav"),
        ],
    },
    {
        "stem": "spectrogram_comparison_three_wav_rain_plane",
        "title": "Spectrogram comparison - rain background (airplane COI)",
        "files": [
            ("coi_gt", "coi_ground_truth_rain_plane.wav"),
            ("bg_gt", "bg_ground_truth_rain_plane.wav"),
            ("mixture_model_input", "mixture_rain_plane.wav"),
            ("sudormrf", "separated_plane_sudormrf_rain_plane.wav"),
            ("clapsep", "separated_plane_clapsep_rain_plane.wav"),
            ("tuss", "separated_plane_tuss_rain_plane.wav"),
        ],
    },
    {
        "stem": "spectrogram_comparison_three_wav_wind_plane",
        "title": "Spectrogram comparison - wind background (airplane COI)",
        "files": [
            ("coi_gt", "coi_ground_truth_wind_plane.wav"),
            ("bg_gt", "bg_ground_truth_wind_plane.wav"),
            ("mixture_model_input", "mixture_wind_plane.wav"),
            ("sudormrf", "separated_plane_sudormrf_wind_plane.wav"),
            ("clapsep", "separated_plane_clapsep_wind_plane.wav"),
            ("tuss", "separated_plane_tuss_wind_plane.wav"),
        ],
    },
    {
        "stem": "spectrogram_tuss_separation_risoux_bird",
        "title": "TUSS spectrogram separation - Risoux field recording",
        "files": [
            ("mixture", "mixture_risoux_bird.wav"),
            ("separated_airplane", "separated_airplane_risoux_bird.wav"),
            ("separated_birds", "separated_birds_risoux_bird.wav"),
            ("separated_background", "separated_background_risoux_bird.wav"),
        ],
    },
    {
        "stem": "spectrogram_tuss_separation_risoux_windy",
        "title": "TUSS spectrogram separation - windy Risoux field recording",
        "files": [
            ("mixture", "mixture_risoux_windy.wav"),
            ("separated_airplane", "separated_airplane_risoux_windy.wav"),
            ("separated_birds", "separated_birds_risoux_windy.wav"),
            ("separated_background", "separated_background_risoux_windy.wav"),
        ],
    },
    {
        "stem": "spectrogram_tuss_separation_three_wav_rain_bird",
        "title": "TUSS spectrogram separation - rain background",
        "files": [
            ("coi_gt", "coi_ground_truth_rain_bird.wav"),
            ("bg_gt", "bg_ground_truth_rain_bird.wav"),
            ("mixture_model_input", "mixture_rain_bird.wav"),
            ("separated_airplane", "separated_airplane_rain_bird.wav"),
            ("separated_birds", "separated_birds_rain_bird.wav"),
            ("separated_background", "separated_background_rain_bird.wav"),
        ],
    },
]

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


def validate_audio_files() -> None:
    missing = []
    for scenario in SCENARIOS:
        for _, filename in scenario["files"]:
            path = AUDIO_DIR / filename
            if not path.exists():
                missing.append(path)
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing qualitative audio files:\n{missing_list}")


def make_paper_spectrogram(scenario: dict) -> None:
    files = scenario["files"]

    # Load audio and compute spectrograms
    specs, metrics_list = [], []
    for _, filename in files:
        wav, sr = load_wav(AUDIO_DIR / filename)
        spec, _ = compute_spectrogram(
            wav, sr=sr, display_sr=SPECTROGRAM_SR, n_mels=N_MELS
        )
        specs.append(spec)
        metrics_list.append(compute_energy_metrics(wav, sr=sr))

    n_plots = len(specs)

    # Paper-optimised dimensions: narrow but readable
    fig = plt.figure(figsize=(7.0, 1.75 * n_plots + 0.5))

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
    cmap = plt.get_cmap("magma")

    im = None
    for row, ((key, _), spec) in enumerate(zip(files, normalized)):
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
            ROW_LABELS[key], fontsize=12, fontweight="bold", pad=4, fontfamily="serif"
        )
        ax.set_ylabel("Freq (Hz)", fontsize=10, fontfamily="serif")
        ax.set_ylim(0, display_max_bin)

        if row == n_plots - 1:
            ax.set_xlabel("Time (s)", fontsize=10, fontfamily="serif")
        else:
            ax.set_xticklabels([])

        # Y-axis Hz ticks
        num_ticks = 5
        mel_ticks = np.linspace(0.0, mel_display_max, num_ticks)
        mel_bin_indices = (mel_ticks / mel_max) * (n_mels_count - 1)
        hz_labels = [fmt_hz(mel_to_hz(m)) for m in mel_ticks]
        ax.set_yticks(mel_bin_indices)
        ax.set_yticklabels(hz_labels, fontsize=8, fontfamily="serif")
        ax.tick_params(axis="x", labelsize=8)

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
    if im is None:
        raise ValueError(f"No spectrograms configured for {scenario['stem']}")

    cbar_ax = fig.add_axes((0.15, -0.02, 0.7, 0.018))
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Magnitude (dB, normalized)", fontsize=9, fontfamily="serif")
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        scenario["title"], fontsize=13, fontweight="bold", fontfamily="serif", y=1.01
    )

    out_path = OUTPUT_DIR / f"paper_{scenario['stem']}.png"
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    validate_audio_files()
    for scenario in SCENARIOS:
        print(f"Generating {scenario['stem']}...")
        make_paper_spectrogram(scenario)
    print("Done.")
