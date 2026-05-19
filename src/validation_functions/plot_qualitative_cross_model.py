#!/usr/bin/env python3
"""
plot_qualitative_cross_model.py — Cross-model spectrogram comparison
for the Results & Discussion chapter (Section 4.1.4).

For each example index k, produces one figure with:
  Rows    = one per model (SuDoRM-RF, CLAPSep, TUSS multi)
  Columns = Mixture input | Clean COI reference | Separated output | Residual

This layout makes it easy to compare how each architecture handles the
same example slot, revealing characteristic artefacts:
  • SuDoRM-RF: time-domain masking artefacts, spectral leakage
  • CLAPSep:   CLAP-conditioning biases, over-smoothing
  • TUSS:      prompt-induced harmonic distortion, cross-talk

Output: ``final_results/dissertation_figures/qualitative/cross_model/``
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import librosa
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
FINAL_RESULTS_DIR = SCRIPT_DIR / "final_results"
OUTPUT_DIR = FINAL_RESULTS_DIR / "dissertation_figures" / "qualitative" / "cross_model"

EXAMPLES = {
    "SuDoRM-RF":    FINAL_RESULTS_DIR / "sudormrf_airplane_examples" / "pann_finetuned" / "mixture_sep",
    "CLAPSep":      FINAL_RESULTS_DIR / "clapsep_airplane_examples" / "pann_finetuned" / "mixture_sep",
    "TUSS (multi)": FINAL_RESULTS_DIR / "tuss_multiclass_airplane_examples" / "pann_finetuned" / "mixture_sep",
}

# ── Style ───────────────────────────────────────────────────────────────────
PALETTE = {
    "sudormrf":     "#636EFA",
    "clapsep":      "#EF553B",
    "tuss_multi":   "#00CC96",
    "text":         "#1f1f1f",
    "bg":           "#FFFFFF",
}

MODEL_COLORS = {
    "SuDoRM-RF":     PALETTE["sudormrf"],
    "CLAPSep":       PALETTE["clapsep"],
    "TUSS (multi)":  PALETTE["tuss_multi"],
}

FONT_FAMILY = "Times New Roman, Nimbus Roman, serif"
FONT = dict(family=FONT_FAMILY, size=11, color=PALETTE["text"])
TITLE_FONT = dict(family=FONT_FAMILY, size=14, color=PALETTE["text"])

LAYOUT_BASE = dict(
    template="plotly_white", font=FONT, title_font=TITLE_FONT,
    paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["bg"],
    margin=dict(l=70, r=20, t=90, b=50),
)

PNG_SCALE = 2

# Spectrogram params (high time resolution)
N_FFT = 2048
HOP_LENGTH = 128
N_MELS = 128


# ── Helpers ─────────────────────────────────────────────────────────────────

def _load_wav(path: Path):
    y, sr = librosa.load(str(path), sr=None, mono=True)
    return y, sr


def _mel_spec(y: np.ndarray, sr: int):
    fmax = sr // 2
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmax=fmax,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=HOP_LENGTH)
    freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=0, fmax=fmax)
    return S_db, times, freqs


def _si_snr(ref: np.ndarray, est: np.ndarray) -> float:
    ref = ref - ref.mean()
    est = est - est.mean()
    denom = float(np.dot(ref, ref))
    if denom <= 1e-12:
        return float("-inf")
    alpha = float(np.dot(est, ref)) / denom
    target = alpha * ref
    noise = est - target
    n = float(np.dot(noise, noise))
    t = float(np.dot(target, target))
    if n <= 1e-12 or t <= 1e-12:
        return float("-inf")
    return 10.0 * np.log10(t / n)


def _si_snri(clean: np.ndarray, mix: np.ndarray, sep: np.ndarray) -> float:
    return _si_snr(clean, sep) - _si_snr(clean, mix)


def build_cross_model_figure(k: int) -> Tuple[go.Figure, str]:
    """Build a (models × 4) spectrogram grid for example index k.

    Returns (figure, case_label) where case_label is 'best'/'typical'/'worst'
    based on the average SI-SNRi across available models.
    """

    rows = []  # list of (model_name, si_snri, S_mix, S_clean, S_sep, S_res, sr)
    all_spectrogram_values = []

    for model_name, ex_dir in EXAMPLES.items():
        if not ex_dir.exists():
            continue
        try:
            mix, sr_m   = _load_wav(ex_dir / f"mixture_created_{k}.wav")
            clean, sr_c = _load_wav(ex_dir / f"mixture_coi_clean_{k}.wav")
            sep, sr_s   = _load_wav(ex_dir / f"mixture_separated_coi_head_{k}.wav")
        except Exception:
            continue

        # Resample to common SR if needed
        if sr_c != sr_m:
            clean = librosa.resample(clean, orig_sr=sr_c, target_sr=sr_m)
        if sr_s != sr_m:
            sep = librosa.resample(sep, orig_sr=sr_s, target_sr=sr_m)

        L = min(len(mix), len(clean), len(sep))
        if L < 1024:
            continue
        mix, clean, sep = mix[:L], clean[:L], sep[:L]
        residual = mix - sep
        snri = _si_snri(clean, mix, sep)

        S_mix, _, _ = _mel_spec(mix, sr_m)
        S_clean, _, _ = _mel_spec(clean, sr_m)
        S_sep, _, _ = _mel_spec(sep, sr_m)
        S_res, _, _ = _mel_spec(residual, sr_m)

        all_spectrogram_values.extend([S_mix, S_clean, S_sep, S_res])
        rows.append((model_name, snri, S_mix, S_clean, S_sep, S_res, sr_m))

    if not rows:
        raise ValueError(f"No valid data for k={k}")

    # Determine case label from average SI-SNRi
    avg_snri = np.mean([r[1] for r in rows])
    if avg_snri >= 3.0:
        case = "best"
    elif avg_snri <= -5.0:
        case = "worst"
    else:
        case = "typical"

    # Shared colour scale across all panels
    all_vals = np.concatenate([S.flatten() for S in all_spectrogram_values])
    zmin, zmax = float(all_vals.min()), float(all_vals.max())

    n_models = len(rows)
    fig = make_subplots(
        rows=n_models, cols=4,
        row_titles=[f"<b>{name}</b><br><span style='font-size:9px'>SI-SNRi = {snri:+.1f} dB</span>"
                    for name, snri, *_ in rows],
        column_titles=["<b>Mixture input</b>", "<b>Clean COI reference</b>",
                       "<b>Separated output</b>", "<b>Residual (mix − sep)</b>"],
        horizontal_spacing=0.02,
        vertical_spacing=0.08,
    )

    for r_idx, (name, snri, S_mix, S_clean, S_sep, S_res, sr) in enumerate(rows, 1):
        for c_idx, S in enumerate([S_mix, S_clean, S_sep, S_res], 1):
            fig.add_trace(go.Heatmap(
                z=S,
                colorscale="Viridis",
                zmin=zmin, zmax=zmax,
                showscale=(r_idx == n_models and c_idx == 4),
                colorbar=dict(title="dB", x=1.01, thickness=10, len=0.25, y=0.12)
                    if (r_idx == n_models and c_idx == 4) else None,
                hovertemplate="%{z:.1f} dB<extra></extra>",
            ), row=r_idx, col=c_idx)

            if r_idx == n_models:
                fig.update_xaxes(title_text="Time (s)", row=r_idx, col=c_idx)
            else:
                fig.update_xaxes(showticklabels=False, row=r_idx, col=c_idx)

        # y-axis only on leftmost column
        fig.update_yaxes(title_text="Frequency (Hz)", row=r_idx, col=1)
        for c in range(2, 5):
            fig.update_yaxes(showticklabels=False, row=r_idx, col=c)

    subtitle = (f"Example index {k} · Average SI-SNRi = {avg_snri:+.1f} dB  ·  "
                "Rows = models (independent validation runs)  ·  "
                "Columns = processing stages")

    fig.update_layout(
        title=dict(text=f"<b>Cross-model separation comparison ({case} case)</b><br><sup>{subtitle}</sup>"),
        height=260 * n_models + 60, width=1200,
        **LAYOUT_BASE,
    )
    return fig, case


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 72)
    print("Cross-model qualitative spectrogram comparison")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find shared indices across all models
    available = {k: v for k, v in EXAMPLES.items() if v.exists()}
    if not available:
        print("No example directories found.")
        return

    shared = set.intersection(*[set(int(f.stem.rsplit("_", 1)[-1])
                                     for f in p.glob("mixture_created_*.wav"))
                                for p in available.values()])
    shared = sorted(shared)
    print(f"Shared example indices: {shared}")
    if not shared:
        print("No shared indices.")
        return

    summary = []
    for k in shared:
        try:
            fig, case = build_cross_model_figure(k)
            fname = f"cross_model_{case}_k{k}"
            png = OUTPUT_DIR / f"{fname}.png"
            html = OUTPUT_DIR / f"{fname}.html"
            try:
                fig.write_image(str(png), scale=PNG_SCALE)
                print(f"  Wrote {png.name}")
            except Exception as e:
                print(f"  [PNG skipped: {e}]")
            fig.write_html(str(html))
            summary.append(f"k={k}  case={case}")
        except Exception as e:
            print(f"  ERROR k={k}: {e}")

    print("\n" + "=" * 72)
    print("Summary")
    for line in summary:
        print(f"  {line}")
    print("=" * 72)


if __name__ == "__main__":
    main()
