#!/usr/bin/env python3
"""
plot_qualitative_spectrograms.py — Spectrogram-based qualitative analysis
for the Results & Discussion chapter (Section 4.1.4).

Generates comparison spectrograms from saved example WAV files to visually
inspect separation quality per architecture.  For each representative example
we show:

  1. Mixture input          (what the separator receives)
  2. Clean COI reference    (ground truth — the target sound in isolation)
  3. Separated COI output   (what the separator produces)
  4. Residual / leakage     (mixture minus separated — shows unremoved background
                             or missing COI energy)

Examples are curated per model into three illustrative categories:
  * best     — highest SI-SNRi on that example
  * typical  — median SI-SNRi
  * worst    — lowest SI-SNRi

Output: ``final_results/dissertation_figures/qualitative/``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import librosa
import librosa.display

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
FINAL_RESULTS_DIR = SCRIPT_DIR / "final_results"
OUTPUT_DIR = FINAL_RESULTS_DIR / "dissertation_figures" / "qualitative"

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
FONT = dict(family=FONT_FAMILY, size=12, color=PALETTE["text"])
TITLE_FONT = dict(family=FONT_FAMILY, size=15, color=PALETTE["text"])

LAYOUT_BASE = dict(
    template="plotly_white", font=FONT, title_font=TITLE_FONT,
    paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["bg"],
    margin=dict(l=60, r=20, t=70, b=50),
)

PNG_SCALE = 2

N_FFT = 2048
HOP_LENGTH = 128
N_MELS = 128


# ── Helpers ─────────────────────────────────────────────────────────────────

def _load_wav(path: Path) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(str(path), sr=None, mono=True)
    return y, sr


def _mel_spec(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute log-scaled mel spectrogram. Returns (S_db, times, freqs)."""
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


# ── Curation ────────────────────────────────────────────────────────────────

def _shared_indices(model_dirs: Dict[str, Path]) -> List[int]:
    sets = []
    for p in model_dirs.values():
        files = list(p.glob("mixture_created_*.wav"))
        indices = {int(f.stem.rsplit("_", 1)[-1]) for f in files}
        sets.append(indices)
    return sorted(set.intersection(*sets))


def _rank_examples(model_name: str, ex_dir: Path, shared: List[int]) -> List[Tuple[int, float]]:
    scored = []
    for k in shared:
        try:
            clean, sr_c = _load_wav(ex_dir / f"mixture_coi_clean_{k}.wav")
            mix, sr_m   = _load_wav(ex_dir / f"mixture_created_{k}.wav")
            sep, sr_s   = _load_wav(ex_dir / f"mixture_separated_coi_head_{k}.wav")
        except Exception:
            continue
        if not (sr_c == sr_m == sr_s):
            continue
        L = min(len(clean), len(mix), len(sep))
        if L < 1024:
            continue
        clean, mix, sep = clean[:L], mix[:L], sep[:L]
        snri = _si_snri(clean, mix, sep)
        if not np.isinf(snri):
            scored.append((k, float(snri)))
    scored.sort(key=lambda t: t[1])
    return scored


def _curate(scored: List[Tuple[int, float]]) -> List[Tuple[str, int, float]]:
    if not scored:
        return []
    n = len(scored)
    if n == 1:
        return [("typical", *scored[0])]
    if n == 2:
        return [("worst", *scored[0]), ("best", *scored[-1])]
    return [("worst", *scored[0]), ("typical", *scored[n // 2]), ("best", *scored[-1])]


# ── Figure builder ──────────────────────────────────────────────────────────

def build_example_figure(
    model_name: str, case: str, k: int, si_snri: float, ex_dir: Path
) -> go.Figure:

    mix, sr     = _load_wav(ex_dir / f"mixture_created_{k}.wav")
    clean, sr_c = _load_wav(ex_dir / f"mixture_coi_clean_{k}.wav")
    sep, sr_s   = _load_wav(ex_dir / f"mixture_separated_coi_head_{k}.wav")

    # All should be same SR; if not, resample clean and sep to mix SR
    if sr_c != sr:
        clean = librosa.resample(clean, orig_sr=sr_c, target_sr=sr)
    if sr_s != sr:
        sep = librosa.resample(sep, orig_sr=sr_s, target_sr=sr)

    L = min(len(mix), len(clean), len(sep))
    mix, clean, sep = mix[:L], clean[:L], sep[:L]
    residual = mix - sep

    S_mix,   t_mix,   f_mix   = _mel_spec(mix, sr)
    S_clean, t_clean, f_clean = _mel_spec(clean, sr)
    S_sep,   t_sep,   f_sep   = _mel_spec(sep, sr)
    S_res,   t_res,   f_res   = _mel_spec(residual, sr)

    all_vals = np.concatenate([S_mix.flatten(), S_clean.flatten(), S_sep.flatten(), S_res.flatten()])
    zmin, zmax = float(all_vals.min()), float(all_vals.max())

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=(
            "<b>Mixture input</b>",
            "<b>Clean COI reference</b>",
            "<b>Separated output</b>",
            "<b>Residual (mix − sep)</b>",
        ),
        horizontal_spacing=0.03,
    )

    specs = [(S_mix, t_mix, f_mix), (S_clean, t_clean, f_clean),
             (S_sep, t_sep, f_sep), (S_res, t_res, f_res)]

    for col, (S, t, f) in enumerate(specs, 1):
        fig.add_trace(go.Heatmap(
            z=S, x=t, y=f,
            colorscale="Viridis",
            zmin=zmin, zmax=zmax,
            showscale=(col == 4),  # colour bar on the rightmost panel
            colorbar=dict(title="dB", x=1.02, thickness=12) if col == 4 else None,
            hovertemplate="Time: %{x:.2f}s<br>Freq: %{y:.0f} Hz<extra></extra>",
        ), row=1, col=col)
        fig.update_xaxes(title_text="Time (s)", row=1, col=col)

    fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
    for c in range(2, 5):
        fig.update_yaxes(showticklabels=False, row=1, col=c)

    fig.update_layout(
        title=dict(
            text=f"<b>{model_name}</b> · {case.upper()} example (index {k}) · SI-SNRi = {si_snri:+.1f} dB"
        ),
        height=380, width=1400,
        **LAYOUT_BASE,
    )
    return fig


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 72)
    print("Qualitative spectrogram gallery")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 72)

    available = {k: v for k, v in EXAMPLES.items() if v.exists()}
    if not available:
        print("No example directories found.")
        return

    shared = _shared_indices(available)
    print(f"Shared example indices: {shared}")
    if not shared:
        print("No shared indices.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for model_name, ex_dir in available.items():
        print(f"\n— {model_name} —")
        scored = _rank_examples(model_name, ex_dir, shared)
        if not scored:
            print("  No scorable examples.")
            continue
        curated = _curate(scored)
        print(f"  Selected {len(curated)} example(s):")
        for case, k, snri in curated:
            print(f"    {case:>7s}  k={k}  SI-SNRi={snri:+.1f} dB")
            fig = build_example_figure(model_name, case, k, snri, ex_dir)
            fname = f"qual_{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_{case}_k{k}"
            png = OUTPUT_DIR / f"{fname}.png"
            html = OUTPUT_DIR / f"{fname}.html"
            try:
                fig.write_image(str(png), scale=PNG_SCALE)
                print(f"    Wrote {png.name}")
            except Exception as e:
                print(f"    [PNG skipped: {e}]")
            fig.write_html(str(html))
            summary.append(f"{model_name:12s} {case:>7s} k={k} SI-SNRi={snri:+.1f} dB")

    print("\n" + "=" * 72)
    print("Summary")
    for line in summary:
        print(f"  {line}")
    print("=" * 72)


if __name__ == "__main__":
    main()
