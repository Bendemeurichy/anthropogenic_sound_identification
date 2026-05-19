#!/usr/bin/env python3
"""
plot_qualitative_gallery.py — Plotly-based cross-model spectrogram gallery
for the dissertation Results chapter.

Spectrogram computation follows the same pipeline as demo_separation.py
(torchaudio MelSpectrogram → dBFS with ref=1.0) so the visual output is
consistent with the user's normal workflow, but rendered with Plotly for
publication-quality figures and interactive HTML.

For each curated example index ``k``, produces one figure that vertically
stacks spectrograms with an energy-metrics sidebar:

  1. Clean COI target  (reference)
  2. Created mixture   (separator input)
  3. Separated COI from each available model

ΔRMS and ΔSEL are reported relative to the clean COI panel.

Output (PNG @ 2× + interactive HTML):
  ``final_results/dissertation_figures/gallery/gallery_<group>_<case>_k<k>.png``
  ``final_results/dissertation_figures/gallery/gallery_<group>_<case>_k<k>.html``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import soundfile as sf
import torch
import torchaudio
import warnings

# Suppress torchaudio mel-filterbank warning for high-sr files
warnings.filterwarnings(
    "ignore",
    message="At least one mel filterbank has all zero values",
    category=UserWarning,
)

SCRIPT_DIR = Path(__file__).parent
FINAL_RESULTS_DIR = SCRIPT_DIR / "final_results"
OUTPUT_DIR = FINAL_RESULTS_DIR / "dissertation_figures" / "gallery"

# ── Configuration ──────────────────────────────────────────────────────────

N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 128

FONT_FAMILY = "Times New Roman, Nimbus Roman, serif"
QUALITATIVE = px.colors.qualitative.Plotly
MODEL_COLORS = {
    "SuDoRM-RF":    QUALITATIVE[0],
    "CLAPSep":      QUALITATIVE[1],
    "TUSS (multi)": QUALITATIVE[2],
}

TEXT_COLOR = "#1f1f1f"
BG_COLOR = "#FFFFFF"

FONT = dict(family=FONT_FAMILY, size=11, color=TEXT_COLOR)
TITLE_FONT = dict(family=FONT_FAMILY, size=14, color=TEXT_COLOR)
AXIS_TITLE_FONT = dict(family=FONT_FAMILY, size=12, color=TEXT_COLOR)
TICK_FONT = dict(family=FONT_FAMILY, size=10, color=TEXT_COLOR)

LAYOUT_BASE = dict(
    template="plotly_white",
    font=FONT,
    title_font=TITLE_FONT,
    paper_bgcolor=BG_COLOR,
    plot_bgcolor=BG_COLOR,
    margin=dict(l=80, r=120, t=100, b=60),
)

PNG_SCALE = 2

# ── Group definitions ──────────────────────────────────────────────────────

GROUPS: Dict[str, List[Tuple[str, Path, str]]] = {
    "airplane": [
        ("SuDoRM-RF",    FINAL_RESULTS_DIR / "sudormrf_airplane_examples",        "pann_finetuned"),
        ("CLAPSep",      FINAL_RESULTS_DIR / "clapsep_airplane_examples",         "pann_finetuned"),
        ("TUSS (multi)", FINAL_RESULTS_DIR / "tuss_multiclass_airplane_examples", "pann_finetuned"),
    ],
    "bird": [
        ("TUSS (multi)", FINAL_RESULTS_DIR / "tuss_multiclass_bird_examples", "bird_mae"),
    ],
}


# ── Audio helpers (matching demo_separation.py) ────────────────────────────

def _load_wav(path: Path, target_sr: Optional[int] = None) -> Tuple[torch.Tensor, int]:
    wav, sr = sf.read(str(path), dtype="float32")
    wav = torch.from_numpy(wav)
    if wav.ndim == 2:
        wav = wav.T
    else:
        wav = wav.unsqueeze(0)
    if target_sr is not None and sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        sr = target_sr
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0)
    else:
        wav = wav.squeeze(0)
    return wav, sr


def _mel_spec(wav: torch.Tensor, sr: int) -> Tuple[np.ndarray, float]:
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    mel_spec = mel_transform(wav.unsqueeze(0))[0]
    mel_spec_db = 10.0 * torch.log10(mel_spec + 1e-8)
    return mel_spec_db.cpu().numpy(), sr


def _energy_metrics(wav: torch.Tensor, sr: int) -> Dict[str, float]:
    eps = 1e-12
    x = wav.detach().float()
    rms = x.pow(2).mean().sqrt()
    rms_db = 20.0 * float(torch.log10(rms + eps))
    sel = x.pow(2).sum() / sr
    sel_db = 10.0 * float(torch.log10(sel + eps))
    return {"rms_db": rms_db, "sel_db": sel_db}


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


# ── Mel ↔ Hz helpers (matching demo_separation.py) ─────────────────────────

def _mel_to_hz(mel_val: float) -> float:
    return 700.0 * (10.0 ** (mel_val / 2595.0) - 1.0)


def _hz_ticks(sr: int, n_mels: int = N_MELS, num_ticks: int = 6) -> Tuple[List[float], List[str]]:
    nyquist = sr / 2.0
    display_max_hz = min(20000, nyquist)
    mel_max = 2595.0 * np.log10(1.0 + nyquist / 700.0)
    mel_display_max = 2595.0 * np.log10(1.0 + display_max_hz / 700.0)
    display_max_bin = (mel_display_max / mel_max) * (n_mels - 1)

    mel_ticks = np.linspace(0.0, mel_display_max, num_ticks)
    mel_bin_indices = (mel_ticks / mel_max) * (n_mels - 1)

    def fmt_hz(hz: float) -> str:
        return f"{hz / 1000:.1f}k" if hz >= 1000 else f"{int(hz)}"

    hz_labels = [fmt_hz(_mel_to_hz(m)) for m in mel_ticks]
    return mel_bin_indices.tolist(), hz_labels, display_max_bin


# ── WAV layout helpers ─────────────────────────────────────────────────────

def _list_indices(examples_dir: Path, classifier: str) -> List[str]:
    mix_dir = examples_dir / classifier / "mixture_sep"
    if not mix_dir.exists():
        return []
    return sorted(
        {p.stem.rsplit("_", 1)[-1]
         for p in mix_dir.glob("mixture_created_*.wav")},
        key=lambda s: int(s) if s.isdigit() else s,
    )


def _separated_path(examples_dir: Path, classifier: str, k: str) -> Optional[Path]:
    mix_dir = examples_dir / classifier / "mixture_sep"
    for fname in (f"mixture_separated_coi_head_{k}.wav",):
        p = mix_dir / fname
        if p.exists():
            return p
    extras = list(mix_dir.glob(f"*separated*_{k}.wav"))
    return extras[0] if extras else None


def _reference_paths(group: List[Tuple[str, Path, str]], k: str
                     ) -> Tuple[Optional[Path], Optional[Path]]:
    for _, ed, cls in group:
        mix_dir = ed / cls / "mixture_sep"
        coi = mix_dir / f"mixture_coi_clean_{k}.wav"
        mix = mix_dir / f"mixture_created_{k}.wav"
        if coi.exists() and mix.exists():
            return coi, mix
    return None, None


def _bg_path(group: List[Tuple[str, Path, str]], k: str) -> Optional[Path]:
    for _, ed, cls in group:
        bg = ed / cls / "mixture_sep" / f"mixture_bg_clean_{k}.wav"
        if bg.exists():
            return bg
    return None


# ── Curation: rank shared indices by mean SI-SNRi ──────────────────────────

def _rank_indices(group: List[Tuple[str, Path, str]]) -> List[Tuple[str, float]]:
    available = [(label, ed, cls) for label, ed, cls in group
                 if _list_indices(ed, cls)]
    if not available:
        return []
    sets = [set(_list_indices(ed, cls)) for _, ed, cls in available]
    shared = sorted(set.intersection(*sets),
                    key=lambda s: int(s) if s.isdigit() else s)
    if not shared:
        return []

    scored: List[Tuple[str, float]] = []
    for k in shared:
        coi, mix = _reference_paths(group, k)
        if coi is None or mix is None:
            continue
        per_model = []
        for _, ed, cls in available:
            sep = _separated_path(ed, cls, k)
            if sep is None:
                continue
            wc, sr_c = _load_wav(coi)
            wm, sr_m = _load_wav(mix, target_sr=sr_c)
            ws, sr_s = _load_wav(sep, target_sr=sr_c)
            L = min(len(wc), len(wm), len(ws))
            if L < 16:
                continue
            per_model.append(_si_snri(wc[:L].numpy(), wm[:L].numpy(), ws[:L].numpy()))
        per_model = [v for v in per_model if not np.isnan(v) and v != float("-inf")]
        if per_model:
            scored.append((k, float(np.mean(per_model))))
    scored.sort(key=lambda t: t[1])
    return scored


def _curated(scored: List[Tuple[str, float]]) -> List[Tuple[str, str, float]]:
    if not scored:
        return []
    if len(scored) == 1:
        return [("typical", *scored[0])]
    if len(scored) == 2:
        return [("worst", *scored[0]), ("best", *scored[-1])]
    mid = scored[len(scored) // 2]
    return [
        ("worst",   *scored[0]),
        ("typical", *mid),
        ("best",    *scored[-1]),
    ]


# ── Plotly rendering ───────────────────────────────────────────────────────

def _build_gallery_figure(
    wav_paths: Sequence[Path],
    titles: Sequence[str],
    ref_idx: int,
    delta_indices: Sequence[int],
) -> go.Figure:
    n_panels = len(wav_paths)

    specs = [[{"type": "heatmap"}, {"type": "table"}] for _ in range(n_panels)]
    fig = make_subplots(
        rows=n_panels, cols=2,
        column_widths=[0.82, 0.18],
        row_titles=[f"<b>{t}</b>" for t in titles],
        horizontal_spacing=0.02,
        vertical_spacing=0.06,
        specs=specs,
    )

    # Compute all spectrograms and metrics
    all_specs: List[np.ndarray] = []
    all_metrics: List[Dict[str, float]] = []
    sr_used: Optional[int] = None

    for p in wav_paths:
        wav, sr = _load_wav(p)
        if sr_used is None:
            sr_used = sr
        elif sr != sr_used:
            # Resample to common SR
            wav = torchaudio.transforms.Resample(sr, sr_used)(wav.unsqueeze(0)).squeeze(0)
        spec, _ = _mel_spec(wav, sr_used)
        all_specs.append(spec)
        all_metrics.append(_energy_metrics(wav, sr_used))

    assert sr_used is not None
    time_sec = all_specs[0].shape[1] * HOP_LENGTH / sr_used
    n_mels = all_specs[0].shape[0]

    # Global color scale (normalized to relative dB for consistent display)
    global_vmin = min(spec.min() for spec in all_specs)
    global_vmax = max(spec.max() for spec in all_specs)

    ref_metrics = all_metrics[ref_idx]

    # Y-axis tick positions and labels
    y_positions, hz_labels, display_max_bin = _hz_ticks(sr_used, n_mels)

    for row, (spec, metrics) in enumerate(zip(all_specs, all_metrics), start=1):
        # Spectrogram heatmap — use mel-bin indices as y (same as demo_separation)
        fig.add_trace(go.Heatmap(
            z=spec,
            x=np.linspace(0, time_sec, spec.shape[1]),
            y=np.linspace(0, n_mels - 1, n_mels),
            colorscale="Magma",
            zmin=global_vmin,
            zmax=global_vmax,
            showscale=(row == n_panels),
            colorbar=dict(
                title="dB",
                x=1.01,
                thickness=12,
                len=0.25,
                y=0.12,
                title_font=AXIS_TITLE_FONT,
                tickfont=TICK_FONT,
            ) if row == n_panels else None,
            hovertemplate="Time: %{x:.2f}s<br>Mag: %{z:.1f} dB<extra></extra>",
        ), row=row, col=1)

        # Y-axis
        if row == (n_panels + 1) // 2:
            fig.update_yaxes(
                title_text="Frequency (Hz)",
                row=row, col=1,
                title_font=AXIS_TITLE_FONT, tickfont=TICK_FONT,
                tickmode="array",
                tickvals=y_positions,
                ticktext=hz_labels,
                range=[0, display_max_bin],
            )
        else:
            fig.update_yaxes(showticklabels=False, row=row, col=1)

        # X-axis
        if row == n_panels:
            fig.update_xaxes(
                title_text="Time (s)",
                row=row, col=1,
                title_font=AXIS_TITLE_FONT, tickfont=TICK_FONT,
            )
        else:
            fig.update_xaxes(showticklabels=False, row=row, col=1)

        # Metrics text panel
        lines = [
            f"RMS:  {metrics['rms_db']:.1f} dBFS",
            f"SEL:  {metrics['sel_db']:.1f} dBFS",
        ]
        if row in delta_indices and ref_metrics is not None:
            delta_rms = metrics["rms_db"] - ref_metrics["rms_db"]
            delta_sel = metrics["sel_db"] - ref_metrics["sel_db"]
            lines += [
                "",
                f"<b>ΔRMS: {delta_rms:+.1f} dB</b>",
                f"<b>ΔSEL: {delta_sel:+.1f} dB</b>",
            ]

        fig.add_trace(go.Table(
            header=dict(values=[""], height=0, fill_color="rgba(0,0,0,0)",
                        line_color="rgba(0,0,0,0)"),
            cells=dict(
                values=["<br>".join(lines)],
                align="left",
                font=dict(family="Courier New, monospace", size=10, color=TEXT_COLOR),
                fill_color="rgba(245,245,245,0.9)",
                line_color="rgba(0,0,0,0)",
                height=90,
            ),
            columnwidth=[1],
        ), row=row, col=2)

    fig.update_layout(
        height=200 * n_panels + 60,
        width=1100,
        showlegend=False,
        **LAYOUT_BASE,
    )
    return fig


# ── Main rendering loop ────────────────────────────────────────────────────

def render_group(group_name: str, group: List[Tuple[str, Path, str]]
                 ) -> Tuple[int, List[str]]:
    available = [(label, ed, cls) for label, ed, cls in group
                 if _list_indices(ed, cls)]
    missing = [label for label, ed, cls in group if not _list_indices(ed, cls)]
    if not available:
        return 0, [label for label, _, _ in group]

    print(f"  Ranking shared indices by mean SI-SNRi across "
          f"{len(available)} model(s)...")
    scored = _rank_indices(group)
    if not scored:
        print(f"  No shared indices with computable SI-SNRi.")
        return 0, missing
    curated = _curated(scored)
    print("  Selected:")
    for case, k, sni in curated:
        print(f"    {case:>7s}  k={k}  SI-SNRi={sni:+.2f} dB")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for case, k, sni in curated:
        coi_path, mix_path = _reference_paths(group, k)
        if coi_path is None or mix_path is None:
            continue
        bg_path = _bg_path(group, k)

        wav_paths: List[Path] = [coi_path]
        titles: List[str] = ["Clean COI target (reference)"]
        if bg_path is not None:
            wav_paths.append(bg_path)
            titles.append("Clean background (added interferer)")
        wav_paths.append(mix_path)
        titles.append("Created mixture (separator input)")

        for label, ed, cls in available:
            sep = _separated_path(ed, cls, k)
            if sep is None:
                continue
            wav_paths.append(sep)
            titles.append(f"Separated COI  —  {label}")

        if len(wav_paths) <= 3:
            continue

        delta_indices = list(range(3 if bg_path is not None else 2,
                                   len(wav_paths)))

        try:
            fig = _build_gallery_figure(
                wav_paths=wav_paths,
                titles=titles,
                ref_idx=0,
                delta_indices=delta_indices,
            )
            out_png = OUTPUT_DIR / f"gallery_{group_name}_{case}_k{k}.png"
            out_html = OUTPUT_DIR / f"gallery_{group_name}_{case}_k{k}.html"
            fig.write_image(str(out_png), scale=PNG_SCALE)
            fig.write_html(str(out_html))
            print(f"  Wrote {out_png.name}")
            written += 1
        except Exception as e:
            import traceback
            print(f"  ERROR rendering k={k}: {e}")
            traceback.print_exc()
    return written, missing


def main() -> None:
    print("=" * 72)
    print("Cross-model qualitative gallery (Plotly)")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 72)

    summary: Dict[str, Tuple[int, List[str]]] = {}
    for group_name, group in GROUPS.items():
        print(f"\n— {group_name} —")
        written, missing = render_group(group_name, group)
        summary[group_name] = (written, missing)
        print(f"  Rendered {written} figure(s).")

    print("\n" + "=" * 72)
    print("Summary")
    for name, (n, missing) in summary.items():
        print(f"  {name:>10}: {n} figure(s)" +
              (f"  (missing examples for: {', '.join(missing)})" if missing else ""))
    print("=" * 72)


if __name__ == "__main__":
    main()