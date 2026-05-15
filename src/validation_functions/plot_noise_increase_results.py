"""
plot_noise_increase_results.py — Chapter-grade visualisation of the
energy-domain noise-robustness sweep.

Replaces the previous multi-figure dashboard with two artefacts that the
dissertation actually cites:

  * ``noise_robustness.png``     — degradation-vs-SNR line plot with error
                                   bars (ΔRMS and ΔSEL).
  * ``noise_absolute_energy.png``— absolute mixture and separated RMS / SEL
                                   per SNR level (sanity-check companion).
  * ``noise_robustness.tex``     — booktabs LaTeX table of the per-SNR
                                   means used in the chapter.

Usage::

    python plot_noise_increase_results.py [results_file.json]

If no file is given, the most recent
``final_results/noise_increase_results/noise_increase_energy_*.json`` is
used.  Outputs are written next to that JSON so the source of truth stays
co-located with its figures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).parent
DEFAULT_DIR = SCRIPT_DIR / "final_results" / "noise_increase_results"

TEMPLATE = "plotly_white"
FONT = dict(family="Times New Roman, serif", size=13, color="#1a1a1a")
LAYOUT_BASE = dict(template=TEMPLATE, font=FONT,
                   margin=dict(l=70, r=30, t=60, b=60))

COL_RMS = "#4c78a8"
COL_SEL = "#f58518"
COL_MIX = "#9e9e9e"
COL_SEP = "#e45756"

PNG_SCALE = 2


# ── I/O ─────────────────────────────────────────────────────────────────────

def _resolve_results_file(arg: Optional[str]) -> Path:
    if arg:
        return Path(arg)
    if not DEFAULT_DIR.exists():
        raise SystemExit(f"Results directory not found: {DEFAULT_DIR}")
    candidates = sorted(DEFAULT_DIR.glob("noise_increase_energy_*.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(f"No noise_increase_energy_*.json found in {DEFAULT_DIR}")
    return candidates[0]


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _model_label(results: dict) -> str:
    cfg = results.get("config", {}) or {}
    name = cfg.get("model_type", "model")
    return str(name).replace("_", " ")


# ── Figures ─────────────────────────────────────────────────────────────────

def fig_degradation(results: dict, title_suffix: str) -> go.Figure:
    snr_results = sorted(results.get("snr_results", []),
                         key=lambda r: r["snr_db"], reverse=True)
    if not snr_results:
        raise ValueError("snr_results is empty")

    snr = [r["snr_db"] for r in snr_results]
    rms = [r["mean_rms_degradation_db"] for r in snr_results]
    sel = [r["mean_sel_degradation_db"] for r in snr_results]
    rms_std = [r.get("std_rms_degradation_db", 0.0) for r in snr_results]
    sel_std = [r.get("std_sel_degradation_db", 0.0) for r in snr_results]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=snr, y=rms, mode="lines+markers", name="ΔRMS",
        line=dict(color=COL_RMS, width=2.5),
        marker=dict(size=9, symbol="circle"),
        error_y=dict(type="data", array=rms_std, thickness=1, width=4,
                     color=COL_RMS),
    ))
    fig.add_trace(go.Scatter(
        x=snr, y=sel, mode="lines+markers", name="ΔSEL",
        line=dict(color=COL_SEL, width=2.5, dash="dash"),
        marker=dict(size=9, symbol="diamond"),
        error_y=dict(type="data", array=sel_std, thickness=1, width=4,
                     color=COL_SEL),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="grey",
                  annotation_text="Clean reference",
                  annotation_position="bottom right")

    fig.update_layout(
        title=f"<b>Energy-domain degradation vs. additive-noise SNR</b><br>"
              f"<sup>{title_suffix}</sup>",
        xaxis=dict(title="Input SNR (dB) — easier ←  → harder",
                   autorange="reversed"),
        yaxis=dict(title="Δ(separated, clean) energy (dB)"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.20,
                    xanchor="center", x=0.5),
        height=480, width=900,
        **LAYOUT_BASE,
    )
    return fig


def fig_absolute_energy(results: dict, title_suffix: str) -> go.Figure:
    snr_results = sorted(results.get("snr_results", []),
                         key=lambda r: r["snr_db"], reverse=True)
    snr = [r["snr_db"] for r in snr_results]
    mix_rms = [r.get("mean_mixture_rms_db", float("nan")) for r in snr_results]
    sep_rms = [r.get("mean_separated_noisy_rms_db", float("nan")) for r in snr_results]
    mix_sel = [r.get("mean_mixture_sel_db", float("nan")) for r in snr_results]
    sep_sel = [r.get("mean_separated_noisy_sel_db", float("nan")) for r in snr_results]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("RMS (dBFS)", "SEL (dBFS)"),
                        horizontal_spacing=0.10)
    fig.add_trace(go.Scatter(x=snr, y=mix_rms, mode="lines+markers",
                             name="Mixture",   line=dict(color=COL_MIX, width=2.5)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=snr, y=sep_rms, mode="lines+markers",
                             name="Separated", line=dict(color=COL_SEP, width=2.5)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=snr, y=mix_sel, mode="lines+markers",
                             showlegend=False, line=dict(color=COL_MIX, width=2.5)),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=snr, y=sep_sel, mode="lines+markers",
                             showlegend=False, line=dict(color=COL_SEP, width=2.5)),
                  row=1, col=2)
    for c in (1, 2):
        fig.update_xaxes(title_text="Input SNR (dB)", autorange="reversed",
                         row=1, col=c)
    fig.update_yaxes(title_text="dBFS", row=1, col=1)

    fig.update_layout(
        title=f"<b>Absolute energy of mixture vs. separated output</b><br>"
              f"<sup>{title_suffix}</sup>",
        legend=dict(orientation="h", yanchor="bottom", y=-0.22,
                    xanchor="center", x=0.5),
        height=460, width=1050,
        **LAYOUT_BASE,
    )
    return fig


# ── LaTeX table ─────────────────────────────────────────────────────────────

def _fmt(v, spec=".2f", dash="—") -> str:
    if v is None:
        return dash
    try:
        f = float(v)
    except (TypeError, ValueError):
        return dash
    if np.isnan(f):
        return dash
    return f"{f:{spec}}"


def write_latex_table(results: dict, out_path: Path, model_label: str) -> None:
    snr_results = sorted(results.get("snr_results", []),
                         key=lambda r: r["snr_db"], reverse=True)
    columns = [
        "SNR (dB)", "ΔRMS (dB)", "ΔSEL (dB)",
        "Mix RMS (dBFS)", "Sep RMS (dBFS)",
        "Mix SEL (dBFS)", "Sep SEL (dBFS)", "$N$",
    ]
    rows: List[List[str]] = []
    for r in snr_results:
        rows.append([
            _fmt(r.get("snr_db"), ".1f"),
            _fmt(r.get("mean_rms_degradation_db"), "+.2f"),
            _fmt(r.get("mean_sel_degradation_db"), "+.2f"),
            _fmt(r.get("mean_mixture_rms_db")),
            _fmt(r.get("mean_separated_noisy_rms_db")),
            _fmt(r.get("mean_mixture_sel_db")),
            _fmt(r.get("mean_separated_noisy_sel_db")),
            _fmt(r.get("n_segments"), "d"),
        ])

    safe = model_label.lower().replace(" ", "-")
    label = f"tab:noise-robustness-{safe}"
    caption = (f"Energy-domain robustness of {model_label} under additive "
               f"white noise.  Negative $\\Delta$ values denote energy lost "
               f"by the separator; mixture columns give the input reference "
               f"at each SNR.")
    align = "r" * len(columns)

    lines = [
        "% Auto-generated by plot_noise_increase_results.py — do not edit.",
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{align}}}",
        "    \\toprule",
        "    " + " & ".join(columns) + r" \\",
        "    \\midrule",
    ]
    for row in rows:
        lines.append("    " + " & ".join(row) + r" \\")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    out_path.write_text("\n".join(lines) + "\n")


# ── Driver ──────────────────────────────────────────────────────────────────

def main(argv: List[str]) -> None:
    arg = argv[1] if len(argv) > 1 else None
    results_file = _resolve_results_file(arg)
    print(f"Loading: {results_file}")
    results = _load(results_file)

    out_dir = results_file.parent
    model = _model_label(results)
    suffix = f"{model}, {len(results.get('snr_results', []))} SNR levels"

    # Figures
    fig1 = fig_degradation(results, suffix)
    p1 = out_dir / "noise_robustness.png"
    try:
        fig1.write_image(str(p1), scale=PNG_SCALE)
        print(f"  Wrote {p1}")
    except Exception as e:
        print(f"  PNG skipped ({e}): {p1.name}")
    fig1.write_html(str(p1.with_suffix(".html")))

    fig2 = fig_absolute_energy(results, suffix)
    p2 = out_dir / "noise_absolute_energy.png"
    try:
        fig2.write_image(str(p2), scale=PNG_SCALE)
        print(f"  Wrote {p2}")
    except Exception as e:
        print(f"  PNG skipped ({e}): {p2.name}")
    fig2.write_html(str(p2.with_suffix(".html")))

    # LaTeX table
    tex = out_dir / "noise_robustness.tex"
    write_latex_table(results, tex, model)
    print(f"  Wrote {tex}")


if __name__ == "__main__":
    main(sys.argv)
