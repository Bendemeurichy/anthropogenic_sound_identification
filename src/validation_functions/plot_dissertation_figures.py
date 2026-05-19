#!/usr/bin/env python3
"""
plot_dissertation_figures.py — Results & Discussion chapter visualisations.

Single entry point for all quantitative figures and LaTeX tables cited in the
dissertation.  Qualitative spectrograms are handled by plot_qualitative_gallery.py.

Key design principles
---------------------
1. **Delta-first framing**: classification figures show *improvement deltas*
   (separator_on − separator_off) rather than absolute scores, making it clear
   *which model helps classification the most* — the central argument for
   selecting TUSS for the multi-class extension.

2. **Per-classifier best-run selection**: for each (model, classifier) pair,
   the run with the highest ΔF1 is selected.  This avoids conflating
   classifier quality with separator quality.

3. **Both classifiers shown**: AST and PANN for airplane; BirdMAE and
   AudioProtoPNet for bird — minimises architectural bias per the methodology.

4. **Three models only**: SuDoRM-RF, CLAPSep, TUSS (multi-class).
   No TUSS single-class data directory exists.

Outputs (under ``final_results/dissertation_figures/``):
  Figures (PNG @ 2x + HTML interactive):
    fig_4_1_separation_metrics      — SI-SNRi / SDR per model × classifier
    fig_4_2_energy_preservation     — Signed RMS / SEL deviation
    fig_4_3_classification_impact   — Diverging ΔF1 / ΔPrecision / ΔRecall bars
    fig_4_3_classification_deltas_grid — 2×2 grid: clean & mix deltas per model
    fig_4_4_confusion_shifts        — Δ confusion-matrix heatmaps
    fig_4_5_generalisation_gap      — In-dist vs Risoux F1 with gap
    fig_4_6_multiclass_overview      — TUSS airplane + bird heads
    fig_4_7_noise_robustness        — SI-SDR vs input SNR
    fig_4_8_activity_gating          — Quality–efficiency trade-off curve

  LaTeX tables (``tables/*.tex``):
    tab_separation_metrics.tex
    tab_classification_impact.tex
    tab_generalisation.tex
    tab_multiclass_tuss.tex
    tab_best_runs.tex
    tab_noise_robustness.tex
    tab_activity_gating.tex

Usage::

    python plot_dissertation_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
FINAL_RESULTS_DIR = SCRIPT_DIR / "final_results"
OUTPUT_DIR = FINAL_RESULTS_DIR / "dissertation_figures"
TABLE_DIR = OUTPUT_DIR / "tables"

# Model directories (airplane COI)
SUDORMRF_DIR = FINAL_RESULTS_DIR / "sudormrf_airplane_results"
CLAPSEP_DIR = FINAL_RESULTS_DIR / "clapsep_airplane_results"
TUSS_MC_AIR_DIR = FINAL_RESULTS_DIR / "tuss_multiclass_airplane_results"

# Risoux (OOD) directories
SUDORMRF_RISOUX_DIR = FINAL_RESULTS_DIR / "sudormrf_airplane_results_risoux"
CLAPSEP_RISOUX_DIR = FINAL_RESULTS_DIR / "clapsep_airplane_results_risoux"
TUSS_MC_AIR_RISOUX_DIR = FINAL_RESULTS_DIR / "tuss_multiclass_airplane_results_risoux"

# Bird COI
TUSS_MC_BIRD_DIR = FINAL_RESULTS_DIR / "tuss_multiclass_bird_results"
TUSS_MC_BIRD_RISOUX_DIR = FINAL_RESULTS_DIR / "tuss_multiclass_bird_results_risoux"

# Supplementary
NOISE_RESULTS_DIR = FINAL_RESULTS_DIR / "noise_increase_results"
GATING_RESULTS_DIR = FINAL_RESULTS_DIR / "activity_gating_results"

# ---------------------------------------------------------------------------
# Model registry — single source of truth
# ---------------------------------------------------------------------------
SEPARATORS_AIRPLANE: List[Tuple[str, Path, Optional[Path]]] = [
    ("SuDoRM-RF", SUDORMRF_DIR, SUDORMRF_RISOUX_DIR),
    ("CLAPSep", CLAPSEP_DIR, CLAPSEP_RISOUX_DIR),
    ("TUSS (multi)", TUSS_MC_AIR_DIR, TUSS_MC_AIR_RISOUX_DIR),
]

CLASSIFIERS_AIRPLANE = ["ast_finetuned", "pann_finetuned"]
CLASSIFIERS_BIRD = ["bird_mae", "audioprotopnet"]

# ---------------------------------------------------------------------------
# Colour palette — Okabe-Ito (colour-blind safe) + semantic mappings
# ---------------------------------------------------------------------------
OI_BLUE = "#0072B2"
OI_ORANGE = "#E69F00"
OI_VERMILION = "#D55E00"
OI_GREEN = "#009E73"
OI_SKY = "#56B4E9"
OI_YELLOW = "#F0E442"
OI_PURPLE = "#CC79A7"
OI_BLACK = "#1f1f1f"

MODEL_COLORS = {
    "SuDoRM-RF": OI_BLUE,
    "CLAPSep": OI_VERMILION,
    "TUSS (multi)": OI_GREEN,
}

CLASSIFIER_COLORS = {
    "ast_finetuned": OI_SKY,
    "pann_finetuned": OI_ORANGE,
    "bird_mae": OI_PURPLE,
    "audioprotopnet": OI_YELLOW,
}

COL_POS = "#1b9e3f"   # green — separation helps
COL_NEG = "#b22222"    # red — separation hurts
COL_NEU = "#555555"    # grey — neutral

# Typographic constants
FONT_FAMILY = "Times New Roman, Nimbus Roman, serif"
FONT = dict(family=FONT_FAMILY, size=13, color=OI_BLACK)
TITLE_FONT = dict(family=FONT_FAMILY, size=16, color=OI_BLACK)
AXIS_TITLE_FONT = dict(family=FONT_FAMILY, size=13, color=OI_BLACK)
TICK_FONT = dict(family=FONT_FAMILY, size=11, color=OI_BLACK)
AXIS_LINE = OI_BLACK
GRID = "rgba(0,0,0,0.07)"

LAYOUT_BASE = dict(
    template="plotly_white",
    font=FONT,
    title_font=TITLE_FONT,
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    margin=dict(l=72, r=28, t=80, b=72),
    legend=dict(
        font=dict(family=FONT_FAMILY, size=11, color=OI_BLACK),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="rgba(0,0,0,0.12)", borderwidth=1,
        orientation="h", yanchor="bottom", y=-0.22,
        xanchor="center", x=0.5,
    ),
)

PNG_SCALE = 2

# ---------------------------------------------------------------------------
# Classifier display names
# ---------------------------------------------------------------------------
CLS_DISPLAY = {
    "pann_finetuned": "PANN",
    "ast_finetuned": "AST",
    "bird_mae": "BirdMAE",
    "audioprotopnet": "AudioProtoPNet",
}


def _cls_label(name: str) -> str:
    return CLS_DISPLAY.get(name, name.replace("_", " "))


# ---------------------------------------------------------------------------
# Data loading — per-classifier best-run selection
# ---------------------------------------------------------------------------

def _load_all(directory: Path, glob: str = "results_*.json") -> List[dict]:
    if not directory.exists():
        return []
    out = []
    for f in sorted(directory.glob(glob)):
        try:
            with open(f) as fh:
                out.append(json.load(fh))
        except Exception:
            continue
    return out


def _load_best(base_dir: Path, classifier: str,
               risoux: bool = False) -> Optional[dict]:
    """Select the run with the best *classification improvement delta*.

    For in-distribution data (risoux=False):
        delta = F1(mix_sep_cls) - F1(mix_cls)
    For Risoux (risoux=True):
        delta = F1(as_is_sep_cls) - F1(as_is_cls)

    Ties are broken by balanced_accuracy delta, then by test recency.
    """
    sub = base_dir / classifier
    if not sub.exists():
        return None
    pattern = "results_test_risoux_*.json" if risoux else "results_test_*.json"
    candidates = _load_all(sub, pattern)
    if not candidates:
        return None
    base_cond = "as_is_cls" if risoux else "mix_cls"
    sep_cond = "as_is_sep_cls" if risoux else "mix_sep_cls"
    best, best_delta = None, float("-inf")
    best_ba_delta = float("-inf")
    for d in candidates:
        f1_base = d.get(base_cond, {}).get("f1_score", float("nan"))
        f1_sep = d.get(sep_cond, {}).get("f1_score", float("nan"))
        if np.isnan(f1_base) or np.isnan(f1_sep):
            continue
        delta = f1_sep - f1_base
        ba_base = d.get(base_cond, {}).get("balanced_accuracy", float("nan"))
        ba_sep = d.get(sep_cond, {}).get("balanced_accuracy", float("nan"))
        ba_delta = (ba_sep - ba_base) if not (np.isnan(ba_base) or np.isnan(ba_sep)) else float("-inf")
        if delta > best_delta or (delta == best_delta and ba_delta > best_ba_delta):
            best_delta = delta
            best_ba_delta = ba_delta
            best = d
    return best


def _metric(d: Optional[dict], key: str, condition: str) -> float:
    if d is None:
        return float("nan")
    return float(d.get(condition, {}).get(key, float("nan")))


def _signal(d: Optional[dict], key: str,
            condition: str = "mix_sep_cls") -> float:
    if d is None:
        return float("nan")
    sm = d.get(condition, {}).get("signal_metrics") or {}
    val = sm.get(key, float("nan"))
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def _cm(d: Optional[dict], condition: str) -> Optional[dict]:
    if d is None:
        return None
    return d.get(condition, {}).get("confusion_matrix")


def _fmt(v: float, spec: str = ".3f", dash: str = "\u2014") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return dash
    return f"{v:{spec}}"


# ---------------------------------------------------------------------------
# Axis styling helper
# ---------------------------------------------------------------------------

def _apply_axis_style(fig: go.Figure, n_rows: int = 1,
                      n_cols: int = 1) -> None:
    kw = dict(
        showline=True, linecolor=AXIS_LINE, linewidth=1.2,
        gridcolor=GRID, ticks="outside", tickcolor=AXIS_LINE,
        ticklen=4, tickwidth=1, tickfont=TICK_FONT,
        title_font=AXIS_TITLE_FONT, automargin=True,
    )
    uses_grid = True
    try:
        fig.update_xaxes(row=1, col=1, **kw)
        fig.update_yaxes(row=1, col=1, **kw)
    except Exception:
        uses_grid = False

    if uses_grid:
        for r in range(1, n_rows + 1):
            for c in range(1, n_cols + 1):
                if r == 1 and c == 1:
                    continue
                fig.update_xaxes(row=r, col=c, **kw)
                fig.update_yaxes(row=r, col=c, **kw)
    else:
        fig.update_xaxes(**kw)
        fig.update_yaxes(**kw)


def _placeholder(msg: str, title: str = "Data not yet available") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=f"<b>{msg}</b>", xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(family=FONT_FAMILY, size=14, color=COL_NEU))
    fig.update_layout(title=f"<b>{title}</b>", height=400, width=800,
                     **LAYOUT_BASE)
    return fig


# ===================================================================
#  FIGURE 4.1 — Separation Metrics
# ===================================================================

def fig_separation_metrics() -> go.Figure:
    models, si_snri, sdr, rms_err, sel_err = [], [], [], [], []
    available = []
    for name, base, _ in SEPARATORS_AIRPLANE:
        d = _load_best(base, "ast_finetuned")
        if d is None:
            continue
        available.append(name)
        models.append(name)
        si_snri.append(_signal(d, "mean_si_snri_db", "mix_sep_cls"))
        sdr.append(_signal(d, "mean_sdr_db", "mix_sep_cls"))
        rms_err.append(abs(_signal(d, "mean_rms_error_db", "mix_sep_cls")))
        sel_err.append(abs(_signal(d, "mean_sel_error_db", "mix_sep_cls")))
    if not available:
        return _placeholder("No separation data.", "Separation Metrics")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("<b>Separation fidelity (\u2191 better)</b>",
                        "<b>Energy envelope error (\u2193 better)</b>"),
        horizontal_spacing=0.14,
    )
    colors = [MODEL_COLORS.get(m, OI_ORANGE) for m in models]
    bar_kw = dict(textposition="outside", cliponaxis=False,
                  textfont=dict(family=FONT_FAMILY, size=10, color=OI_BLACK),
                  marker_line=dict(color="rgba(0,0,0,0.25)", width=0.6))

    fig.add_trace(go.Bar(x=models, y=si_snri, name="SI-SNRi (dB)",
                         marker_color=colors,
                         text=[_fmt(v, ".1f") for v in si_snri],
                         legendgroup="a", showlegend=True, **bar_kw),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=models, y=sdr, name="SDR (dB)",
                         marker_color=colors, opacity=0.65,
                         text=[_fmt(v, ".1f") for v in sdr],
                         legendgroup="b", showlegend=True, **bar_kw),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=models, y=rms_err, name="|RMS error| (dB)",
                         marker_color=colors,
                         text=[_fmt(v, ".1f") for v in rms_err],
                         legendgroup="c", showlegend=True, **bar_kw),
                  row=1, col=2)
    fig.add_trace(go.Bar(x=models, y=sel_err, name="|SEL error| (dB)",
                         marker_color=colors, opacity=0.65,
                         text=[_fmt(v, ".1f") for v in sel_err],
                         legendgroup="d", showlegend=True, **bar_kw),
                  row=1, col=2)

    missing = [m for m, _, _ in SEPARATORS_AIRPLANE if m not in available]
    subtitle = "Airplane COI \u00b7 AST classifier \u00b7 mixture + separation"
    if missing:
        subtitle += f"  \u00b7  Missing: {', '.join(missing)}"

    fig.update_layout(
        barmode="group", bargap=0.28, bargroupgap=0.08,
        title=dict(text=f"<b>Signal-level separation quality</b><br><sup>{subtitle}</sup>"),
        height=500, width=1000, **LAYOUT_BASE,
    )
    fig.update_yaxes(title_text="dB", row=1, col=1)
    fig.update_yaxes(title_text="|dB|", row=1, col=2)
    _apply_axis_style(fig, 1, 2)
    return fig


# ===================================================================
#  FIGURE 4.2 — Energy Preservation
# ===================================================================

def fig_energy_preservation() -> go.Figure:
    models, rms_mix, sel_mix = [], [], []
    rms_clean, sel_clean = [], []
    available = []
    for name, base, _ in SEPARATORS_AIRPLANE:
        d = _load_best(base, "ast_finetuned")
        if d is None:
            continue
        available.append(name)
        models.append(name)
        rms_mix.append(_signal(d, "mean_rms_error_db", "mix_sep_cls"))
        sel_mix.append(_signal(d, "mean_sel_error_db", "mix_sep_cls"))
        rms_clean.append(_signal(d, "mean_rms_error_db", "clean_sep_cls"))
        sel_clean.append(_signal(d, "mean_sel_error_db", "clean_sep_cls"))
    if not available:
        return _placeholder("No energy data.", "Energy Preservation")

    fig = go.Figure()
    colors = [MODEL_COLORS.get(m, OI_ORANGE) for m in models]
    for i, m in enumerate(models):
        for cond, vals, label_suffix in [
            ("mixture+sep", [rms_mix[i], sel_mix[i]], ""),
            ("clean+sep", [rms_clean[i], sel_clean[i]], " (clean)"),
        ]:
            fig.add_trace(go.Bar(
                y=[f"{m}  (RMS){label_suffix}", f"{m}  (SEL){label_suffix}"],
                x=vals, orientation="h",
                name=f"{m}{label_suffix}" if i == 0 else None,
                marker_color=colors[i],
                text=[_fmt(v, "+.1f") for v in vals],
                textposition="outside",
                textfont=dict(size=10),
                marker_line=dict(color="rgba(0,0,0,0.25)", width=0.6),
            ))

    fig.add_vline(x=0, line_dash="dot", line_color=OI_BLACK, line_width=1.2,
                  annotation_text="Perfect preservation",
                  annotation_position="top right",
                  annotation_font=dict(size=10, color=COL_NEU))
    fig.update_layout(
        barmode="group", bargap=0.25, bargroupgap=0.12,
        title=dict(text="<b>Energy preservation deviation</b>"
                       "<br><sup>\u0394 energy = separated \u2212 clean-COI reference"
                       "  \u00b7  Negative = energy lost</sup>"),
        xaxis=dict(title="\u0394 energy (dB)"),
        height=max(380, 90 * len(models) * 2), width=900,
        **LAYOUT_BASE,
    )
    _apply_axis_style(fig)
    return fig


# ===================================================================
#  FIGURE 4.3 — Classification Impact (KEY FIGURE)
# ===================================================================

def fig_classification_impact() -> go.Figure:
    metrics = [("f1_score", "F1"), ("precision", "Precision"),
               ("recall", "Recall")]
    rows_data: List[List[dict]] = [[] for _ in metrics]
    any_data = False

    for cls in CLASSIFIERS_AIRPLANE:
        for name, base, _ in SEPARATORS_AIRPLANE:
            d = _load_best(base, cls)
            if d is None:
                continue
            for i, (key, label) in enumerate(metrics):
                v_clean_cls = _metric(d, key, "clean_cls")
                v_clean_sep = _metric(d, key, "clean_sep_cls")
                v_mix_cls = _metric(d, key, "mix_cls")
                v_mix_sep = _metric(d, key, "mix_sep_cls")
                delta_clean = v_clean_sep - v_clean_cls if not (np.isnan(v_clean_cls) or np.isnan(v_clean_sep)) else float("nan")
                delta_mix = v_mix_sep - v_mix_cls if not (np.isnan(v_mix_cls) or np.isnan(v_mix_sep)) else float("nan")
                rows_data[i].append({
                    "label": f"{name} \u00b7 {_cls_label(cls)}",
                    "delta_clean": delta_clean,
                    "delta_mix": delta_mix,
                })
                if not np.isnan(delta_mix):
                    any_data = True

    if not any_data:
        return _placeholder("No classification data.", "Classification Impact")

    fig = make_subplots(
        rows=3, cols=1, vertical_spacing=0.10,
        subplot_titles=[f"<b>\u0394 {label}</b>  (with separator minus without)"
                        for _, label in metrics],
    )

    for row_idx, entries in enumerate(rows_data, 1):
        if not entries:
            continue
        entries_sorted = sorted(entries,
                                key=lambda x: abs(x["delta_mix"]) if not np.isnan(x["delta_mix"]) else 0,
                                reverse=True)
        y_labels = [e["label"] for e in entries_sorted]
        delta_c = [e["delta_clean"] for e in entries_sorted]
        delta_m = [e["delta_mix"] for e in entries_sorted]

        fig.add_trace(go.Bar(
            y=y_labels, x=delta_c, orientation="h",
            name="Clean COI \u0394" if row_idx == 1 else None,
            marker_color=OI_SKY, showlegend=(row_idx == 1),
            text=[_fmt(v, "+.3f") for v in delta_c],
            textposition="outside",
            textfont=dict(family=FONT_FAMILY, size=9, color=OI_BLACK),
            marker_line=dict(color="rgba(0,0,0,0.15)", width=0.5),
            hovertemplate="%{y}<br>\u0394 clean = %{x:+.3f}<extra></extra>",
        ), row=row_idx, col=1)
        fig.add_trace(go.Bar(
            y=y_labels, x=delta_m, orientation="h",
            name="Mixture \u0394" if row_idx == 1 else None,
            marker_color=OI_VERMILION, showlegend=(row_idx == 1),
            text=[_fmt(v, "+.3f") for v in delta_m],
            textposition="outside",
            textfont=dict(family=FONT_FAMILY, size=9, color=OI_BLACK),
            marker_line=dict(color="rgba(0,0,0,0.15)", width=0.5),
            hovertemplate="%{y}<br>\u0394 mix = %{x:+.3f}<extra></extra>",
        ), row=row_idx, col=1)

        fig.add_vline(x=0, line_dash="solid", line_color=OI_BLACK,
                      line_width=1, row=row_idx, col=1)

        helps_c = sum(1 for v in delta_c if v > 0.005)
        hurts_c = sum(1 for v in delta_c if v < -0.005)
        helps_m = sum(1 for v in delta_m if v > 0.005)
        hurts_m = sum(1 for v in delta_m if v < -0.005)
        fig.add_annotation(
            xref=f"x{row_idx}", yref=f"y{row_idx}",
            x=0.98, y=0.98, xanchor="right", yanchor="top",
            text=f"Clean: helps {helps_c}/hurts {hurts_c}  \u00b7  "
                 f"Mix: helps {helps_m}/hurts {hurts_m}",
            showarrow=False, font=dict(size=8, color=OI_BLACK),
            bgcolor="rgba(255,255,255,0.9)", borderpad=3,
            bordercolor="rgba(0,0,0,0.1)", borderwidth=1,
        )

    fig.update_layout(
        title=dict(
            text="<b>Impact of separation on downstream classification</b>"
                 "<br><sup>Light blue = \u0394 on clean COI; terracotta = \u0394 on mixture."
                 "  Positive = separation helps; negative = hurts.</sup>",
        ),
        height=720, width=1050, **LAYOUT_BASE,
    )
    fig.update_xaxes(title_text="\u0394 metric", row=3, col=1)
    _apply_axis_style(fig, 3, 1)
    return fig


# ===================================================================
#  FIGURE 4.4 — Confusion Shifts
# ===================================================================

def _norm_cm(cm: dict) -> np.ndarray:
    tp, tn, fp, fn = cm.get("tp", 0), cm.get("tn", 0), cm.get("fp", 0), cm.get("fn", 0)
    total = tp + tn + fp + fn
    return np.array([[tn, fp], [fn, tp]], dtype=float) / (total or 1) * 100.0


def fig_confusion_shifts() -> go.Figure:
    panels = []
    for cls in CLASSIFIERS_AIRPLANE:
        for name, base, _ in SEPARATORS_AIRPLANE:
            d = _load_best(base, cls)
            if d is None:
                continue
            cm_cls = _cm(d, "mix_cls")
            cm_sep = _cm(d, "mix_sep_cls")
            if cm_cls is None or cm_sep is None:
                continue
            n_cls = _norm_cm(cm_cls)
            n_sep = _norm_cm(cm_sep)
            delta = n_sep - n_cls
            text = [[f"{v:+.1f}pp" for v in row] for row in delta.tolist()]
            panels.append((name, _cls_label(cls), delta, text, cm_cls))

    if not panels:
        return _placeholder("No confusion data.", "Confusion Shifts")

    n = len(panels)
    n_cols = 2
    n_rows = (n + n_cols - 1) // n_cols
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"<b>{m}</b><br><span style='font-size:10px'>{c}</span>"
                        for m, c, _, _, _ in panels],
        horizontal_spacing=0.12, vertical_spacing=0.28 if n_rows > 1 else 0.18,
    )
    div_scale = [[0.0, COL_NEG], [0.5, "#FFFFFF"], [1.0, COL_POS]]
    zmax = max(abs(p[2]).max() for p in panels)
    zmax = max(zmax, 5.0)

    for idx, (name, cls_label, delta, text, cm_cls) in enumerate(panels):
        r, c = idx // n_cols + 1, idx % n_cols + 1
        total = sum(cm_cls.values())
        fig.add_trace(go.Heatmap(
            z=delta, x=["Pred. 0", "Pred. 1"], y=["True 0", "True 1"],
            text=text, texttemplate="%{text}",
            textfont=dict(size=11, color=OI_BLACK),
            colorscale=div_scale, zmid=0, zmin=-zmax, zmax=zmax,
            showscale=(idx == 0),
            colorbar=dict(title=dict(text="\u0394pp", side="right"), thickness=12) if idx == 0 else None,
        ), row=r, col=c)
        fig.update_xaxes(side="bottom", row=r, col=c)
        fig.update_yaxes(autorange="reversed", row=r, col=c)
        fig.add_annotation(
            xref=f"x{idx+1}", yref=f"y{idx+1}", x=0.5, y=-0.28,
            xanchor="center", text=f"Baseline N = {total}", showarrow=False,
            font=dict(size=9, color=COL_NEU),
        )

    fig.update_layout(
        title=dict(
            text="<b>How separation shifts the confusion pattern</b>"
                 "<br><sup>\u0394 = (mix + sep) \u2212 (mix only), in percentage points.</sup>",
        ),
        height=max(420, 380 * n_rows), width=max(800, 460 * n_cols),
        **LAYOUT_BASE,
    )
    _apply_axis_style(fig, n_rows, n_cols)
    return fig


# ===================================================================
#  FIGURE 4.5 — Generalisation Gap
# ===================================================================

def fig_generalisation_gap() -> go.Figure:
    rows_data = []
    for cls in CLASSIFIERS_AIRPLANE:
        for name, base, risoux_dir in SEPARATORS_AIRPLANE:
            d_in = _load_best(base, cls)
            d_out = _load_best(risoux_dir, cls, risoux=True) if risoux_dir else None
            rows_data.append({
                "model": name,
                "classifier": _cls_label(cls),
                "f1_in": _metric(d_in, "f1_score", "mix_sep_cls"),
                "ba_in": _metric(d_in, "balanced_accuracy", "mix_sep_cls"),
                "f1_out": _metric(d_out, "f1_score", "as_is_sep_cls") if d_out else float("nan"),
                "ba_out": _metric(d_out, "balanced_accuracy", "as_is_sep_cls") if d_out else float("nan"),
                "f1_base_in": _metric(d_in, "f1_score", "mix_cls"),
                "f1_base_out": _metric(d_out, "f1_score", "as_is_cls") if d_out else float("nan"),
            })

    if not any(not np.isnan(r["f1_in"]) for r in rows_data):
        return _placeholder("No generalisation data.", "Generalisation Gap")

    labels = [f"{r['model']}<br>{r['classifier']}" for r in rows_data]
    f1_in = [r["f1_in"] for r in rows_data]
    f1_out = [r["f1_out"] for r in rows_data]
    delta_f1 = [r["f1_out"] - r["f1_base_out"] if not (np.isnan(r["f1_out"]) or np.isnan(r["f1_base_out"])) else float("nan")
                for r in rows_data]

    fig = go.Figure()
    bar_kw = dict(textposition="outside", cliponaxis=False,
                  textfont=dict(family=FONT_FAMILY, size=10, color=OI_BLACK),
                  marker_line=dict(color="rgba(0,0,0,0.2)", width=0.6))

    fig.add_trace(go.Bar(x=labels, y=f1_in, name="In-dist test (mix+sep)",
                         marker_color=[MODEL_COLORS.get(r["model"], OI_ORANGE) for r in rows_data],
                         text=[_fmt(v, ".2f") for v in f1_in],
                         legendgroup="in", showlegend=True, **bar_kw))
    fig.add_trace(go.Bar(x=labels, y=f1_out, name="Risoux (OOD, as-is+sep)",
                         marker_color=[MODEL_COLORS.get(r["model"], OI_ORANGE) for r in rows_data],
                         opacity=0.65,
                         text=[_fmt(v, ".2f") for v in f1_out],
                         legendgroup="out", showlegend=True, **bar_kw))

    for i, d_f1 in enumerate(delta_f1):
        if np.isnan(d_f1):
            continue
        y_anchor = max(f1_in[i] if not np.isnan(f1_in[i]) else 0,
                       f1_out[i] if not np.isnan(f1_out[i]) else 0) + 0.05
        color = COL_NEG if d_f1 < -0.05 else COL_POS if d_f1 > 0.05 else COL_NEU
        label = f"\u0394 {d_f1:+.2f}"
        fig.add_annotation(x=labels[i], y=y_anchor, text=f"<b>{label}</b>",
                          showarrow=False, yanchor="bottom",
                          font=dict(family=FONT_FAMILY, size=11, color=color))

    fig.update_layout(
        barmode="group", bargap=0.28, bargroupgap=0.08,
        title=dict(text="<b>Generalisation gap: in-distribution vs. Risoux field recordings</b>"
                       "<br><sup>F1 after separation; \u0394 = Risoux improvement over baseline."
                       "  Negative \u0394 = separation helps less on OOB data.</sup>"),
        yaxis=dict(title="F1 score", range=[0, 1.05]),
        height=520, width=850, **LAYOUT_BASE,
    )
    _apply_axis_style(fig)
    return fig


# ===================================================================
#  FIGURE 4.6 — Multi-Class Overview (TUSS airplane + bird)
# ===================================================================

def fig_multiclass_overview() -> go.Figure:
    air_data = []
    for cls in CLASSIFIERS_AIRPLANE:
        for name, base, _ in SEPARATORS_AIRPLANE:
            d = _load_best(base, cls)
            if d is None:
                continue
            air_data.append({
                "model": name,
                "classifier": _cls_label(cls),
                "f1_mix": _metric(d, "f1_score", "mix_cls"),
                "f1_sep": _metric(d, "f1_score", "mix_sep_cls"),
                "delta": _metric(d, "f1_score", "mix_sep_cls") - _metric(d, "f1_score", "mix_cls"),
            })

    bird_data = []
    for cls in CLASSIFIERS_BIRD:
        d = _load_best(TUSS_MC_BIRD_DIR, cls)
        if d is not None:
            bird_data.append({
                "classifier": _cls_label(cls),
                "f1_mix": _metric(d, "f1_score", "mix_cls"),
                "f1_sep": _metric(d, "f1_score", "mix_sep_cls"),
                "delta": _metric(d, "f1_score", "mix_sep_cls") - _metric(d, "f1_score", "mix_cls"),
            })

    if not air_data and not bird_data:
        return _placeholder("No multi-class data.", "Multi-Class Separation")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("<b>Airplane COI</b><br><span style='font-size:10px'>"
                        "All architectures</span>",
                        "<b>Bird COI</b><br><span style='font-size:10px'>"
                        "TUSS multi-class only</span>"),
        horizontal_spacing=0.14,
    )

    if air_data:
        labels = [f"{d['model']}<br>{d['classifier']}" for d in air_data]
        deltas = [d["delta"] for d in air_data]
        colors = [COL_POS if v > 0.005 else COL_NEG if v < -0.005 else COL_NEU
                  for v in deltas]
        fig.add_trace(go.Bar(x=labels, y=deltas,
                             name="\u0394F1 (sep \u2212 no sep)",
                             marker_color=colors,
                             text=[_fmt(v, "+.3f") for v in deltas],
                             textposition="outside", showlegend=True,
                             textfont=dict(size=9),
                             marker_line=dict(color="rgba(0,0,0,0.2)", width=0.6)),
                      row=1, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color=OI_BLACK, line_width=1,
                      row=1, col=1)

    if bird_data:
        labels = [d["classifier"] for d in bird_data]
        deltas = [d["delta"] for d in bird_data]
        colors = [COL_POS if v > 0.005 else COL_NEG if v < -0.005 else COL_NEU
                  for v in deltas]
        fig.add_trace(go.Bar(x=labels, y=deltas,
                             name="\u0394F1 (sep \u2212 no sep)",
                             marker_color=colors,
                             text=[_fmt(v, "+.3f") for v in deltas],
                             textposition="outside", showlegend=False,
                             textfont=dict(size=10),
                             marker_line=dict(color="rgba(0,0,0,0.2)", width=0.6)),
                      row=1, col=2)
        fig.add_hline(y=0, line_dash="dot", line_color=OI_BLACK, line_width=1,
                      row=1, col=2)

    fig.update_layout(
        barmode="group", bargap=0.30, bargroupgap=0.10,
        title=dict(
            text="<b>\u0394F1: does separation help classification?</b>"
                 "<br><sup>Teal = helps; terracotta = hurts.  Airplane (left) all models;"
                 "  Bird (right) TUSS multi-class only.</sup>",
        ),
        yaxis=dict(title="\u0394F1"),
        height=520, width=1000, **LAYOUT_BASE,
    )
    _apply_axis_style(fig, 1, 2)
    return fig


# ===================================================================
#  FIGURES 4.7 & 4.8 — Noise robustness and Activity gating
# ===================================================================

def fig_noise_robustness() -> go.Figure:
    noise_files = sorted(NOISE_RESULTS_DIR.glob("noise_increase_energy_*.json"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
    if not noise_files:
        return _placeholder("No noise data.\nRun test_noise_increase.py first.",
                            "Noise Robustness")

    fig = go.Figure()
    any_trace = False
    has_si_sdr = False

    for nf in noise_files:
        with open(nf) as f:
            data = json.load(f)
        model = data.get("config", {}).get("model_type", "model")
        snr_results = sorted(data.get("snr_results", []),
                              key=lambda x: x["snr_db"], reverse=True)
        if not snr_results:
            continue
        snr = [r["snr_db"] for r in snr_results]
        use_si = "mean_si_sdr_noisy_vs_clean_db" in snr_results[0]
        has_si_sdr |= use_si

        if use_si:
            y = [r["mean_si_sdr_noisy_vs_clean_db"] for r in snr_results]
            y_std = [r.get("std_si_sdr_noisy_vs_clean_db", 0) for r in snr_results]
            y_label = "SI-SDR (dB)"
        else:
            y = [r["mean_rms_degradation_db"] for r in snr_results]
            y_std = [r.get("std_rms_degradation_db", 0) for r in snr_results]
            y_label = "\u0394RMS (dB)"

        color = MODEL_COLORS.get(model, OI_ORANGE)
        any_trace = True
        fig.add_trace(go.Scatter(
            x=snr, y=y, mode="lines+markers", name=f"{model} ({y_label})",
            line=dict(color=color, width=2.5),
            marker=dict(size=8, symbol="circle"),
            error_y=dict(type="data", array=y_std, thickness=1, width=4,
                         color=color),
        ))

    if not any_trace:
        return _placeholder("Noise JSONs empty.", "Noise Robustness")

    fig.add_vrect(x0=-5, x1=10, fillcolor=OI_ORANGE, opacity=0.10,
                  layer="below", line_width=0)
    fig.add_annotation(x=2.5, y=0.95, xref="x", yref="paper",
                       text="<b>Realistic deployment</b>", showarrow=False,
                       font=dict(size=10, color=OI_BLACK))

    if has_si_sdr:
        fig.add_hline(y=0, line_dash="dot", line_color=OI_BLACK,
                      annotation_text="Clean reference",
                      annotation_position="bottom right",
                      annotation_font=dict(size=10, color=COL_NEU))

    fig.update_layout(
        title=dict(
            text="<b>Separation robustness under additive white noise</b>"
                 "<br><sup><< easier \u2192 harder SNR \u00b7 shaded = realistic deployment range</sup>"),
        xaxis=dict(title="Input SNR (dB)", autorange="reversed"),
        yaxis=dict(title="SI-SDR (dB)" if has_si_sdr else "\u0394RMS (dB)"),
        height=520, width=900, **LAYOUT_BASE,
    )
    _apply_axis_style(fig)
    return fig


def fig_activity_gating() -> go.Figure:
    json_path = GATING_RESULTS_DIR / "sweep_results.json"
    if not json_path.exists():
        return _placeholder("No activity-gating data.", "Activity Gating")

    with open(json_path) as f:
        data = json.load(f)
    sweep = sorted(data.get("sweep", []), key=lambda r: r["threshold"])
    baseline = data.get("baseline_no_recycling", {})
    th = [r["threshold"] for r in sweep]
    f1 = [r.get("f1_score", float("nan")) for r in sweep]
    sni = [r.get("mean_si_snri_db", float("nan")) for r in sweep]
    hit = [r.get("cache_hit_rate", float("nan")) * 100 for r in sweep]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=th, y=f1, mode="lines+markers", name="F1",
                             line=dict(color=OI_VERMILION, width=2.5),
                             marker=dict(size=7)), secondary_y=False)
    if any(not np.isnan(v) for v in sni):
        fig.add_trace(go.Scatter(x=th, y=sni, mode="lines+markers",
                                 name="SI-SNRi (dB)",
                                 line=dict(color=OI_BLUE, width=2.5, dash="dash"),
                                 marker=dict(size=7)), secondary_y=False)
    fig.add_trace(go.Scatter(x=th, y=hit, mode="lines+markers",
                             name="Cache hit rate (%)",
                             line=dict(color=OI_GREEN, width=2.5, dash="dot"),
                             marker=dict(size=7, symbol="diamond")), secondary_y=True)

    bf1 = baseline.get("f1_score", float("nan"))
    if not np.isnan(bf1):
        fig.add_hline(y=bf1, line_dash="dot", line_color=OI_VERMILION,
                      annotation_text=f"Baseline F1 = {bf1:.2f}",
                      annotation_position="top left")

    zone_th = [t for t, fi, hi in zip(th, f1, hit)
               if not np.isnan(fi) and not np.isnan(hi) and fi >= bf1 - 0.02 and hi >= 50]
    if zone_th:
        fig.add_vrect(x0=min(zone_th), x1=max(zone_th),
                      fillcolor=OI_GREEN, opacity=0.10, layer="below", line_width=0)
        fig.add_annotation(x=(min(zone_th) + max(zone_th)) / 2, y=0.05,
                           xref="x", yref="paper", text="<b>Recommended zone</b>",
                           showarrow=False, font=dict(size=9, color=OI_BLACK))

    fig.update_layout(
        title=dict(
            text="<b>Activity gating: quality\u2013efficiency trade-off</b>"
                 "<br><sup>F1 / SI-SNRi (left) and cache hit rate (right)"
                 " vs. cosine-similarity threshold</sup>",
        ),
        xaxis=dict(title="Cosine-similarity threshold"),
        height=520, width=900, **LAYOUT_BASE,
    )
    fig.update_yaxes(title_text="F1 / SI-SNRi (dB)", secondary_y=False,
                     showline=True, linecolor=AXIS_LINE, linewidth=1.2,
                     gridcolor=GRID, ticks="outside", tickcolor=AXIS_LINE,
                     ticklen=4, tickwidth=1, tickfont=TICK_FONT,
                     title_font=AXIS_TITLE_FONT)
    fig.update_yaxes(title_text="Cache hit rate (%)", range=[0, 105],
                     secondary_y=True,
                     showline=True, linecolor=AXIS_LINE, linewidth=1.2,
                     ticks="outside", tickcolor=AXIS_LINE,
                     ticklen=4, tickwidth=1,
                     tickfont=TICK_FONT, title_font=AXIS_TITLE_FONT)
    _apply_axis_style(fig)
    return fig


# ===================================================================
#  LaTeX tables
# ===================================================================

def _write_latex_table(path: Path, columns: Sequence[str],
                       rows: Sequence[Sequence[str]],
                       caption: str, label: str,
                       col_align: Optional[str] = None) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if col_align is None:
        col_align = "l" + "r" * (len(columns) - 1)
    lines = [
        "% Auto-generated by plot_dissertation_figures.py \u2014 do not edit by hand.",
        "\\begin{table}[htbp]", "  \\centering",
        f"  \\caption{{{caption}}}", f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{col_align}}}",
        "    \\toprule",
        "    " + " & ".join(columns) + r" \\",
        "    \\midrule",
    ]
    for row in rows:
        lines.append("    " + " & ".join(row) + r" \\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"  Wrote {path.relative_to(FINAL_RESULTS_DIR)}")


def table_best_runs() -> None:
    header = ["Separator", "Classifier", "Run condition",
               "F1 base", "F1 sep", "\u0394F1",
               "Bal.Acc base", "Bal.Acc sep", "\u0394BA",
               "SI-SNRi (dB)"]
    rows = []
    for name, base, _ in SEPARATORS_AIRPLANE:
        for cls in CLASSIFIERS_AIRPLANE:
            d = _load_best(base, cls)
            if d is None:
                continue
            f1_mix = _metric(d, "f1_score", "mix_cls")
            f1_sep = _metric(d, "f1_score", "mix_sep_cls")
            ba_mix = _metric(d, "balanced_accuracy", "mix_cls")
            ba_sep = _metric(d, "balanced_accuracy", "mix_sep_cls")
            snri = _signal(d, "mean_si_snri_db", "mix_sep_cls")
            delta_f1 = f1_sep - f1_mix if not (np.isnan(f1_mix) or np.isnan(f1_sep)) else float("nan")
            delta_ba = ba_sep - ba_mix if not (np.isnan(ba_mix) or np.isnan(ba_sep)) else float("nan")
            rows.append([
                name, _cls_label(cls), "Mixture",
                _fmt(f1_mix), _fmt(f1_sep), _fmt(delta_f1, "+.3f"),
                _fmt(ba_mix), _fmt(ba_sep), _fmt(delta_ba, "+.3f"),
                _fmt(snri, ".2f"),
            ])
            f1_clean = _metric(d, "f1_score", "clean_cls")
            f1_clean_sep = _metric(d, "f1_score", "clean_sep_cls")
            ba_clean = _metric(d, "balanced_accuracy", "clean_cls")
            ba_clean_sep = _metric(d, "balanced_accuracy", "clean_sep_cls")
            delta_f1_c = f1_clean_sep - f1_clean if not (np.isnan(f1_clean) or np.isnan(f1_clean_sep)) else float("nan")
            delta_ba_c = ba_clean_sep - ba_clean if not (np.isnan(ba_clean) or np.isnan(ba_clean_sep)) else float("nan")
            rows.append([
                name, _cls_label(cls), "Clean COI",
                _fmt(f1_clean), _fmt(f1_clean_sep), _fmt(delta_f1_c, "+.3f"),
                _fmt(ba_clean), _fmt(ba_clean_sep), _fmt(delta_ba_c, "+.3f"),
                "\u2014",
            ])
    _write_latex_table(
        TABLE_DIR / "tab_best_runs.tex",
        header, rows,
        caption="Best classification run per (separator, classifier) pair, "
                "selected by maximum \u0394F1 improvement. "
                "Base = without separation; Sep = with separation; "
                "\u0394 = Sep $-$ Base.",
        label="tab:best-runs", col_align="lllrrrrrrr",
    )


def table_separation_metrics() -> None:
    rows = []
    for name, base, _ in SEPARATORS_AIRPLANE:
        for cls in CLASSIFIERS_AIRPLANE:
            d = _load_best(base, cls)
            if d is None:
                continue
            rows.append([
                name, _cls_label(cls),
                _fmt(_signal(d, "mean_si_snr_db", "mix_sep_cls"), ".2f"),
                _fmt(_signal(d, "mean_si_snri_db", "mix_sep_cls"), ".2f"),
                _fmt(_signal(d, "mean_sdr_db", "mix_sep_cls"), ".2f"),
                _fmt(_signal(d, "mean_rms_error_db", "mix_sep_cls"), "+.2f"),
                _fmt(_signal(d, "mean_sel_error_db", "mix_sep_cls"), "+.2f"),
            ])
    if not rows:
        print("  [skipped \u2014 no data]")
        return
    _write_latex_table(
        TABLE_DIR / "tab_separation_metrics.tex",
        ["Model", "Classifier", "SI-SNR (dB)", "SI-SNRi (dB)", "SDR (dB)",
         "RMS err (dB)", "SEL err (dB)"],
        rows,
        caption="Signal-level separation metrics on held-out artificial mixtures "
                "(mixture + separation condition).",
        label="tab:separation-metrics", col_align="llrrrrr",
    )


def table_classification_impact() -> None:
    rows = []
    for name, base, _ in SEPARATORS_AIRPLANE:
        for cls in CLASSIFIERS_AIRPLANE:
            d = _load_best(base, cls)
            if d is None:
                continue
            for cond_base, cond_sep, cond_label in [
                ("mix_cls", "mix_sep_cls", "Mixture"),
                ("clean_cls", "clean_sep_cls", "Clean COI"),
            ]:
                for metric_key, metric_label in [("f1_score", "F1"),
                                                  ("precision", "Prec."),
                                                  ("recall", "Rec.")]:
                    v_base = _metric(d, metric_key, cond_base)
                    v_sep = _metric(d, metric_key, cond_sep)
                    delta = v_sep - v_base if not (np.isnan(v_base) or np.isnan(v_sep)) else float("nan")
                    rows.append([
                        name, _cls_label(cls), cond_label, metric_label,
                        _fmt(v_base), _fmt(v_sep), _fmt(delta, "+.3f"),
                    ])
    if not rows:
        print("  [skipped \u2014 no data]")
        return
    _write_latex_table(
        TABLE_DIR / "tab_classification_impact.tex",
        ["Separator", "Classifier", "Input", "Metric",
         "Base", "+Separation", "\u0394"],
        rows,
        caption="Downstream classification impact of separation. "
                "\u0394 = (with separation) $-$ (without).",
        label="tab:classification-impact", col_align="llllrrrr",
    )


def table_generalisation() -> None:
    rows = []
    for name, base, risoux_dir in SEPARATORS_AIRPLANE:
        for cls in CLASSIFIERS_AIRPLANE:
            d_in = _load_best(base, cls)
            d_out = _load_best(risoux_dir, cls, risoux=True) if risoux_dir else None
            if d_in is None:
                continue
            f1_in_val = _metric(d_in, "f1_score", "mix_sep_cls")
            f1_out_val = _metric(d_out, "f1_score", "as_is_sep_cls") if d_out else float("nan")
            f1_base_in = _metric(d_in, "f1_score", "mix_cls")
            f1_base_out = _metric(d_out, "f1_score", "as_is_cls") if d_out else float("nan")
            delta_in = f1_in_val - f1_base_in if not (np.isnan(f1_in_val) or np.isnan(f1_base_in)) else float("nan")
            delta_out = f1_out_val - f1_base_out if not (np.isnan(f1_out_val) or np.isnan(f1_base_out)) else float("nan")
            gap = (f1_in_val - f1_out_val) if not (np.isnan(f1_in_val) or np.isnan(f1_out_val)) else float("nan")
            rows.append([
                name, _cls_label(cls),
                _fmt(f1_in_val), _fmt(delta_in, "+.3f"),
                _fmt(f1_out_val), _fmt(delta_out, "+.3f"),
                _fmt(gap, "+.3f"),
            ])
    if not rows:
        print("  [skipped \u2014 no data]")
        return
    _write_latex_table(
        TABLE_DIR / "tab_generalisation.tex",
        ["Model", "Classifier",
         "In-dist F1", "In-dist \u0394F1",
         "Risoux F1", "Risoux \u0394F1",
         "Gap (in \u2212 OOD)"],
        rows,
        caption="Out-of-distribution generalisation: F1 and \u0394F1 on artificial mixtures vs.\\ "
                "Risoux forest field recordings. "
                "\u0394F1 = (with separation) $-$ (without).",
        label="tab:generalisation", col_align="llrrrrrr",
    )


def table_multiclass() -> None:
    rows = []
    for cls in CLASSIFIERS_AIRPLANE:
        d = _load_best(TUSS_MC_AIR_DIR, cls)
        if d is not None:
            f1_mix = _metric(d, "f1_score", "mix_cls")
            f1_sep = _metric(d, "f1_score", "mix_sep_cls")
            delta = f1_sep - f1_mix if not (np.isnan(f1_mix) or np.isnan(f1_sep)) else float("nan")
            rows.append([
                "Airplane", _cls_label(cls), "TUSS (multi)",
                _fmt(f1_mix), _fmt(f1_sep), _fmt(delta, "+.3f"),
                _fmt(_signal(d, "mean_si_snri_db", "mix_sep_cls"), ".2f"),
            ])
    for cls in CLASSIFIERS_BIRD:
        d = _load_best(TUSS_MC_BIRD_DIR, cls)
        if d is None:
            continue
        f1_mix = _metric(d, "f1_score", "mix_cls")
        f1_sep = _metric(d, "f1_score", "mix_sep_cls")
        delta = f1_sep - f1_mix if not (np.isnan(f1_mix) or np.isnan(f1_sep)) else float("nan")
        rows.append([
            "Bird", _cls_label(cls), "TUSS (multi)",
            _fmt(f1_mix), _fmt(f1_sep), _fmt(delta, "+.3f"),
            _fmt(_signal(d, "mean_si_snri_db", "mix_sep_cls"), ".2f"),
        ])
    if not rows:
        print("  [skipped \u2014 no data]")
        return
    _write_latex_table(
        TABLE_DIR / "tab_multiclass_tuss.tex",
        ["COI", "Classifier", "Model",
         "F1 base", "F1 sep", "\u0394F1", "SI-SNRi (dB)"],
        rows,
        caption="TUSS multi-class performance per COI head. "
                "\u0394F1 = \u0394(separation $-$ no separation).",
        label="tab:multiclass-tuss", col_align="llllrrrr",
    )


def table_noise_robustness() -> None:
    files = sorted(NOISE_RESULTS_DIR.glob("noise_increase_energy_*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        print("  [skipped \u2014 no data]")
        return
    rows = []
    for nf in files:
        with open(nf) as f:
            data = json.load(f)
        model = data.get("config", {}).get("model_type", "model")
        for r in sorted(data.get("snr_results", []), key=lambda x: x["snr_db"], reverse=True):
            has_si = "mean_si_sdr_noisy_vs_clean_db" in r
            rows.append([
                model, _fmt(r.get("snr_db"), ".1f"),
                _fmt(r.get("mean_si_sdr_noisy_vs_clean_db"), "+.2f") if has_si else "\u2014",
                _fmt(r.get("mean_rms_degradation_db"), "+.2f"),
                _fmt(r.get("mean_sel_degradation_db"), "+.2f"),
                _fmt(r.get("n_segments"), "d"),
            ])
    if not rows:
        print("  [skipped \u2014 empty JSONs]")
        return
    _write_latex_table(
        TABLE_DIR / "tab_noise_robustness.tex",
        ["Model", "SNR (dB)", "SI-SDR (dB)", "\u0394RMS (dB)", "\u0394SEL (dB)", "$N$"],
        rows,
        caption="Robustness under additive white noise. SI-SDR is scale-invariant; "
                "RMS/SEL contain rescaling artefact.",
        label="tab:noise-robustness", col_align="lrrrrr",
    )


def table_activity_gating() -> None:
    json_path = GATING_RESULTS_DIR / "sweep_results.json"
    if not json_path.exists():
        print("  [skipped \u2014 no data]")
        return
    with open(json_path) as f:
        data = json.load(f)
    sweep = sorted(data.get("sweep", []), key=lambda r: r["threshold"])
    baseline = data.get("baseline_no_recycling", {})
    rows = []
    for r in sweep:
        rows.append([
            _fmt(r.get("threshold"), ".2f"),
            _fmt(r.get("f1_score"), ".3f"),
            _fmt(r.get("mean_si_snri_db"), ".2f"),
            _fmt(r.get("cache_hit_rate", float("nan")) * 100, ".1f"),
        ])
    rows.append([
        "baseline", _fmt(baseline.get("f1_score"), ".3f"),
        _fmt(baseline.get("mean_si_snri_db"), ".2f"), "0.0",
    ])
    _write_latex_table(
        TABLE_DIR / "tab_activity_gating.tex",
        ["Threshold", "F1", "SI-SNRi (dB)", "Hit rate (%)"],
        rows,
        caption="Activity-gating sweep: quality\u2013efficiency trade-off.",
        label="tab:activity-gating", col_align="lrrr",
    )


# ===================================================================
#  I/O driver
# ===================================================================

def save_figure(fig: go.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / f"{name}.png"
    try:
        fig.write_image(str(png), scale=PNG_SCALE)
        print(f"  Wrote {png.relative_to(FINAL_RESULTS_DIR)}")
    except Exception as e:
        print(f"  [PNG skipped \u2014 {e}]")
    html = OUTPUT_DIR / f"{name}.html"
    fig.write_html(str(html))
    print(f"  Wrote {html.relative_to(FINAL_RESULTS_DIR)}")


def main() -> None:
    print("=" * 72)
    print("Building dissertation figures and LaTeX tables")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 72)

    figures = [
        ("fig_4_1_separation_metrics", fig_separation_metrics),
        ("fig_4_2_energy_preservation", fig_energy_preservation),
        ("fig_4_3_classification_impact", fig_classification_impact),
        ("fig_4_4_confusion_shifts", fig_confusion_shifts),
        ("fig_4_5_generalisation_gap", fig_generalisation_gap),
        ("fig_4_6_multiclass_overview", fig_multiclass_overview),
        ("fig_4_7_noise_robustness", fig_noise_robustness),
        ("fig_4_8_activity_gating", fig_activity_gating),
    ]

    print("\n\u2014 Figures \u2014")
    for name, fn in figures:
        try:
            print(f"  Building {name} \u2026")
            save_figure(fn(), name)
        except Exception as e:
            import traceback
            print(f"  ERROR {name}: {e}")
            traceback.print_exc()

    print("\n\u2014 LaTeX tables \u2014")
    for tbl_fn in (table_best_runs, table_separation_metrics,
                   table_classification_impact, table_generalisation,
                   table_multiclass, table_noise_robustness,
                   table_activity_gating):
        try:
            tbl_fn()
        except Exception as e:
            import traceback
            print(f"  ERROR {tbl_fn.__name__}: {e}")
            traceback.print_exc()

    print(f"\nDone.\n  Figures: {OUTPUT_DIR}\n  Tables : {TABLE_DIR}")


if __name__ == "__main__":
    main()