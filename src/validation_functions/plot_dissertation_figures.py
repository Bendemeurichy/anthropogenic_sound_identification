#!/usr/bin/env python3
"""
plot_dissertation_figures.py — Results & Discussion chapter visualisations.

Key design changes (v2):
- Matched runs: signal and classification figures use the SAME timestamp
  across both classifiers (AST + PANN), averaging signal metrics and
  showing per-classifier classification results side-by-side.
- Fig 4.7: only the latest noise-robustness JSON is plotted.
- Fig 4.8: split into two clear subplots (F1 vs threshold, hit rate vs threshold).
- Fig 4.4: shows absolute confusion values with a neutral diverging scale.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
FINAL_RESULTS_DIR = SCRIPT_DIR / "final_results"
OUTPUT_DIR = FINAL_RESULTS_DIR / "dissertation_figures"
TABLE_DIR = OUTPUT_DIR / "tables"

SUDORMRF_DIR = FINAL_RESULTS_DIR / "sudormrf_airplane_results"
CLAPSEP_DIR = FINAL_RESULTS_DIR / "clapsep_airplane_results"
TUSS_MC_AIR_DIR = FINAL_RESULTS_DIR / "tuss_multiclass_airplane_results"

SUDORMRF_RISOUX_DIR = FINAL_RESULTS_DIR / "sudormrf_airplane_results_risoux"
CLAPSEP_RISOUX_DIR = FINAL_RESULTS_DIR / "clapsep_airplane_results_risoux"
TUSS_MC_AIR_RISOUX_DIR = FINAL_RESULTS_DIR / "tuss_multiclass_airplane_results_risoux"

TUSS_MC_BIRD_DIR = FINAL_RESULTS_DIR / "tuss_multiclass_bird_results"
TUSS_MC_BIRD_RISOUX_DIR = FINAL_RESULTS_DIR / "tuss_multiclass_bird_results_risoux"

NOISE_RESULTS_DIR = FINAL_RESULTS_DIR / "noise_increase_results"
GATING_RESULTS_DIR = FINAL_RESULTS_DIR / "activity_gating_results"

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
SEPARATORS_AIRPLANE: List[Tuple[str, Path, Optional[Path]]] = [
    ("SuDoRM-RF", SUDORMRF_DIR, SUDORMRF_RISOUX_DIR),
    ("CLAPSep", CLAPSEP_DIR, CLAPSEP_RISOUX_DIR),
    ("TUSS (multi)", TUSS_MC_AIR_DIR, TUSS_MC_AIR_RISOUX_DIR),
]

CLASSIFIERS_AIRPLANE = ["ast_finetuned", "pann_finetuned"]
CLASSIFIERS_BIRD = ["bird_mae", "audioprotopnet"]

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
QUAL = px.colors.qualitative.Plotly

MODEL_COLORS = {
    "SuDoRM-RF": QUAL[0],
    "CLAPSep": QUAL[1],
    "TUSS (multi)": QUAL[2],
}

CLASSIFIER_COLORS = {
    "ast_finetuned": QUAL[3],
    "pann_finetuned": QUAL[4],
    "bird_mae": QUAL[5],
    "audioprotopnet": QUAL[6],
}

COL_POS = "#2ca02c"
COL_NEG = "#d62728"
COL_NEU = "#7f7f7f"

FONT_FAMILY = "Times New Roman, Nimbus Roman, serif"
FONT = dict(family=FONT_FAMILY, size=13, color="#1f1f1f")
TITLE_FONT = dict(family=FONT_FAMILY, size=18, color="#1f1f1f")
AXIS_TITLE_FONT = dict(family=FONT_FAMILY, size=13, color="#1f1f1f")
TICK_FONT = dict(family=FONT_FAMILY, size=11, color="#1f1f1f")

LAYOUT_BASE = dict(
    template="plotly_white",
    font=FONT,
    title_font=TITLE_FONT,
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    margin=dict(l=80, r=40, t=100, b=80),
)

PNG_SCALE = 2

CLS_DISPLAY = {
    "pann_finetuned": "PANN",
    "ast_finetuned": "AST",
    "bird_mae": "BirdMAE",
    "audioprotopnet": "AudioProtoPNet",
}


def _cls_label(name: str) -> str:
    return CLS_DISPLAY.get(name, name.replace("_", " "))


# ---------------------------------------------------------------------------
# Data loading — matched runs (same timestamp across classifiers)
# ---------------------------------------------------------------------------

def _extract_timestamp(path: Path) -> str:
    """Extract timestamp from results_test_*.json filename."""
    stem = path.stem
    parts = stem.split("_")
    # results_test_TIMESTAMP.json  or  results_test_risoux_TIMESTAMP.json
    if "risoux" in stem:
        return "_".join(parts[3:])
    return "_".join(parts[2:])


def _load_best_matched(base_dir: Path, classifiers: Sequence[str],
                       risoux: bool = False) -> Optional[dict]:
    """Find the run with the best average ΔF1 across *all* classifiers,
    requiring the SAME timestamp to exist in every classifier directory.

    Returns a dict with keys:
        timestamp, and one key per classifier (e.g. 'ast_finetuned')
    Signal metrics are averaged across classifiers (stored under key
    '__avg_signal').
    """
    pattern = "results_test_risoux_*.json" if risoux else "results_test_*.json"

    # Collect files per classifier
    files_by_cls: Dict[str, Dict[str, Path]] = {}
    for cls in classifiers:
        sub = base_dir / cls
        if not sub.exists():
            return None
        files_by_cls[cls] = {_extract_timestamp(f): f for f in sorted(sub.glob(pattern))}

    # Find common timestamps
    common = set.intersection(*[set(files_by_cls[c].keys()) for c in classifiers])
    if not common:
        return None

    base_cond = "as_is_cls" if risoux else "mix_cls"
    sep_cond = "as_is_sep_cls" if risoux else "mix_sep_cls"

    best_run: Optional[dict] = None
    best_score = float("-inf")

    for ts in sorted(common):
        loaded: Dict[str, dict] = {}
        for cls in classifiers:
            with open(files_by_cls[cls][ts]) as fh:
                loaded[cls] = json.load(fh)

        # Average ΔF1 across classifiers
        deltas = []
        for cls in classifiers:
            d = loaded[cls]
            f1_base = d.get(base_cond, {}).get("f1_score", float("nan"))
            f1_sep = d.get(sep_cond, {}).get("f1_score", float("nan"))
            if not (np.isnan(f1_base) or np.isnan(f1_sep)):
                deltas.append(f1_sep - f1_base)
        if not deltas:
            continue
        score = float(np.mean(deltas))

        if score > best_score:
            best_score = score
            # Average signal metrics across classifiers
            avg_signal: Dict[str, Dict[str, float]] = {}
            for cond in ["mix_sep_cls", "clean_sep_cls"]:
                avg_signal[cond] = {}
                for key in ["mean_si_snr_db", "mean_si_snri_db", "mean_sdr_db",
                            "mean_rms_error_db", "mean_sel_error_db"]:
                    vals = []
                    for cls in classifiers:
                        sm = loaded[cls].get(cond, {}).get("signal_metrics") or {}
                        v = sm.get(key, float("nan"))
                        if not np.isnan(v):
                            vals.append(float(v))
                    avg_signal[cond][key] = float(np.mean(vals)) if vals else float("nan")

            best_run = {
                "timestamp": ts,
                "__avg_signal": avg_signal,
            }
            best_run.update(loaded)

    return best_run


def _load_best(base_dir: Path, classifier: str,
               risoux: bool = False) -> Optional[dict]:
    """Legacy: select best run for a single classifier (used for bird)."""
    sub = base_dir / classifier
    if not sub.exists():
        return None
    pattern = "results_test_risoux_*.json" if risoux else "results_test_*.json"
    candidates = []
    for f in sorted(sub.glob(pattern)):
        try:
            with open(f) as fh:
                candidates.append(json.load(fh))
        except Exception:
            continue
    if not candidates:
        return None
    base_cond = "as_is_cls" if risoux else "mix_cls"
    sep_cond = "as_is_sep_cls" if risoux else "mix_sep_cls"
    best, best_delta = None, float("-inf")
    for d in candidates:
        f1_base = d.get(base_cond, {}).get("f1_score", float("nan"))
        f1_sep = d.get(sep_cond, {}).get("f1_score", float("nan"))
        if np.isnan(f1_base) or np.isnan(f1_sep):
            continue
        if f1_sep - f1_base > best_delta:
            best_delta = f1_sep - f1_base
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
    # Try averaged signal first (from matched runs)
    if "__avg_signal" in d:
        return float(d["__avg_signal"].get(condition, {}).get(key, float("nan")))
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


# ===================================================================
# FIGURE 4.1 — Separation Metrics (averaged across classifiers)
# ===================================================================

def fig_separation_metrics() -> go.Figure:
    models, si_snri, sdr = [], [], []
    for name, base, _ in SEPARATORS_AIRPLANE:
        d = _load_best_matched(base, CLASSIFIERS_AIRPLANE)
        if d is None:
            continue
        models.append(name)
        si_snri.append(_signal(d, "mean_si_snri_db", "mix_sep_cls"))
        sdr.append(_signal(d, "mean_sdr_db", "mix_sep_cls"))

    if not models:
        return _placeholder("No separation data.", "Separation Metrics")

    colors = [MODEL_COLORS[m] for m in models]
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=models, y=si_snri, name="SI-SNRi (dB)",
        marker_color=colors,
        text=[_fmt(v, ".1f") for v in si_snri],
        textposition="outside", textfont=dict(size=11),
        marker_line=dict(color="rgba(0,0,0,0.3)", width=1),
    ))
    fig.add_trace(go.Bar(
        x=models, y=sdr, name="SDR (dB)",
        marker_color=colors, opacity=0.55,
        text=[_fmt(v, ".1f") for v in sdr],
        textposition="outside", textfont=dict(size=11),
        marker_line=dict(color="rgba(0,0,0,0.3)", width=1),
    ))

    fig.update_layout(
        barmode="group", bargap=0.25, bargroupgap=0.15,
        title=dict(
            text="<b>Signal-level separation quality</b><br>"
                 "<sup>Averaged across AST + PANN classifiers · Same run per model</sup>"
        ),
        yaxis=dict(title="dB", gridcolor="rgba(0,0,0,0.07)",
                   showline=True, linecolor="#1f1f1f", linewidth=1),
        xaxis=dict(showline=True, linecolor="#1f1f1f", linewidth=1),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18,
                    xanchor="center", x=0.5, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="rgba(0,0,0,0.1)", borderwidth=1),
        height=520, width=750,
        **LAYOUT_BASE,
    )
    return fig


# ===================================================================
# FIGURE 4.2 — Energy Preservation (averaged across classifiers)
# ===================================================================

def fig_energy_preservation() -> go.Figure:
    models, rms_mix, sel_mix = [], [], []
    rms_clean, sel_clean = [], []
    for name, base, _ in SEPARATORS_AIRPLANE:
        d = _load_best_matched(base, CLASSIFIERS_AIRPLANE)
        if d is None:
            continue
        models.append(name)
        rms_mix.append(_signal(d, "mean_rms_error_db", "mix_sep_cls"))
        sel_mix.append(_signal(d, "mean_sel_error_db", "mix_sep_cls"))
        rms_clean.append(_signal(d, "mean_rms_error_db", "clean_sep_cls"))
        sel_clean.append(_signal(d, "mean_sel_error_db", "clean_sep_cls"))

    if not models:
        return _placeholder("No energy data.", "Energy Preservation")

    colors = [MODEL_COLORS[m] for m in models]
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=models, y=rms_mix, name="RMS (mix)",
        marker_color=colors, opacity=0.95,
        text=[_fmt(v, "+.1f") for v in rms_mix],
        textposition="outside", textfont=dict(size=10),
        marker_line=dict(color="rgba(0,0,0,0.3)", width=1),
    ))
    fig.add_trace(go.Bar(
        x=models, y=sel_mix, name="SEL (mix)",
        marker_color=colors, opacity=0.50,
        text=[_fmt(v, "+.1f") for v in sel_mix],
        textposition="outside", textfont=dict(size=10),
        marker_line=dict(color="rgba(0,0,0,0.3)", width=1),
    ))

    fig.add_hline(y=0, line_dash="dot", line_color="#1f1f1f", line_width=1.5,
                  annotation_text="Perfect preservation",
                  annotation_position="top left",
                  annotation_font=dict(size=10, color=COL_NEU))

    fig.update_layout(
        barmode="group", bargap=0.25, bargroupgap=0.12,
        title=dict(
            text="<b>Energy preservation deviation</b><br>"
                 "<sup>Averaged across AST + PANN · Δ = separated − clean-COI reference</sup>"
        ),
        yaxis=dict(title="Δ energy (dB)", gridcolor="rgba(0,0,0,0.07)",
                   showline=True, linecolor="#1f1f1f", linewidth=1, zeroline=False),
        xaxis=dict(showline=True, linecolor="#1f1f1f", linewidth=1),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18,
                    xanchor="center", x=0.5, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="rgba(0,0,0,0.1)", borderwidth=1),
        height=520, width=750,
        **LAYOUT_BASE,
    )
    return fig


# ===================================================================
# FIGURE 4.3 — Classification Impact (matched runs)
# ===================================================================

def fig_classification_impact() -> go.Figure:
    metrics = [("f1_score", "F1"), ("precision", "Precision"), ("recall", "Recall")]

    # Load matched runs for all models
    matched: Dict[str, dict] = {}
    for name, base, _ in SEPARATORS_AIRPLANE:
        d = _load_best_matched(base, CLASSIFIERS_AIRPLANE)
        if d is not None:
            matched[name] = d

    if not matched:
        return _placeholder("No classification data.", "Classification Impact")

    fig = make_subplots(
        rows=3, cols=2,
        row_titles=[f"<b>Δ {label}</b>" for _, label in metrics],
        column_titles=[f"<b>{_cls_label(c)}</b>" for c in CLASSIFIERS_AIRPLANE],
        vertical_spacing=0.10, horizontal_spacing=0.10,
        shared_yaxes="rows",
    )

    for row_idx, (key, metric_label) in enumerate(metrics, 1):
        for col_idx, cls in enumerate(CLASSIFIERS_AIRPLANE, 1):
            y_labels = []
            delta_c = []
            delta_m = []
            for name, _, _ in SEPARATORS_AIRPLANE:
                if name not in matched:
                    continue
                d = matched[name]
                dc = _metric(d.get(cls), key, "clean_sep_cls") - _metric(d.get(cls), key, "clean_cls")
                dm = _metric(d.get(cls), key, "mix_sep_cls") - _metric(d.get(cls), key, "mix_cls")
                y_labels.append(name)
                delta_c.append(dc)
                delta_m.append(dm)

            colors = [MODEL_COLORS[m] for m in y_labels]

            fig.add_trace(go.Bar(
                y=y_labels, x=delta_c, orientation="h",
                name="Clean COI Δ", marker_color=colors, opacity=0.45,
                text=[_fmt(v, "+.3f") for v in delta_c],
                textposition="outside", textfont=dict(size=9),
                marker_line=dict(color="rgba(0,0,0,0.3)", width=1),
                showlegend=(row_idx == 1 and col_idx == 1),
                legendgroup="clean",
            ), row=row_idx, col=col_idx)

            fig.add_trace(go.Bar(
                y=y_labels, x=delta_m, orientation="h",
                name="Mixture Δ", marker_color=colors, opacity=0.95,
                text=[_fmt(v, "+.3f") for v in delta_m],
                textposition="outside", textfont=dict(size=9),
                marker_line=dict(color="rgba(0,0,0,0.3)", width=1),
                showlegend=(row_idx == 1 and col_idx == 1),
                legendgroup="mix",
            ), row=row_idx, col=col_idx)

            fig.add_vline(x=0, line_dash="solid", line_color="#1f1f1f",
                          line_width=1.2, row=row_idx, col=col_idx)

            fig.update_xaxes(gridcolor="rgba(0,0,0,0.07)", showline=True,
                             linecolor="#1f1f1f", linewidth=1, zeroline=False,
                             row=row_idx, col=col_idx)
            if col_idx == 1:
                fig.update_yaxes(showline=True, linecolor="#1f1f1f", linewidth=1,
                                 row=row_idx, col=col_idx)
            else:
                fig.update_yaxes(showticklabels=False, row=row_idx, col=col_idx)

    fig.update_layout(
        barmode="group", bargap=0.25, bargroupgap=0.12,
        title=dict(
            text="<b>Impact of separation on downstream classification</b><br>"
                 "<sup>Light = clean COI Δ · Dark = mixture Δ · Same run across classifiers</sup>"
        ),
        height=900, width=1050,
        legend=dict(orientation="h", yanchor="bottom", y=-0.06,
                    xanchor="center", x=0.5, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="rgba(0,0,0,0.1)", borderwidth=1),
        **LAYOUT_BASE,
    )
    return fig


# ===================================================================
# FIGURE 4.4 — Confusion Shifts (actual values, neutral scale)
# ===================================================================

def _norm_cm(cm: dict) -> Tuple[np.ndarray, float]:
    tp, tn, fp, fn = cm.get("tp", 0), cm.get("tn", 0), cm.get("fp", 0), cm.get("fn", 0)
    total = tp + tn + fp + fn
    arr = np.array([[tn, fp], [fn, tp]], dtype=float)
    return arr, float(total)


def fig_confusion_shifts() -> go.Figure:
    panels = []
    for name, base, _ in SEPARATORS_AIRPLANE:
        d = _load_best_matched(base, CLASSIFIERS_AIRPLANE)
        if d is None:
            continue
        # Use AST as representative (signal metrics averaged, confusion is per-classifier)
        # Show both classifiers
        for cls in CLASSIFIERS_AIRPLANE:
            cls_data = d.get(cls)
            if cls_data is None:
                continue
            cm_cls = _cm(cls_data, "mix_cls")
            cm_sep = _cm(cls_data, "mix_sep_cls")
            if cm_cls is None or cm_sep is None:
                continue
            arr_cls, total_cls = _norm_cm(cm_cls)
            arr_sep, total_sep = _norm_cm(cm_sep)
            # Percentages
            pct_cls = arr_cls / (total_cls or 1) * 100.0
            pct_sep = arr_sep / (total_sep or 1) * 100.0
            delta = pct_sep - pct_cls

            # Annotate with absolute counts + delta
            text = []
            for r in range(2):
                row_text = []
                for c in range(2):
                    row_text.append(
                        f"{int(arr_sep[r,c])}<br><span style='font-size:8px'>"
                        f"({delta[r,c]:+.1f}pp)</span>"
                    )
                text.append(row_text)

            panels.append((name, _cls_label(cls), delta, text, arr_sep, total_sep))

    if not panels:
        return _placeholder("No confusion data.", "Confusion Shifts")

    n = len(panels)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"<b>{m}</b><br><span style='font-size:10px'>{c}</span>"
                        for m, c, _, _, _, _ in panels],
        horizontal_spacing=0.10, vertical_spacing=0.22 if n_rows > 1 else 0.15,
    )

    # Neutral diverging: purple-white-orange (no green/red semantic)
    div_scale = [[0.0, "#54278f"], [0.5, "#FFFFFF"], [1.0, "#d94801"]]
    zmax = max(abs(p[2]).max() for p in panels)
    zmax = max(zmax, 5.0)

    for idx, (name, cls_label, delta, text, arr_sep, total_sep) in enumerate(panels):
        r, c = idx // n_cols + 1, idx % n_cols + 1
        fig.add_trace(go.Heatmap(
            z=delta, x=["Pred. 0", "Pred. 1"], y=["True 0", "True 1"],
            text=text, texttemplate="%{text}",
            textfont=dict(size=11, color="#1f1f1f"),
            colorscale=div_scale, zmid=0, zmin=-zmax, zmax=zmax,
            showscale=(idx == 0),
            colorbar=dict(title=dict(text="Δpp", side="right"),
                          thickness=12, x=1.02) if idx == 0 else None,
        ), row=r, col=c)
        fig.update_xaxes(side="bottom", row=r, col=c)
        fig.update_yaxes(autorange="reversed", row=r, col=c)
        fig.add_annotation(
            xref=f"x{idx+1}", yref=f"y{idx+1}",
            x=0.5, y=-0.32, xanchor="center",
            text=f"N = {int(total_sep)}", showarrow=False,
            font=dict(size=9, color=COL_NEU),
        )

    fig.update_layout(
        title=dict(
            text="<b>How separation shifts the confusion pattern</b><br>"
                 "<sup>Cell = absolute count (baseline Δ in pp) · Purple = decrease · Orange = increase</sup>"
        ),
        height=max(400, 380 * n_rows), width=max(700, 360 * n_cols),
        **LAYOUT_BASE,
    )
    return fig


# ===================================================================
# FIGURE 4.5 — Generalisation Gap (matched runs)
# ===================================================================

def fig_generalisation_gap() -> go.Figure:
    data_by_cls: Dict[str, List[dict]] = {c: [] for c in CLASSIFIERS_AIRPLANE}

    for cls in CLASSIFIERS_AIRPLANE:
        for name, base, risoux_dir in SEPARATORS_AIRPLANE:
            # Load matched run, but classification result is per-classifier
            d_matched = _load_best_matched(base, CLASSIFIERS_AIRPLANE, risoux=False)
            d_matched_risoux = _load_best_matched(risoux_dir, CLASSIFIERS_AIRPLANE, risoux=True) if risoux_dir else None
            if d_matched is None:
                continue
            data_by_cls[cls].append({
                "model": name,
                "f1_in": _metric(d_matched.get(cls), "f1_score", "mix_sep_cls"),
                "f1_base_in": _metric(d_matched.get(cls), "f1_score", "mix_cls"),
                "f1_out": _metric(d_matched_risoux.get(cls), "f1_score", "as_is_sep_cls") if d_matched_risoux else float("nan"),
                "f1_base_out": _metric(d_matched_risoux.get(cls), "f1_score", "as_is_cls") if d_matched_risoux else float("nan"),
            })

    if not any(data_by_cls.values()):
        return _placeholder("No generalisation data.", "Generalisation Gap")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"<b>{_cls_label(c)}</b>" for c in CLASSIFIERS_AIRPLANE],
        horizontal_spacing=0.14,
    )

    for col_idx, cls in enumerate(CLASSIFIERS_AIRPLANE, 1):
        entries = data_by_cls[cls]
        if not entries:
            continue

        models = [e["model"] for e in entries]
        colors = [MODEL_COLORS[m] for m in models]
        f1_in = [e["f1_in"] for e in entries]
        f1_out = [e["f1_out"] for e in entries]
        delta_f1 = [e["f1_out"] - e["f1_base_out"] if not (np.isnan(e["f1_out"]) or np.isnan(e["f1_base_out"])) else float("nan")
                    for e in entries]

        bar_kw = dict(textposition="outside", cliponaxis=False,
                      textfont=dict(size=10),
                      marker_line=dict(color="rgba(0,0,0,0.2)", width=0.6))

        fig.add_trace(go.Bar(
            x=models, y=f1_in, name="In-dist (mix+sep)",
            marker_color=colors, opacity=0.95,
            text=[_fmt(v, ".2f") for v in f1_in],
            showlegend=(col_idx == 1), legendgroup="in",
            **bar_kw,
        ), row=1, col=col_idx)
        fig.add_trace(go.Bar(
            x=models, y=f1_out, name="Risoux (as-is+sep)",
            marker_color=colors, opacity=0.50,
            text=[_fmt(v, ".2f") for v in f1_out],
            showlegend=(col_idx == 1), legendgroup="out",
            **bar_kw,
        ), row=1, col=col_idx)

        for i, d_f1 in enumerate(delta_f1):
            if np.isnan(d_f1):
                continue
            y_anchor = max(f1_in[i] if not np.isnan(f1_in[i]) else 0,
                           f1_out[i] if not np.isnan(f1_out[i]) else 0) + 0.04
            color = COL_NEG if d_f1 < -0.05 else COL_POS if d_f1 > 0.05 else COL_NEU
            fig.add_annotation(
                x=models[i], y=y_anchor, text=f"<b>Δ {d_f1:+.2f}</b>",
                showarrow=False, yanchor="bottom",
                font=dict(size=10, color=color),
                row=1, col=col_idx,
            )

        fig.update_yaxes(title_text="F1 score" if col_idx == 1 else None,
                         range=[0, 1.10], gridcolor="rgba(0,0,0,0.07)",
                         showline=True, linecolor="#1f1f1f", linewidth=1,
                         row=1, col=col_idx)
        fig.update_xaxes(showline=True, linecolor="#1f1f1f", linewidth=1,
                         row=1, col=col_idx)

    fig.update_layout(
        barmode="group", bargap=0.28, bargroupgap=0.08,
        title=dict(
            text="<b>Generalisation gap: in-distribution vs. Risoux</b><br>"
                 "<sup>Same matched run for both classifiers · Δ = Risoux improvement over baseline</sup>"
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18,
                    xanchor="center", x=0.5, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="rgba(0,0,0,0.1)", borderwidth=1),
        height=520, width=950,
        **LAYOUT_BASE,
    )
    return fig


# ===================================================================
# FIGURE 4.6 — Multi-Class Overview
# ===================================================================

def fig_multiclass_overview() -> go.Figure:
    rows = []

    # Airplane — matched runs
    for name, base, _ in SEPARATORS_AIRPLANE:
        d = _load_best_matched(base, CLASSIFIERS_AIRPLANE)
        if d is None:
            continue
        for cls in CLASSIFIERS_AIRPLANE:
            delta = _metric(d.get(cls), "f1_score", "mix_sep_cls") - _metric(d.get(cls), "f1_score", "mix_cls")
            rows.append({
                "group": f"Airplane · {name}",
                "classifier": _cls_label(cls),
                "delta": delta,
            })

    # Bird — matched runs
    d_bird = _load_best_matched(TUSS_MC_BIRD_DIR, CLASSIFIERS_BIRD)
    if d_bird is not None:
        for cls in CLASSIFIERS_BIRD:
            delta = _metric(d_bird.get(cls), "f1_score", "mix_sep_cls") - _metric(d_bird.get(cls), "f1_score", "mix_cls")
            rows.append({
                "group": "Bird · TUSS (multi)",
                "classifier": _cls_label(cls),
                "delta": delta,
            })

    if not rows:
        return _placeholder("No multi-class data.", "Multi-Class Separation")

    labels = [f"{r['group']}<br><span style='font-size:9px'>{r['classifier']}</span>" for r in rows]
    deltas = [r["delta"] for r in rows]
    colors = [COL_POS if v > 0.005 else COL_NEG if v < -0.005 else COL_NEU for v in deltas]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=deltas,
        marker_color=colors,
        text=[_fmt(v, "+.3f") for v in deltas],
        textposition="outside",
        textfont=dict(size=9),
        marker_line=dict(color="rgba(0,0,0,0.2)", width=0.6),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#1f1f1f", line_width=1.2)

    fig.update_layout(
        title=dict(
            text="<b>Does separation help classification?</b><br>"
                 "<sup>ΔF1 = F1(sep) − F1(no sep) · Same matched run per model</sup>"
        ),
        yaxis=dict(title="ΔF1", gridcolor="rgba(0,0,0,0.07)",
                   showline=True, linecolor="#1f1f1f", linewidth=1, zeroline=False),
        xaxis=dict(showline=True, linecolor="#1f1f1f", linewidth=1),
        height=520, width=max(800, 120 * len(labels)),
        **LAYOUT_BASE,
    )
    return fig


# ===================================================================
# FIGURE 4.7 — Noise Robustness (latest run only)
# ===================================================================

def fig_noise_robustness() -> go.Figure:
    noise_files = sorted(NOISE_RESULTS_DIR.glob("noise_increase_energy_*.json"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
    if not noise_files:
        return _placeholder("No noise data.\nRun test_noise_increase.py first.",
                            "Noise Robustness")

    # Only the latest file
    latest = noise_files[0]
    with open(latest) as f:
        data = json.load(f)

    model = data.get("config", {}).get("model_type", "model")
    # Sort in ASCENDING order (low SNR/hard on left, high SNR/easy on right)
    snr_results = sorted(data.get("snr_results", []),
                         key=lambda x: x["snr_db"])
    if not snr_results:
        return _placeholder("Latest noise JSON empty.", "Noise Robustness")

    snr = [r["snr_db"] for r in snr_results]
    use_si = "mean_si_sdr_noisy_vs_clean_db" in snr_results[0]

    if use_si:
        y = [r["mean_si_sdr_noisy_vs_clean_db"] for r in snr_results]
        y_std = [r.get("std_si_sdr_noisy_vs_clean_db", 0) for r in snr_results]
        y_label = "SI-SDR noisy vs. clean (dB)"
    else:
        y = [r["mean_rms_degradation_db"] for r in snr_results]
        y_std = [r.get("std_rms_degradation_db", 0) for r in snr_results]
        y_label = "RMS degradation (dB)"

    color = MODEL_COLORS.get(model, QUAL[2])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=snr, y=y, mode="lines+markers",
        name=f"{model}",
        line=dict(color=color, width=2.5),
        marker=dict(size=9, symbol="circle"),
        error_y=dict(type="data", array=y_std, thickness=1.2, width=5,
                     color=color),
    ))

    # Shade realistic deployment range (SNR: -5 to 10 dB)
    fig.add_vrect(x0=-5, x1=10, fillcolor=QUAL[4], opacity=0.08,
                  layer="below", line_width=0)
    fig.add_annotation(x=2.5, y=0.95, xref="x", yref="paper",
                       text="<b>Realistic deployment</b>", showarrow=False,
                       font=dict(size=10, color="#1f1f1f"))

    if use_si:
        fig.add_hline(y=0, line_dash="dot", line_color="#1f1f1f", line_width=1.5,
                      annotation_text="No degradation (clean reference)",
                      annotation_position="bottom right",
                      annotation_font=dict(size=10, color=COL_NEU))

    fig.update_layout(
        title=dict(
            text="<b>Separation robustness under additive white noise</b><br>"
                 f"<sup>{model} · Latest run · harder ← → easier</sup>"
        ),
        xaxis=dict(title="Input SNR (dB)",
                   gridcolor="rgba(0,0,0,0.07)", showline=True,
                   linecolor="#1f1f1f", linewidth=1),
        yaxis=dict(title=y_label,
                   gridcolor="rgba(0,0,0,0.07)", showline=True,
                   linecolor="#1f1f1f", linewidth=1),
        height=520, width=800,
        **LAYOUT_BASE,
    )
    return fig


# ===================================================================
# FIGURE 4.8 — Activity Gating (split into 2 clear panels)
# ===================================================================

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
    hit = [r.get("cache_hit_rate", float("nan")) * 100 for r in sweep]

    bf1 = baseline.get("f1_score", float("nan"))

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=["<b>F1 score vs. threshold</b>", "<b>Cache hit rate vs. threshold</b>"],
    )

    # Top: F1
    fig.add_trace(go.Scatter(
        x=th, y=f1, mode="lines+markers", name="F1",
        line=dict(color=QUAL[1], width=2.5),
        marker=dict(size=8),
    ), row=1, col=1)
    if not np.isnan(bf1):
        fig.add_hline(y=bf1, line_dash="dot", line_color=QUAL[1],
                      annotation_text=f"Baseline F1 = {bf1:.2f}",
                      annotation_position="top left", row=1, col=1)

    zone_th = [t for t, fi in zip(th, f1)
               if not np.isnan(fi) and fi >= bf1 - 0.02]
    if zone_th:
        fig.add_vrect(x0=min(zone_th), x1=max(zone_th),
                      fillcolor=QUAL[2], opacity=0.10, layer="below",
                      line_width=0, row=1, col=1)
        fig.add_annotation(x=(min(zone_th) + max(zone_th)) / 2, y=0.05,
                           xref="x", yref="paper",
                           text="<b>F1 ≥ baseline − 0.02</b>",
                           showarrow=False, font=dict(size=9, color="#1f1f1f"),
                           row=1, col=1)

    # Bottom: Cache hit rate
    fig.add_trace(go.Scatter(
        x=th, y=hit, mode="lines+markers", name="Cache hit rate (%)",
        line=dict(color=QUAL[2], width=2.5),
        marker=dict(size=8, symbol="diamond"),
    ), row=2, col=1)

    zone_th_hit = [t for t, fi, hi in zip(th, f1, hit)
                   if not np.isnan(fi) and not np.isnan(hi)
                   and fi >= bf1 - 0.02 and hi >= 50]
    if zone_th_hit:
        fig.add_vrect(x0=min(zone_th_hit), x1=max(zone_th_hit),
                      fillcolor=QUAL[2], opacity=0.10, layer="below",
                      line_width=0, row=2, col=1)
        fig.add_annotation(x=(min(zone_th_hit) + max(zone_th_hit)) / 2, y=0.05,
                           xref="x", yref="paper",
                           text="<b>Recommended zone (F1 ≥ baseline − 0.02 & hit ≥ 50%)</b>",
                           showarrow=False, font=dict(size=9, color="#1f1f1f"),
                           row=2, col=1)

    fig.update_xaxes(title_text="Cosine-similarity threshold",
                     gridcolor="rgba(0,0,0,0.07)", showline=True,
                     linecolor="#1f1f1f", linewidth=1, row=2, col=1)
    fig.update_yaxes(title_text="F1 score", range=[0, 1.05],
                     gridcolor="rgba(0,0,0,0.07)", showline=True,
                     linecolor="#1f1f1f", linewidth=1, row=1, col=1)
    fig.update_yaxes(title_text="Hit rate (%)", range=[0, 105],
                     gridcolor="rgba(0,0,0,0.07)", showline=True,
                     linecolor="#1f1f1f", linewidth=1, row=2, col=1)

    fig.update_layout(
        title=dict(
            text="<b>Activity gating: quality–efficiency trade-off</b><br>"
                 "<sup>Top: classification quality · Bottom: computational savings</sup>"
        ),
        height=700, width=800,
        showlegend=False,
        **LAYOUT_BASE,
    )
    return fig


# ===================================================================
# Placeholder helper
# ===================================================================

def _placeholder(msg: str, title: str = "Data not yet available") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=f"<b>{msg}</b>", xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(family=FONT_FAMILY, size=14, color=COL_NEU))
    fig.update_layout(title=f"<b>{title}</b>", height=400, width=800,
                      **LAYOUT_BASE)
    return fig


# ===================================================================
# LaTeX tables
# ===================================================================

def _write_latex_table(path: Path, columns: Sequence[str],
                       rows: Sequence[Sequence[str]],
                       caption: str, label: str,
                       col_align: Optional[str] = None) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if col_align is None:
        col_align = "l" + "r" * (len(columns) - 1)
    lines = [
        "% Auto-generated — do not edit by hand.",
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
               "F1 base", "F1 sep", "ΔF1",
               "Bal.Acc base", "Bal.Acc sep", "ΔBA",
               "SI-SNRi (dB)"]
    rows = []
    for name, base, _ in SEPARATORS_AIRPLANE:
        d = _load_best_matched(base, CLASSIFIERS_AIRPLANE)
        if d is None:
            continue
        for cls in CLASSIFIERS_AIRPLANE:
            cls_data = d.get(cls)
            if cls_data is None:
                continue
            f1_mix = _metric(cls_data, "f1_score", "mix_cls")
            f1_sep = _metric(cls_data, "f1_score", "mix_sep_cls")
            ba_mix = _metric(cls_data, "balanced_accuracy", "mix_cls")
            ba_sep = _metric(cls_data, "balanced_accuracy", "mix_sep_cls")
            snri = _signal(d, "mean_si_snri_db", "mix_sep_cls")
            delta_f1 = f1_sep - f1_mix if not (np.isnan(f1_mix) or np.isnan(f1_sep)) else float("nan")
            delta_ba = ba_sep - ba_mix if not (np.isnan(ba_mix) or np.isnan(ba_sep)) else float("nan")
            rows.append([
                name, _cls_label(cls), "Mixture",
                _fmt(f1_mix), _fmt(f1_sep), _fmt(delta_f1, "+.3f"),
                _fmt(ba_mix), _fmt(ba_sep), _fmt(delta_ba, "+.3f"),
                _fmt(snri, ".2f"),
            ])
            f1_clean = _metric(cls_data, "f1_score", "clean_cls")
            f1_clean_sep = _metric(cls_data, "f1_score", "clean_sep_cls")
            ba_clean = _metric(cls_data, "balanced_accuracy", "clean_cls")
            ba_clean_sep = _metric(cls_data, "balanced_accuracy", "clean_sep_cls")
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
                "selected by maximum average ΔF1 across classifiers.",
        label="tab:best-runs", col_align="lllrrrrrrr",
    )


def table_separation_metrics() -> None:
    rows = []
    for name, base, _ in SEPARATORS_AIRPLANE:
        d = _load_best_matched(base, CLASSIFIERS_AIRPLANE)
        if d is None:
            continue
        rows.append([
            name,
            _fmt(_signal(d, "mean_si_snr_db", "mix_sep_cls"), ".2f"),
            _fmt(_signal(d, "mean_si_snri_db", "mix_sep_cls"), ".2f"),
            _fmt(_signal(d, "mean_sdr_db", "mix_sep_cls"), ".2f"),
            _fmt(_signal(d, "mean_rms_error_db", "mix_sep_cls"), "+.2f"),
            _fmt(_signal(d, "mean_sel_error_db", "mix_sep_cls"), "+.2f"),
        ])
    if not rows:
        print("  [skipped — no data]")
        return
    _write_latex_table(
        TABLE_DIR / "tab_separation_metrics.tex",
        ["Model", "SI-SNR (dB)", "SI-SNRi (dB)", "SDR (dB)",
         "RMS err (dB)", "SEL err (dB)"],
        rows,
        caption="Signal-level separation metrics (averaged across AST + PANN).",
        label="tab:separation-metrics", col_align="lrrrrr",
    )


def table_classification_impact() -> None:
    rows = []
    for name, base, _ in SEPARATORS_AIRPLANE:
        d = _load_best_matched(base, CLASSIFIERS_AIRPLANE)
        if d is None:
            continue
        for cls in CLASSIFIERS_AIRPLANE:
            cls_data = d.get(cls)
            if cls_data is None:
                continue
            for cond_base, cond_sep, cond_label in [
                ("mix_cls", "mix_sep_cls", "Mixture"),
                ("clean_cls", "clean_sep_cls", "Clean COI"),
            ]:
                for metric_key, metric_label in [("f1_score", "F1"),
                                                  ("precision", "Prec."),
                                                  ("recall", "Rec.")]:
                    v_base = _metric(cls_data, metric_key, cond_base)
                    v_sep = _metric(cls_data, metric_key, cond_sep)
                    delta = v_sep - v_base if not (np.isnan(v_base) or np.isnan(v_sep)) else float("nan")
                    rows.append([
                        name, _cls_label(cls), cond_label, metric_label,
                        _fmt(v_base), _fmt(v_sep), _fmt(delta, "+.3f"),
                    ])
    if not rows:
        print("  [skipped — no data]")
        return
    _write_latex_table(
        TABLE_DIR / "tab_classification_impact.tex",
        ["Separator", "Classifier", "Input", "Metric",
         "Base", "+Separation", "Δ"],
        rows,
        caption="Downstream classification impact of separation. "
                "Δ = (with separation) − (without).",
        label="tab:classification-impact", col_align="llllrrr",
    )


def table_generalisation() -> None:
    rows = []
    for name, base, risoux_dir in SEPARATORS_AIRPLANE:
        d_in = _load_best_matched(base, CLASSIFIERS_AIRPLANE, risoux=False)
        d_out = _load_best_matched(risoux_dir, CLASSIFIERS_AIRPLANE, risoux=True) if risoux_dir else None
        if d_in is None:
            continue
        for cls in CLASSIFIERS_AIRPLANE:
            cls_in = d_in.get(cls)
            cls_out = d_out.get(cls) if d_out else None
            f1_in_val = _metric(cls_in, "f1_score", "mix_sep_cls")
            f1_out_val = _metric(cls_out, "f1_score", "as_is_sep_cls")
            f1_base_in = _metric(cls_in, "f1_score", "mix_cls")
            f1_base_out = _metric(cls_out, "f1_score", "as_is_cls")
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
        print("  [skipped — no data]")
        return
    _write_latex_table(
        TABLE_DIR / "tab_generalisation.tex",
        ["Model", "Classifier",
         "In-dist F1", "In-dist ΔF1",
         "Risoux F1", "Risoux ΔF1",
         "Gap (in − OOD)"],
        rows,
        caption="Out-of-distribution generalisation (matched runs).",
        label="tab:generalisation", col_align="llrrrrr",
    )


def table_multiclass() -> None:
    rows = []
    d_air = _load_best_matched(TUSS_MC_AIR_DIR, CLASSIFIERS_AIRPLANE)
    if d_air is not None:
        for cls in CLASSIFIERS_AIRPLANE:
            cls_data = d_air.get(cls)
            if cls_data is None:
                continue
            f1_mix = _metric(cls_data, "f1_score", "mix_cls")
            f1_sep = _metric(cls_data, "f1_score", "mix_sep_cls")
            delta = f1_sep - f1_mix if not (np.isnan(f1_mix) or np.isnan(f1_sep)) else float("nan")
            rows.append([
                "Airplane", _cls_label(cls), "TUSS (multi)",
                _fmt(f1_mix), _fmt(f1_sep), _fmt(delta, "+.3f"),
                _fmt(_signal(d_air, "mean_si_snri_db", "mix_sep_cls"), ".2f"),
            ])
    d_bird = _load_best_matched(TUSS_MC_BIRD_DIR, CLASSIFIERS_BIRD)
    if d_bird is not None:
        for cls in CLASSIFIERS_BIRD:
            cls_data = d_bird.get(cls)
            if cls_data is None:
                continue
            f1_mix = _metric(cls_data, "f1_score", "mix_cls")
            f1_sep = _metric(cls_data, "f1_score", "mix_sep_cls")
            delta = f1_sep - f1_mix if not (np.isnan(f1_mix) or np.isnan(f1_sep)) else float("nan")
            rows.append([
                "Bird", _cls_label(cls), "TUSS (multi)",
                _fmt(f1_mix), _fmt(f1_sep), _fmt(delta, "+.3f"),
                _fmt(_signal(d_bird, "mean_si_snri_db", "mix_sep_cls"), ".2f"),
            ])
    if not rows:
        print("  [skipped — no data]")
        return
    _write_latex_table(
        TABLE_DIR / "tab_multiclass_tuss.tex",
        ["COI", "Classifier", "Model",
         "F1 base", "F1 sep", "ΔF1", "SI-SNRi (dB)"],
        rows,
        caption="TUSS multi-class performance per COI head (matched runs).",
        label="tab:multiclass-tuss", col_align="lllrrrr",
    )


def table_noise_robustness() -> None:
    files = sorted(NOISE_RESULTS_DIR.glob("noise_increase_energy_*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        print("  [skipped — no data]")
        return
    # Latest only
    latest = files[0]
    with open(latest) as f:
        data = json.load(f)
    model = data.get("config", {}).get("model_type", "model")
    rows = []
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
        print("  [skipped — empty JSONs]")
        return
    _write_latex_table(
        TABLE_DIR / "tab_noise_robustness.tex",
        ["Model", "SNR (dB)", "SI-SDR (dB)", "ΔRMS (dB)", "ΔSEL (dB)", "$N$"],
        rows,
        caption="Robustness under additive white noise (latest run).",
        label="tab:noise-robustness", col_align="lrrrrr",
    )


def table_activity_gating() -> None:
    json_path = GATING_RESULTS_DIR / "sweep_results.json"
    if not json_path.exists():
        print("  [skipped — no data]")
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
            _fmt(r.get("cache_hit_rate", float("nan")) * 100, ".1f"),
        ])
    rows.append([
        "baseline", _fmt(baseline.get("f1_score"), ".3f"), "0.0",
    ])
    _write_latex_table(
        TABLE_DIR / "tab_activity_gating.tex",
        ["Threshold", "F1", "Hit rate (%)"],
        rows,
        caption="Activity-gating sweep: quality–efficiency trade-off.",
        label="tab:activity-gating", col_align="lrrr",
    )


# ===================================================================
# I/O driver
# ===================================================================

def save_figure(fig: go.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / f"{name}.png"
    try:
        fig.write_image(str(png), scale=PNG_SCALE)
        print(f"  Wrote {png.relative_to(FINAL_RESULTS_DIR)}")
    except Exception as e:
        print(f"  [PNG skipped — {e}]")
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

    print("\n— Figures —")
    for name, fn in figures:
        try:
            print(f"  Building {name} …")
            save_figure(fn(), name)
        except Exception as e:
            import traceback
            print(f"  ERROR {name}: {e}")
            traceback.print_exc()

    print("\n— LaTeX tables —")
    table_best_runs()
    table_separation_metrics()
    table_classification_impact()
    table_generalisation()
    table_multiclass()
    table_noise_robustness()
    table_activity_gating()

    print(f"\nDone.")
    print(f"  Figures: {OUTPUT_DIR}")
    print(f"  Tables : {TABLE_DIR}")


if __name__ == "__main__":
    main()