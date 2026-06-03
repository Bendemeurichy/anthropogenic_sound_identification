#!/usr/bin/env python3
"""
plot_paper_figures.py — 2-column paper-optimised figures for extended abstract.
Generates EN + NL versions with larger fonts and narrower layouts.
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
OUTPUT_DIR = FINAL_RESULTS_DIR / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUDORMRF_DIR = FINAL_RESULTS_DIR / "sudormrf_airplane_results"
CLAPSEP_DIR = FINAL_RESULTS_DIR / "clapsep_airplane_results"
TUSS_MC_AIR_DIR = FINAL_RESULTS_DIR / "tuss_multiclass_airplane_results"
NOISE_RESULTS_DIR = FINAL_RESULTS_DIR / "noise_increase_results"
TUSS_MC_BIRD_DIR = FINAL_RESULTS_DIR / "tuss_multiclass_bird_results"
TUSS_MC_BIRD_RISOUX_DIR = FINAL_RESULTS_DIR / "tuss_multiclass_bird_results_risoux"

SEPARATORS_AIRPLANE: List[Tuple[str, Path]] = [
    ("SuDoRM-RF", SUDORMRF_DIR),
    ("CLAPSep", CLAPSEP_DIR),
    ("TUSS", TUSS_MC_AIR_DIR),
]

CLASSIFIERS_AIRPLANE = ["ast_finetuned", "pann_finetuned"]
CLASSIFIERS_BIRD = ["bird_mae", "audioprotopnet"]

QUAL = px.colors.qualitative.Plotly
MODEL_COLORS = {
    "SuDoRM-RF": QUAL[0],
    "CLAPSep": QUAL[1],
    "TUSS": QUAL[2],
}

CLS_DISPLAY = {
    "pann_finetuned": "PANN",
    "ast_finetuned": "AST",
    "bird_mae": "BirdMAE",
    "audioprotopnet": "AudioProtoPNet",
}

# ---------------------------------------------------------------------------
# 2-column paper layout constants
# ---------------------------------------------------------------------------
FONT_FAMILY = "Times New Roman, Nimbus Roman, serif"
PAPER_TITLE_SIZE = 16
PAPER_AXIS_SIZE = 13
PAPER_TICK_SIZE = 11
PAPER_ANNOT_SIZE = 10
PAPER_LEGEND_SIZE = 11
COL_WIDTH_PX = 560
ASPECT_SINGLE = 0.65  # height/width for single-panel
ASPECT_DUAL = 0.55  # for two-panel
PNG_SCALE = 3

# ---------------------------------------------------------------------------
# Language labels
# ---------------------------------------------------------------------------
LANG = {
    "en": {
        "si_snri": "SI-SNRi (dB)",
        "sdr": "SDR (dB)",
        "db": "dB",
        "title_sep": "Signal-level separation quality",
        "x_model": "Model",
        "title_cls": "Impact of separation on downstream classification",
        "clean_coi": "Clean COI",
        "mixture": "Mixture",
        "delta_f1": "ΔF₁",
        "title_noise": "Noise robustness",
        "input_snr": "Input SNR (dB)",
        "si_sdr": "SI-SDR noisy vs. clean (dB)",
        "no_degradation": "No degradation",
        "realistic_range": "Typical ecoacoustic range",
        "title_bird_sep": "TUSS separation: Airplane vs. Bird",
        "airplane": "Airplane",
        "bird": "Bird",
        "baseline": "Baseline (no recycling)",
        "positive_zone": "Positive separation",
        "title_activity": "Activity gating: quality vs. efficiency",
        "hit_rate": "Cache hit rate (%)",
        "si_snri_label": "SI-SNRi (dB)",
        "threshold": "Cosine-similarity threshold",
        "title_bird_cls": "Bird multi-class separation impact",
        "delta_f1_clean": "ΔF₁ (clean COI)",
        "delta_f1_mixture": "ΔF₁ (mixture)",
        "delta_ba_in": "ΔBA (in-domain mixtures)",
        "delta_ba_ood": "ΔBA (Risoux)",
        "left_panel": "ΔF₁ (in-domain)",
        "right_panel": "ΔBA (in-domain vs. Risoux)",
    },
    "nl": {
        "si_snri": "SI-SNRi (dB)",
        "sdr": "SDR (dB)",
        "db": "dB",
        "title_sep": "Signaalkwaliteit van scheiding",
        "x_model": "Model",
        "title_cls": "Impact van scheiding op classificatie",
        "clean_coi": "Schoon doelgeluid",
        "mixture": "Mengsel",
        "delta_f1": "ΔF₁",
        "title_noise": "Ruisrobuustheid",
        "input_snr": "Invoer-SNR (dB)",
        "si_sdr": "SI-SDR met ruis vs. schoon (dB)",
        "no_degradation": "Geen degradatie",
        "realistic_range": "Typisch ecoakoestisch bereik",
        "title_bird_sep": "TUSS-scheiding: Vliegtuig vs. Vogel",
        "airplane": "Vliegtuig",
        "bird": "Vogel",
        "baseline": "Basislijn (geen caching)",
        "positive_zone": "Positieve scheiding",
        "title_activity": "Activity gating: kwaliteit vs. efficiëntie",
        "hit_rate": "Cache hit rate (%)",
        "si_snri_label": "SI-SNRi (dB)",
        "threshold": "Cosine similarity-drempel",
        "title_bird_cls": "Impact van vogelscheiding op classificatie",
        "delta_f1_clean": "ΔF₁ (schoon doelgeluid)",
        "delta_f1_mixture": "ΔF₁ (mengsel)",
        "delta_ba_in": "ΔBA (in-domain mengsels)",
        "delta_ba_ood": "ΔBA (Risoux)",
        "left_panel": "ΔF₁ (in-domain)",
        "right_panel": "ΔBA (in-domain vs. Risoux)",
    },
}

# ---------------------------------------------------------------------------
# Data loading (re-used from dissertation script)
# ---------------------------------------------------------------------------


def _load_classifier_result(classifier_dir: Path, risoux: bool = False) -> Optional[dict]:
    if not classifier_dir.exists():
        return None
    pattern = "results_test_risoux_*.json" if risoux else "results_test_*.json"
    files = list(classifier_dir.glob(pattern))
    if not files:
        return None
    try:
        with open(files[0]) as fh:
            return json.load(fh)
    except Exception:
        return None


def _load_results(base_dir: Path, classifiers: Sequence[str], risoux: bool = False) -> Optional[dict]:
    loaded: Dict[str, dict] = {}
    for cls in classifiers:
        result = _load_classifier_result(base_dir / cls, risoux)
        if result is None:
            return None
        loaded[cls] = result

    avg_signal: Dict[str, Dict[str, float]] = {}
    for cond in ["mix_sep_cls", "clean_sep_cls"]:
        avg_signal[cond] = {}
        for key in [
            "mean_si_snr_db",
            "mean_si_snri_db",
            "mean_sdr_db",
            "mean_rms_error_db",
            "mean_sel_error_db",
        ]:
            vals = []
            for cls in classifiers:
                sm = loaded[cls].get(cond, {}).get("signal_metrics") or {}
                v = sm.get(key, float("nan"))
                if not np.isnan(v):
                    vals.append(float(v))
            avg_signal[cond][key] = float(np.mean(vals)) if vals else float("nan")

    result = {"__avg_signal": avg_signal}
    result.update(loaded)
    return result


def _signal(d: Optional[dict], key: str, condition: str = "mix_sep_cls") -> float:
    if d is None:
        return float("nan")
    if "__avg_signal" in d:
        return float(d["__avg_signal"].get(condition, {}).get(key, float("nan")))
    sm = d.get(condition, {}).get("signal_metrics") or {}
    try:
        return float(sm.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def _metric(d: Optional[dict], key: str, condition: str) -> float:
    if d is None:
        return float("nan")
    return float(d.get(condition, {}).get(key, float("nan")))


def _fmt(v: float, spec: str = ".1f", dash: str = "\u2014") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return dash
    return f"{v:{spec}}"


def _cls_label(name: str) -> str:
    return CLS_DISPLAY.get(name, name.replace("_", " "))


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _paper_layout(fig: go.Figure, lang: str, height: int, width: int) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        font=dict(family=FONT_FAMILY, size=PAPER_AXIS_SIZE, color="#1f1f1f"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=50, r=25, t=60, b=60),
        height=height,
        width=width,
        legend=dict(
            font=dict(size=PAPER_LEGEND_SIZE),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
        ),
    )
    return fig


def _save(fig: go.Figure, name: str, lang: str) -> None:
    path = OUTPUT_DIR / f"{name}_{lang}.png"
    fig.write_image(str(path), scale=PNG_SCALE)
    print(f"  Saved {path}")


# ===================================================================
# FIGURE 1 — Separation Metrics (2-column bar chart)
# ===================================================================


def fig_separation_metrics(lang: str = "en") -> go.Figure:
    t = LANG[lang]
    models, si_snri, sdr = [], [], []
    for name, base in SEPARATORS_AIRPLANE:
        d = _load_results(base, CLASSIFIERS_AIRPLANE)
        if d is None:
            continue
        models.append(name)
        si_snri.append(_signal(d, "mean_si_snri_db", "mix_sep_cls"))
        sdr.append(_signal(d, "mean_sdr_db", "mix_sep_cls"))

    colors = [MODEL_COLORS[m] for m in models]
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=models,
            y=si_snri,
            name=t["si_snri"],
            marker_color=colors,
            marker_line=dict(color="rgba(0,0,0,0.3)", width=0.8),
            text=[_fmt(v) for v in si_snri],
            textposition="outside",
            textfont=dict(size=PAPER_TICK_SIZE + 1, color="#1f1f1f"),
            cliponaxis=False,
        )
    )
    fig.add_trace(
        go.Bar(
            x=models,
            y=sdr,
            name=t["sdr"],
            marker_color=colors,
            opacity=0.50,
            marker_line=dict(color="rgba(0,0,0,0.3)", width=0.8),
            text=[_fmt(v) for v in sdr],
            textposition="outside",
            textfont=dict(size=PAPER_TICK_SIZE + 1, color="#1f1f1f"),
            cliponaxis=False,
        )
    )

    height = int(COL_WIDTH_PX * ASPECT_SINGLE)
    fig = _paper_layout(fig, lang, height, COL_WIDTH_PX)
    fig.update_layout(
        barmode="group",
        bargap=0.22,
        bargroupgap=0.12,
        title=dict(text=f"<b>{t['title_sep']}</b>", font=dict(size=PAPER_TITLE_SIZE)),
        yaxis=dict(
            title=t["db"],
            gridcolor="rgba(0,0,0,0.07)",
            showline=True,
            linecolor="#1f1f1f",
            linewidth=1,
            tickfont=dict(size=PAPER_TICK_SIZE),
        ),
        xaxis=dict(
            showline=True,
            linecolor="#1f1f1f",
            linewidth=1,
            tickfont=dict(size=PAPER_TICK_SIZE),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5
        ),
    )
    return fig


# ===================================================================
# FIGURE 2 — Classification Impact (simplified: ΔF1 only, 1 row x 2 cols)
# ===================================================================


def fig_classification_impact(lang: str = "en") -> go.Figure:
    t = LANG[lang]

    matched: Dict[str, dict] = {}
    for name, base in SEPARATORS_AIRPLANE:
        d = _load_results(base, CLASSIFIERS_AIRPLANE)
        if d is not None:
            matched[name] = d

    fig = make_subplots(
        rows=1,
        cols=2,
        column_titles=[f"<b>{_cls_label(c)}</b>" for c in CLASSIFIERS_AIRPLANE],
        horizontal_spacing=0.15,
    )

    bar_kw = dict(
        orientation="h",
        cliponaxis=False,
        marker_line=dict(color="rgba(0,0,0,0.2)", width=0.5),
        textfont=dict(size=PAPER_TICK_SIZE, color="#1f1f1f"),
    )

    for col_idx, cls in enumerate(CLASSIFIERS_AIRPLANE, 1):
        y_labels = []
        delta_c, delta_m = [], []
        for name, _ in SEPARATORS_AIRPLANE:
            if name not in matched:
                continue
            d_m = matched[name]
            dc = _metric(d_m.get(cls), "f1_score", "clean_sep_cls") - _metric(
                d_m.get(cls), "f1_score", "clean_cls"
            )
            dm = _metric(d_m.get(cls), "f1_score", "mix_sep_cls") - _metric(
                d_m.get(cls), "f1_score", "mix_cls"
            )
            y_labels.append(name)
            delta_c.append(dc)
            delta_m.append(dm)

        colors = [MODEL_COLORS[m] for m in y_labels]

        fig.add_trace(
            go.Bar(
                y=y_labels,
                x=delta_c,
                name=t["clean_coi"],
                marker_color=colors,
                opacity=0.45,
                text=[_fmt(v, ".3f") for v in delta_c],
                textposition="inside",
                showlegend=(col_idx == 1),
                legendgroup="clean",
                **bar_kw,
            ),
            row=1,
            col=col_idx,
        )
        fig.add_trace(
            go.Bar(
                y=y_labels,
                x=delta_m,
                name=t["mixture"],
                marker_color=colors,
                opacity=0.95,
                text=[_fmt(v, ".3f") for v in delta_m],
                textposition="inside",
                showlegend=(col_idx == 1),
                legendgroup="mix",
                **bar_kw,
            ),
            row=1,
            col=col_idx,
        )

        fig.add_vline(
            x=0,
            line_dash="solid",
            line_color="#1f1f1f",
            line_width=1.2,
            row=1,
            col=col_idx,
        )

        # Set text color based on bar darkness
        all_vals = [v for v in (delta_c + delta_m) if not np.isnan(v)]
        y_range = max(abs(v) for v in all_vals) * 1.15 if all_vals else 0.1

        fig.update_xaxes(
            range=[-y_range, y_range],
            title_text=t["delta_f1"] if col_idx == 1 else None,
            gridcolor="rgba(0,0,0,0.07)",
            showline=True,
            linecolor="#1f1f1f",
            linewidth=1,
            zeroline=False,
            tickfont=dict(size=PAPER_TICK_SIZE),
            row=1,
            col=col_idx,
        )
        fig.update_yaxes(
            showline=True,
            linecolor="#1f1f1f",
            linewidth=1,
            tickfont=dict(size=PAPER_TICK_SIZE),
            row=1,
            col=col_idx,
        )

    height = int(COL_WIDTH_PX * ASPECT_SINGLE * 0.85)
    fig = _paper_layout(fig, lang, height, COL_WIDTH_PX)
    fig.update_layout(
        barmode="group",
        bargap=0.20,
        bargroupgap=0.10,
        title=dict(text=f"<b>{t['title_cls']}</b>", font=dict(size=PAPER_TITLE_SIZE)),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5
        ),
    )
    # Center column titles
    for ann in fig.layout.annotations:
        ann.font = dict(size=PAPER_AXIS_SIZE, color="#1f1f1f")
        ann.xanchor = "center"
    return fig


# ===================================================================
# FIGURE 3 — Noise Robustness
# ===================================================================


def fig_noise_robustness(lang: str = "en") -> go.Figure:
    t = LANG[lang]
    noise_files = sorted(
        NOISE_RESULTS_DIR.glob("noise_increase_energy_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not noise_files:
        return _placeholder("No noise data available")

    with open(noise_files[0]) as f:
        data = json.load(f)

    snr_results = sorted(
        data.get("snr_results", []), key=lambda x: x["snr_db"], reverse=True
    )
    snr = [r["snr_db"] for r in snr_results]

    use_si = "mean_si_sdr_noisy_vs_clean_db" in snr_results[0]
    y = [
        r["mean_si_sdr_noisy_vs_clean_db" if use_si else "mean_rms_degradation_db"]
        for r in snr_results
    ]
    y_std = [
        r.get("std_si_sdr_noisy_vs_clean_db" if use_si else "std_rms_degradation_db", 0)
        for r in snr_results
    ]
    y_label = t["si_sdr"] if use_si else "RMS degradation (dB)"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=snr,
            y=y,
            mode="lines+markers",
            name="TUSS",
            line=dict(color=MODEL_COLORS["TUSS"], width=2.5),
            marker=dict(size=8, symbol="circle"),
            error_y=dict(
                type="data",
                array=y_std,
                thickness=1,
                width=4,
                color=MODEL_COLORS["TUSS"],
            ),
        )
    )

    # Shade realistic deployment range
    fig.add_vrect(
        x0=-5,
        x1=10,
        fillcolor=QUAL[4],
        opacity=0.08,
        layer="below",
        line_width=0,
        annotation_text=t["realistic_range"],
        annotation_font=dict(size=PAPER_ANNOT_SIZE, color="#555"),
        annotation_position="top left",
    )

    if use_si:
        fig.add_hline(
            y=0,
            line_dash="dot",
            line_color="#1f1f1f",
            line_width=1.5,
            annotation_text=t["no_degradation"],
            annotation_position="bottom right",
            annotation_font=dict(size=PAPER_ANNOT_SIZE, color="#555"),
        )

    height = int(COL_WIDTH_PX * ASPECT_SINGLE)
    fig = _paper_layout(fig, lang, height, COL_WIDTH_PX)
    fig.update_layout(
        title=dict(text=f"<b>{t['title_noise']}</b>", font=dict(size=PAPER_TITLE_SIZE)),
        xaxis=dict(
            title=t["input_snr"],
            gridcolor="rgba(0,0,0,0.07)",
            showline=True,
            linecolor="#1f1f1f",
            linewidth=1,
            autorange="reversed",
            tickfont=dict(size=PAPER_TICK_SIZE),
        ),
        yaxis=dict(
            title=y_label,
            gridcolor="rgba(0,0,0,0.07)",
            showline=True,
            linecolor="#1f1f1f",
            linewidth=1,
            tickfont=dict(size=PAPER_TICK_SIZE),
        ),
        showlegend=False,
    )
    return fig


# ===================================================================
# FIGURE 4 — Bird classifier improvements (ΔF₁ & ΔBA)
# ===================================================================


def fig_classifier_improvements(lang: str = "en") -> go.Figure:
    t = LANG[lang]
    d_bird = _load_results(TUSS_MC_BIRD_DIR, CLASSIFIERS_BIRD)
    d_bird_risoux = _load_results(
        TUSS_MC_BIRD_RISOUX_DIR, CLASSIFIERS_BIRD, risoux=True
    )

    if d_bird is None and d_bird_risoux is None:
        return _placeholder("No bird data")

    cls_labels = []
    delta_f1_clean, delta_f1_mix = [], []
    delta_ba_in, delta_ba_ood = [], []

    for cls in CLASSIFIERS_BIRD:
        m_data = d_bird.get(cls) if d_bird else None
        r_data = d_bird_risoux.get(cls) if d_bird_risoux else None
        if m_data is None and r_data is None:
            continue

        f1_mix_base = _metric(m_data, "f1_score", "mix_cls")
        f1_mix_sep = _metric(m_data, "f1_score", "mix_sep_cls")
        f1_clean_base = _metric(m_data, "f1_score", "clean_cls")
        f1_clean_sep = _metric(m_data, "f1_score", "clean_sep_cls")
        ba_in_base = _metric(m_data, "balanced_accuracy", "mix_cls")
        ba_in_sep = _metric(m_data, "balanced_accuracy", "mix_sep_cls")
        ba_ood_base = _metric(r_data, "balanced_accuracy", "as_is_cls")
        ba_ood_sep = _metric(r_data, "balanced_accuracy", "as_is_sep_cls")

        cls_labels.append(_cls_label(cls))
        delta_f1_clean.append(
            f1_clean_sep - f1_clean_base
            if not (np.isnan(f1_clean_base) or np.isnan(f1_clean_sep))
            else float("nan")
        )
        delta_f1_mix.append(
            f1_mix_sep - f1_mix_base
            if not (np.isnan(f1_mix_base) or np.isnan(f1_mix_sep))
            else float("nan")
        )
        delta_ba_in.append(
            ba_in_sep - ba_in_base
            if not (np.isnan(ba_in_base) or np.isnan(ba_in_sep))
            else float("nan")
        )
        delta_ba_ood.append(
            ba_ood_sep - ba_ood_base
            if not (np.isnan(ba_ood_base) or np.isnan(ba_ood_sep))
            else float("nan")
        )

    if not cls_labels:
        return _placeholder("No bird data")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"<b>{t['left_panel']}</b>",
            f"<b>{t['right_panel']}</b>",
        ],
        horizontal_spacing=0.18,
    )

    bar_kw = dict(
        cliponaxis=False,
        marker_line=dict(color="rgba(0,0,0,0.2)", width=0.5),
        textposition="outside",
        textfont=dict(size=PAPER_TICK_SIZE, color="#1f1f1f"),
    )

    COL_CLEAN = QUAL[0]
    COL_MIX = QUAL[1]
    COL_OOD = QUAL[2]

    # Left panel: ΔF₁ — clean COI vs. mixture
    fig.add_trace(
        go.Bar(
            x=cls_labels,
            y=delta_f1_clean,
            name=t["delta_f1_clean"],
            marker_color=COL_CLEAN,
            opacity=0.65,
            text=[_fmt(v, "+.3f") for v in delta_f1_clean],
            showlegend=True,
            legendgroup="f1_clean",
            **bar_kw,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=cls_labels,
            y=delta_f1_mix,
            name=t["delta_f1_mixture"],
            marker_color=COL_MIX,
            opacity=0.95,
            text=[_fmt(v, "+.3f") for v in delta_f1_mix],
            showlegend=True,
            legendgroup="f1_mix",
            **bar_kw,
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=0, line_dash="solid", line_color="#1f1f1f", line_width=1.2, row=1, col=1
    )

    # Right panel: ΔBA — in-domain vs. OOD
    fig.add_trace(
        go.Bar(
            x=cls_labels,
            y=delta_ba_in,
            name=t["delta_ba_in"],
            marker_color=COL_MIX,
            opacity=0.95,
            text=[_fmt(v, "+.3f") for v in delta_ba_in],
            showlegend=True,
            legend="legend2",
            legendgroup="ba_in",
            **bar_kw,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=cls_labels,
            y=delta_ba_ood,
            name=t["delta_ba_ood"],
            marker_color=COL_OOD,
            opacity=0.95,
            text=[_fmt(v, "+.3f") for v in delta_ba_ood],
            showlegend=True,
            legend="legend2",
            legendgroup="ba_ood",
            **bar_kw,
        ),
        row=1,
        col=2,
    )
    fig.add_hline(
        y=0, line_dash="solid", line_color="#1f1f1f", line_width=1.2, row=1, col=2
    )

    # Y-axis padding so outside text labels fit
    all_left = [v for v in (delta_f1_clean + delta_f1_mix) if not np.isnan(v)]
    y_min_l = min(all_left + [0])
    y_max_l = max(all_left + [0])
    padding_l = max((y_max_l - y_min_l) * 0.15, 0.005)
    fig.update_yaxes(range=[y_min_l - padding_l, y_max_l + padding_l], row=1, col=1)
    all_right = [v for v in (delta_ba_in + delta_ba_ood) if not np.isnan(v)]
    y_min_r = min(all_right + [0])
    y_max_r = max(all_right + [0])
    padding_r = max((y_max_r - y_min_r) * 0.15, 0.005)
    fig.update_yaxes(range=[y_min_r - padding_r, y_max_r + padding_r], row=1, col=2)

    # Axes
    for c in (1, 2):
        fig.update_yaxes(
            gridcolor="rgba(0,0,0,0.07)",
            showline=True,
            linecolor="#1f1f1f",
            linewidth=1,
            zeroline=False,
            tickfont=dict(size=PAPER_TICK_SIZE),
            row=1,
            col=c,
        )
        fig.update_xaxes(
            showline=True,
            linecolor="#1f1f1f",
            linewidth=1,
            tickfont=dict(size=PAPER_TICK_SIZE),
            row=1,
            col=c,
        )
    fig.update_yaxes(title_text="ΔF₁", row=1, col=1)
    fig.update_yaxes(title_text="ΔBA", row=1, col=2)

    height = int(COL_WIDTH_PX * ASPECT_DUAL)
    fig = _paper_layout(fig, lang, height, COL_WIDTH_PX)
    fig.update_layout(
        barmode="group",
        bargap=0.22,
        bargroupgap=0.10,
        title=dict(
            text=f"<b>{t['title_bird_cls']}</b>", font=dict(size=PAPER_TITLE_SIZE)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.30,
            xanchor="center",
            x=0.18 if lang == "nl" else 0.25,
            font=dict(size=PAPER_LEGEND_SIZE),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
        ),
        legend2=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.30,
            xanchor="center",
            x=0.88 if lang == "nl" else 0.85,
            font=dict(size=PAPER_LEGEND_SIZE),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
        ),
    )
    for ann in fig.layout.annotations:
        ann.font = dict(size=PAPER_AXIS_SIZE - 1, color="#1f1f1f")
    return fig


# ===================================================================
# FIGURE 5 — Activity Gating (simplified)
# ===================================================================


def fig_activity_gating(lang: str = "en") -> go.Figure:
    t = LANG[lang]
    json_path = FINAL_RESULTS_DIR / "activity_gating_results" / "sweep_results.json"
    if not json_path.exists():
        return _placeholder("No activity-gating data")

    with open(json_path) as f:
        data = json.load(f)
    sweep = sorted(data.get("sweep", []), key=lambda r: r["threshold"])
    baseline = data.get("baseline_no_recycling", {})

    th = [r["threshold"] for r in sweep]
    has_si = "mean_si_snri_db" in sweep[0] if sweep else False
    y_key = "mean_si_snri_db" if has_si else "mean_si_snr_db"
    snr_vals = [r.get(y_key, float("nan")) for r in sweep]
    hit = [r.get("cache_hit_rate", float("nan")) * 100 for r in sweep]
    b_snr = baseline.get(y_key, float("nan"))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        subplot_titles=[
            f"<b>{t['si_snri_label']} vs. {t['threshold'].lower()}</b>",
            f"<b>{t['hit_rate']} vs. {t['threshold'].lower()}</b>",
        ],
    )

    # Top: SI-SNRi
    fig.add_trace(
        go.Scatter(
            x=th,
            y=snr_vals,
            mode="lines+markers",
            name=t["si_snri_label"],
            line=dict(color=QUAL[1], width=2.5),
            marker=dict(size=7),
        ),
        row=1,
        col=1,
    )
    if not np.isnan(b_snr):
        fig.add_hline(
            y=b_snr,
            line_dash="dot",
            line_color=QUAL[1],
            annotation_text=f"{t['baseline']} = {b_snr:.1f} dB",
            annotation_font=dict(size=PAPER_ANNOT_SIZE, color="#555"),
            row=1,
            col=1,
        )

    fig.add_hline(
        y=0,
        line_dash="solid",
        line_color="#888",
        line_width=0.8,
        annotation_text=t["positive_zone"],
        annotation_font=dict(size=PAPER_ANNOT_SIZE, color="#888"),
        row=1,
        col=1,
    )

    zone_th = [t_val for t_val, s in zip(th, snr_vals) if not np.isnan(s) and s >= 0]
    if zone_th:
        fig.add_vrect(
            x0=min(zone_th),
            x1=max(zone_th),
            fillcolor=QUAL[2],
            opacity=0.06,
            layer="below",
            line_width=0,
            row=1,
            col=1,
        )

    # Bottom: hit rate
    fig.add_trace(
        go.Scatter(
            x=th,
            y=hit,
            mode="lines+markers",
            name=t["hit_rate"],
            line=dict(color=QUAL[3], width=2.5),
            marker=dict(size=7, symbol="diamond"),
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(
        title_text=t["threshold"],
        gridcolor="rgba(0,0,0,0.07)",
        showline=True,
        linecolor="#1f1f1f",
        linewidth=1,
        tickfont=dict(size=PAPER_TICK_SIZE),
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text=t["si_snri_label"],
        gridcolor="rgba(0,0,0,0.07)",
        showline=True,
        linecolor="#1f1f1f",
        linewidth=1,
        tickfont=dict(size=PAPER_TICK_SIZE),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text=t["hit_rate"],
        range=[0, 105],
        gridcolor="rgba(0,0,0,0.07)",
        showline=True,
        linecolor="#1f1f1f",
        linewidth=1,
        tickfont=dict(size=PAPER_TICK_SIZE),
        row=2,
        col=1,
    )

    height = int(COL_WIDTH_PX * ASPECT_SINGLE * 1.3)
    fig = _paper_layout(fig, lang, height, COL_WIDTH_PX)
    fig.update_layout(
        title=dict(
            text=f"<b>{t['title_activity']}</b>", font=dict(size=PAPER_TITLE_SIZE)
        ),
        showlegend=False,
    )
    for ann in fig.layout.annotations:
        ann.font = dict(size=PAPER_AXIS_SIZE - 1, color="#1f1f1f")
    return fig


# ---------------------------------------------------------------------------
# Placeholder
# ---------------------------------------------------------------------------


def _placeholder(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=f"<b>{msg}</b>",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(family=FONT_FAMILY, size=14, color="#7f7f7f"),
    )
    fig.update_layout(height=300, width=COL_WIDTH_PX)
    return fig


# ===================================================================
# Main
# ===================================================================

FIG_GENERATORS = {
    "separation_metrics": fig_separation_metrics,
    "classification_impact": fig_classification_impact,
    "noise_robustness": fig_noise_robustness,
    "classifier_improvements": fig_classifier_improvements,
    "activity_gating": fig_activity_gating,
}

if __name__ == "__main__":
    import sys

    langs = ["en", "nl"]
    figs_to_gen = sys.argv[1:] if len(sys.argv) > 1 else list(FIG_GENERATORS.keys())

    for lang in langs:
        print(f"\n--- {lang.upper()} ---")
        for name in figs_to_gen:
            if name not in FIG_GENERATORS:
                print(f"  Unknown figure: {name}")
                continue
            print(f"  Generating {name}...")
            fig = FIG_GENERATORS[name](lang)
            _save(fig, name, lang)

    print(f"\nDone. Figures saved to {OUTPUT_DIR}")
