"""
plot_results_chapter.py — Dissertation Results & Discussion figures.

Reads validation JSON outputs from the runner scripts and produces 6 publication-
quality Plotly figures saved as PNG (and interactive HTML) in ./chapter_figures/.

Figure layout:
  Fig 1  — Classification performance table (all models, both conditions)
  Fig 2  — Classification bar chart: F1 / Recall / Precision by model
  Fig 3  — Risoux generalisation: in-distribution vs. out-of-distribution F1
  Fig 4  — Multi-class vs. single-class TUSS comparison (airplane + bird heads)
  Fig 5  — Noise robustness: RMS / SEL degradation vs. white-noise SNR level
  Fig 6  — Activity gating sweep: F1 + SI-SNRi vs. threshold + cache hit rate

Usage:
    python plot_results_chapter.py

All result directories are expected relative to this script's location.
Adjust the path constants below if your output directories differ.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Path configuration ──────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent

# Validation result directories (created by runner scripts)
SUDORMRF_DIR       = SCRIPT_DIR / "sudormrf_airplane_results"
TUSS_SC_DIR        = SCRIPT_DIR / "tuss_singleclass_airplane_results"
CLAPSEP_DIR        = SCRIPT_DIR / "clapsep_airplane_results"
TUSS_MC_AIR_DIR    = SCRIPT_DIR / "tuss_multiclass_airplane_results"
TUSS_SC_BIRD_DIR   = SCRIPT_DIR / "tuss_singleclass_bird_results"   # existing run_bird_validation
TUSS_MC_BIRD_DIR   = SCRIPT_DIR / "tuss_multiclass_bird_results"

# Risoux result subdirectories (appended with _risoux by runners)
SUDORMRF_RISOUX_DIR    = SCRIPT_DIR / "sudormrf_airplane_results_risoux"
TUSS_SC_RISOUX_DIR     = SCRIPT_DIR / "tuss_singleclass_airplane_results_risoux"
CLAPSEP_RISOUX_DIR     = SCRIPT_DIR / "clapsep_airplane_results_risoux"
TUSS_MC_AIR_RISOUX_DIR = SCRIPT_DIR / "tuss_multiclass_airplane_results_risoux"

# Noise robustness JSON (from test_noise_increase.py)
NOISE_RESULTS_DIR  = SCRIPT_DIR / "noise_increase_results"

# Activity gating sweep JSON (from run_activity_gating_sweep.py)
GATING_SWEEP_JSON  = SCRIPT_DIR / "activity_gating_results" / "sweep_results.json"

OUTPUT_DIR = SCRIPT_DIR / "chapter_figures"

# ── Plotly style constants ───────────────────────────────────────────────────
TEMPLATE = "plotly_white"
FONT_FAMILY = "Arial"
FONT_SIZE = 13
COLORS = [
    "#636EFA",  # blue
    "#EF553B",  # red / orange
    "#00CC96",  # green
    "#AB63FA",  # purple
    "#FFA15A",  # orange
    "#19D3F3",  # cyan
    "#FF6692",  # pink
    "#B6E880",  # light green
]
LAYOUT_BASE = dict(
    template=TEMPLATE,
    font=dict(family=FONT_FAMILY, size=FONT_SIZE),
    margin=dict(l=60, r=30, t=60, b=60),
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_latest_json(directory: Path, glob: str = "results_test_*.json") -> Optional[dict]:
    """Load the most recently modified JSON matching glob in directory."""
    candidates = sorted(directory.glob(glob), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    with open(candidates[0]) as f:
        return json.load(f)


def _load_classifier_json(base_dir: Path, cls_name: str, risoux: bool = False) -> Optional[dict]:
    """Load result JSON for a specific classifier under base_dir/cls_name/."""
    sub = base_dir / cls_name
    if not sub.exists():
        return None
    glob = "results_test_risoux_test_*.json" if risoux else "results_test_*.json"
    return _load_latest_json(sub, glob)


def _get_metric(d: Optional[dict], key: str, condition: str = "mix_sep_cls") -> float:
    """Extract a scalar metric from a result dict. Returns NaN if unavailable."""
    if d is None:
        return float("nan")
    cond = d.get(condition, {})
    return float(cond.get(key, float("nan")))


def _get_signal_metric(d: Optional[dict], key: str, condition: str = "mix_sep_cls") -> float:
    """Extract a signal_metrics sub-key from a result dict."""
    if d is None:
        return float("nan")
    sm = d.get(condition, {}).get("signal_metrics", {})
    return float(sm.get(key, float("nan")))


# ── Figure 1 — Classification performance table ───────────────────────────

def fig1_metrics_table() -> go.Figure:
    """Table: all models × classifiers, mix_sep_cls condition."""
    models = [
        ("SuDoRM-RF", "plane",          SUDORMRF_DIR,    False),
        ("SuDoRM-RF", "ast_finetuned",  SUDORMRF_DIR,    False),
        ("TUSS s/c",  "plane",          TUSS_SC_DIR,     False),
        ("TUSS s/c",  "ast_finetuned",  TUSS_SC_DIR,     False),
        ("CLAPSep",   "plane",          CLAPSEP_DIR,     False),
        ("CLAPSep",   "ast_finetuned",  CLAPSEP_DIR,     False),
        ("TUSS m/c",  "plane",          TUSS_MC_AIR_DIR, False),
        ("TUSS m/c",  "ast_finetuned",  TUSS_MC_AIR_DIR, False),
    ]

    header_vals = ["Model", "Classifier", "F1", "Recall", "Precision", "Bal. Acc.", "SI-SNR (dB)", "SI-SNRi (dB)", "RMS err (dB)"]
    rows = {h: [] for h in header_vals}

    for model_name, cls_name, base_dir, _ in models:
        d = _load_classifier_json(base_dir, cls_name)
        rows["Model"].append(model_name)
        rows["Classifier"].append(cls_name.replace("_", " "))
        rows["F1"].append(f"{_get_metric(d, 'f1_score'):.3f}")
        rows["Recall"].append(f"{_get_metric(d, 'recall'):.3f}")
        rows["Precision"].append(f"{_get_metric(d, 'precision'):.3f}")
        rows["Bal. Acc."].append(f"{_get_metric(d, 'balanced_accuracy'):.3f}")
        si_snr = _get_signal_metric(d, "mean_si_snr_db")
        rows["SI-SNR (dB)"].append(f"{si_snr:+.2f}" if not np.isnan(si_snr) else "—")
        si_snri = _get_signal_metric(d, "mean_si_snri_db")
        rows["SI-SNRi (dB)"].append(f"{si_snri:+.2f}" if not np.isnan(si_snri) else "—")
        rms = _get_signal_metric(d, "mean_rms_error_db")
        rows["RMS err (dB)"].append(f"{rms:+.2f}" if not np.isnan(rms) else "—")

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in header_vals],
            fill_color="#4C78A8",
            font=dict(color="white", family=FONT_FAMILY, size=FONT_SIZE),
            align="center",
        ),
        cells=dict(
            values=[rows[h] for h in header_vals],
            fill_color=[["#F0F4FF" if i % 2 == 0 else "white"] * len(rows["Model"]) for _ in header_vals],
            font=dict(family=FONT_FAMILY, size=FONT_SIZE - 1),
            align=["left", "left"] + ["center"] * (len(header_vals) - 2),
        ),
    )])
    fig.update_layout(
        title="<b>Classification Performance — Mixture Condition (mix_sep_cls)</b>",
        **LAYOUT_BASE,
    )
    return fig


# ── Figure 2 — Classification bar chart ──────────────────────────────────

def fig2_classification_bars() -> go.Figure:
    """Grouped bar chart: F1 by model, split by classifier."""
    model_labels = ["SuDoRM-RF", "TUSS s/c", "CLAPSep", "TUSS m/c"]
    dirs = [SUDORMRF_DIR, TUSS_SC_DIR, CLAPSEP_DIR, TUSS_MC_AIR_DIR]
    classifiers = ["plane", "ast_finetuned"]
    cls_display = {"plane": "PlaneClassifier", "ast_finetuned": "AST fine-tuned"}
    metrics = ["f1_score", "recall", "precision"]
    metric_display = {"f1_score": "F1", "recall": "Recall", "precision": "Precision"}
    conditions = ["clean_sep_cls", "mix_sep_cls"]
    cond_display = {"clean_sep_cls": "Clean", "mix_sep_cls": "Mixture"}

    fig = make_subplots(
        rows=1, cols=len(classifiers),
        subplot_titles=[f"Classifier: {cls_display[c]}" for c in classifiers],
        shared_yaxes=True,
    )

    for col_idx, cls_name in enumerate(classifiers, 1):
        for m_idx, metric in enumerate(metrics):
            for cond_idx, cond in enumerate(conditions):
                y_vals = []
                for base_dir in dirs:
                    d = _load_classifier_json(base_dir, cls_name)
                    y_vals.append(_get_metric(d, metric, condition=cond))

                dash = "solid" if cond == "mix_sep_cls" else "dot"
                fig.add_trace(
                    go.Bar(
                        name=f"{metric_display[metric]} ({cond_display[cond]})",
                        x=model_labels,
                        y=y_vals,
                        marker_color=COLORS[m_idx * 2 + cond_idx],
                        opacity=0.85,
                        showlegend=(col_idx == 1),
                    ),
                    row=1, col=col_idx,
                )

    fig.update_layout(
        barmode="group",
        title="<b>Classification Performance by Model and Classifier</b>",
        yaxis_title="Score",
        yaxis_range=[0, 1],
        legend=dict(orientation="h", y=-0.2),
        **LAYOUT_BASE,
    )
    return fig


# ── Figure 3 — Risoux generalisation ─────────────────────────────────────

def fig3_risoux_generalisation() -> go.Figure:
    """Bar chart comparing in-distribution (mix_sep_cls) vs Risoux F1."""
    model_labels = ["SuDoRM-RF", "TUSS s/c", "CLAPSep", "TUSS m/c"]
    dirs     = [SUDORMRF_DIR,       TUSS_SC_DIR,       CLAPSEP_DIR,       TUSS_MC_AIR_DIR]
    ris_dirs = [SUDORMRF_RISOUX_DIR, TUSS_SC_RISOUX_DIR, CLAPSEP_RISOUX_DIR, TUSS_MC_AIR_RISOUX_DIR]
    # Use the plane classifier (primary)
    cls_name = "plane"

    in_dist_f1  = [_get_metric(_load_classifier_json(d, cls_name), "f1_score", "mix_sep_cls") for d in dirs]
    risoux_f1   = [_get_metric(_load_classifier_json(d, cls_name, risoux=True), "f1_score", "mix_sep_cls") for d in ris_dirs]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="In-distribution (test set)", x=model_labels, y=in_dist_f1,
                         marker_color=COLORS[0], opacity=0.85))
    fig.add_trace(go.Bar(name="Risoux (out-of-distribution)", x=model_labels, y=risoux_f1,
                         marker_color=COLORS[1], opacity=0.85))

    fig.update_layout(
        barmode="group",
        title="<b>Generalisation: In-Distribution vs. Risoux Test Set (F1, PlaneClassifier)</b>",
        yaxis_title="F1 Score",
        yaxis_range=[0, 1],
        legend=dict(orientation="h", y=-0.15),
        **LAYOUT_BASE,
    )
    return fig


# ── Figure 4 — Multi-class vs. single-class TUSS ──────────────────────────

def fig4_multiclass_comparison() -> go.Figure:
    """Side-by-side comparison of TUSS single-class vs. multi-class heads."""
    # Airplane: single-class=TUSS_SC_DIR, multi-class=TUSS_MC_AIR_DIR
    # Bird: single-class=TUSS_SC_BIRD_DIR, multi-class=TUSS_MC_BIRD_DIR
    configs = [
        # (label, single_dir, multi_dir, cls_name)
        ("Airplane (PlaneClassifier)", TUSS_SC_DIR, TUSS_MC_AIR_DIR, "plane"),
        ("Airplane (AST)",             TUSS_SC_DIR, TUSS_MC_AIR_DIR, "ast_finetuned"),
        ("Bird (BirdMAE)",             TUSS_SC_BIRD_DIR, TUSS_MC_BIRD_DIR, "bird_mae"),
        ("Bird (AudioProtoPNet)",      TUSS_SC_BIRD_DIR, TUSS_MC_BIRD_DIR, "audioprotopnet"),
    ]
    metrics = ["f1_score", "recall", "precision"]
    metric_display = {"f1_score": "F1", "recall": "Recall", "precision": "Precision"}

    labels = [c[0] for c in configs]
    fig = make_subplots(rows=1, cols=len(metrics),
                        subplot_titles=[metric_display[m] for m in metrics],
                        shared_yaxes=True)

    for col_idx, metric in enumerate(metrics, 1):
        sc_vals = [_get_metric(_load_classifier_json(sc, cls), metric) for _, sc, _, cls in configs]
        mc_vals = [_get_metric(_load_classifier_json(mc, cls), metric) for _, _, mc, cls in configs]
        fig.add_trace(go.Bar(name="Single-class TUSS", x=labels, y=sc_vals,
                             marker_color=COLORS[0], showlegend=(col_idx == 1)), row=1, col=col_idx)
        fig.add_trace(go.Bar(name="Multi-class TUSS", x=labels, y=mc_vals,
                             marker_color=COLORS[1], showlegend=(col_idx == 1)), row=1, col=col_idx)

    fig.update_layout(
        barmode="group",
        title="<b>Single-Class vs. Multi-Class TUSS — Airplane & Bird Heads</b>",
        yaxis_range=[0, 1],
        legend=dict(orientation="h", y=-0.2),
        **LAYOUT_BASE,
    )
    return fig


# ── Figure 5 — Noise robustness ───────────────────────────────────────────

def fig5_noise_robustness() -> go.Figure:
    """Line chart: RMS / SEL degradation vs. white-noise SNR level."""
    # Find the most recently modified noise_increase JSON
    jsons = sorted(NOISE_RESULTS_DIR.glob("noise_increase_energy_*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True) if NOISE_RESULTS_DIR.exists() else []

    if not jsons:
        fig = go.Figure()
        fig.add_annotation(text="No noise robustness results found.<br>Run test_noise_increase.py first.",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="<b>Noise Robustness (data not yet available)</b>", **LAYOUT_BASE)
        return fig

    with open(jsons[0]) as f:
        data = json.load(f)

    snr_results = data.get("snr_results", [])
    snr_levels  = [r["snr_db"] for r in snr_results]
    rms_degrad  = [r["mean_rms_degradation_db"] for r in snr_results]
    sel_degrad  = [r["mean_sel_degradation_db"] for r in snr_results]

    # Clean baseline reference lines
    cb = data.get("clean_baseline", {})
    cb_rms = cb.get("mean_rms_preservation_db", None)
    cb_sel = cb.get("mean_sel_preservation_db", None)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=snr_levels, y=rms_degrad, mode="lines+markers",
                             name="RMS degradation", marker_color=COLORS[0], line_width=2))
    fig.add_trace(go.Scatter(x=snr_levels, y=sel_degrad, mode="lines+markers",
                             name="SEL degradation", marker_color=COLORS[1], line_width=2))

    if cb_rms is not None:
        fig.add_hline(y=cb_rms, line_dash="dot", line_color=COLORS[0],
                      annotation_text=f"Clean RMS: {cb_rms:+.2f} dB", annotation_position="top left")
    if cb_sel is not None:
        fig.add_hline(y=cb_sel, line_dash="dot", line_color=COLORS[1],
                      annotation_text=f"Clean SEL: {cb_sel:+.2f} dB", annotation_position="bottom left")

    fig.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1)

    fig.update_layout(
        title=f"<b>Noise Robustness — Energy Degradation vs. White-Noise SNR (TUSS)</b>",
        xaxis_title="Added noise SNR (dB)",
        yaxis_title="Degradation relative to clean separation (dB)",
        legend=dict(orientation="h", y=-0.15),
        **LAYOUT_BASE,
    )
    return fig


# ── Figure 6 — Activity gating sweep ─────────────────────────────────────

def fig6_activity_gating() -> go.Figure:
    """Dual-axis: F1 + SI-SNRi vs. threshold, with hit rate on secondary axis."""
    if not GATING_SWEEP_JSON.exists():
        fig = go.Figure()
        fig.add_annotation(text="No activity gating sweep results found.<br>Run run_activity_gating_sweep.py first.",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="<b>Activity Gating Sweep (data not yet available)</b>", **LAYOUT_BASE)
        return fig

    with open(GATING_SWEEP_JSON) as f:
        data = json.load(f)

    sweep = data.get("sweep", [])
    baseline = data.get("baseline_no_recycling", {})

    thresholds = [r["threshold"] for r in sweep]
    f1_vals    = [r.get("f1_score", float("nan")) for r in sweep]
    snri_vals  = [r.get("mean_si_snri_db", float("nan")) for r in sweep]
    hit_rates  = [r.get("cache_hit_rate", float("nan")) for r in sweep]

    baseline_f1   = baseline.get("f1_score", float("nan"))
    baseline_snri = baseline.get("mean_si_snri_db", float("nan"))

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # F1 score
    fig.add_trace(go.Scatter(x=thresholds, y=f1_vals, mode="lines+markers",
                             name="F1 score", line=dict(color=COLORS[0], width=2)),
                  secondary_y=False)
    if not np.isnan(baseline_f1):
        fig.add_hline(y=baseline_f1, line_dash="dot", line_color=COLORS[0],
                      annotation_text="Baseline F1 (no recycling)", annotation_position="right",
                      secondary_y=False)

    # SI-SNRi
    if any(not np.isnan(v) for v in snri_vals):
        fig.add_trace(go.Scatter(x=thresholds, y=snri_vals, mode="lines+markers",
                                 name="SI-SNRi (dB)", line=dict(color=COLORS[2], width=2, dash="dash")),
                      secondary_y=False)
        if not np.isnan(baseline_snri):
            fig.add_hline(y=baseline_snri, line_dash="dot", line_color=COLORS[2],
                          annotation_text="Baseline SI-SNRi", annotation_position="right",
                          secondary_y=False)

    # Cache hit rate (secondary axis)
    fig.add_trace(go.Scatter(x=thresholds, y=[h * 100 for h in hit_rates], mode="lines+markers",
                             name="Cache hit rate (%)", line=dict(color=COLORS[3], width=2, dash="dot")),
                  secondary_y=True)

    fig.update_layout(
        title="<b>Activity Gating: F1 & SI-SNRi vs. Similarity Threshold</b>",
        xaxis_title="Similarity threshold",
        legend=dict(orientation="h", y=-0.2),
        **LAYOUT_BASE,
    )
    fig.update_yaxes(title_text="F1 / SI-SNRi (dB)", secondary_y=False)
    fig.update_yaxes(title_text="Cache hit rate (%)", secondary_y=True, range=[0, 100])
    return fig


# ── Main ──────────────────────────────────────────────────────────────────

def save_figure(fig: go.Figure, name: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = OUTPUT_DIR / f"{name}.html"
    fig.write_html(str(html_path))
    print(f"  Saved: {html_path}")
    try:
        png_path = OUTPUT_DIR / f"{name}.png"
        fig.write_image(str(png_path), width=1200, height=700, scale=2)
        print(f"  Saved: {png_path}")
    except Exception as e:
        print(f"  [PNG skipped — install kaleido for static export: {e}]")


def main():
    print("=" * 70)
    print("PLOTTING DISSERTATION RESULTS CHAPTER FIGURES")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}\n")

    figures = [
        ("fig1_metrics_table",        fig1_metrics_table),
        ("fig2_classification_bars",  fig2_classification_bars),
        ("fig3_risoux_generalisation",fig3_risoux_generalisation),
        ("fig4_multiclass_comparison",fig4_multiclass_comparison),
        ("fig5_noise_robustness",     fig5_noise_robustness),
        ("fig6_activity_gating",      fig6_activity_gating),
    ]

    for name, fn in figures:
        print(f"Generating {name} ...")
        try:
            fig = fn()
            save_figure(fig, name)
        except Exception as e:
            import traceback
            print(f"  ERROR generating {name}: {e}")
            traceback.print_exc()

    print(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
