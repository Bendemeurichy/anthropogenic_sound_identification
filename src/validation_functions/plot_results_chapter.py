"""
plot_results_chapter.py — Dissertation Results & Discussion artefacts.

Produces the figures and LaTeX tables used in the Results chapter.  Output is
deliberately minimal and chapter-aligned; rasterised metrics tables have been
replaced by ``\\input``-able ``.tex`` files.

Outputs (under ``final_results/chapter_figures/``):

  Figures (PNG via Plotly + kaleido):
    fig_classification_delta.png    — Per-classifier F1 across the three
                                      operating conditions (mix-only,
                                      clean+sep, mix+sep) for every separator.
                                      Conveys both the gain over a no-separation
                                      baseline and the artefact cost on clean
                                      audio.
    fig_risoux_generalisation.png   — In-distribution vs. Risoux F1 (single
                                      panel, single classifier per COI).
    fig_multiclass_tuss.png         — TUSS single- vs multi-class F1 for
                                      airplane and bird heads.
    fig_noise_robustness.png        — Energy-degradation curve vs. SNR
                                      (delegates to plot_noise_increase_results).
    fig_activity_gating.png         — F1 / SI-SNRi / cache hit-rate sweep.

  LaTeX tables (``.tex``):
    tab_separation_metrics.tex      — Signal-level metrics (SI-SNR, SI-SNRi,
                                      SDR, RMS-err, SEL-err) per model × COI.
    tab_classification_per_cond.tex — Classification metrics per condition for
                                      every model × classifier.
    tab_risoux_generalisation.tex   — In-distribution vs. Risoux F1.
    tab_multiclass_tuss.tex         — Single- vs. multi-class TUSS metrics.

Run::

    python plot_results_chapter.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
FINAL_RESULTS_DIR = SCRIPT_DIR / "final_results"

SUDORMRF_DIR        = FINAL_RESULTS_DIR / "sudormrf_airplane_results"
TUSS_SC_DIR         = FINAL_RESULTS_DIR / "tuss_singleclass_airplane_results"
CLAPSEP_DIR         = FINAL_RESULTS_DIR / "clapsep_airplane_results"
TUSS_MC_AIR_DIR     = FINAL_RESULTS_DIR / "tuss_multiclass_airplane_results"
TUSS_SC_BIRD_DIR    = FINAL_RESULTS_DIR / "tuss_singleclass_bird_results"
TUSS_MC_BIRD_DIR    = FINAL_RESULTS_DIR / "tuss_multiclass_bird_results"

SUDORMRF_RISOUX_DIR    = FINAL_RESULTS_DIR / "sudormrf_airplane_results_risoux"
TUSS_SC_RISOUX_DIR     = FINAL_RESULTS_DIR / "tuss_singleclass_airplane_results_risoux"
CLAPSEP_RISOUX_DIR     = FINAL_RESULTS_DIR / "clapsep_airplane_results_risoux"
TUSS_MC_AIR_RISOUX_DIR = FINAL_RESULTS_DIR / "tuss_multiclass_airplane_results_risoux"

NOISE_RESULTS_DIR = FINAL_RESULTS_DIR / "noise_increase_results"
GATING_SWEEP_JSON = FINAL_RESULTS_DIR / "activity_gating_results" / "sweep_results.json"

OUTPUT_DIR = FINAL_RESULTS_DIR / "chapter_figures"
TABLE_DIR = OUTPUT_DIR / "tables"

# ── Plotly style ────────────────────────────────────────────────────────────
TEMPLATE = "plotly_white"
FONT = dict(family="Times New Roman, serif", size=13, color="#1a1a1a")
LAYOUT_BASE = dict(template=TEMPLATE, font=FONT,
                   margin=dict(l=70, r=30, t=60, b=60))

# Muted, print-friendly palette (Tableau 10 reordered)
COL_MIX_ONLY    = "#9e9e9e"   # baseline (no separation)
COL_CLEAN_SEP   = "#4c78a8"   # separation cost on clean audio
COL_MIX_SEP     = "#e45756"   # primary: mixture + separation
COL_RISOUX      = "#54a24b"
COL_SECONDARY   = "#f58518"
COL_TERTIARY    = "#72b7b2"

PNG_SCALE = 2

# ── Classifier name mapping ─────────────────────────────────────────────────
CLS_DIR_MAP: Dict[str, str] = {"plane": "pann_finetuned"}
CLS_DISPLAY: Dict[str, str] = {
    "pann_finetuned":  "PANN (fine-tuned)",
    "ast_finetuned":   "AST (fine-tuned)",
    "bird_mae":        "BirdMAE",
    "audioprotopnet":  "AudioProtoPNet",
}


def _cls_dir(name: str) -> str:
    return CLS_DIR_MAP.get(name, name)


def _cls_label(name: str) -> str:
    resolved = _cls_dir(name)
    return CLS_DISPLAY.get(resolved, resolved.replace("_", " "))


# ── JSON loading ────────────────────────────────────────────────────────────

def _load_latest(directory: Path, glob: str) -> Optional[dict]:
    if not directory.exists():
        return None
    candidates = sorted(directory.glob(glob), key=lambda p: p.stat().st_mtime,
                        reverse=True)
    if not candidates:
        return None
    with open(candidates[0]) as f:
        return json.load(f)


def _load_classifier_json(base: Path, cls: str, risoux: bool = False) -> Optional[dict]:
    sub = base / _cls_dir(cls)
    glob = "results_test_risoux_*.json" if risoux else "results_test_*.json"
    return _load_latest(sub, glob)


def _metric(d: Optional[dict], key: str, condition: str) -> float:
    if d is None:
        return float("nan")
    return float(d.get(condition, {}).get(key, float("nan")))


def _signal(d: Optional[dict], key: str, condition: str = "mix_sep_cls") -> float:
    if d is None:
        return float("nan")
    sm = d.get(condition, {}).get("signal_metrics", {})
    val = sm.get(key, float("nan"))
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(v: float, spec: str = "+.2f", dash: str = "—") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return dash
    return f"{v:{spec}}"


# ── LaTeX table writer ──────────────────────────────────────────────────────

def _write_latex_table(
    path: Path,
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
    caption: str,
    label: str,
    col_align: Optional[str] = None,
) -> None:
    """Write a booktabs-style LaTeX tabular environment."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if col_align is None:
        col_align = "l" + "r" * (len(columns) - 1)

    lines = [
        "% Auto-generated by plot_results_chapter.py — do not edit by hand.",
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{col_align}}}",
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
    path.write_text("\n".join(lines) + "\n")
    print(f"  Wrote {path.relative_to(FINAL_RESULTS_DIR)}")


# ── Model registry (single source of truth) ─────────────────────────────────
SEPARATORS_AIRPLANE: List[Tuple[str, Path, Path]] = [
    # (display, results_dir, risoux_dir)
    ("SuDoRM-RF",         SUDORMRF_DIR,    SUDORMRF_RISOUX_DIR),
    ("TUSS (single)",     TUSS_SC_DIR,     TUSS_SC_RISOUX_DIR),
    ("CLAPSep",           CLAPSEP_DIR,     CLAPSEP_RISOUX_DIR),
    ("TUSS (multi)",      TUSS_MC_AIR_DIR, TUSS_MC_AIR_RISOUX_DIR),
]

CLASSIFIERS_AIRPLANE = ["plane", "ast_finetuned"]
CLASSIFIERS_BIRD = ["bird_mae", "audioprotopnet"]


# ── Figure 1: Classification across conditions ──────────────────────────────

def fig_classification_delta() -> go.Figure:
    """One subplot per classifier, four separator models on the x-axis,
    three F1 bars per model: ``mix_cls`` (no separation), ``clean_sep_cls``
    (separation on clean COI — artefact cost) and ``mix_sep_cls`` (separation
    on mixture — operational condition).

    The mix-only baseline is the same for every separator (same classifier,
    same input); plotting it alongside makes the gain/cost trade-off obvious
    at a glance.
    """
    classifiers = CLASSIFIERS_AIRPLANE
    fig = make_subplots(
        rows=1, cols=len(classifiers),
        subplot_titles=[_cls_label(c) for c in classifiers],
        shared_yaxes=True, horizontal_spacing=0.06,
    )

    sep_labels = [m[0] for m in SEPARATORS_AIRPLANE]

    conditions = [
        ("mix_cls",       "No separation (baseline)", COL_MIX_ONLY),
        ("clean_sep_cls", "Separation on clean COI",  COL_CLEAN_SEP),
        ("mix_sep_cls",   "Separation on mixture",    COL_MIX_SEP),
    ]

    for col_idx, cls in enumerate(classifiers, 1):
        first = col_idx == 1
        # Pre-load one JSON per separator
        loaded = [_load_classifier_json(sep[1], cls) for sep in SEPARATORS_AIRPLANE]

        for cond_key, cond_label, color in conditions:
            ys = [_metric(d, "f1_score", cond_key) for d in loaded]
            fig.add_trace(
                go.Bar(
                    x=sep_labels, y=ys,
                    name=cond_label,
                    marker_color=color,
                    text=[f"{y:.2f}" if not np.isnan(y) else "" for y in ys],
                    textposition="outside",
                    cliponaxis=False,
                    showlegend=first,
                    legendgroup=cond_key,
                ),
                row=1, col=col_idx,
            )

    fig.update_yaxes(title_text="F1 score", range=[0, 1.05], row=1, col=1)
    for c in range(2, len(classifiers) + 1):
        fig.update_yaxes(range=[0, 1.05], row=1, col=c)

    fig.update_layout(
        barmode="group",
        title="<b>Downstream classification F1 — separator × classifier × condition</b>",
        legend=dict(orientation="h", yanchor="bottom", y=-0.20,
                    xanchor="center", x=0.5),
        height=480, width=1100,
        **LAYOUT_BASE,
    )
    return fig


# ── Figure 2: Risoux generalisation ─────────────────────────────────────────

def fig_risoux_generalisation() -> go.Figure:
    cls = "plane"
    sep_labels = [m[0] for m in SEPARATORS_AIRPLANE]

    in_f1  = [_metric(_load_classifier_json(d, cls), "f1_score", "mix_sep_cls")
              for _, d, _ in SEPARATORS_AIRPLANE]
    out_f1 = [_metric(_load_classifier_json(rd, cls, risoux=True),
                      "f1_score", "mix_sep_cls")
              for _, _, rd in SEPARATORS_AIRPLANE]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=sep_labels, y=in_f1,
                         name="In-distribution test set",
                         marker_color=COL_MIX_SEP,
                         text=[f"{v:.2f}" for v in in_f1],
                         textposition="outside", cliponaxis=False))
    fig.add_trace(go.Bar(x=sep_labels, y=out_f1,
                         name="Risoux (out-of-distribution)",
                         marker_color=COL_RISOUX,
                         text=[f"{v:.2f}" for v in out_f1],
                         textposition="outside", cliponaxis=False))
    fig.update_layout(
        barmode="group",
        title=f"<b>Generalisation gap: in-distribution vs. Risoux ({_cls_label(cls)})</b>",
        yaxis=dict(title="F1 score (mix + separation)", range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.20,
                    xanchor="center", x=0.5),
        height=480, width=900,
        **LAYOUT_BASE,
    )
    return fig


# ── Figure 3: TUSS single vs multi-class ────────────────────────────────────

def fig_multiclass_tuss() -> go.Figure:
    configs = [
        ("Airplane / PANN",          TUSS_SC_DIR,      TUSS_MC_AIR_DIR,  "plane"),
        ("Airplane / AST",           TUSS_SC_DIR,      TUSS_MC_AIR_DIR,  "ast_finetuned"),
        ("Bird / BirdMAE",           TUSS_SC_BIRD_DIR, TUSS_MC_BIRD_DIR, "bird_mae"),
        ("Bird / AudioProtoPNet",    TUSS_SC_BIRD_DIR, TUSS_MC_BIRD_DIR, "audioprotopnet"),
    ]
    labels = [c[0] for c in configs]
    sc_f1 = [_metric(_load_classifier_json(sc, cls), "f1_score", "mix_sep_cls")
             for _, sc, _, cls in configs]
    mc_f1 = [_metric(_load_classifier_json(mc, cls), "f1_score", "mix_sep_cls")
             for _, _, mc, cls in configs]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=sc_f1, name="Single-class TUSS",
                         marker_color=COL_CLEAN_SEP,
                         text=[f"{v:.2f}" for v in sc_f1],
                         textposition="outside", cliponaxis=False))
    fig.add_trace(go.Bar(x=labels, y=mc_f1, name="Multi-class TUSS",
                         marker_color=COL_MIX_SEP,
                         text=[f"{v:.2f}" for v in mc_f1],
                         textposition="outside", cliponaxis=False))
    fig.update_layout(
        barmode="group",
        title="<b>TUSS single-class vs. multi-class — downstream F1 (mix + separation)</b>",
        yaxis=dict(title="F1 score", range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.20,
                    xanchor="center", x=0.5),
        height=480, width=1000,
        **LAYOUT_BASE,
    )
    return fig


# ── Figure 4: Noise robustness ──────────────────────────────────────────────

def fig_noise_robustness() -> go.Figure:
    if not NOISE_RESULTS_DIR.exists():
        return _placeholder("Noise robustness data not available — "
                            "run test_noise_increase.py first.")
    files = sorted(NOISE_RESULTS_DIR.glob("noise_increase_energy_*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return _placeholder("Noise robustness data not available.")

    with open(files[0]) as f:
        data = json.load(f)
    snr_results = sorted(data.get("snr_results", []),
                         key=lambda x: x["snr_db"], reverse=True)
    snr = [r["snr_db"] for r in snr_results]
    rms = [r["mean_rms_degradation_db"] for r in snr_results]
    sel = [r["mean_sel_degradation_db"] for r in snr_results]
    rms_std = [r.get("std_rms_degradation_db", 0) for r in snr_results]
    sel_std = [r.get("std_sel_degradation_db", 0) for r in snr_results]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=snr, y=rms, mode="lines+markers", name="ΔRMS",
        line=dict(color=COL_CLEAN_SEP, width=2.5),
        marker=dict(size=8, symbol="circle"),
        error_y=dict(type="data", array=rms_std, thickness=1, width=4,
                     color=COL_CLEAN_SEP),
    ))
    fig.add_trace(go.Scatter(
        x=snr, y=sel, mode="lines+markers", name="ΔSEL",
        line=dict(color=COL_SECONDARY, width=2.5, dash="dash"),
        marker=dict(size=8, symbol="diamond"),
        error_y=dict(type="data", array=sel_std, thickness=1, width=4,
                     color=COL_SECONDARY),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="grey",
                  annotation_text="Clean reference",
                  annotation_position="bottom right")

    fig.update_layout(
        title="<b>Energy-domain degradation vs. additive-noise SNR</b>",
        xaxis=dict(title="Input SNR (dB) — easier ←  → harder",
                   autorange="reversed"),
        yaxis=dict(title="Δ(separated, clean) energy (dB)"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.20,
                    xanchor="center", x=0.5),
        height=480, width=900,
        **LAYOUT_BASE,
    )
    return fig


# ── Figure 5: Activity gating sweep ─────────────────────────────────────────

def fig_activity_gating() -> go.Figure:
    if not GATING_SWEEP_JSON.exists():
        return _placeholder("Activity gating sweep data not available — "
                            "run run_activity_gating_sweep.py first.")
    with open(GATING_SWEEP_JSON) as f:
        data = json.load(f)

    sweep = sorted(data.get("sweep", []), key=lambda r: r["threshold"])
    baseline = data.get("baseline_no_recycling", {})
    th  = [r["threshold"] for r in sweep]
    f1  = [r.get("f1_score", float("nan")) for r in sweep]
    sni = [r.get("mean_si_snri_db", float("nan")) for r in sweep]
    hit = [r.get("cache_hit_rate", float("nan")) * 100 for r in sweep]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=th, y=f1, mode="lines+markers", name="F1",
                             line=dict(color=COL_MIX_SEP, width=2.5)),
                  secondary_y=False)
    if any(not np.isnan(v) for v in sni):
        fig.add_trace(go.Scatter(x=th, y=sni, mode="lines+markers", name="SI-SNRi (dB)",
                                 line=dict(color=COL_CLEAN_SEP, width=2.5, dash="dash")),
                      secondary_y=False)
    fig.add_trace(go.Scatter(x=th, y=hit, mode="lines+markers", name="Cache hit rate (%)",
                             line=dict(color=COL_RISOUX, width=2.5, dash="dot")),
                  secondary_y=True)

    bf1 = baseline.get("f1_score", float("nan"))
    if not np.isnan(bf1):
        fig.add_hline(y=bf1, line_dash="dot", line_color=COL_MIX_SEP,
                      annotation_text=f"Baseline F1 = {bf1:.2f}",
                      annotation_position="top left")

    fig.update_layout(
        title="<b>Activity gating: quality–efficiency trade-off</b>",
        xaxis_title="Cosine-similarity threshold",
        legend=dict(orientation="h", yanchor="bottom", y=-0.20,
                    xanchor="center", x=0.5),
        height=480, width=900,
        **LAYOUT_BASE,
    )
    fig.update_yaxes(title_text="F1 / SI-SNRi (dB)", secondary_y=False)
    fig.update_yaxes(title_text="Cache hit rate (%)", range=[0, 100],
                     secondary_y=True)
    return fig


def _placeholder(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=14, color="grey"))
    fig.update_layout(title="<b>Data not yet available</b>",
                      height=400, width=800, **LAYOUT_BASE)
    return fig


# ── LaTeX tables ────────────────────────────────────────────────────────────

def table_separation_metrics() -> None:
    """Signal-level separation metrics per model (single-class baseline rows)."""
    rows: List[List[str]] = []
    entries = [
        ("airplane", "SuDoRM-RF",     SUDORMRF_DIR),
        ("airplane", "TUSS (single)", TUSS_SC_DIR),
        ("airplane", "CLAPSep",       CLAPSEP_DIR),
        ("airplane", "TUSS (multi)",  TUSS_MC_AIR_DIR),
        ("bird",     "TUSS (single)", TUSS_SC_BIRD_DIR),
        ("bird",     "TUSS (multi)",  TUSS_MC_BIRD_DIR),
    ]
    # Use the primary classifier per COI just to locate a JSON; signal_metrics
    # are condition-level, so any classifier works.
    primary = {"airplane": "plane", "bird": "bird_mae"}

    for coi, model, base in entries:
        d = _load_classifier_json(base, primary[coi])
        rows.append([
            coi.capitalize(), model,
            _fmt(_signal(d, "mean_si_snr_db")),
            _fmt(_signal(d, "mean_si_snri_db")),
            _fmt(_signal(d, "mean_sdr_db")),
            _fmt(_signal(d, "mean_rms_error_db")),
            _fmt(_signal(d, "mean_sel_error_db")),
        ])
    _write_latex_table(
        TABLE_DIR / "tab_separation_metrics.tex",
        ["COI", "Model", "SI-SNR (dB)", "SI-SNRi (dB)", "SDR (dB)",
         "RMS err (dB)", "SEL err (dB)"],
        rows,
        caption="Signal-level separation metrics on the held-out test set, "
                "evaluated on the mixture + separation condition.",
        label="tab:separation-metrics",
        col_align="llrrrrr",
    )


def table_classification_per_condition() -> None:
    rows: List[List[str]] = []
    cond_order = [("mix_cls", "Mix only"),
                  ("clean_sep_cls", "Clean+sep"),
                  ("mix_sep_cls", "Mix+sep")]

    targets: List[Tuple[str, str, Path, str]] = []
    for sep_name, base, _ in SEPARATORS_AIRPLANE:
        for cls in CLASSIFIERS_AIRPLANE:
            targets.append(("Airplane", sep_name, base, cls))
    # bird COI
    for sep_name, base in [("TUSS (single)", TUSS_SC_BIRD_DIR),
                           ("TUSS (multi)",  TUSS_MC_BIRD_DIR)]:
        for cls in CLASSIFIERS_BIRD:
            targets.append(("Bird", sep_name, base, cls))

    for coi, sep, base, cls in targets:
        d = _load_classifier_json(base, cls)
        for cond_key, cond_lab in cond_order:
            rows.append([
                coi, sep, _cls_label(cls), cond_lab,
                _fmt(_metric(d, "f1_score",        cond_key), ".3f", "—"),
                _fmt(_metric(d, "precision",       cond_key), ".3f", "—"),
                _fmt(_metric(d, "recall",          cond_key), ".3f", "—"),
                _fmt(_metric(d, "balanced_accuracy", cond_key), ".3f", "—"),
            ])

    _write_latex_table(
        TABLE_DIR / "tab_classification_per_cond.tex",
        ["COI", "Separator", "Classifier", "Condition",
         "F1", "Precision", "Recall", "Bal.Acc."],
        rows,
        caption="Downstream classification metrics across operating "
                "conditions. \\emph{Mix only} omits separation entirely "
                "(no-separation baseline); \\emph{Clean+sep} measures the "
                "artefact cost of separation on already-clean COI; "
                "\\emph{Mix+sep} is the deployment condition.",
        label="tab:classification-per-cond",
        col_align="llllrrrr",
    )


def table_risoux() -> None:
    rows = []
    for sep_name, base, ris in SEPARATORS_AIRPLANE:
        for cls in CLASSIFIERS_AIRPLANE:
            d_in  = _load_classifier_json(base, cls)
            d_out = _load_classifier_json(ris,  cls, risoux=True)
            f1_in  = _metric(d_in,  "f1_score", "mix_sep_cls")
            f1_out = _metric(d_out, "f1_score", "mix_sep_cls")
            gap    = (f1_in - f1_out) if not (np.isnan(f1_in) or np.isnan(f1_out)) else float("nan")
            rows.append([
                sep_name, _cls_label(cls),
                _fmt(f1_in,  ".3f", "—"),
                _fmt(f1_out, ".3f", "—"),
                _fmt(gap,    "+.3f", "—"),
            ])
    _write_latex_table(
        TABLE_DIR / "tab_risoux_generalisation.tex",
        ["Separator", "Classifier", "In-dist. F1", "Risoux F1", "Δ (in − Risoux)"],
        rows,
        caption="Out-of-distribution generalisation: F1 on the in-distribution "
                "test set vs.\\ the Risoux forest field recordings, all under "
                "the mixture + separation condition.",
        label="tab:risoux-generalisation",
        col_align="llrrr",
    )


def table_multiclass() -> None:
    configs = [
        ("Airplane", "PANN",            TUSS_SC_DIR,      TUSS_MC_AIR_DIR,  "plane"),
        ("Airplane", "AST",             TUSS_SC_DIR,      TUSS_MC_AIR_DIR,  "ast_finetuned"),
        ("Bird",     "BirdMAE",         TUSS_SC_BIRD_DIR, TUSS_MC_BIRD_DIR, "bird_mae"),
        ("Bird",     "AudioProtoPNet",  TUSS_SC_BIRD_DIR, TUSS_MC_BIRD_DIR, "audioprotopnet"),
    ]
    rows = []
    for coi, cls_name, sc, mc, cls in configs:
        d_sc = _load_classifier_json(sc, cls)
        d_mc = _load_classifier_json(mc, cls)
        for label, d in [("Single", d_sc), ("Multi", d_mc)]:
            rows.append([
                coi, cls_name, label,
                _fmt(_metric(d, "f1_score", "mix_sep_cls"), ".3f", "—"),
                _fmt(_signal(d, "mean_si_snri_db")),
                _fmt(_signal(d, "mean_rms_error_db")),
                _fmt(_signal(d, "mean_sel_error_db")),
            ])
    _write_latex_table(
        TABLE_DIR / "tab_multiclass_tuss.tex",
        ["COI", "Classifier", "TUSS head", "F1", "SI-SNRi (dB)",
         "RMS err (dB)", "SEL err (dB)"],
        rows,
        caption="Single-class vs.\\ multi-class TUSS heads, evaluated under "
                "the mixture + separation condition.",
        label="tab:multiclass-tuss",
        col_align="lllrrrr",
    )


# ── I/O ─────────────────────────────────────────────────────────────────────

def save_figure(fig: go.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / f"{name}.png"
    try:
        fig.write_image(str(png), scale=PNG_SCALE)
        print(f"  Wrote {png.relative_to(FINAL_RESULTS_DIR)}")
    except Exception as e:
        print(f"  [PNG skipped — install kaleido: {e}]")
    html = OUTPUT_DIR / f"{name}.html"
    fig.write_html(str(html))


def main() -> None:
    print("=" * 72)
    print("Building chapter figures and LaTeX tables")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 72)

    print("\n— Figures —")
    figure_specs = [
        ("fig_classification_delta",    fig_classification_delta),
        ("fig_risoux_generalisation",   fig_risoux_generalisation),
        ("fig_multiclass_tuss",         fig_multiclass_tuss),
        ("fig_noise_robustness",        fig_noise_robustness),
        ("fig_activity_gating",         fig_activity_gating),
    ]
    for name, fn in figure_specs:
        try:
            save_figure(fn(), name)
        except Exception as e:
            import traceback
            print(f"  ERROR {name}: {e}")
            traceback.print_exc()

    print("\n— LaTeX tables —")
    for tbl_fn in (table_separation_metrics,
                   table_classification_per_condition,
                   table_risoux,
                   table_multiclass):
        try:
            tbl_fn()
        except Exception as e:
            import traceback
            print(f"  ERROR {tbl_fn.__name__}: {e}")
            traceback.print_exc()

    print(f"\nDone. Figures: {OUTPUT_DIR}\n      Tables : {TABLE_DIR}")


if __name__ == "__main__":
    main()
