"""
plot_confusion_matrices.py — Per-run confusion matrix grids and LaTeX metrics
tables for the dissertation.

For every ``results_*.json`` discovered under ``final_results/<run>/<classifier>/``
this script emits, into a sibling ``plots/`` directory:

  * ``confusion_matrices.png`` — a 2×2 grid containing one binary confusion
    matrix per operating condition (``clean_cls``, ``mix_cls``,
    ``clean_sep_cls``, ``mix_sep_cls``).  All four heatmaps share a single
    colour scale so cell darkness is directly comparable across panels.
  * ``metrics.tex`` — booktabs tabular environment summarising the
    classification metrics (and signal metrics where applicable) for every
    condition.  Designed to be ``\\input``-ed from the chapter.

Top-misclassified bar charts and rasterised metric tables produced by
previous revisions have been removed: confusion-matrix counts plus the
LaTeX table cover every diagnostic the chapter actually cites.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Constants ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
FINAL_RESULTS_DIR = SCRIPT_DIR / "final_results"

CONDITION_ORDER = ["clean_cls", "mix_cls", "clean_sep_cls", "mix_sep_cls"]
CONDITION_LABELS = {
    "clean_cls":     "Clean COI · classifier only",
    "mix_cls":       "Mixture · classifier only",
    "clean_sep_cls": "Clean COI · separation + classifier",
    "mix_sep_cls":   "Mixture · separation + classifier",
}

CLS_DISPLAY = {
    "pann_finetuned":  "PANN (fine-tuned)",
    "ast_finetuned":   "AST (fine-tuned)",
    "bird_mae":        "BirdMAE",
    "audioprotopnet":  "AudioProtoPNet",
}

PNG_SCALE = 2
TEMPLATE = "plotly_white"
FONT = dict(family="Times New Roman, serif", size=12, color="#1a1a1a")


# ── Helpers ─────────────────────────────────────────────────────────────────

def _cls_label(name: str) -> str:
    return CLS_DISPLAY.get(name, name.replace("_", " "))


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _separator_label(results: dict) -> str:
    """Best-effort, human-readable separator name from checkpoint paths."""
    sep = results.get("checkpoint_paths", {}).get("separator", "") or ""
    s = sep.lower()
    if "clapsep" in s:        return "CLAPSep"
    if "sudormrf" in s:       return "SuDoRM-RF"
    if "tuss" in s and "multi" in s:  return "TUSS (multi)"
    if "tuss" in s and "single" in s: return "TUSS (single)"
    if "tuss" in s:           return "TUSS"
    return "Separator"


def _classifier_label_from_path(results: dict, fallback: str) -> str:
    clf = results.get("classifier") or results.get("checkpoint_paths", {}).get("classifier", "")
    if isinstance(clf, str):
        for key, disp in CLS_DISPLAY.items():
            if key in clf.lower():
                return disp
    return _cls_label(fallback)


def _fmt(v, spec: str = ".3f", dash: str = "—") -> str:
    if v is None:
        return dash
    try:
        f = float(v)
    except (TypeError, ValueError):
        return dash
    if np.isnan(f):
        return dash
    return f"{f:{spec}}"


# ── Confusion matrix figure ─────────────────────────────────────────────────

def build_cm_figure(results: dict, title: str) -> go.Figure:
    conditions = [c for c in CONDITION_ORDER if isinstance(results.get(c), dict)
                  and "confusion_matrix" in results[c]]
    if not conditions:
        raise ValueError("No condition contains a confusion_matrix")

    # Pad to 2x2 layout regardless of how many conditions exist
    grid = conditions + [None] * (4 - len(conditions))
    grid = grid[:4]

    titles = [CONDITION_LABELS.get(c, c) if c else "" for c in grid]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=titles,
        horizontal_spacing=0.18,
        vertical_spacing=0.20,
    )

    # Shared colour scale for visual comparability
    z_max = max(
        max(results[c]["confusion_matrix"].values())
        for c in conditions
    )

    for idx, cond in enumerate(grid):
        if cond is None:
            continue
        r, c = idx // 2 + 1, idx % 2 + 1
        cm = results[cond]["confusion_matrix"]
        tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
        z = np.array([[tn, fp], [fn, tp]])
        total = z.sum() or 1
        text = [
            [f"TN<br>{tn}<br>({tn / total:.1%})",
             f"FP<br>{fp}<br>({fp / total:.1%})"],
            [f"FN<br>{fn}<br>({fn / total:.1%})",
             f"TP<br>{tp}<br>({tp / total:.1%})"],
        ]
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=["Pred. negative", "Pred. positive"],
                y=["True negative",  "True positive"],
                text=text, texttemplate="%{text}",
                textfont=dict(size=11),
                colorscale="Blues",
                zmin=0, zmax=z_max,
                showscale=(idx == 0),
                hovertemplate="True: %{y}<br>Pred: %{x}<br>n = %{z}<extra></extra>",
            ),
            row=r, col=c,
        )
        fig.update_xaxes(side="bottom", row=r, col=c)
        fig.update_yaxes(autorange="reversed", row=r, col=c)

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=15)),
        template=TEMPLATE, font=FONT,
        width=900, height=780,
        margin=dict(l=70, r=40, t=80, b=50),
    )
    return fig


# ── LaTeX metrics table ─────────────────────────────────────────────────────

def write_metrics_table(
    results: dict, out_path: Path, separator: str, classifier: str,
    coi: Optional[str] = None,
) -> None:
    metric_keys = [
        ("accuracy",          "Accuracy"),
        ("precision",         "Precision"),
        ("recall",            "Recall"),
        ("f1_score",          "F1"),
        ("specificity",       "Specificity"),
        ("balanced_accuracy", "Bal.Acc."),
        ("mcc",               "MCC"),
    ]
    sig_keys = [
        ("mean_si_snr_db",   "SI-SNR (dB)"),
        ("mean_si_snri_db",  "SI-SNRi (dB)"),
        ("mean_sdr_db",      "SDR (dB)"),
        ("mean_rms_error_db","RMS err (dB)"),
        ("mean_sel_error_db","SEL err (dB)"),
    ]

    columns = ["Condition"] + [m[1] for m in metric_keys] + [s[1] for s in sig_keys]
    rows: List[List[str]] = []
    for cond in CONDITION_ORDER:
        d = results.get(cond)
        if not isinstance(d, dict) or "confusion_matrix" not in d:
            continue
        row = [CONDITION_LABELS[cond]]
        for k, _ in metric_keys:
            row.append(_fmt(d.get(k)))
        sig = d.get("signal_metrics") or {}
        for k, _ in sig_keys:
            row.append(_fmt(sig.get(k), ".2f") if sig else "—")
        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    align = "l" + "r" * (len(columns) - 1)
    label = (f"tab:cm-{separator}-{classifier}-{coi}".lower()
             .replace(" ", "").replace("(", "").replace(")", "")
             .replace("·", "").replace("--", "-"))
    caption = (f"Per-condition classification and signal metrics — "
               f"{separator} + {classifier}"
               + (f" ({coi})" if coi else "") + ".")

    lines = [
        "% Auto-generated by plot_confusion_matrices.py — do not edit.",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\small",
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

def process_results_file(json_path: Path) -> None:
    results = _load(json_path)
    if not any(isinstance(results.get(c), dict) and "confusion_matrix" in results[c]
               for c in CONDITION_ORDER):
        print(f"  · skipped (no confusion_matrix): {json_path.name}")
        return

    classifier_dir = json_path.parent
    classifier_key = classifier_dir.name
    run_dir = classifier_dir.parent
    out_dir = classifier_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    sep = _separator_label(results)
    clf = _classifier_label_from_path(results, classifier_key)
    coi = "Risoux" if "risoux" in json_path.stem.lower() else None
    title = f"Confusion matrices — {sep} + {clf}" + (f"  ({coi})" if coi else "")

    fig = build_cm_figure(results, title)
    suffix = "_risoux" if coi else ""
    png_path = out_dir / f"confusion_matrices{suffix}.png"
    try:
        fig.write_image(str(png_path), scale=PNG_SCALE)
        print(f"  · {png_path.relative_to(FINAL_RESULTS_DIR)}")
    except Exception as e:
        print(f"  · PNG skipped ({e}): {png_path.name}")
    fig.write_html(str(png_path.with_suffix(".html")))

    tex_path = out_dir / f"metrics{suffix}.tex"
    write_metrics_table(results, tex_path, sep, clf, coi)
    print(f"  · {tex_path.relative_to(FINAL_RESULTS_DIR)}")


def main() -> None:
    if not FINAL_RESULTS_DIR.exists():
        print(f"final_results/ not found at {FINAL_RESULTS_DIR}")
        return
    files = sorted(FINAL_RESULTS_DIR.rglob("results_*.json"))
    # Skip noise/gating sweep JSONs — those have their own scripts
    files = [f for f in files
             if "noise_increase_results" not in f.parts
             and "activity_gating_results" not in f.parts]
    if not files:
        print("No classifier result files found.")
        return
    print(f"Processing {len(files)} result files…")
    for f in files:
        try:
            process_results_file(f)
        except Exception as e:
            import traceback
            print(f"  ERROR {f}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
