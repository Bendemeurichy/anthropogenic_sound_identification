"""
Visualization script for noise increase experiment results.

This script creates comprehensive plots showing how separation affects
classification robustness across different SNR levels.

Usage:
    python plot_noise_increase_results.py [results_file.json]
    
If no file is specified, the script will auto-detect the most recent
results file in the noise_increase_results directory.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_results(results_path: str | Path) -> dict:
    """Load noise increase results from JSON file.
    
    Args:
        results_path: Path to JSON results file
        
    Returns:
        Dictionary containing experiment results and configuration
    """
    with open(results_path, "r") as f:
        return json.load(f)


def extract_model_info(results: dict) -> dict:
    """Extract model type and configuration from results.
    
    Args:
        results: Dictionary containing experiment results
        
    Returns:
        Dictionary with model information (type, prompts, etc.)
    """
    config = results.get("config", {})
    model_info = {
        "model_type": config.get("model_type", "Unknown"),
        "noise_type": config.get("noise_type", "artificial_white_noise"),
        "max_samples": config.get("max_samples", "all"),
        "seed": config.get("seed", 42),
    }
    
    # Add TUSS-specific info if present
    if "tuss_coi_prompt" in config:
        model_info["tuss_coi_prompt"] = config["tuss_coi_prompt"]
        model_info["tuss_bg_prompt"] = config["tuss_bg_prompt"]
    
    return model_info


def create_recall_vs_snr_plot(
    results: dict, model_info: dict | None = None
) -> go.Figure:
    """Create main recall vs SNR comparison plot.
    
    Shows how recall changes with SNR for both classification-only and
    separation+classification approaches, highlighting the improvement.
    
    Args:
        results: Dictionary containing experiment results
        model_info: Optional model configuration info
        
    Returns:
        Plotly Figure object
    """
    snr_data = results.get("snr_results", [])
    
    if not snr_data:
        raise ValueError("No SNR results found in the data")
    
    # Extract data
    snr_levels = [d["snr_db"] for d in snr_data]
    cls_only_recall = [d["cls_only_recall"] for d in snr_data]
    sep_cls_recall = [d["sep_cls_recall"] for d in snr_data]
    
    # Calculate improvement
    improvement = [s - c for s, c in zip(sep_cls_recall, cls_only_recall)]
    
    # Create figure
    fig = go.Figure()
    
    # Add classification-only line
    fig.add_trace(
        go.Scatter(
            x=snr_levels,
            y=cls_only_recall,
            mode="lines+markers",
            name="Classification Only",
            line=dict(color="steelblue", width=3),
            marker=dict(size=8, symbol="circle"),
            hovertemplate="<b>Classification Only</b><br>"
            + "SNR: %{x:.1f} dB<br>"
            + "Recall: %{y:.3f}<extra></extra>",
        )
    )
    
    # Add separation+classification line
    fig.add_trace(
        go.Scatter(
            x=snr_levels,
            y=sep_cls_recall,
            mode="lines+markers",
            name="Separation + Classification",
            line=dict(color="darkorange", width=3),
            marker=dict(size=8, symbol="diamond"),
            hovertemplate="<b>Separation + Classification</b><br>"
            + "SNR: %{x:.1f} dB<br>"
            + "Recall: %{y:.3f}<extra></extra>",
        )
    )
    
    # Add shaded area showing improvement
    fig.add_trace(
        go.Scatter(
            x=snr_levels + snr_levels[::-1],
            y=sep_cls_recall + cls_only_recall[::-1],
            fill="toself",
            fillcolor="rgba(34, 139, 34, 0.2)",  # Semi-transparent green
            line=dict(width=0),
            showlegend=True,
            name="Separation Improvement",
            hoverinfo="skip",
        )
    )
    
    # Build title
    title_text = "Recall vs SNR: Impact of Separation on Classification"
    if model_info:
        model_type = model_info.get("model_type", "")
        if model_type:
            title_text += f" ({model_type})"
    
    # Update layout
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=18)),
        xaxis=dict(
            title="SNR (dB)",
            gridcolor="lightgray",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="gray",
            zerolinewidth=2,
        ),
        yaxis=dict(
            title="Recall (True Positive Rate)",
            gridcolor="lightgray",
            gridwidth=1,
            range=[0, 1.05],
        ),
        plot_bgcolor="white",
        width=1000,
        height=600,
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.02,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
        ),
        hovermode="x unified",
    )
    
    return fig


def create_confidence_vs_snr_plot(
    results: dict, model_info: dict | None = None
) -> go.Figure:
    """Create confidence vs SNR comparison plot.
    
    Shows how model confidence changes with SNR for both approaches.
    
    Args:
        results: Dictionary containing experiment results
        model_info: Optional model configuration info
        
    Returns:
        Plotly Figure object
    """
    snr_data = results.get("snr_results", [])
    
    if not snr_data:
        raise ValueError("No SNR results found in the data")
    
    # Extract data
    snr_levels = [d["snr_db"] for d in snr_data]
    cls_only_conf = [d["cls_only_mean_conf"] for d in snr_data]
    sep_cls_conf = [d["sep_cls_mean_conf"] for d in snr_data]
    cls_only_std = [d["cls_only_std_conf"] for d in snr_data]
    sep_cls_std = [d["sep_cls_std_conf"] for d in snr_data]
    
    # Create figure with error bars
    fig = go.Figure()
    
    # Add classification-only line with error bars
    fig.add_trace(
        go.Scatter(
            x=snr_levels,
            y=cls_only_conf,
            mode="lines+markers",
            name="Classification Only",
            line=dict(color="steelblue", width=3),
            marker=dict(size=8, symbol="circle"),
            error_y=dict(
                type="data",
                array=cls_only_std,
                visible=True,
                color="steelblue",
                thickness=1.5,
                width=4,
            ),
            hovertemplate="<b>Classification Only</b><br>"
            + "SNR: %{x:.1f} dB<br>"
            + "Mean Confidence: %{y:.3f}<extra></extra>",
        )
    )
    
    # Add separation+classification line with error bars
    fig.add_trace(
        go.Scatter(
            x=snr_levels,
            y=sep_cls_conf,
            mode="lines+markers",
            name="Separation + Classification",
            line=dict(color="darkorange", width=3),
            marker=dict(size=8, symbol="diamond"),
            error_y=dict(
                type="data",
                array=sep_cls_std,
                visible=True,
                color="darkorange",
                thickness=1.5,
                width=4,
            ),
            hovertemplate="<b>Separation + Classification</b><br>"
            + "SNR: %{x:.1f} dB<br>"
            + "Mean Confidence: %{y:.3f}<extra></extra>",
        )
    )
    
    # Build title
    title_text = "Mean Confidence vs SNR"
    if model_info:
        model_type = model_info.get("model_type", "")
        if model_type:
            title_text += f" ({model_type})"
    
    # Update layout
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=18)),
        xaxis=dict(
            title="SNR (dB)",
            gridcolor="lightgray",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="gray",
            zerolinewidth=2,
        ),
        yaxis=dict(
            title="Mean Confidence",
            gridcolor="lightgray",
            gridwidth=1,
            range=[0, 1.05],
        ),
        plot_bgcolor="white",
        width=1000,
        height=600,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
        ),
        hovermode="x unified",
    )
    
    return fig


def create_separation_gain_plot(
    results: dict, model_info: dict | None = None
) -> go.Figure:
    """Create bar chart showing separation gain per SNR level.
    
    Shows the absolute improvement in recall when using separation.
    
    Args:
        results: Dictionary containing experiment results
        model_info: Optional model configuration info
        
    Returns:
        Plotly Figure object
    """
    snr_data = results.get("snr_results", [])
    
    if not snr_data:
        raise ValueError("No SNR results found in the data")
    
    # Extract data
    snr_levels = [f"{d['snr_db']:.1f} dB" for d in snr_data]
    cls_only_recall = [d["cls_only_recall"] for d in snr_data]
    sep_cls_recall = [d["sep_cls_recall"] for d in snr_data]
    
    # Calculate improvement
    improvement = [s - c for s, c in zip(sep_cls_recall, cls_only_recall)]
    
    # Color bars based on positive/negative improvement
    colors = ["forestgreen" if imp > 0 else "crimson" for imp in improvement]
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=snr_levels,
            y=improvement,
            marker_color=colors,
            text=[f"{imp:+.3f}" for imp in improvement],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>"
            + "Recall Improvement: %{y:+.3f}<extra></extra>",
        )
    )
    
    # Build title
    title_text = "Separation Gain (Recall Improvement)"
    if model_info:
        model_type = model_info.get("model_type", "")
        if model_type:
            title_text += f" ({model_type})"
    
    # Update layout
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=18)),
        xaxis=dict(
            title="SNR Level",
            gridcolor="lightgray",
            gridwidth=1,
        ),
        yaxis=dict(
            title="Recall Improvement (Separation - Baseline)",
            gridcolor="lightgray",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=2,
        ),
        plot_bgcolor="white",
        width=1000,
        height=600,
        showlegend=False,
    )
    
    return fig


def create_summary_table(results: dict, model_info: dict | None = None) -> go.Figure:
    """Create summary table showing key statistics.
    
    Args:
        results: Dictionary containing experiment results
        model_info: Optional model configuration info
        
    Returns:
        Plotly Figure object with table
    """
    snr_data = results.get("snr_results", [])
    summary = results.get("summary", {})
    config = results.get("config", {})
    dataset_stats = results.get("dataset_stats", {})
    
    if not snr_data:
        raise ValueError("No SNR results found in the data")
    
    # Build header
    header_values = [
        "SNR (dB)",
        "Cls Only<br>Recall",
        "Sep+Cls<br>Recall",
        "Improvement",
        "Cls Only<br>Conf",
        "Sep+Cls<br>Conf",
        "Actual<br>SNR (dB)",
        "N Samples",
    ]
    
    # Build cell values
    cell_values = []
    for d in snr_data:
        improvement = d["sep_cls_recall"] - d["cls_only_recall"]
        cell_values.append(
            [
                f"{d['snr_db']:.1f}",
                f"{d['cls_only_recall']:.3f}",
                f"{d['sep_cls_recall']:.3f}",
                f"{improvement:+.3f}",
                f"{d['cls_only_mean_conf']:.3f}",
                f"{d['sep_cls_mean_conf']:.3f}",
                f"{d['mean_actual_snr_db']:.1f}",
                f"{d['n_samples']}",
            ]
        )
    
    # Transpose for table format
    cell_values_transposed = list(zip(*cell_values)) if cell_values else []
    
    # Color code improvement column
    improvements = [d["sep_cls_recall"] - d["cls_only_recall"] for d in snr_data]
    improvement_colors = [
        "lightgreen" if imp > 0 else "lightcoral" for imp in improvements
    ]
    
    # Build column colors
    cell_fill_colors = [
        ["lavender"] * len(snr_data),  # SNR
        ["lightblue"] * len(snr_data),  # Cls Only Recall
        ["lightyellow"] * len(snr_data),  # Sep+Cls Recall
        improvement_colors,  # Improvement (color-coded)
        ["lightblue"] * len(snr_data),  # Cls Only Conf
        ["lightyellow"] * len(snr_data),  # Sep+Cls Conf
        ["lavender"] * len(snr_data),  # Actual SNR
        ["lavender"] * len(snr_data),  # N Samples
    ]
    
    # Create table
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=header_values,
                    fill_color="paleturquoise",
                    align="center",
                    font=dict(color="black", size=12, family="Arial"),
                    height=40,
                ),
                cells=dict(
                    values=cell_values_transposed if cell_values else [[] * len(header_values)],
                    fill_color=cell_fill_colors,
                    align="center",
                    font=dict(color="black", size=11, family="Arial"),
                    height=30,
                ),
            )
        ]
    )
    
    # Build title with configuration info
    title_lines = ["Noise Increase Experiment - Detailed Results"]
    
    if model_info:
        model_type = model_info.get("model_type", "")
        if model_type:
            title_lines.append(f"Model: {model_type}")
    
    if config:
        noise_type = config.get("noise_type", "")
        if noise_type:
            title_lines.append(f"Noise: {noise_type.replace('_', ' ').title()}")
    
    if summary:
        best_gain = summary.get("best_sep_gain_recall", 0)
        mean_gain = summary.get("mean_sep_gain_recall", 0)
        title_lines.append(
            f"Best Gain: {best_gain:+.3f} | Mean Gain: {mean_gain:+.3f}"
        )
    
    if dataset_stats:
        n_coi = dataset_stats.get("n_coi_samples", "?")
        n_contam = dataset_stats.get("n_contaminated_removed", 0)
        if n_contam > 0:
            title_lines.append(f"Samples: {n_coi} COI ({n_contam} contaminated removed)")
        else:
            title_lines.append(f"Samples: {n_coi} COI")
    
    title_text = "<br>".join(title_lines)
    
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14)),
        height=max(400, 80 + len(snr_data) * 30),
        width=1200,
    )
    
    return fig


def create_combined_dashboard(
    results: dict, model_info: dict | None = None
) -> go.Figure:
    """Create combined interactive dashboard with all plots.
    
    Args:
        results: Dictionary containing experiment results
        model_info: Optional model configuration info
        
    Returns:
        Plotly Figure object with subplots
    """
    snr_data = results.get("snr_results", [])
    
    if not snr_data:
        raise ValueError("No SNR results found in the data")
    
    # Extract data
    snr_levels = [d["snr_db"] for d in snr_data]
    cls_only_recall = [d["cls_only_recall"] for d in snr_data]
    sep_cls_recall = [d["sep_cls_recall"] for d in snr_data]
    cls_only_conf = [d["cls_only_mean_conf"] for d in snr_data]
    sep_cls_conf = [d["sep_cls_mean_conf"] for d in snr_data]
    improvement = [s - c for s, c in zip(sep_cls_recall, cls_only_recall)]
    
    # Create subplots: 2x2 grid
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Recall vs SNR",
            "Confidence vs SNR",
            "Separation Gain",
            "Recall Comparison (Both Methods)",
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.15,
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "scatter"}],
        ],
    )
    
    # Row 1, Col 1: Recall vs SNR
    fig.add_trace(
        go.Scatter(
            x=snr_levels,
            y=cls_only_recall,
            mode="lines+markers",
            name="Cls Only",
            line=dict(color="steelblue", width=2),
            marker=dict(size=6),
            legendgroup="cls",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=snr_levels,
            y=sep_cls_recall,
            mode="lines+markers",
            name="Sep+Cls",
            line=dict(color="darkorange", width=2),
            marker=dict(size=6),
            legendgroup="sep",
        ),
        row=1,
        col=1,
    )
    
    # Row 1, Col 2: Confidence vs SNR
    fig.add_trace(
        go.Scatter(
            x=snr_levels,
            y=cls_only_conf,
            mode="lines+markers",
            name="Cls Only",
            line=dict(color="steelblue", width=2),
            marker=dict(size=6),
            showlegend=False,
            legendgroup="cls",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=snr_levels,
            y=sep_cls_conf,
            mode="lines+markers",
            name="Sep+Cls",
            line=dict(color="darkorange", width=2),
            marker=dict(size=6),
            showlegend=False,
            legendgroup="sep",
        ),
        row=1,
        col=2,
    )
    
    # Row 2, Col 1: Separation Gain (Bar Chart)
    colors = ["forestgreen" if imp > 0 else "crimson" for imp in improvement]
    fig.add_trace(
        go.Bar(
            x=[f"{snr:.1f}" for snr in snr_levels],
            y=improvement,
            marker_color=colors,
            name="Gain",
            showlegend=False,
            text=[f"{imp:+.2f}" for imp in improvement],
            textposition="outside",
        ),
        row=2,
        col=1,
    )
    
    # Row 2, Col 2: Grouped bar chart for both methods
    snr_labels = [f"{snr:.0f}" for snr in snr_levels]
    fig.add_trace(
        go.Bar(
            x=snr_labels,
            y=cls_only_recall,
            name="Cls Only",
            marker_color="steelblue",
            showlegend=False,
            legendgroup="cls",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=snr_labels,
            y=sep_cls_recall,
            name="Sep+Cls",
            marker_color="darkorange",
            showlegend=False,
            legendgroup="sep",
        ),
        row=2,
        col=2,
    )
    
    # Update axes
    fig.update_xaxes(title_text="SNR (dB)", row=1, col=1)
    fig.update_yaxes(title_text="Recall", range=[0, 1.05], row=1, col=1)
    
    fig.update_xaxes(title_text="SNR (dB)", row=1, col=2)
    fig.update_yaxes(title_text="Confidence", range=[0, 1.05], row=1, col=2)
    
    fig.update_xaxes(title_text="SNR (dB)", row=2, col=1)
    fig.update_yaxes(title_text="Recall Gain", row=2, col=1)
    
    fig.update_xaxes(title_text="SNR (dB)", row=2, col=2)
    fig.update_yaxes(title_text="Recall", range=[0, 1.05], row=2, col=2)
    
    # Build title
    title_text = "Noise Increase Experiment Dashboard"
    if model_info:
        model_type = model_info.get("model_type", "")
        if model_type:
            title_text += f" - {model_type}"
    
    # Update layout
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=18)),
        width=1400,
        height=1000,
        plot_bgcolor="white",
        barmode="group",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
    )
    
    return fig


def find_latest_results_file(results_dir: Path) -> Path | None:
    """Find the most recent results JSON file in the given directory.
    
    Args:
        results_dir: Directory to search for results files
        
    Returns:
        Path to the most recent JSON file, or None if not found
    """
    json_files = list(results_dir.glob("noise_increase_results_*.json"))
    
    if not json_files:
        return None
    
    # Sort by modification time, most recent first
    json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return json_files[0]


def main():
    """Main entry point for the visualization script."""
    # Determine results file
    if len(sys.argv) > 1:
        results_path = Path(sys.argv[1])
        if not results_path.exists():
            print(f"Error: File not found: {results_path}")
            sys.exit(1)
    else:
        # Auto-detect most recent results file
        results_dir = Path(__file__).parent / "noise_increase_results"
        if not results_dir.exists():
            print(f"Error: Results directory not found: {results_dir}")
            print("Please specify a results file as an argument.")
            sys.exit(1)
        
        results_path = find_latest_results_file(results_dir)
        if results_path is None:
            print(f"Error: No results files found in {results_dir}")
            print("Please specify a results file as an argument.")
            sys.exit(1)
        
        print(f"Auto-detected most recent results file: {results_path.name}")
    
    # Load results
    print(f"Loading results from: {results_path}")
    results = load_results(results_path)
    
    # Extract model info
    model_info = extract_model_info(results)
    model_type = model_info.get("model_type", "unknown")
    
    print(f"Model type: {model_type}")
    print(f"Number of SNR levels: {len(results.get('snr_results', []))}")
    
    # Create output directory
    output_dir = results_path.parent / "plots"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate timestamp for this plotting run
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\nGenerating visualizations...")
    
    # 1. Main recall vs SNR plot
    print("  [1/5] Creating recall vs SNR plot...")
    recall_fig = create_recall_vs_snr_plot(results, model_info)
    recall_path = output_dir / f"recall_vs_snr_{model_type.lower()}_{ts}.png"
    recall_fig.write_image(str(recall_path), scale=2)
    print(f"    Saved: {recall_path}")
    
    # 2. Confidence vs SNR plot
    print("  [2/5] Creating confidence vs SNR plot...")
    conf_fig = create_confidence_vs_snr_plot(results, model_info)
    conf_path = output_dir / f"confidence_vs_snr_{model_type.lower()}_{ts}.png"
    conf_fig.write_image(str(conf_path), scale=2)
    print(f"    Saved: {conf_path}")
    
    # 3. Separation gain bar chart
    print("  [3/5] Creating separation gain plot...")
    gain_fig = create_separation_gain_plot(results, model_info)
    gain_path = output_dir / f"separation_gain_{model_type.lower()}_{ts}.png"
    gain_fig.write_image(str(gain_path), scale=2)
    print(f"    Saved: {gain_path}")
    
    # 4. Summary table
    print("  [4/5] Creating summary table...")
    table_fig = create_summary_table(results, model_info)
    table_path = output_dir / f"summary_table_{model_type.lower()}_{ts}.png"
    table_fig.write_image(str(table_path), scale=2)
    print(f"    Saved: {table_path}")
    
    # 5. Combined interactive dashboard
    print("  [5/5] Creating interactive dashboard...")
    dashboard_fig = create_combined_dashboard(results, model_info)
    dashboard_path = output_dir / f"dashboard_{model_type.lower()}_{ts}.html"
    dashboard_fig.write_html(str(dashboard_path))
    print(f"    Saved: {dashboard_path}")
    
    print(f"\n✓ All visualizations saved to: {output_dir}")
    print(f"\nSummary Statistics:")
    summary = results.get("summary", {})
    if summary:
        print(f"  Best separation gain (recall):  {summary.get('best_sep_gain_recall', 0):+.3f}")
        print(f"  Mean separation gain (recall):  {summary.get('mean_sep_gain_recall', 0):+.3f}")
    
    dataset_stats = results.get("dataset_stats", {})
    if dataset_stats:
        n_coi = dataset_stats.get("n_coi_samples", "?")
        n_contam = dataset_stats.get("n_contaminated_removed", 0)
        print(f"  COI samples evaluated:           {n_coi}")
        if n_contam > 0:
            print(f"  Contaminated backgrounds removed: {n_contam}")


if __name__ == "__main__":
    main()
