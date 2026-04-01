"""
Visualization script for energy-based noise increase experiment results.

This script creates comprehensive plots showing how separation preserves energy
(RMS and SEL) across different SNR levels.

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
        "experiment_type": config.get("experiment_type", "energy_preservation"),
        "seed": config.get("seed", 42),
    }
    
    # Add TUSS-specific info if present
    if "tuss_coi_prompt" in config:
        model_info["tuss_coi_prompt"] = config["tuss_coi_prompt"]
        model_info["tuss_bg_prompt"] = config["tuss_bg_prompt"]
    
    return model_info


def create_energy_degradation_plot(
    results: dict, model_info: dict | None = None
) -> go.Figure:
    """Create main energy degradation vs SNR plot.
    
    Shows how RMS and SEL energy degrade with increasing noise levels
    (decreasing SNR).
    
    Args:
        results: Dictionary containing experiment results
        model_info: Optional model configuration info
        
    Returns:
        Plotly Figure object
    """
    snr_data = results.get("snr_results", [])
    clean_baseline = results.get("clean_baseline", {})
    
    if not snr_data:
        raise ValueError("No SNR results found in the data")
    
    # Sort by SNR level (descending: +25 dB → -20 dB, easy → hard)
    snr_data = sorted(snr_data, key=lambda x: x["snr_db"], reverse=True)
    
    # Extract data
    snr_levels = [d["snr_db"] for d in snr_data]
    rms_degradation = [d["mean_rms_degradation_db"] for d in snr_data]
    sel_degradation = [d["mean_sel_degradation_db"] for d in snr_data]
    rms_std = [d["std_rms_degradation_db"] for d in snr_data]
    sel_std = [d["std_sel_degradation_db"] for d in snr_data]
    
    # Create figure
    fig = go.Figure()
    
    # Add RMS degradation line with error bars
    fig.add_trace(
        go.Scatter(
            x=snr_levels,
            y=rms_degradation,
            mode="lines+markers",
            name="RMS Degradation",
            line=dict(color="steelblue", width=3),
            marker=dict(size=8, symbol="circle"),
            error_y=dict(
                type="data",
                array=rms_std,
                visible=True,
                color="steelblue",
                thickness=1.5,
                width=4,
            ),
            hovertemplate="<b>RMS</b><br>"
            + "SNR: %{x:.1f} dB<br>"
            + "Degradation: %{y:+.2f} dB<extra></extra>",
        )
    )
    
    # Add SEL degradation line with error bars
    fig.add_trace(
        go.Scatter(
            x=snr_levels,
            y=sel_degradation,
            mode="lines+markers",
            name="SEL Degradation",
            line=dict(color="darkorange", width=3),
            marker=dict(size=8, symbol="diamond"),
            error_y=dict(
                type="data",
                array=sel_std,
                visible=True,
                color="darkorange",
                thickness=1.5,
                width=4,
            ),
            hovertemplate="<b>SEL</b><br>"
            + "SNR: %{x:.1f} dB<br>"
            + "Degradation: %{y:+.2f} dB<extra></extra>",
        )
    )
    
    # Add horizontal zero line (perfect preservation)
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        opacity=0.7,
        annotation_text="Perfect Preservation",
        annotation_position="right",
    )
    
    # Build title
    title_text = "Energy Degradation vs SNR: Separation Quality"
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
            autorange="reversed",  # High SNR (easy) on left, low SNR (hard) on right
        ),
        yaxis=dict(
            title="Energy Degradation from Clean Baseline (dB)",
            gridcolor="lightgray",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=2,
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


def create_absolute_energy_plot(
    results: dict, model_info: dict | None = None
) -> go.Figure:
    """Create absolute energy levels vs SNR plot.
    
    Shows the absolute RMS and SEL values (in dBFS) at different SNR levels
    for both mixture and separated signals.
    
    Args:
        results: Dictionary containing experiment results
        model_info: Optional model configuration info
        
    Returns:
        Plotly Figure object
    """
    snr_data = results.get("snr_results", [])
    clean_baseline = results.get("clean_baseline", {})
    
    if not snr_data:
        raise ValueError("No SNR results found in the data")
    
    # Sort by SNR level (descending)
    snr_data = sorted(snr_data, key=lambda x: x["snr_db"], reverse=True)
    
    # Extract data
    snr_levels = [d["snr_db"] for d in snr_data]
    mixture_rms = [d["mean_mixture_rms_db"] for d in snr_data]
    mixture_sel = [d["mean_mixture_sel_db"] for d in snr_data]
    separated_rms = [d["mean_separated_noisy_rms_db"] for d in snr_data]
    separated_sel = [d["mean_separated_noisy_sel_db"] for d in snr_data]
    
    # Get clean baseline values
    baseline_orig_rms = clean_baseline.get("mean_original_rms_db", None)
    baseline_orig_sel = clean_baseline.get("mean_original_sel_db", None)
    baseline_sep_rms = clean_baseline.get("mean_separated_rms_db", None)
    baseline_sep_sel = clean_baseline.get("mean_separated_sel_db", None)
    
    # Create subplots for RMS and SEL
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("RMS Energy Levels", "SEL Energy Levels"),
        horizontal_spacing=0.12,
    )
    
    # RMS subplot
    fig.add_trace(
        go.Scatter(
            x=snr_levels,
            y=mixture_rms,
            mode="lines+markers",
            name="Mixture RMS",
            line=dict(color="lightcoral", width=2, dash="dot"),
            marker=dict(size=6),
            legendgroup="rms",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=snr_levels,
            y=separated_rms,
            mode="lines+markers",
            name="Separated RMS",
            line=dict(color="steelblue", width=3),
            marker=dict(size=8),
            legendgroup="rms",
        ),
        row=1,
        col=1,
    )
    
    # Add clean baseline reference for RMS
    if baseline_orig_rms is not None:
        fig.add_hline(
            y=baseline_orig_rms,
            line_dash="dash",
            line_color="green",
            opacity=0.5,
            annotation_text=f"Clean Baseline: {baseline_orig_rms:.1f} dBFS",
            annotation_position="top right",
            row=1,
            col=1,
        )
    
    # SEL subplot
    fig.add_trace(
        go.Scatter(
            x=snr_levels,
            y=mixture_sel,
            mode="lines+markers",
            name="Mixture SEL",
            line=dict(color="lightcoral", width=2, dash="dot"),
            marker=dict(size=6),
            legendgroup="sel",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=snr_levels,
            y=separated_sel,
            mode="lines+markers",
            name="Separated SEL",
            line=dict(color="darkorange", width=3),
            marker=dict(size=8),
            legendgroup="sel",
        ),
        row=1,
        col=2,
    )
    
    # Add clean baseline reference for SEL
    if baseline_orig_sel is not None:
        fig.add_hline(
            y=baseline_orig_sel,
            line_dash="dash",
            line_color="green",
            opacity=0.5,
            annotation_text=f"Clean Baseline: {baseline_orig_sel:.1f} dBFS",
            annotation_position="top right",
            row=1,
            col=2,
        )
    
    # Build title
    title_text = "Absolute Energy Levels vs SNR"
    if model_info:
        model_type = model_info.get("model_type", "")
        if model_type:
            title_text += f" ({model_type})"
    
    # Update axes
    fig.update_xaxes(title_text="SNR (dB)", autorange="reversed", row=1, col=1)
    fig.update_xaxes(title_text="SNR (dB)", autorange="reversed", row=1, col=2)
    fig.update_yaxes(title_text="RMS (dBFS)", row=1, col=1)
    fig.update_yaxes(title_text="SEL (dBFS)", row=1, col=2)
    
    # Update layout
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=18)),
        width=1400,
        height=600,
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
        hovermode="x unified",
    )
    
    return fig


def create_clean_separation_quality_plot(
    results: dict, model_info: dict | None = None
) -> go.Figure:
    """Create bar chart showing separation quality on clean signals.
    
    Shows how much energy changes when separating clean (noise-free) signals.
    Ideally should be close to zero (perfect preservation).
    
    Args:
        results: Dictionary containing experiment results
        model_info: Optional model configuration info
        
    Returns:
        Plotly Figure object
    """
    clean_baseline = results.get("clean_baseline", {})
    
    if not clean_baseline:
        raise ValueError("No clean baseline found in the data")
    
    # Extract clean separation preservation metrics
    rms_preservation = clean_baseline.get("mean_rms_preservation_db", 0)
    sel_preservation = clean_baseline.get("mean_sel_preservation_db", 0)
    
    # Create bar chart
    fig = go.Figure()
    
    metrics = ["RMS", "SEL"]
    values = [rms_preservation, sel_preservation]
    colors = ["steelblue", "darkorange"]
    
    fig.add_trace(
        go.Bar(
            x=metrics,
            y=values,
            marker_color=colors,
            text=[f"{v:+.2f} dB" for v in values],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>"
            + "Energy Change: %{y:+.2f} dB<extra></extra>",
        )
    )
    
    # Add horizontal zero line (perfect preservation)
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        opacity=0.7,
    )
    
    # Build title
    title_text = "Separation Quality on Clean Signals (No Noise)"
    if model_info:
        model_type = model_info.get("model_type", "")
        if model_type:
            title_text += f" ({model_type})"
    
    # Update layout
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=18)),
        xaxis=dict(
            title="Energy Metric",
            gridcolor="lightgray",
            gridwidth=1,
        ),
        yaxis=dict(
            title="Energy Change (dB)<br>(Separated - Original)",
            gridcolor="lightgray",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=2,
        ),
        plot_bgcolor="white",
        width=600,
        height=500,
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
    clean_baseline = results.get("clean_baseline", {})
    
    if not snr_data:
        raise ValueError("No SNR results found in the data")
    
    # Sort by SNR level (descending)
    snr_data = sorted(snr_data, key=lambda x: x["snr_db"], reverse=True)
    
    # Build header
    header_values = [
        "SNR (dB)",
        "RMS<br>Degradation",
        "SEL<br>Degradation",
        "Mixture<br>RMS (dBFS)",
        "Separated<br>RMS (dBFS)",
        "Mixture<br>SEL (dBFS)",
        "Separated<br>SEL (dBFS)",
        "N Segments",
    ]
    
    # Build cell values
    cell_values = []
    for d in snr_data:
        cell_values.append(
            [
                f"{d['snr_db']:.1f}",
                f"{d['mean_rms_degradation_db']:+.2f} dB",
                f"{d['mean_sel_degradation_db']:+.2f} dB",
                f"{d['mean_mixture_rms_db']:.2f}",
                f"{d['mean_separated_noisy_rms_db']:.2f}",
                f"{d['mean_mixture_sel_db']:.2f}",
                f"{d['mean_separated_noisy_sel_db']:.2f}",
                f"{d['n_segments']}",
            ]
        )
    
    # Transpose for table format
    cell_values_transposed = list(zip(*cell_values)) if cell_values else []
    
    # Color code degradation columns (indices 1, 2)
    rms_degs = [d["mean_rms_degradation_db"] for d in snr_data]
    sel_degs = [d["mean_sel_degradation_db"] for d in snr_data]
    
    # Color scale: green (near 0) to red (large negative)
    def degradation_color(deg_val):
        if deg_val >= -1:
            return "lightgreen"
        elif deg_val >= -3:
            return "lightyellow"
        elif deg_val >= -6:
            return "lightsalmon"
        else:
            return "lightcoral"
    
    rms_colors = [degradation_color(d) for d in rms_degs]
    sel_colors = [degradation_color(d) for d in sel_degs]
    
    # Build column colors
    cell_fill_colors = [
        ["lavender"] * len(snr_data),  # SNR
        rms_colors,  # RMS Degradation (color-coded)
        sel_colors,  # SEL Degradation (color-coded)
        ["lightblue"] * len(snr_data),  # Mixture RMS
        ["lightyellow"] * len(snr_data),  # Separated RMS
        ["lightblue"] * len(snr_data),  # Mixture SEL
        ["lightyellow"] * len(snr_data),  # Separated SEL
        ["lavender"] * len(snr_data),  # N Segments
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
    title_lines = ["Energy Preservation Experiment - Detailed Results"]
    
    if model_info:
        model_type = model_info.get("model_type", "")
        if model_type:
            title_lines.append(f"Model: {model_type}")
    
    if config:
        noise_type = config.get("noise_type", "")
        if noise_type:
            title_lines.append(f"Noise: {noise_type.replace('_', ' ').title()}")
    
    # Add clean baseline info
    if clean_baseline:
        rms_pres = clean_baseline.get("mean_rms_preservation_db", 0)
        sel_pres = clean_baseline.get("mean_sel_preservation_db", 0)
        title_lines.append(
            f"Clean Baseline - RMS: {rms_pres:+.2f} dB | SEL: {sel_pres:+.2f} dB"
        )
    
    if summary:
        max_rms_deg = summary.get("max_rms_degradation_db", 0)
        max_sel_deg = summary.get("max_sel_degradation_db", 0)
        title_lines.append(
            f"Max Degradation - RMS: {max_rms_deg:+.2f} dB | SEL: {max_sel_deg:+.2f} dB"
        )
    
    if dataset_stats:
        n_coi = dataset_stats.get("n_coi_samples", "?")
        title_lines.append(f"Samples: {n_coi} COI")
    
    title_text = "<br>".join(title_lines)
    
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14)),
        height=max(400, 80 + len(snr_data) * 30),
        width=1400,
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
    clean_baseline = results.get("clean_baseline", {})
    
    if not snr_data:
        raise ValueError("No SNR results found in the data")
    
    # Sort by SNR level (descending)
    snr_data = sorted(snr_data, key=lambda x: x["snr_db"], reverse=True)
    
    # Extract data
    snr_levels = [d["snr_db"] for d in snr_data]
    rms_degradation = [d["mean_rms_degradation_db"] for d in snr_data]
    sel_degradation = [d["mean_sel_degradation_db"] for d in snr_data]
    separated_rms = [d["mean_separated_noisy_rms_db"] for d in snr_data]
    separated_sel = [d["mean_separated_noisy_sel_db"] for d in snr_data]
    
    # Get clean baseline
    baseline_orig_rms = clean_baseline.get("mean_original_rms_db", None)
    baseline_orig_sel = clean_baseline.get("mean_original_sel_db", None)
    rms_preservation = clean_baseline.get("mean_rms_preservation_db", 0)
    sel_preservation = clean_baseline.get("mean_sel_preservation_db", 0)
    
    # Create subplots: 2x2 grid
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Energy Degradation vs SNR",
            "Absolute RMS Levels",
            "Absolute SEL Levels",
            "Clean Separation Quality",
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.15,
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "bar"}],
        ],
    )
    
    # Row 1, Col 1: Energy degradation
    fig.add_trace(
        go.Scatter(
            x=snr_levels,
            y=rms_degradation,
            mode="lines+markers",
            name="RMS Degradation",
            line=dict(color="steelblue", width=2),
            marker=dict(size=6),
            legendgroup="rms",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=snr_levels,
            y=sel_degradation,
            mode="lines+markers",
            name="SEL Degradation",
            line=dict(color="darkorange", width=2),
            marker=dict(size=6),
            legendgroup="sel",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    
    # Row 1, Col 2: Absolute RMS levels
    fig.add_trace(
        go.Scatter(
            x=snr_levels,
            y=separated_rms,
            mode="lines+markers",
            name="Separated RMS",
            line=dict(color="steelblue", width=2),
            marker=dict(size=6),
            showlegend=False,
            legendgroup="rms",
        ),
        row=1,
        col=2,
    )
    if baseline_orig_rms is not None:
        fig.add_hline(
            y=baseline_orig_rms,
            line_dash="dash",
            line_color="green",
            opacity=0.5,
            row=1,
            col=2,
        )
    
    # Row 2, Col 1: Absolute SEL levels
    fig.add_trace(
        go.Scatter(
            x=snr_levels,
            y=separated_sel,
            mode="lines+markers",
            name="Separated SEL",
            line=dict(color="darkorange", width=2),
            marker=dict(size=6),
            showlegend=False,
            legendgroup="sel",
        ),
        row=2,
        col=1,
    )
    if baseline_orig_sel is not None:
        fig.add_hline(
            y=baseline_orig_sel,
            line_dash="dash",
            line_color="green",
            opacity=0.5,
            row=2,
            col=1,
        )
    
    # Row 2, Col 2: Clean separation quality (bar chart)
    metrics = ["RMS", "SEL"]
    values = [rms_preservation, sel_preservation]
    colors = ["steelblue", "darkorange"]
    
    fig.add_trace(
        go.Bar(
            x=metrics,
            y=values,
            marker_color=colors,
            name="Clean Sep Quality",
            showlegend=False,
            text=[f"{v:+.2f}" for v in values],
            textposition="outside",
        ),
        row=2,
        col=2,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=2)
    
    # Update axes
    fig.update_xaxes(title_text="SNR (dB)", autorange="reversed", row=1, col=1)
    fig.update_yaxes(title_text="Degradation (dB)", row=1, col=1)
    
    fig.update_xaxes(title_text="SNR (dB)", autorange="reversed", row=1, col=2)
    fig.update_yaxes(title_text="RMS (dBFS)", row=1, col=2)
    
    fig.update_xaxes(title_text="SNR (dB)", autorange="reversed", row=2, col=1)
    fig.update_yaxes(title_text="SEL (dBFS)", row=2, col=1)
    
    fig.update_xaxes(title_text="Metric", row=2, col=2)
    fig.update_yaxes(title_text="Change (dB)", row=2, col=2)
    
    # Build title
    title_text = "Energy Preservation Experiment Dashboard"
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
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5,
        ),
    )
    
    return fig


def find_latest_results_file(results_dir: Path) -> Path | None:
    """Find the most recent energy results JSON file in the given directory.
    
    Args:
        results_dir: Directory to search for results files
        
    Returns:
        Path to the most recent JSON file, or None if not found
    """
    # Look for energy-specific results files first
    json_files = list(results_dir.glob("noise_increase_energy_*.json"))
    
    # Fall back to any noise increase results
    if not json_files:
        json_files = list(results_dir.glob("noise_increase_*.json"))
    
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
    experiment_type = model_info.get("experiment_type", "unknown")
    
    print(f"Model type: {model_type}")
    print(f"Experiment type: {experiment_type}")
    print(f"Number of SNR levels: {len(results.get('snr_results', []))}")
    
    # Create output directory
    output_dir = results_path.parent / "plots"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate timestamp for this plotting run
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\nGenerating visualizations...")
    
    # 1. Main energy degradation plot
    print("  [1/6] Creating energy degradation vs SNR plot...")
    deg_fig = create_energy_degradation_plot(results, model_info)
    deg_path = output_dir / f"energy_degradation_{model_type.lower()}_{ts}.png"
    deg_fig.write_image(str(deg_path), scale=2)
    print(f"    Saved: {deg_path}")
    
    # 2. Absolute energy levels plot
    print("  [2/6] Creating absolute energy levels plot...")
    abs_fig = create_absolute_energy_plot(results, model_info)
    abs_path = output_dir / f"absolute_energy_{model_type.lower()}_{ts}.png"
    abs_fig.write_image(str(abs_path), scale=2)
    print(f"    Saved: {abs_path}")
    
    # 3. Clean separation quality plot
    print("  [3/6] Creating clean separation quality plot...")
    clean_fig = create_clean_separation_quality_plot(results, model_info)
    clean_path = output_dir / f"clean_separation_{model_type.lower()}_{ts}.png"
    clean_fig.write_image(str(clean_path), scale=2)
    print(f"    Saved: {clean_path}")
    
    # 4. Summary table
    print("  [4/6] Creating summary table...")
    table_fig = create_summary_table(results, model_info)
    table_path = output_dir / f"summary_table_{model_type.lower()}_{ts}.png"
    table_fig.write_image(str(table_path), scale=2)
    print(f"    Saved: {table_path}")
    
    # 5. Combined interactive dashboard
    print("  [5/6] Creating interactive dashboard...")
    dashboard_fig = create_combined_dashboard(results, model_info)
    dashboard_path = output_dir / f"dashboard_{model_type.lower()}_{ts}.html"
    dashboard_fig.write_html(str(dashboard_path))
    print(f"    Saved: {dashboard_path}")
    
    # 6. Save individual plots as HTML for interactivity
    print("  [6/6] Saving interactive HTML versions...")
    deg_fig.write_html(str(output_dir / f"energy_degradation_{model_type.lower()}_{ts}.html"))
    abs_fig.write_html(str(output_dir / f"absolute_energy_{model_type.lower()}_{ts}.html"))
    
    print(f"\n✓ All visualizations saved to: {output_dir}")
    print(f"\nSummary Statistics:")
    
    # Print clean baseline if available
    clean_baseline = results.get("clean_baseline", {})
    if clean_baseline:
        print(f"  Clean Baseline (no noise, n={clean_baseline.get('n_segments', 0)} segments):")
        print(f"    Original RMS: {clean_baseline.get('mean_original_rms_db', 0):.2f} dBFS")
        print(f"    Original SEL: {clean_baseline.get('mean_original_sel_db', 0):.2f} dBFS")
        print(f"    RMS preservation: {clean_baseline.get('mean_rms_preservation_db', 0):+.2f} dB")
        print(f"    SEL preservation: {clean_baseline.get('mean_sel_preservation_db', 0):+.2f} dB")
    
    summary = results.get("summary", {})
    if summary:
        print(f"  Energy Degradation Range:")
        print(f"    RMS: {summary.get('min_rms_degradation_db', 0):+.2f} to {summary.get('max_rms_degradation_db', 0):+.2f} dB")
        print(f"    SEL: {summary.get('min_sel_degradation_db', 0):+.2f} to {summary.get('max_sel_degradation_db', 0):+.2f} dB")
    
    dataset_stats = results.get("dataset_stats", {})
    if dataset_stats:
        n_coi = dataset_stats.get("n_coi_samples", "?")
        print(f"  COI samples evaluated: {n_coi}")


if __name__ == "__main__":
    main()
