"""
Script to visualize confusion matrices from validation results using Plotly.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_results(results_path: str | Path) -> dict:
    """Load validation results from JSON file."""
    with open(results_path, "r") as f:
        return json.load(f)


def create_confusion_matrix_figure(
    cm_data: dict, title: str, show_percentages: bool = True, model_info: dict = None
) -> go.Figure:
    """
    Create a Plotly confusion matrix heatmap.

    The figure now contains only the heatmap itself; any auxiliary table showing
    misclassification counts has been removed.

    Args:
        cm_data: Dictionary with tp, tn, fp, fn values
        title: Title for the plot
        show_percentages: Whether to show percentages alongside counts

    Returns:
        Plotly Figure object
    """
    # Extract values
    tp = cm_data["tp"]
    tn = cm_data["tn"]
    fp = cm_data["fp"]
    fn = cm_data["fn"]

    # Create confusion matrix array
    # Format: [[TN, FP], [FN, TP]]
    cm = np.array([[tn, fp], [fn, tp]])
    total = cm.sum()

    # Create heatmap text annotations
    if show_percentages:
        text = [
            [
                f"TN: {tn}<br>({tn / total * 100:.1f}%)",
                f"FP: {fp}<br>({fp / total * 100:.1f}%)",
            ],
            [
                f"FN: {fn}<br>({fn / total * 100:.1f}%)",
                f"TP: {tp}<br>({tp / total * 100:.1f}%)",
            ],
        ]
    else:
        text = [[f"TN: {tn}", f"FP: {fp}"], [f"FN: {fn}", f"TP: {tp}"]]

    # the small misclassification‑count table is no longer part of the figure,
    # so we drop the previous logic that assembled labels and counts.

    # construct figure containing only the heatmap
    fig = go.Figure()

    # add heatmap trace
    heatmap = go.Heatmap(
        z=cm,
        x=["Predicted Negative", "Predicted Positive"],
        y=["Actual Negative", "Actual Positive"],
        text=text,
        texttemplate="%{text}",
        textfont={"size": 14},
        colorscale="Blues",
        showscale=True,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    )
    fig.add_trace(heatmap)

    full_title = title
    if model_info:
        separator = model_info.get("separator", "")
        classifier = model_info.get("classifier", "")
        if separator and classifier:
            full_title += f" ({separator} + {classifier})"

    fig.update_layout(
        title=dict(text=full_title, font=dict(size=16)),
        width=700,
        height=450,
    )

    # update only the heatmap axes
    fig.update_xaxes(title="Predicted Label", side="bottom")
    fig.update_yaxes(title="Actual Label", autorange="reversed")

    return fig


def create_combined_figure(results: dict, model_info: dict = None) -> go.Figure:
    """
    Create a combined figure with all confusion matrices, ensuring consistent scale.

    Misclassification counts are not shown; previous versions added text
    annotations under each subplot describing false positives/negatives, but
    those have been removed.

    Args:
        results: Dictionary containing all test results
        model_info: Optional dictionary with model information

    Returns:
        Plotly Figure object with subplots
    """
    # Filter to only include test results (exclude checkpoint_paths, etc.)
    test_names = [
        k
        for k in results.keys()
        if isinstance(results[k], dict) and "confusion_matrix" in results[k]
    ]
    n_tests = len(test_names)

    # Calculate grid dimensions
    n_cols = 2
    n_rows = (n_tests + 1) // 2

    # Create subplots
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[name.replace("_", " ").title() for name in test_names],
        horizontal_spacing=0.15,
        vertical_spacing=0.15,
    )

    # Determine consistent z-axis range across all confusion matrices
    max_value = max(
        max(data["confusion_matrix"].values())
        for data in results.values()
        if isinstance(data, dict) and "confusion_matrix" in data
    )

    for idx, name in enumerate(test_names):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        data = results[name]
        cm_data = data["confusion_matrix"]
        tp = cm_data["tp"]
        tn = cm_data["tn"]
        fp = cm_data["fp"]
        fn = cm_data["fn"]

        cm = np.array([[tn, fp], [fn, tp]])
        total = cm.sum()

        text = [
            [
                f"TN: {tn}<br>({tn / total * 100:.1f}%)",
                f"FP: {fp}<br>({fp / total * 100:.1f}%)",
            ],
            [
                f"FN: {fn}<br>({fn / total * 100:.1f}%)",
                f"TP: {tp}<br>({tp / total * 100:.1f}%)",
            ],
        ]

        heatmap = go.Heatmap(
            z=cm,
            x=["Pred Neg", "Pred Pos"],
            y=["Act Neg", "Act Pos"],
            text=text,
            texttemplate="%{text}",
            textfont={"size": 11},
            colorscale="Blues",
            zmin=0,
            zmax=max_value,  # Ensure consistent scale
            showscale=idx == 0,  # Only show colorscale for first plot
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        )

        fig.add_trace(heatmap, row=row, col=col)

        # Update axes for this subplot
        fig.update_xaxes(title_text="Predicted", row=row, col=col)
        fig.update_yaxes(title_text="Actual", autorange="reversed", row=row, col=col)

        # previously the function added a text annotation below each subplot
        # listing misclassification counts.  Those annotations have been removed
        # so the heatmap alone is shown.

    title_text = "Confusion Matrices - Classifying Trains"
    if model_info:
        separator = model_info.get("separator", "")
        classifier = model_info.get("classifier", "")
        if separator and classifier:
            title_text += f" ({separator} + {classifier})"

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=20)),
        width=900,
        height=800,
        # give extra room at the bottom so our mis‑classification annotations
        # added below each subplot aren’t cut off
        margin=dict(b=100),
    )

    return fig


def create_top_misclassified_figure(
    results: dict, model_info: dict = None
) -> go.Figure:
    """
    Create a combined bar chart showing the top 10 misclassified raw labels
    for each test. Each subplot corresponds to one run/test and displays the
    ten labels with the highest misclassification counts (from
    `misclassified_raw_counts`).

    Args:
        results: Dictionary containing all test results
        model_info: Optional dictionary with model information

    Returns:
        Plotly Figure object with subplots.
    """
    # Filter to only include test results (exclude checkpoint_paths, etc.)
    test_names = [
        k
        for k in results.keys()
        if isinstance(results[k], dict) and "confusion_matrix" in results[k]
    ]
    n_tests = len(test_names)

    # Calculate grid dimensions
    n_cols = 2
    n_rows = (n_tests + 1) // 2

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[name.replace("_", " ").title() for name in test_names],
        horizontal_spacing=0.25,
        vertical_spacing=0.15,
    )

    for idx, name in enumerate(test_names):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        data = results[name]
        mis_raw = data.get("misclassified_raw_counts", {})
        if mis_raw:
            items = sorted(mis_raw.items(), key=lambda kv: kv[1], reverse=True)[:10]
            labels = [str(k) for k, _ in items]
            counts = [v for _, v in items]
        else:
            labels = []
            counts = []

        bar = go.Bar(
            x=counts,
            y=labels,
            orientation="h",
            marker_color="darkorange",
            text=counts,
            textposition="auto",
            showlegend=False,
        )
        fig.add_trace(bar, row=row, col=col)
        fig.update_xaxes(title_text="Count", row=row, col=col)
        # reverse y‑axis so highest values appear at the top
        fig.update_yaxes(autorange="reversed", row=row, col=col)

    title_text = "Top misclassified labels"
    if model_info:
        separator = model_info.get("separator", "")
        classifier = model_info.get("classifier", "")
        if separator and classifier:
            title_text += f" ({separator} + {classifier})"

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=20)),
        width=1100,
        height=800,
    )

    return fig


def create_metrics_table(results: dict, model_info: dict = None) -> go.Figure:
    """
    Create a table showing all metrics for each test.

    Args:
        results: Dictionary containing all test results
        model_info: Optional dictionary with model information

    Returns:
        Plotly Figure object with table
    """
    # Filter to only include test results
    test_names = [
        k
        for k in results.keys()
        if isinstance(results[k], dict) and "confusion_matrix" in results[k]
    ]

    # Metrics to display
    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "specificity",
        "balanced_accuracy",
        "mcc",
    ]

    # Signal metrics for separation tests
    signal_metrics = [
        "mean_si_snr_db",
        "mean_sdr_db",
        "mean_si_sdr_db",
    ]

    # Prepare table data
    rows = []
    for name in test_names:
        data = results[name]
        row = [name.replace("_", " ").title()]
        for metric in metrics:
            value = data.get(metric, 0)
            row.append(f"{value:.4f}")

        # Add signal metrics if they exist (for separation runs)
        if "signal_metrics" in data:
            signal_data = data["signal_metrics"]
            for sig_metric in signal_metrics:
                value = signal_data.get(sig_metric, 0)
                row.append(f"{value:.2f}")
        else:
            # Add empty cells for non-separation runs
            for _ in signal_metrics:
                row.append("-")

        rows.append(row)

    # Build header
    header_values = ["Test"] + [m.replace("_", " ").title() for m in metrics]
    header_values += [m.replace("_", " ").title() for m in signal_metrics]

    # Create cell fill colors: different color for signal metrics columns
    num_metric_cols = len(metrics) + 1  # +1 for test name
    num_signal_cols = len(signal_metrics)

    cell_fill_colors = []
    for row_idx in range(len(rows)):
        row_colors = ["lavender"] * num_metric_cols + ["lightgreen"] * num_signal_cols
        cell_fill_colors.append(row_colors)

    # Transpose for table format
    cell_fill_transposed = list(zip(*cell_fill_colors)) if cell_fill_colors else []

    # Create table
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=header_values,
                    fill_color=["paleturquoise"] * num_metric_cols
                    + ["lightseagreen"] * num_signal_cols,
                    align="center",
                    font=dict(color="black", size=11),
                ),
                cells=dict(
                    values=list(zip(*rows))
                    if rows
                    else [
                        [],
                    ]
                    * len(header_values),
                    fill_color=cell_fill_transposed,
                    align="center",
                    font=dict(color="black", size=10),
                ),
            )
        ]
    )

    title_text = "Performance Metrics by Test Type"
    if model_info:
        separator = model_info.get("separator", "")
        classifier = model_info.get("classifier", "")
        if separator and classifier:
            title_text += f" ({separator} + {classifier})"

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=16)),
        height=300,
        width=1600,
    )

    return fig


def create_metrics_bar_chart(results: dict, model_info: dict = None) -> go.Figure:
    """
    Create a grouped bar chart showing key metrics across all tests.

    Args:
        results: Dictionary containing all test results
        model_info: Optional dictionary with model information

    Returns:
        Plotly Figure object with grouped bar chart
    """
    # Filter to only include test results
    test_names = [
        k
        for k in results.keys()
        if isinstance(results[k], dict) and "confusion_matrix" in results[k]
    ]

    # Important metrics to display
    important_metrics = ["accuracy", "precision", "recall", "f1_score"]
    metrics_labels = [m.replace("_", " ").title() for m in important_metrics]

    # Prepare data
    test_labels = [name.replace("_", " ").title() for name in test_names]

    # Create subplots for each metric
    n_metrics = len(important_metrics)
    n_rows = 2
    n_cols = 2

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=metrics_labels,
        horizontal_spacing=0.12,
        vertical_spacing=0.15,
    )

    colors = ["royalblue", "orange", "green", "red"]

    for idx, metric in enumerate(important_metrics):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        # Get values for this metric across all tests
        values = []
        for name in test_names:
            value = results[name].get(metric, 0)
            values.append(value)

        # Add bar trace
        fig.add_trace(
            go.Bar(
                x=test_labels,
                y=values,
                name=metric,
                marker_color=colors[idx],
                text=[f"{v:.3f}" for v in values],
                textposition="outside",
                showlegend=False,
                hovertemplate="<b>%{x}</b><br>"
                + metric.replace("_", " ").title()
                + ": %{y:.4f}<extra></extra>",
            ),
            row=row,
            col=col,
        )

        # Update axes
        fig.update_xaxes(tickangle=-45, row=row, col=col)
        fig.update_yaxes(range=[0, 1], row=row, col=col)

    title_text = "Key Metrics Across All Tests"
    if model_info:
        separator = model_info.get("separator", "")
        classifier = model_info.get("classifier", "")
        if separator and classifier:
            title_text += f" ({separator} + {classifier})"

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=18)),
        width=1000,
        height=700,
        showlegend=False,
    )

    return fig


def extract_model_info(results: dict) -> dict:
    """
    Extract model names from checkpoint paths.

    Args:
        results: Dictionary containing all test results and checkpoint_paths

    Returns:
        Dictionary with model information
    """
    checkpoint_paths = results.get("checkpoint_paths", {})

    model_info = {}

    # Extract separator model name
    sep_path = checkpoint_paths.get("separator", "")
    if sep_path:
        # Normalize path separators
        normalized_path = sep_path.replace("\\", "/")
        path_parts = normalized_path.split("/")

        # Look for known model names
        if "CLAPSep" in sep_path:
            model_info["separator"] = "CLAPSep"
        elif "sudormrf" in normalized_path.lower():
            # Extract timestamp for version differentiation
            timestamp = None
            for i, part in enumerate(path_parts):
                if part.lower() == "sudormrf" and i + 2 < len(path_parts):
                    # Check if next directory after sudormrf/checkpoints is a timestamp
                    potential_timestamp = path_parts[i + 2]
                    if (
                        potential_timestamp
                        and len(potential_timestamp) == 15
                        and potential_timestamp.replace("_", "").isdigit()
                    ):
                        timestamp = potential_timestamp
                        break

            model_info["separator"] = (
                f"SudoRMRF_{timestamp}" if timestamp else "SudoRMRF"
            )
        else:
            # Fallback: find the first directory in models or similar pattern
            # Try to find a model name in the path
            for i, part in enumerate(path_parts):
                if part.lower() in ["models"]:
                    # Next part should be the model name
                    if i + 1 < len(path_parts):
                        model_info["separator"] = path_parts[i + 1].title()
                    break

            # If still not found, use last meaningful directory before checkpoint/checkpoints
            if "separator" not in model_info:
                for i, part in enumerate(path_parts):
                    if part.lower() in ["checkpoint", "checkpoints"]:
                        if i > 0:
                            model_info["separator"] = path_parts[i - 1].title()
                        break

    # Extract classifier model name
    clf_path = checkpoint_paths.get("classifier", "")
    if clf_path:
        # Extract model name from filename
        if (
            "plane_classifier" in clf_path.lower()
            or "plane_clasifier" in clf_path.lower()
        ):
            model_info["classifier"] = "PlaneClassifier"
        else:
            # Fallback: extract from directory name before checkpoints
            normalized_path = clf_path.replace("\\", "/")
            path_parts = normalized_path.split("/")
            for i, part in enumerate(path_parts):
                if part.lower() in ["checkpoint", "checkpoints"]:
                    if i > 0:
                        model_info["classifier"] = path_parts[i - 1].title()
                    break

    return model_info


def main():
    # Path to results file
    results_dir = Path(__file__).parent / "validation_examples_planes"
    results_file = results_dir / "results_20260222_231459.json"

    #  SudoRMRF
    # "results_20260211_003754.json"
    # "results_20260210_234618.json"
    # Clapsep
    # "results_20260211_111014.json"
    # "results_20260211_232530.json"

    # Load results
    results = load_results(results_file)

    print(f"Loaded results for {len(results)} entries:")
    for name in results.keys():
        print(f"  - {name}")

    # Extract model information
    model_info = extract_model_info(results)
    model_dir_name = (
        f"{model_info.get('separator', 'Sep')}_{model_info.get('classifier', 'Clf')}"
        if model_info
        else "plots"
    )

    # Create output directory for plots with model names
    output_dir = results_dir / "plots" / model_dir_name
    output_dir.mkdir(exist_ok=True, parents=True)

    # Filter to only include test results (exclude checkpoint_paths, etc.)
    test_results = {
        k: v
        for k, v in results.items()
        if isinstance(v, dict) and "confusion_matrix" in v
    }

    # Create individual confusion matrix figures
    for name, data in test_results.items():
        test_label = name.replace("_", " ").title()
        title = f"Confusion Matrix - {test_label}"
        if model_info:
            title += f" ({model_info.get('separator', '')} + {model_info.get('classifier', '')})"

        fig = create_confusion_matrix_figure(
            data["confusion_matrix"],
            title=title,
            model_info=model_info,
        )

        # Save as PNG image (model info in directory name)
        output_path = output_dir / f"confusion_matrix_{name}.png"
        fig.write_image(str(output_path), scale=2)
        print(f"Saved: {output_path}")

    # Create combined figure with all confusion matrices
    combined_fig = create_combined_figure(test_results, model_info)
    combined_path = output_dir / "confusion_matrices_combined.png"
    combined_fig.write_image(str(combined_path), scale=2)
    print(f"Saved: {combined_path}")

    # Create metrics table
    metrics_table_fig = create_metrics_table(test_results, model_info)
    metrics_table_path = output_dir / "metrics_table.png"
    metrics_table_fig.write_image(str(metrics_table_path), scale=2)
    print(f"Saved: {metrics_table_path}")

    # Create metrics bar chart
    metrics_bar_fig = create_metrics_bar_chart(test_results, model_info)
    metrics_bar_path = output_dir / "metrics_bar_chart.png"
    metrics_bar_fig.write_image(str(metrics_bar_path), scale=2)
    print(f"Saved: {metrics_bar_path}")

    # Create top misclassified labels figure (combined across runs)
    top_mis_fig = create_top_misclassified_figure(test_results, model_info)
    top_mis_path = output_dir / "top_misclassified_labels.png"
    top_mis_fig.write_image(str(top_mis_path), scale=2)
    print(f"Saved: {top_mis_path}")


if __name__ == "__main__":
    main()
