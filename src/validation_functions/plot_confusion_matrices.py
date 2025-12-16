"""
Script to visualize confusion matrices from validation results using Plotly.
"""

import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def load_results(results_path: str | Path) -> dict:
    """Load validation results from JSON file."""
    with open(results_path, "r") as f:
        return json.load(f)


def create_confusion_matrix_figure(
    cm_data: dict, title: str, show_percentages: bool = True
) -> go.Figure:
    """
    Create a Plotly confusion matrix heatmap.

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
    # Rows = Actual (Negative, Positive), Cols = Predicted (Negative, Positive)
    cm = np.array([[tn, fp], [fn, tp]])

    total = cm.sum()

    # Create text annotations
    if show_percentages:
        text = [
            [
                f"TN: {tn}<br>({tn/total*100:.1f}%)",
                f"FP: {fp}<br>({fp/total*100:.1f}%)",
            ],
            [
                f"FN: {fn}<br>({fn/total*100:.1f}%)",
                f"TP: {tp}<br>({tp/total*100:.1f}%)",
            ],
        ]
    else:
        text = [[f"TN: {tn}", f"FP: {fp}"], [f"FN: {fn}", f"TP: {tp}"]]

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
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
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title="Predicted Label", side="bottom"),
        yaxis=dict(title="Actual Label", autorange="reversed"),
        width=500,
        height=450,
    )

    return fig


def create_combined_figure(results: dict) -> go.Figure:
    """
    Create a combined figure with all confusion matrices, ensuring consistent scale.

    Args:
        results: Dictionary containing all test results

    Returns:
        Plotly Figure object with subplots
    """
    test_names = list(results.keys())
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
    max_value = max(max(data["confusion_matrix"].values()) for data in results.values())

    for idx, (name, data) in enumerate(results.items()):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        cm_data = data["confusion_matrix"]
        tp = cm_data["tp"]
        tn = cm_data["tn"]
        fp = cm_data["fp"]
        fn = cm_data["fn"]

        cm = np.array([[tn, fp], [fn, tp]])
        total = cm.sum()

        text = [
            [
                f"TN: {tn}<br>({tn/total*100:.1f}%)",
                f"FP: {fp}<br>({fp/total*100:.1f}%)",
            ],
            [
                f"FN: {fn}<br>({fn/total*100:.1f}%)",
                f"TP: {tp}<br>({tp/total*100:.1f}%)",
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

    fig.update_layout(
        title=dict(
            text="Confusion Matrices - Classifying Airplanes", font=dict(size=20)
        ),
        width=900,
        height=800,
    )

    return fig


def create_grouped_bar_chart(results: dict) -> go.Figure:
    """
    Create a grouped bar chart showing key metrics for clean audios and mixtures.

    Args:
        results: Dictionary containing all test results

    Returns:
        Plotly Figure object with grouped bar chart
    """
    # Define important metrics and labels
    important_metrics = ["accuracy", "precision", "recall", "f1_score"]
    metrics_labels = [m.replace("_", " ").title() for m in important_metrics]

    # Expected result keys
    clean_class = "Clean sounds classification"
    clean_sep = "Clean sounds separation + classification"
    mix_class = "Mixtures classification"
    mix_sep = "Mixtures separation + classification"

    # Colors for the two bar types
    colors = {"classification": "royalblue", "separation": "orange"}

    # Grid layout (2x2) for the 4 metrics
    from plotly.subplots import make_subplots

    n_metrics = len(important_metrics)
    n_rows = 2
    n_cols = 2
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=["" for _ in range(n_metrics)],
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
    )

    categories = ["Clean", "Mixture"]

    for idx, (metric, metric_label) in enumerate(
        zip(important_metrics, metrics_labels)
    ):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        # Get values safely (default 0 if missing)
        c_class = results.get(clean_class, {}).get(metric, 0)
        c_sep = results.get(clean_sep, {}).get(metric, 0)
        m_class = results.get(mix_class, {}).get(metric, 0)
        m_sep = results.get(mix_sep, {}).get(metric, 0)

        # Two traces: classification and separation+classification
        fig.add_trace(
            go.Bar(
                x=[c_class, m_class],
                y=categories,
                name="Classification",
                marker_color=colors["classification"],
                orientation="h",
                text=[f"{c_class:.3f}", f"{m_class:.3f}"],
                textposition="outside",
                showlegend=(idx == 0),
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Bar(
                x=[c_sep, m_sep],
                y=categories,
                name="Separation+Classification",
                marker_color=colors["separation"],
                orientation="h",
                text=[f"{c_sep:.3f}", f"{m_sep:.3f}"],
                textposition="outside",
                showlegend=(idx == 0),
            ),
            row=row,
            col=col,
        )

        # y-axis labels should show the category names
        fig.update_yaxes(title_text=None, row=row, col=col)
        fig.update_xaxes(title_text=metric_label, row=row, col=col)

    fig.update_layout(
        title=dict(text="Key Metrics by Test Type", font=dict(size=20)),
        width=1100,
        height=700,
        barmode="group",
        legend_title_text="Test Type",
        showlegend=True,
    )
    return fig


def main():
    # Path to results file
    results_dir = Path(__file__).parent / "validation_results"
    results_file = results_dir / "results_20251216_094131.json"

    # Load results
    results = load_results(results_file)

    print(f"Loaded results for {len(results)} test executions:")
    for name in results.keys():
        print(f"  - {name}")

    # Create output directory for plots
    output_dir = results_dir / "plots"
    output_dir.mkdir(exist_ok=True)

    # Create individual confusion matrix figures
    for name, data in results.items():
        fig = create_confusion_matrix_figure(
            data["confusion_matrix"],
            title=f"Confusion Matrix - {name.replace('_', ' ').title()}",
        )

        # Save as PNG image
        output_path = output_dir / f"confusion_matrix_{name}.png"
        fig.write_image(str(output_path), scale=2)
        print(f"Saved: {output_path}")

    # Create combined figure with all confusion matrices
    combined_fig = create_combined_figure(results)
    combined_path = output_dir / "confusion_matrices_combined.png"
    combined_fig.write_image(str(combined_path), scale=2)
    print(f"Saved: {combined_path}")

    # Create and save a combined grouped bar chart for all metrics
    combined_bar_fig = create_grouped_bar_chart(results)
    combined_bar_path = output_dir / "bar_chart_metrics_combined.png"
    combined_bar_fig.write_image(str(combined_bar_path), scale=2)
    print(f"Saved: {combined_bar_path}")


if __name__ == "__main__":
    main()
