# Noise Increase Experiment - Usage Guide

This guide explains how to run the noise increase experiment and visualize the results.

## Overview

The noise increase experiment evaluates how separation improves classification robustness across different Signal-to-Noise Ratio (SNR) levels by:
1. Adding artificial white noise to COI (Class of Interest) samples
2. Testing classification with and without separation
3. Measuring recall (detection rate) at each SNR level

## Files

- **`test_noise_increase.py`** - Main experiment script
- **`plot_noise_increase_results.py`** - Visualization script
- **`noise_increase_results/`** - Output directory for results
- **`noise_increase_results/plots/`** - Generated visualizations

## Running the Experiment

### 1. Configure the Experiment

Edit the configuration section in `test_noise_increase.py` (lines 197-248):

```python
# Model selection
USE_TUSS = False  # Set True for TUSS, False for SudoRM-RF

# Dataset paths
DATA_CSV = str(PROJECT_ROOT / "path/to/separation_dataset.csv")
SEP_CHECKPOINT = str(PROJECT_ROOT / "path/to/best_model.pt")
CLS_WEIGHTS = str(PROJECT_ROOT / "path/to/final_model.weights.h5")

# Experiment parameters
SNR_START = 25      # High SNR (easy conditions)
SNR_END = -10       # Low SNR (challenging conditions)
NUM_STEPS = 8       # Number of SNR levels to test
MAX_SAMPLES = 200   # Maximum COI samples to use (None = all)
SEED = 42          # Random seed for reproducibility
```

### 2. Run the Experiment

```bash
# Activate virtual environment
cd /home/bendm/Thesis/project/code
source .venv/bin/activate

# Run experiment
cd src/validation_functions
python test_noise_increase.py
```

### 3. Monitor Progress

The script will display:
- Dataset statistics (COI samples, background samples, contamination filtering)
- SNR sweep configuration
- Progress for each SNR level

### 4. Output Files

Results are saved to `noise_increase_results/`:
- **JSON**: `noise_increase_results_artificial_{model}_{timestamp}.json` - Full results with metadata
- **CSV**: `noise_increase_results_artificial_{model}_{timestamp}.csv` - SNR sweep data table

## Visualizing Results

### Option 1: Auto-detect Latest Results

```bash
python plot_noise_increase_results.py
```

This automatically finds and plots the most recent results file.

### Option 2: Specify Results File

```bash
python plot_noise_increase_results.py noise_increase_results/noise_increase_results_artificial_sudormrf_20260322_123456.json
```

### Generated Visualizations

The script creates 5 visualizations in `noise_increase_results/plots/`:

1. **`recall_vs_snr_*.png`** - Main comparison plot
   - Shows recall vs SNR for classification-only (blue) and separation+classification (orange)
   - Green shaded area highlights separation improvement
   
2. **`confidence_vs_snr_*.png`** - Confidence trends
   - Shows mean classifier confidence at each SNR level
   - Error bars show standard deviation
   
3. **`separation_gain_*.png`** - Bar chart
   - Shows recall improvement per SNR level
   - Green bars = positive gain, Red bars = negative (rare)
   
4. **`summary_table_*.png`** - Detailed metrics table
   - Recall, confidence, and actual SNR for each level
   - Color-coded improvement column
   
5. **`dashboard_*.html`** - Interactive dashboard
   - Combined view with all plots
   - Hover for detailed values
   - Open in web browser for exploration

## Understanding the Results

### Key Metrics

- **Recall (True Positive Rate)**: Fraction of COI samples correctly detected
- **Confidence**: Mean classifier confidence score (0-1)
- **Separation Gain**: Improvement in recall when using separation
- **Actual SNR**: Achieved SNR after noise scaling (may differ from target due to clamping)

### Interpreting Plots

**High SNR (e.g., 25 dB)**:
- Both methods perform well (recall ≈ 0.95-1.0)
- Small separation gain (noise is minimal)

**Medium SNR (e.g., 0-10 dB)**:
- Classification-only performance degrades
- Separation provides increasing benefit
- **Peak separation gain typically occurs here**

**Low SNR (e.g., -10 to -20 dB)**:
- Both methods struggle (recall < 0.5)
- Separation still helps but with diminishing returns
- Noise overwhelms signal

### Summary Statistics

Printed at the end and in the table:
- **Best separation gain**: Maximum recall improvement across all SNR levels
- **Mean separation gain**: Average improvement across all tested SNRs
- **Contaminated backgrounds removed**: Number of background samples filtered out (should be 0 for clean datasets)

## Contamination Filtering

The experiment now includes the same contamination filtering as `test_pipeline.py`:

### What It Does

Checks background samples for COI synonyms (e.g., "plane", "airplane", "aircraft") in the `orig_label` column and removes them. This prevents:
- False negatives when contaminated backgrounds are misclassified as negative
- Inflated performance metrics from incorrectly labeled data

### When It Applies

- **Current experiment**: Uses artificial white noise, so filtering has no effect on the experiment itself
- **Future extensions**: If you modify the code to use real background samples, filtering becomes critical

### Output

The script reports:
```
Dataset Statistics:
  COI samples:        200
  Background samples: 450 total, 445 clean
  Contaminated removed: 5
```

If contamination is found, detailed breakdown by split (train/val/test) and example labels are shown.

## Tips

### Choosing SNR Range

- **Wide range (-20 to +25 dB)**: Shows full performance curve, good for publications
- **Narrow range (-5 to +10 dB)**: Focuses on realistic conditions where separation matters most
- **More steps (12-16)**: Smoother curves but slower runtime
- **Fewer steps (6-8)**: Faster, sufficient for initial experiments

### Sample Size

- **MAX_SAMPLES = 200**: Fast, good for testing and iteration
- **MAX_SAMPLES = None**: Uses all available COI samples, better statistics but slower

### Multiple Models

To compare models:
1. Run experiment with SudoRM-RF (`USE_TUSS = False`)
2. Run experiment with TUSS (`USE_TUSS = True`)
3. Load both JSON files and compare side-by-side

## Troubleshooting

### "No COI samples available"

Check that your CSV has samples with `label=1` or `orig_label` containing COI synonyms.

### "No results files found"

Ensure you've run `test_noise_increase.py` first. Results must match pattern `noise_increase_results_*.json`.

### Plots look weird

Check that:
- Recall values are in range [0, 1]
- SNR levels are ordered correctly
- All required fields are present in JSON

### Out of memory

Reduce `MAX_SAMPLES` or test fewer SNR levels (`NUM_STEPS`).

## Example Workflow

```bash
# 1. Edit configuration in test_noise_increase.py
vim test_noise_increase.py

# 2. Run experiment
python test_noise_increase.py

# 3. Visualize results (auto-detect latest)
python plot_noise_increase_results.py

# 4. Open interactive dashboard
firefox noise_increase_results/plots/dashboard_*.html

# 5. Use PNG plots in paper/presentation
ls noise_increase_results/plots/*.png
```

## Citation

If you use this experiment in your research, please cite both:
- The separation model (SudoRM-RF or TUSS)
- The plane classifier

See the main project README for citation details.
