# Validation pipeline

Evaluates separation model effectiveness via downstream classification metrics.

## COI synonym system

The pipeline automatically re-binarizes dataset labels based on which classifier you use:

| `classifier_type` | COI synonyms used | Detects |
|-------------------|-------------------|---------|
| `"plane"`, `"pann"`, `"ast"`, `"pann_finetuned"` | `AIRPLANE_SYNONYMS` | Airplane/aircraft |
| `"bird_mae"`, `"birdnet"` | `BIRD_SYNONYMS` | Birds/avian |

The same dataset CSV can be used for both — COI synonyms are swapped automatically. Synonyms are defined in `src/label_loading/coi_labels.py`. Pass custom synonyms via `ValidationPipeline(coi_synonyms={...})`.

## Class balancing

Enabled by default — downsamples the majority class when imbalance ratio exceeds 2.0. This produces fair confusion matrices. Disable with `balance_classes=False`.

## Test stages (standard set)

1. **Clean COI — Classification Only** — Baseline, no interference
2. **Clean COI — Separation + Classification** — Separator preserves COI?
3. **Synthetic Mixtures — Classification Only** — Baseline with interference
4. **Synthetic Mixtures — Separation + Classification** — Separator removes interference?

Risoux is always excluded from synthetic mixtures and evaluated separately as-is.

## Running validation

The unified `scripts/run_validation.py` supports all separators and classifiers:

```bash
# TUSS airplane
python scripts/run_validation.py --separator tuss --tuss-checkpoint path/to/best_model.pt --data-csv path/to/dataset.csv --classifier plane

# TUSS birds
python scripts/run_validation.py --separator tuss --tuss-checkpoint path/to/best_model.pt --tuss-coi-prompt birds --data-csv path/to/dataset.csv --classifier bird_mae audioprotopnet

# CLAPSep airplane (PANN + AST)
python scripts/run_validation.py --separator clapsep --clapsep-checkpoint path/to/best_model.ckpt --data-csv path/to/dataset.csv --classifier pann_finetuned ast_finetuned
```

## Noise robustness experiment

Evaluates separation benefit across SNR levels:

```bash
cd src/validation_functions
# Edit SNR/MAX_SAMPLES/classifier config in test_noise_increase.py
python test_noise_increase.py
python plot_noise_increase_results.py  # Auto-detects latest results
```

Outputs go to `noise_increase_results/`: JSON data, CSV table, recall-vs-SNR plots, interactive HTML dashboard.

## Dataset format

CSV must include `orig_label` column with raw label strings:

```csv
filename,label,orig_label,split
audio1.wav,1,"airplane",train
audio2.wav,0,"wind",test
```

## Key files

| File | Purpose |
|------|---------|
| `test_pipeline.py` | Main validation pipeline (COI re-binarization, contamination filtering, metric computation) |
| `test_noise_increase.py` | SNR sweep experiment |
| `metrics.py` | Metrics container and contamination filtering |
| `demo_separation.py` | Spectrogram visualization utilities |
| `classification_models/` | Classifier wrappers (see `classification_models/README.md`) |
