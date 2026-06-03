# Sound Separation & Classification

Audio source separation and classification pipeline — separating
airplane noise and bird sounds from field recordings.

Master thesis — Machine Learning for Biodiversity.

## Setup

```bash
# Requires Python 3.11–3.12 and Poetry
poetry install
```

## Environment

Optional environment variables (defaults are relative to project root):

| Variable | Default | Purpose |
|---|---|---|
| `THESIS_DATA_DIR` | `../datasets` | Root of audio datasets |
| `THESIS_OUTPUT_DIR` | `<root>/outputs` | Output directory |
| `THESIS_CHECKPOINTS_DIR` | `<root>/checkpoints` | Model checkpoints |

Or use `src/common/paths.py` to override defaults in code.

## Project Structure

```
├── configs/            YAML configs for hyperparameter tuning
├── scripts/            Utility and HPC scripts
│   ├── analysis/       Analysis & visualization
│   ├── data/           Data preparation (labels, webdataset)
│   ├── debug/          Debugging & tracing
│   ├── demo/           Demo entry points
│   ├── diagnostics/    Diagnostics & reports
│   ├── hpc/            HPC cluster job scripts (PBS)
│   ├── tuning/         Optuna hyperparameter tuning
│   └── run_validation.py  Unified validation runner
├── src/                Main package
│   ├── common/         Shared utilities (audio, paths, augmentations)
│   ├── activity_filter/  Audio activity detection
│   ├── label_loading/  Dataset-specific label loaders (10+ datasets)
│   ├── models/         Separation models (see per-model README)
│   │   ├── base.py     Shared BaseSeparator ABC
│   │   ├── clapsep/    CLAPSep (text-prompt separation)
│   │   ├── sudormrf/   SuDoRM-RF (time-domain separation)
│   │   └── tuss/       TUSS (universal source separation)
│   ├── orchestration/  Training orchestration & runner
│   ├── pipeline/       End-to-end separation pipeline
│   ├── validation_functions/  Validation & diagnostics
│   └── tag_analysis/   Tag/label analysis
└── tests/              Test suite
    ├── conftest.py     Shared fixtures & path setup
    ├── audio/          Audio processing tests
    ├── data/           Data loading & resampling tests
    ├── models/         Model inference/training tests
    ├── pipeline/       Pipeline & contamination tests
    └── validation/     Validation & mask recycling tests
```

## Training

Models are trained via their respective train scripts, e.g.:

```bash
python src/models/tuss/train.py --config src/models/tuss/training_config.yaml
python src/models/sudormrf/train.py --config src/models/sudormrf/training_config.yaml
python src/models/clapsep/train_text_coi.py --config src/models/clapsep/training_config.yaml
```

Each model directory has a `README.md` with usage details.

HPC cluster jobs are defined in `scripts/hpc/` (PBS format). See `scripts/hpc/README.md` for setup.

## Inference

Each model provides an inference class inheriting from `BaseSeparator`:

```python
from models.tuss.inference import TUSSInference

tuss = TUSSInference.from_checkpoint("checkpoints/best_model.pt")
sources = tuss.separate("audio.wav")
coi = tuss.get_coi_audio(sources)
tuss.save_audio(coi, "separated_coi.wav")
```

## Validation

Unified validation runner (replaces 9 individual scripts):

```bash
# TUSS airplane — single plane CNN classifier
python scripts/run_validation.py \
    --separator tuss \
    --tuss-checkpoint path/to/best_model.pt \
    --data-csv path/to/dataset.csv \
    --classifier plane

# CLAPSep airplane — PANN + AST (both classifiers)
python scripts/run_validation.py \
    --separator clapsep \
    --clapsep-checkpoint path/to/best_model.ckpt \
    --data-csv path/to/dataset.csv \
    --classifier pann_finetuned ast_finetuned

# TUSS birds — bird_mae + audioprotopnet, with Risoux eval
python scripts/run_validation.py \
    --tuss-checkpoint path/to/best_model.pt \
    --tuss-coi-prompt birds \
    --data-csv path/to/dataset.csv \
    --classifier bird_mae audioprotopnet \
    --with-risoux
```

See `src/validation_functions/README.md` for COI synonym system details.

## Testing

```bash
pytest tests/
```
