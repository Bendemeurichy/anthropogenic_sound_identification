# Optuna Hyperparameter Tuning

Hyperparameter search for SuDoRMRF, TUSS, and CLAPSep models on COI separation tasks.

## Quick start

```bash
# Quick test (5 trials, 5 epochs)
python scripts/tuning/optuna_tune_models.py --n-trials 5 --target bird

# Tune a specific model with more trials
python scripts/tuning/optuna_tune_models.py --model sudormrf --n-trials 30 --target bird

# Full tuning run for all three models
python scripts/tuning/optuna_tune_models.py --n-trials 20 --target bird

# Smaller dataset (e.g., airplanes): increase epochs
python scripts/tuning/optuna_tune_models.py --n-trials 20 --max-epochs 50 --target airplane
```

## Models tuned

| Model | Hyperparameters searched |
|-------|-------------------------|
| SuDoRMRF | Architecture (channels, blocks, basis functions), LR, weight decay, class weight |
| TUSS | LR, weight decay, coi_weight, zero_ref_loss_weight, freeze_backbone |
| CLAPSep | Embed dimensions, masker layers, LR, weight decay, freeze_encoder |

## Output

Results saved to `configs/tuned/`:

- `{model}_{target}_best.yaml` — Best config, ready to pass to training scripts
- `{model}_{target}_tuning_stats.txt` — Trial history and summary
- `optuna_coi_tuning.db` — Full study (visualize with `optuna-dashboard sqlite:///optuna_coi_tuning.db`)

## Training with tuned configs

```bash
python src/models/sudormrf/train.py --config configs/tuned/sudormrf_bird_best.yaml
python src/models/tuss/train.py --config configs/tuned/tuss_bird_best.yaml
python src/models/clapsep/train_coi.py --config configs/tuned/clapsep_bird_best.yaml
```

## Multi-GPU

Run different models on separate GPUs in parallel:

```bash
python scripts/tuning/optuna_tune_models.py --model sudormrf --n-trials 20 --gpu 0 &
python scripts/tuning/optuna_tune_models.py --model tuss --n-trials 20 --gpu 1 &
python scripts/tuning/optuna_tune_models.py --model clapsep --n-trials 20 --gpu 2 &
```

## Resume tuning

Same command as before — the SQLite storage lets you resume interrupted studies.

## Tuning configs

Model-specific search spaces are in `configs/tuning/`. To modify hyperparameter ranges, edit these files:

- `configs/tuning/sudormrf_tuning.yaml`
- `configs/tuning/tuss_tuning.yaml`
- `configs/tuning/clapsep_tuning.yaml`
