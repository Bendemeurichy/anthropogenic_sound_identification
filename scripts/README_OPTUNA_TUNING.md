# Optuna Hyperparameter Tuning for COI Separation Models

This guide explains how to use the `optuna_tune_models.py` script to find optimal hyperparameters for comparing SuDoRMRF, TUSS, and CLAPSep models on bird separation (or any COI task).

## Overview

The script automates hyperparameter search for all three models:
- **SuDoRMRF**: Tunes architecture (channels, blocks, basis functions) and training params
- **TUSS**: Tunes learning rate, regularization, prompt freezing, and loss weights
- **CLAPSep**: Tunes embedding dimensions, attention layers, and encoder freezing

**Optimized for Birds**: With abundant bird samples, training converges quickly. The script uses:
- **5 epochs** by default (instead of 100-200)
- **No augmentations** (augment_multiplier=1, reduced background mixing)
- **Faster warmup** (50-200 steps instead of 200-600)

## Prerequisites

1. **Prepared dataset**: You need a CSV file with columns:
   - `filename`: path to audio file
   - `split`: train/val/test
   - `label`: class label (e.g., "bird", "Bird")
   - For TUSS, also needs `coi_class`: integer index

2. **Data preparation**: Each model's training script handles data loading/preparation automatically. You just need to ensure:
   - The CSV file exists (check `data/*.csv`)
   - Audio files are accessible at the paths in the CSV
   - Target class labels match what's in the metadata

3. **Python dependencies**:
   ```bash
   pip install optuna
   ```

## Quick Start

### 1. Quick Test (5 trials, default 5 epochs)
The script is optimized for bird separation with abundant data - uses only 5 epochs and no augmentations by default:
```bash
python scripts/optuna_tune_models.py \
  --n-trials 5 \
  --target bird
```

### 2. Tune Specific Model
Focus on one model with more trials:
```bash
python scripts/optuna_tune_models.py \
  --model sudormrf \
  --n-trials 30 \
  --target bird
```

### 3. Use Specific GPU
Run on a specific GPU (useful for multi-GPU systems):
```bash
# Using GPU 1
python scripts/optuna_tune_models.py \
  --n-trials 20 \
  --target bird \
  --gpu 1

# Or equivalently:
python scripts/optuna_tune_models.py \
  --n-trials 20 \
  --target bird \
  --device cuda:1
```

### 4. Full Tuning Run
Tune all three models (faster with abundant bird data):
```bash
python scripts/optuna_tune_models.py \
  --n-trials 20 \
  --target bird
```

### 5. Override for Smaller Datasets
If you have fewer samples (e.g., airplanes), increase epochs:
```bash
python scripts/optuna_tune_models.py \
  --n-trials 20 \
  --max-epochs 50 \
  --target airplane
```

## Hyperparameters Tuned

### SuDoRMRF
- **Architecture**: `out_channels` (128/256/512), `in_channels` (256/512/768), `num_blocks` (12-20), `enc_num_basis` (512/1024/2048), `num_head_conv_blocks` (1-3)
- **Training**: `lr` (1e-5 to 1e-3), `weight_decay` (1e-5 to 1e-3), `class_weight` (1.0-3.0), `warmup_steps` (50-200), `grad_accum_steps` (8/16/32)
- **Data**: `augment_multiplier=1`, `background_only_prob=0.1`

### TUSS
- **Training**: `lr` (1e-6 to 1e-4), `weight_decay` (1e-3 to 1e-1), `coi_weight` (1.0-3.0), `zero_ref_loss_weight` (0.01-0.5), `warmup_steps` (50-200), `grad_accum_steps` (4/8/16)
- **Model**: `freeze_backbone` (True/False - whether to freeze pretrained weights)
- **Data**: `augment_multiplier=1`, `background_only_prob=0.1`

### CLAPSep
- **Architecture**: `embed_dim` (64/128/256), `encoder_embed_dim` (64/128/256), `n_masker_layer` (2-5), `d_attn` (320/640/1024)
- **Training**: `lr` (1e-5 to 1e-3), `weight_decay` (1e-6 to 1e-4), `class_weight` (1.0-3.0)
- **Model**: `freeze_encoder` (True/False - whether to freeze CLAP encoder)
- **Data**: `augment_multiplier=1`, `background_only_prob=0.1`

## Output Files

All outputs are saved to `configs/tuned/`:

1. **Best configurations**: `{model}_{target}_best.yaml`
   - Ready-to-use config files with optimal hyperparameters
   - Can be passed directly to training scripts

2. **Tuning statistics**: `{model}_{target}_tuning_stats.txt`
   - Summary of all trials
   - Best hyperparameters
   - Trial history

3. **Optuna database**: `optuna_coi_tuning.db`
   - Full study history
   - Can be visualized with Optuna dashboard

## Visualizing Results

View tuning progress and parameter importance:

```bash
# Install optuna dashboard
pip install optuna-dashboard

# Launch dashboard
optuna-dashboard sqlite:///optuna_coi_tuning.db
```

Then open http://localhost:8080 in your browser.

## Training with Tuned Configs

After tuning, use the best configs for final training:

```bash
# SuDoRMRF
python src/models/sudormrf/train.py --config configs/tuned/sudormrf_bird_best.yaml

# TUSS
python src/models/tuss/train.py --config configs/tuned/tuss_bird_best.yaml

# CLAPSep
python src/models/clapsep/train_coi.py --config configs/tuned/clapsep_bird_best.yaml
```

## Advanced Usage

### Custom Target Class
Tune for different classes (e.g., airplane):
```bash
python scripts/optuna_tune_models.py \
  --target airplane \
  --n-trials 20
```

### Resume Tuning
The script uses SQLite storage, so you can resume interrupted studies:
```bash
# Same command as before - it will continue from where it left off
python scripts/optuna_tune_models.py \
  --model sudormrf \
  --n-trials 50 \
  --target bird
```

### Custom Storage
Use a different database or remote storage:
```bash
python scripts/optuna_tune_models.py \
  --storage postgresql://user:password@localhost/optuna \
  --n-trials 20
```

### Multi-GPU Setup
If you have multiple GPUs, run different models on different GPUs in parallel:
```bash
# Terminal 1 - SuDoRMRF on GPU 0
python scripts/optuna_tune_models.py --model sudormrf --n-trials 20 --gpu 0 &

# Terminal 2 - TUSS on GPU 1
python scripts/optuna_tune_models.py --model tuss --n-trials 20 --gpu 1 &

# Terminal 3 - CLAPSep on GPU 2
python scripts/optuna_tune_models.py --model clapsep --n-trials 20 --gpu 2 &
```

This can reduce total tuning time from ~4 hours to ~1.5 hours!

## Tips

1. **Start small**: Use `--n-trials 5` to test first (default 5 epochs is already fast)
2. **GPU memory**: If you get OOM errors, reduce `batch_size` in the base configs
3. **Time estimation**: Each trial takes ~2-5 minutes with birds (5 epochs, no augmentations)
4. **Parallel tuning**: Run different models on different GPUs simultaneously (see Multi-GPU Setup above)
5. **Early stopping**: Each model uses early stopping, so failed configs will finish quickly
6. **For other datasets**: If you have fewer samples, increase `--max-epochs` (e.g., 50 for airplanes)
7. **GPU selection**: Use `--gpu N` for specific GPU or `--device cuda:N` for more control

## Expected Results

With birds dataset (easy separation task):
- **SuDoRMRF**: ~12-18 dB SI-SNR
- **TUSS**: ~15-20 dB SNR (benefits from pretrained model)
- **CLAPSep**: ~10-15 dB SI-SNR

The tuning will help identify which model architecture works best for your specific data.

## Troubleshooting

### "Dataset not found"
Ensure your CSV file exists and is referenced in the base config:
```bash
# Check what dataset the base config expects
grep df_path src/models/*/training_config.yaml
```

### "Audio file not found"
Verify audio paths in the CSV are correct:
```bash
head -5 data/your_dataset.csv
```

### Parse errors
If the script can't extract metrics, check that the training scripts print:
- "Best Val SI-SNR: X.XX dB" (sudormrf, clapsep)
- "Best Val SNR: X.XX dB" (tuss)

## Next Steps

After tuning:
1. Review `configs/tuned/*_stats.txt` to understand which hyperparameters matter most
2. Train final models with best configs
3. Run inference/evaluation to compare model performance
4. Use Optuna dashboard to visualize hyperparameter importance
