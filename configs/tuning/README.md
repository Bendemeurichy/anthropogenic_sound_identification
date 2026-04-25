# Hyperparameter Tuning Configurations

This directory contains dedicated tuning configurations for Optuna hyperparameter optimization, completely separate from production training configs.

## Quick Verification

All models are configured for **5 epochs per trial** by default:
- ✅ SuDoRMRF: 5 epochs
- ✅ TUSS: 5 epochs  
- ✅ CLAPSep: 5 epochs

You can verify this by checking the `tuning.num_epochs` field in each config file.

## SuDoRMRF Architecture Tuning

SuDoRMRF now tunes **both** the shared backbone and separation branches:

### Shared Backbone
- **num_blocks**: 12 to 20 (default: 16)
  - Number of SuDoRM-RF blocks in the shared feature extraction backbone
  - More blocks = more capacity but slower and more memory usage

### Separation Branches  
- **num_head_conv_blocks**: 1 to 4 (default: 2)
  - Number of UConvBlocks in each separation head (COI and background)
  - 0 = simple architecture (PReLU + Conv1d)
  - >0 = enhanced architecture with N UConvBlocks per branch

This allows exploring the trade-off between:
- Shared representation capacity (num_blocks)
- Task-specific feature extraction (num_head_conv_blocks)

## Epoch Configuration

The script will print a **prominent warning** at startup:

```
================================================================================
Starting Optuna hyperparameter tuning for SUDORMRF
================================================================================
Target class: bird
Number of trials: 20
⚠️  EPOCHS PER TRIAL: 5  ⚠️
Device: cuda
...
================================================================================
```

Each trial also prints its configuration:

```
================================================================================
Trial 0: SUDORMRF on cuda
Epochs: 5 | LR: 5.00e-04
Architecture: num_blocks=16, num_head_conv_blocks=2
================================================================================
```

## Override Epochs (if needed)

For smaller datasets or longer convergence times, you can override:

```bash
python scripts/optuna_tune_models.py --max-epochs 10 --target airplane
```

This will show:
```
⚠️  EPOCHS PER TRIAL: 10  ⚠️
⚠️  WARNING: Using 10 epochs (default is 5) ⚠️
```

## Why 5 Epochs is Enough for Tuning

1. **Fast iteration**: Each trial takes 2-5 minutes instead of hours
2. **Relative comparison**: We're comparing hyperparameters, not training to convergence
3. **Early trends**: Good hyperparameters show improvement within 5 epochs
4. **Final training**: After tuning, retrain with best config for full 200-400 epochs

## Preventing Accidents

If you accidentally ran the old script (before redesign) that loaded production configs:
- ❌ OLD: Would use 400 epochs for SuDoRMRF (from training_config.yaml)
- ✅ NEW: Always uses 5 epochs (from tuning configs)

The new script **never** loads production training configs, so this can't happen anymore.
