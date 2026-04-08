# Birds Dataset Optimization Summary

## Key Changes for Fast Tuning

Since birds dataset has abundant samples (16,482 annotations), training converges quickly. The tuning script has been optimized:

### Training Speed
- **Epochs**: 5 (default) instead of 100-200
  - Birds converge in 4-5 epochs due to data abundance
  - Each trial: ~2-5 minutes instead of 30-60 minutes
  
### Data Configuration
- **augment_multiplier**: 1 (disabled)
  - No time stretching, noise, or gain augmentations
  - Not needed with 16k+ samples
  
- **background_only_prob**: 0.1 (reduced from 0.2-0.3)
  - Less background-only training examples
  - Faster convergence to optimal hyperparameters

### Warmup Steps
- **SuDoRMRF**: 50-200 (was 200-600)
- **TUSS**: 50-200 (was 100-500)
- No change for CLAPSep (doesn't use warmup)

## Performance Expectations

With these optimizations:
- **Time per trial**: ~2-5 minutes (vs 30-60 min with full augmentations)
- **Trials needed**: 20-30 for good coverage
- **Total time for all 3 models**: ~2-4 hours (20 trials each)

## Usage Examples

```bash
# Quick test (5 trials × 5 epochs × 3 models ≈ 15-30 min)
python scripts/optuna_tune_models.py --n-trials 5 --target bird

# Recommended (20 trials × 5 epochs × 3 models ≈ 2-4 hours)
python scripts/optuna_tune_models.py --n-trials 20 --target bird

# Thorough search (30 trials × 5 epochs × 3 models ≈ 3-6 hours)
python scripts/optuna_tune_models.py --n-trials 30 --target bird

# Single model focused tuning
python scripts/optuna_tune_models.py --model tuss --n-trials 50 --target bird

# Use specific GPU (useful for multi-GPU systems)
python scripts/optuna_tune_models.py --n-trials 20 --target bird --gpu 1

# Multi-GPU parallel tuning (all models in parallel, ~1.5 hours total)
python scripts/optuna_tune_models.py --model sudormrf --n-trials 20 --gpu 0 &
python scripts/optuna_tune_models.py --model tuss --n-trials 20 --gpu 1 &
python scripts/optuna_tune_models.py --model clapsep --n-trials 20 --gpu 2 &
```

## When to Use More Epochs

If you're working with a different dataset that has:
- Fewer samples (< 1000)
- More challenging separation (lower SNR)
- Different domain (not birds)

Then override the epochs:
```bash
python scripts/optuna_tune_models.py --n-trials 20 --max-epochs 50 --target airplane
```

## Expected Metrics (Birds)

With 5 epochs on birds dataset:
- **SuDoRMRF**: 12-18 dB SI-SNR
- **TUSS**: 15-20 dB SNR (benefits from pretrained model)
- **CLAPSep**: 10-15 dB SI-SNR

These are sufficient to identify best hyperparameters - final models can train longer for production.
