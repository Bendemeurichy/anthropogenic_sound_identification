# CLAPSep LoRA Implementation

## Overview

CLAPSep now supports proper LoRA (Low-Rank Adaptation) fine-tuning for the CLAP encoder, making it much more parameter-efficient while maintaining performance.

## What Changed

### Before (Misleading)
- Docstring claimed "optional LoRA fine-tuning"
- Only implemented freeze/unfreeze (all-or-nothing)
- `freeze_encoder=False` → **All** encoder weights trainable (~50-100M params)
- `freeze_encoder=True` → **Zero** encoder weights trainable

### After (Proper LoRA)
- True LoRA implementation using `loralib`
- Low-rank adapters applied to WindowAttention layers
- `freeze_encoder=False, use_lora=True` → **Only LoRA params** trainable (~0.5-2M params)
- Follows original CLAPSep paper's approach

## Usage

### Installation
```bash
pip install loralib
```

### Three Training Modes

#### Mode 1: Frozen Encoder (Fastest, Baseline)
```bash
python src/models/clapsep/train_coi.py \
  --df-path data/birds_data.csv \
  --clap-checkpoint checkpoints/music_audioset_epoch_15_esc_90.14.pt
  # freeze_encoder=True by default, use_lora=False
```
- **Trainable**: Decoder only (~10M params)
- **Speed**: Fastest
- **Memory**: Lowest
- **Use case**: Quick experiments, limited data

#### Mode 2: LoRA Fine-tuning (Recommended)
```bash
python src/models/clapsep/train_coi.py \
  --df-path data/birds_data.csv \
  --clap-checkpoint checkpoints/music_audioset_epoch_15_esc_90.14.pt \
  --no-freeze-encoder \
  --use-lora \
  --lora-rank 8
```
- **Trainable**: Decoder + LoRA adapters (~11-12M params)
- **Speed**: Fast
- **Memory**: Medium
- **Use case**: Recommended for most cases (good balance)

#### Mode 3: Full Fine-tuning (Most Parameters)
```bash
python src/models/clapsep/train_coi.py \
  --df-path data/birds_data.csv \
  --clap-checkpoint checkpoints/music_audioset_epoch_15_esc_90.14.pt \
  --no-freeze-encoder
  # use_lora=False by default
```
- **Trainable**: Decoder + full encoder (~60M params)
- **Speed**: Slower
- **Memory**: Highest
- **Use case**: Lots of data, very different domain

## LoRA Rank

The `--lora-rank` parameter controls the number of trainable parameters:

- **rank=4**: ~0.3M LoRA params (most parameter-efficient)
- **rank=8**: ~0.5M LoRA params (recommended default)
- **rank=16**: ~1.0M LoRA params (higher capacity)

Lower rank = fewer parameters, faster training, but potentially lower capacity.

## Config File

In `training_config.yaml`:

```yaml
model:
  # LoRA fine-tuning (recommended)
  freeze_encoder: false
  use_lora: true
  lora_rank: 8
  
  # Or decoder-only (faster baseline)
  # freeze_encoder: true
  # use_lora: false
```

## Implementation Details

The LoRA implementation:

1. **Targets WindowAttention layers** only (as in original CLAPSep)
2. **Replaces nn.Linear** with `lora.Linear` in attention
3. **Freezes base weights**, only trains LoRA matrices (A and B)
4. **Preserves pretrained knowledge** while adapting to COI task

### Code Location

- `src/models/clapsep/train_coi.py:54-98`: LoRA utility functions
- `src/models/clapsep/train_coi.py:153-170`: LoRA application in model init
- Based on original CLAPSep: `src/models/clapsep/base/model/CLAPSep.py:40-52`

## Parameter Comparison

Example for HTSAT-base encoder:

| Mode | Encoder Params | Trainable | Percentage |
|------|---------------|-----------|------------|
| Frozen | ~50M | 0M | 0% |
| LoRA (r=4) | ~50M | ~0.3M | 0.6% |
| LoRA (r=8) | ~50M | ~0.5M | 1.0% |
| LoRA (r=16) | ~50M | ~1.0M | 2.0% |
| Full | ~50M | ~50M | 100% |

*Plus ~10M decoder parameters in all cases*

## Optuna Tuning

The Optuna script now tunes LoRA settings:

```python
# Automatically searches:
- use_lora: [True, False]
- lora_rank: [4, 8, 16]  # if use_lora=True
```

This finds the optimal trade-off between:
- Parameter efficiency (fewer params)
- Model capacity (better performance)
- Training speed

## Expected Results

With birds dataset (5 epochs):
- **Frozen encoder**: 10-13 dB SI-SNR (baseline)
- **LoRA (r=8)**: 12-15 dB SI-SNR (recommended)
- **Full fine-tuning**: 13-16 dB SI-SNR (best, but 50x more params)

LoRA typically gets 80-95% of full fine-tuning performance with 1-2% of trainable parameters.

## Troubleshooting

### "ModuleNotFoundError: No module named 'loralib'"
```bash
pip install loralib
```

### "use_lora requires no_freeze_encoder"
LoRA only works when encoder is trainable:
```bash
--no-freeze-encoder --use-lora  # Correct
--use-lora                       # Auto-enables --no-freeze-encoder
```

### Out of Memory
Try smaller LoRA rank or frozen encoder:
```bash
--no-freeze-encoder --use-lora --lora-rank 4  # Smaller LoRA
# Or go back to frozen:
# (remove --no-freeze-encoder and --use-lora flags)
```

## References

- LoRA paper: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- Original CLAPSep: Uses LoRA for parameter-efficient adaptation
- loralib: [microsoft/LoRA](https://github.com/microsoft/LoRA)
