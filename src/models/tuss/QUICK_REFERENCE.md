# Quick Reference: Extending TUSS Models

## Common Commands

### Inspect a checkpoint
```bash
python inspect_checkpoint.py checkpoints/tuss/best_model.pt
python inspect_checkpoint.py checkpoints/tuss/best_model.pt -c training_config.yaml
```

### Train from scratch
```bash
# Edit training_config.yaml:
#   coi_prompts: ["airplane"]
#   resume_from: ""  # or comment out
python train.py
```

### Continue fine-tuning
```bash
# Edit training_config.yaml:
#   coi_prompts: ["airplane"]  # Same as checkpoint
#   resume_from: "checkpoints/tuss/best_model.pt"
python train.py
```

### Extend model
```bash
# Edit training_config.yaml:
#   coi_prompts: ["airplane", "bird"]  # Add new class
#   resume_from: "checkpoints/tuss/best_model.pt"
python train.py
```

## Configuration Quick Reference

### Key Settings in `training_config.yaml`

```yaml
model:
  coi_prompts: ["airplane"]  # List of sound classes
  bg_prompt: "background"    # Background/residual class
  freeze_backbone: false     # Set true to freeze encoder/decoder

training:
  lr: 0.0001                           # Base learning rate
  existing_prompt_lr_multiplier: 0.1   # Continuing prompts: 0.00001
  resume_from: "path/to/checkpoint.pt" # Empty/commented = fresh start
  num_epochs: 30
  validate_every_n_epochs: 1
```

## Expected Output Patterns

### Fresh Start
```
Creating model …
📋 Model will use 2 outputs: ['airplane', 'background']

======================================================================
PARAMETER GROUPS
======================================================================

🆕 New prompts (full LR: 1.0e-04):
   airplane, background
   Total: 256,000 parameters

🏗️  Backbone (LR: 1.0e-04):
   Total: 42.5M parameters
======================================================================
```

### Continuing
```
Resuming training state from checkpoint: checkpoints/tuss/best_model.pt
✓ Loaded model weights (strict=False)
✓ Loaded optimizer state
✓ Loaded scheduler state
Resumed at epoch 24, global_step 1200, best_val_loss -8.21

======================================================================
PARAMETER GROUPS
======================================================================

🔄 Continuing prompts (reduced LR: 1.0e-05):
   airplane, background
   Total: 256,000 parameters

❄️  Frozen prompts (no training):
   sfx, sfxbg, bass, drums, vocals, other, speech, musicbg
   Total: 1,024,000 parameters
======================================================================
```

### Extending
```
Resuming training state from checkpoint: checkpoints/tuss/best_model.pt
✓ Loaded model weights (strict=False)
⚠️  Model is being extended with new/frozen prompts
   Skipping optimizer state loading (will start fresh)
⚠️  Starting fresh scheduler (will apply warmup from beginning)
Resumed at epoch 24, global_step 1200, best_val_loss -8.21

======================================================================
PARAMETER GROUPS
======================================================================

🆕 New prompts (full LR: 1.0e-04):
   bird
   Total: 128,000 parameters

🔄 Continuing prompts (reduced LR: 1.0e-05):
   airplane, background
   Total: 256,000 parameters

❄️  Frozen prompts (no training):
   sfx, sfxbg, bass, drums, vocals, other, speech, musicbg
   Total: 1,024,000 parameters
======================================================================
```

## Troubleshooting

### "Checkpoint not found"
- Check path in `resume_from`
- Use absolute path or relative to working directory
- Verify file exists: `ls -lh checkpoints/tuss/best_model.pt`

### "CUDA out of memory"
- Reduce `batch_size` in config
- Reduce `max_clip_duration`
- Enable gradient accumulation: increase `grad_accum_steps`

### "Prompt not found in checkpoint"
- Check spelling in `coi_prompts`
- Inspect checkpoint: `python inspect_checkpoint.py path/to/checkpoint.pt`
- Verify prompt was in original training config

### Learning rate too high/low
- Adjust base `lr` in config
- Adjust `existing_prompt_lr_multiplier` (default: 0.1)
- Check scheduler warmup: increase `warmup_steps`

### Model not improving
- Check if enough data for new class
- Verify data loading: check `sampler.py` logs
- Try longer training: increase `num_epochs`
- Monitor per-class metrics in validation logs

## Three-State Prompt System

| Prompt Location | State | Learning Rate | Gradients | Validated |
|-----------------|-------|---------------|-----------|-----------|
| In config only | 🆕 New | Full (1.0×) | ✅ Yes | ✅ Yes |
| In config + checkpoint | 🔄 Continuing | Reduced (0.1×) | ✅ Yes | ✅ Yes |
| In checkpoint only | ❄️ Frozen | N/A | ❌ No | ❌ No |

## File Locations

```
/home/bendm/Thesis/project/code/src/models/tuss/
├── train.py                              # Main training script
├── training_config.yaml                  # Configuration file
├── inspect_checkpoint.py                 # Checkpoint inspection tool
├── checkpoints/tuss/
│   └── best_model.pt                    # Your trained checkpoint
└── docs/
    ├── FINAL_SUMMARY.md                 # Complete implementation summary
    ├── EXTENDING_PROMPTS.md             # Detailed extending guide
    ├── EXAMPLE_EXTENDING_CHECKPOINT.md  # Practical example
    ├── OPTIMIZER_STATE_FIX.md           # Critical fix details
    ├── DEVICE_HANDLING.md               # GPU/checkpoint loading
    └── QUICK_REFERENCE.md               # This file
```

## Getting Help

1. Read `FINAL_SUMMARY.md` for complete overview
2. Read `EXTENDING_PROMPTS.md` for detailed extending guide
3. Read `EXAMPLE_EXTENDING_CHECKPOINT.md` for practical example
4. Check this quick reference for common commands
5. Inspect your checkpoint before extending: `python inspect_checkpoint.py`

## Tips

- Always inspect checkpoint before extending
- Start with small number of epochs to test setup
- Monitor validation metrics for overfitting
- Keep base learning rate conservative (1e-4)
- Use warmup when extending models
- Save checkpoints frequently
- Test with one new class before adding many
