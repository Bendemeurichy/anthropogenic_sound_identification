# TUSS (Task-Aware Unified Source Separation)

Fine-tuned TUSS model for COI separation. See `train.py` module docstring for architecture overview.

## Quick start

```bash
cd src/models/tuss

# Inspect a checkpoint
python inspect_checkpoint.py checkpoints/tuss/best_model.pt
python inspect_checkpoint.py checkpoints/tuss/best_model.pt -c training_config.yaml

# Train from scratch
# Set training.resume_from: "" in training_config.yaml
python train.py

# Continue fine-tuning (same prompts as checkpoint)
# Set training.resume_from: "checkpoints/tuss/best_model.pt"
python train.py

# Extend model with new prompts
# Add new class to model.coi_prompts list, then:
python train.py
```

## Prompt states

| Prompt location | State | LR multiplier | Gradients |
|-----------------|-------|---------------|-----------|
| In config only | New | 1.0x | Yes |
| In config + checkpoint | Continuing | 0.1x | Yes |
| In checkpoint only | Frozen | N/A | No |

## Extending a trained model

Set `resume_from` and add new prompts to `coi_prompts` in `training_config.yaml`:

```yaml
model:
  coi_prompts: ["airplane", "bird"]  # Add new prompt(s)
  bg_prompt: "background"
  coi_prompt_init_from: "sfx"
training:
  resume_from: "checkpoints/tuss/best_model.pt"
  lr: 1e-5  # Lower LR to prevent catastrophic forgetting
```

- Existing prompts keep their trained weights
- New prompts are freshly initialized from `coi_prompt_init_from`
- Optimizer state is skipped in extend mode (safe — Adam rebuilds momentum)
- To balance old vs new class performance, try `freeze_backbone: true` or two-stage training

## Key config settings

```yaml
model:
  coi_prompts: ["airplane"]
  bg_prompt: "background"
  freeze_backbone: false
  coi_prompt_init_from: "sfx"
  bg_prompt_init_from: "sfxbg"
training:
  lr: 0.0001
  existing_prompt_lr_multiplier: 0.1
  resume_from: ""          # Empty = train from scratch
  num_epochs: 30
  validate_every_n_epochs: 1
```

## Expected output patterns

**Fresh start**: "Creating model" → `New prompts (full LR)` + `Backbone`

**Continuing**: "Resuming training state" → `Continuing prompts (reduced LR)` + `Frozen prompts (no training)` + optimizer/scheduler loaded

**Extending**: "Model is being extended" → `New prompts (full LR)` + `Continuing prompts (reduced LR)` + `Frozen prompts` + optimizer/scheduler skipped

## Files

| File | Purpose |
|------|---------|
| `train.py` | Training script; 3-state prompt system, differential LRs |
| `inference.py` | Inference wrapper implementing `BaseSeparator` |
| `inspect_checkpoint.py` | Checkpoint inspection: prompts, shapes, divergence |
| `training_config.yaml` | Config with all model/data/training settings |
| `config.py` | Config dataclasses with YAML serialization |
| `dataset.py` | PyTorch Dataset for CSV-based loading |
| `losses.py` | SI-SNR and auxiliary loss functions |
| `augmentations.py` | Training-time audio augmentations |
| `utils.py` | Training helpers (schedulers, logging, metric tracking) |
