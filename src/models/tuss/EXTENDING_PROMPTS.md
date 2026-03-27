# Extending a Fine-Tuned TUSS Model with New Prompts

This guide shows you how to add new COI class prompts to an already fine-tuned TUSS model.

## Overview

The training script now supports **incremental prompt injection**: you can load a checkpoint that was trained with some prompts (e.g., "airplane", "train") and add NEW prompts (e.g., "bird", "car") without losing the learned weights for the existing prompts.

## How It Works

When you set `resume_from` in your config and specify NEW prompts in `coi_prompts`:

1. **Existing prompts** (in the checkpoint) are loaded with their trained weights
2. **New prompts** (not in the checkpoint) are freshly initialized and injected
3. Training continues with ALL prompts (old + new)

## Step-by-Step Example

### Scenario: Adding "bird" and "dog" to a model trained on "airplane" and "train"

#### 1. Original Training (airplane + train)

```yaml
# training_config.yaml (original)
model:
  pretrained_path: "base/pretrained_models/tuss.medium.2-4src"
  coi_prompts: ["airplane", "train"]
  bg_prompt: "background"
  coi_prompt_init_from: "sfx"
  bg_prompt_init_from: "sfxbg"

data:
  df_path: "data/airplane_train_data.csv"
  target_classes:
    - ["Airplane", "Aircraft"]
    - ["Train", "Rail transport"]

training:
  resume_from: ""  # Start fresh
  checkpoint_dir: "checkpoints"
  num_epochs: 200
```

Run training:
```bash
python train.py
```

This creates: `checkpoints/YYYYMMDD_HHMMSS/best_model.pt` with prompts: `airplane`, `train`, `background`

#### 2. Extend with New Prompts (bird + dog)

Create new data CSV with **all 4 classes** (airplane, train, bird, dog):

```python
# prepare_extended_dataset.py
import pandas as pd

# Load your new dataset that includes bird and dog samples
# Make sure to set coi_class: 0=airplane, 1=train, 2=bird, 3=dog
df = ...  # your data loading logic

# Critical: bird and dog samples must have coi_class=2 and coi_class=3
df.to_csv("data/extended_4class_data.csv", index=False)
```

Update config to add new prompts and point to the checkpoint:

```yaml
# training_config.yaml (extended)
model:
  pretrained_path: "base/pretrained_models/tuss.medium.2-4src"
  coi_prompts: ["airplane", "train", "bird", "dog"]  # Added bird, dog
  bg_prompt: "background"
  coi_prompt_init_from: "sfx"
  bg_prompt_init_from: "sfxbg"

data:
  df_path: "data/extended_4class_data.csv"  # New CSV with all 4 classes
  target_classes:
    - ["Airplane", "Aircraft"]
    - ["Train", "Rail transport"]
    - ["Bird", "Birds"]  # New
    - ["Dog", "Bark"]    # New

training:
  resume_from: "checkpoints/20260327_143022/best_model.pt"  # Path to your trained checkpoint
  checkpoint_dir: "checkpoints"
  num_epochs: 200
  lr: 1e-5  # Lower LR since we're fine-tuning
```

Run extended training:
```bash
python train.py
```

**What happens:**
- ✅ Loads backbone weights from the checkpoint
- ✅ Loads existing prompts: `airplane`, `train`, `background` (with trained weights)
- ✅ Injects NEW prompts: `bird`, `dog` (freshly initialized from "sfx")
- ✅ Trains ALL 5 prompts together (4 COI + 1 background)

You'll see output like:
```
Preparing to resume from fine-tuned checkpoint: checkpoints/20260327_143022/best_model.pt
  Found 3 existing prompts in checkpoint: ['airplane', 'background', 'train']
  Prompt 'airplane' exists in checkpoint – will be loaded
  Prompt 'train' exists in checkpoint – will be loaded
  Injected NEW prompt 'bird' (init from 'sfx')
  Injected NEW prompt 'dog' (init from 'sfx')
  Prompt 'background' exists in checkpoint – will be loaded
Loading checkpoint weights (newly injected prompts will be preserved)...
  ✓ 2 new prompt(s) preserved: ['bird', 'dog']
```

## Important Notes

### 1. Dataset Requirements

Your extended dataset CSV must include:
- All OLD classes (airplane, train) - so they don't degrade
- All NEW classes (bird, dog) - so they can be learned
- `coi_class` column with correct indices (0, 1, 2, 3, ...)

### 2. Class Index Mapping

**CRITICAL:** The order in `coi_prompts` determines the class index:
```yaml
coi_prompts: ["airplane", "train", "bird", "dog"]
             #     0         1        2      3
```

In your CSV, `coi_class` must match:
- airplane samples → `coi_class=0`
- train samples → `coi_class=1`
- bird samples → `coi_class=2`
- dog samples → `coi_class=3`

### 3. Training Strategy

Consider these options when extending:

**Option A: Lower Learning Rate** (recommended)
```yaml
training:
  lr: 1e-5  # Much lower than original 5e-5
  freeze_backbone: false  # Allow fine-tuning
```
- Prevents catastrophic forgetting of old classes
- New prompts learn gradually

**Option B: Freeze Backbone, Train Prompts Only**
```yaml
model:
  freeze_backbone: true  # Only update prompts
training:
  lr: 5e-5  # Can use higher LR since backbone is frozen
```
- Faster training
- Less risk of degrading existing performance
- New classes may not integrate as well

**Option C: Two-Stage Training**
1. Stage 1: Freeze backbone, train only new prompts (50 epochs)
2. Stage 2: Unfreeze, fine-tune everything (50 epochs with low LR)

### 4. Validation Strategy

Monitor performance on **all classes**:
- Old classes (airplane, train) should maintain performance
- New classes (bird, dog) should improve over epochs

If old classes degrade significantly, try:
- Lower learning rate
- Add more old class samples to the training data
- Use prompt-only training (freeze_backbone=true)

## Testing the Extended Model

After training, test inference with all prompts:

```python
import torch
from nets.model_wrapper import SeparationModel

# Load checkpoint
ckpt = torch.load("checkpoints/YYYYMMDD_HHMMSS/best_model.pt")

# Extract prompt info
prompts = ckpt["all_prompts"]  # ["airplane", "train", "bird", "dog", "background"]
print(f"Model supports {len(prompts)-1} COI classes: {prompts[:-1]}")

# Test inference
model.eval()
with torch.no_grad():
    # Test with all prompts
    output = model(mixture, [prompts])  # Shape: (1, 5, T)
    
    # Extract individual sources
    airplane_audio = output[0, 0]  # First prompt
    train_audio = output[0, 1]     # Second prompt
    bird_audio = output[0, 2]      # Third prompt (NEW)
    dog_audio = output[0, 3]       # Fourth prompt (NEW)
    background = output[0, 4]      # Background
```

## Troubleshooting

### Issue: New prompts don't learn well

**Solution:** Increase training data for new classes, or try higher LR for prompts only:
```yaml
model:
  freeze_backbone: true  # Train only prompts
training:
  lr: 1e-4  # Higher LR for prompt-only training
```

### Issue: Old classes degrade

**Solution:** Lower LR and ensure balanced data:
```yaml
training:
  lr: 5e-6  # Very conservative LR
data:
  augment_multiplier: 3  # More augmentation for stability
```

### Issue: "Missing keys" warning shows unexpected prompts

This means your checkpoint has different prompts than expected. Verify:
```python
import torch
ckpt = torch.load("path/to/checkpoint.pt")
print("Prompts in checkpoint:", ckpt.get("coi_prompts"), ckpt.get("bg_prompt"))
```

## Summary

✅ **You CAN extend a trained model with new prompts**  
✅ **Existing prompts preserve their learned weights**  
✅ **New prompts are initialized fresh and learn alongside old ones**  
✅ **Use lower LR to prevent catastrophic forgetting**  

This allows incremental expansion of your model's capabilities without retraining from scratch!
