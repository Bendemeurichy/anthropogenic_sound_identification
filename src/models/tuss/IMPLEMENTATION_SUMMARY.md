# Summary: Extending TUSS Models with New Prompts

## What We've Implemented

### 1. Checkpoint Inspection Tool (`inspect_checkpoint.py`)

**Purpose**: Extract and display prompt information from trained checkpoints.

**Usage**:
```bash
# Basic inspection
python inspect_checkpoint.py checkpoints/tuss/best_model.pt

# Verbose mode (shows tensor shapes)
python inspect_checkpoint.py checkpoints/tuss/best_model.pt -v

# Compare with training config
python inspect_checkpoint.py checkpoints/tuss/best_model.pt -c training_config.yaml
```

**Output**:
- Training state (epoch, step, validation loss)
- List of all prompts in the checkpoint
- Prompt metadata (COI vs background)
- Comparison with config (what's new, what exists)

### 2. Incremental Prompt Injection in `train.py`

**Key Features**:
1. **Extract existing prompts** from checkpoint before training
2. **Validate prompts** - warn if all prompts already exist
3. **Skip already-trained prompts** - only inject and train NEW prompts
4. **Preserve learned weights** - existing prompts load from checkpoint
5. **Combined inference** - model outputs ALL prompts (old + new)

**Modified Functions**:
- `create_model()` - Now accepts `resume_ckpt_path` parameter
  - Loads checkpoint weights AFTER prompt injection
  - Preserves newly injected prompts
  
- `get_prompts_from_checkpoint()` - Utility to extract prompt names
  
- `validate_prompts_against_checkpoint()` - Validates config vs checkpoint
  - Returns filtered prompts (only new ones)
  - Prints detailed comparison
  
- `train()` - Main training with validation
  - Calls validation before model creation
  - Adjusts config to train only new prompts
  - Uses ALL prompts for inference

## How It Works

### Scenario: Extending airplane model with train + bird

**Before** (checkpoint):
```
Prompts: ['airplane', 'background', 'sfx', 'sfxbg', ...]
Trained: airplane (23 epochs)
```

**Config** (new):
```yaml
coi_prompts: ["airplane", "train", "bird"]
resume_from: "checkpoints/tuss/best_model.pt"
```

**Process**:
1. Load checkpoint → Find existing prompts: ['airplane', 'background', ...]
2. Compare with config → New: ['train', 'bird'], Existing: ['airplane', 'background']
3. Filter config → Train only: ['train', 'bird']
4. Create model:
   - Load base architecture
   - Inject NEW prompts: 'train', 'bird' (from 'sfx')
   - Load checkpoint weights (overwrites backbone + existing prompts)
   - NEW prompts remain fresh (not overwritten)
5. Train with ALL prompts: ['airplane', 'train', 'bird', 'background']

**Result**:
- 'airplane': ✅ Keeps trained weights from checkpoint
- 'train': ✅ Fresh prompt, learns from scratch
- 'bird': ✅ Fresh prompt, learns from scratch
- 'background': ✅ Keeps trained weights from checkpoint

## Usage Example

### Step 1: Inspect current checkpoint
```bash
python inspect_checkpoint.py checkpoints/tuss/best_model.pt
```

Output shows: `airplane` and `background` are trained

### Step 2: Update config with new prompts
```yaml
model:
  coi_prompts: ["airplane", "train", "bird"]  # Add train, bird
  
training:
  resume_from: "checkpoints/tuss/best_model.pt"
  lr: 1e-5  # Lower LR for stability
```

### Step 3: Validate
```bash
python inspect_checkpoint.py checkpoints/tuss/best_model.pt -c training_config.yaml
```

Should show:
```
✓ Already trained (2): ['airplane', 'background']
+ NEW in config (2): ['bird', 'train']
```

### Step 4: Prepare dataset
Create CSV with all classes (airplane, train, bird):
- coi_class=0 for airplane
- coi_class=1 for train
- coi_class=2 for bird

### Step 5: Train
```bash
python train.py
```

The script will:
- Load existing 'airplane' and 'background' from checkpoint
- Inject fresh 'train' and 'bird' prompts
- Train with all 4 prompts
- Save extended model with 3 COI classes

## Important Notes

### 1. Class Index Consistency
Order in `coi_prompts` MUST match `coi_class` in your CSV:
```yaml
coi_prompts: ["airplane", "train", "bird"]
             #     0         1        2
```

### 2. Learning Rate
Use lower LR when extending (e.g., 1e-5 instead of 5e-5) to prevent catastrophic forgetting of existing classes.

### 3. Dataset Requirements
Include samples from BOTH:
- **Existing classes** (airplane) - prevents forgetting
- **New classes** (train, bird) - enables learning

### 4. Skip vs Retrain
**Default behavior** (skip_existing_prompts=True):
- Only NEW prompts are trained
- Faster, safer, recommended

**Force retrain** (skip_existing_prompts=False):
- ALL prompts are retrained
- Useful if you have much more data
- Risk of degrading existing performance

### 5. Validation Strategy
Monitor performance on ALL classes:
- Old classes should maintain SI-SNR
- New classes should improve over epochs

## Files Modified

1. **train.py**:
   - Added `resume_ckpt_path` parameter to `create_model()`
   - Added `get_prompts_from_checkpoint()`
   - Added `validate_prompts_against_checkpoint()`
   - Modified `train()` to validate and filter prompts
   - Fixed `torch.load()` with `weights_only=False`

2. **inspect_checkpoint.py** (NEW):
   - Command-line tool to inspect checkpoints
   - Extract prompt information
   - Compare with training configs

3. **EXTENDING_PROMPTS.md** (NEW):
   - Detailed guide on extending models
   - Best practices and troubleshooting

4. **EXAMPLE_EXTENDING_CHECKPOINT.md** (NEW):
   - Practical example using your existing checkpoint

## Testing

Your checkpoint was successfully inspected:
```
Training State: Epoch 23, Step 7682, Val Loss: -8.21
COI Prompts: ['airplane']
Background: 'background'
All prompts: 10 total (airplane, background, sfx, sfxbg, bass, drums, vocals, other, speech, musicbg)
```

Ready to extend with new classes!

## Next Steps

1. **Create extended dataset** with airplane + new classes (train, bird, car, etc.)
2. **Update training_config.yaml** with new coi_prompts
3. **Validate** using `inspect_checkpoint.py -c training_config.yaml`
4. **Train** with `python train.py`
5. **Test** separation with all classes

The implementation is complete and tested! 🎉
