# Example: Extending the existing checkpoint with new prompts

## Current checkpoint status
The checkpoint at `checkpoints/tuss/best_model.pt` contains:
- **1 COI class**: `airplane` (trained for 23 epochs)
- **1 Background**: `background`
- **8 pretrained prompts**: bass, drums, vocals, other, speech, musicbg, sfx, sfxbg

## Example: Add "train" and "bird" as new COI classes

### 1. Check what's in the checkpoint
```bash
source .venv/bin/activate
cd src/models/tuss
python inspect_checkpoint.py checkpoints/tuss/best_model.pt
```

### 2. Prepare extended dataset
You need a CSV with airplane, train, and bird samples:
```csv
filename,split,label,coi_class
/path/to/airplane1.wav,train,1,0
/path/to/train1.wav,train,1,1
/path/to/bird1.wav,train,1,2
/path/to/background1.wav,train,0,-1
...
```

**Important**: coi_class mapping:
- airplane → 0 (existing)
- train → 1 (NEW)
- bird → 2 (NEW)
- background → -1 (or any negative value)

### 3. Update training_config.yaml
```yaml
model:
  pretrained_path: "base/pretrained_models/tuss.medium.2-4src"
  coi_prompts: ["airplane", "train", "bird"]  # Extended from 1 to 3 classes
  bg_prompt: "background"
  coi_prompt_init_from: "sfx"
  bg_prompt_init_from: "sfxbg"

data:
  df_path: "data/airplane_train_bird_data.csv"  # Your new CSV
  target_classes:
    - ["Airplane", "Aircraft"]
    - ["Train", "Rail transport"]
    - ["Bird", "Birds chirping"]

training:
  resume_from: "checkpoints/tuss/best_model.pt"  # Load existing checkpoint
  checkpoint_dir: "checkpoints"
  num_epochs: 100
  lr: 1e-5  # Lower LR for fine-tuning
```

### 4. Validate before training
```bash
python inspect_checkpoint.py checkpoints/tuss/best_model.pt -c training_config.yaml
```

Expected output:
```
Analysis:
  ✓ Already trained (2): ['airplane', 'background']
  + NEW in config (2): ['bird', 'train']
```

### 5. Run training
```bash
python train.py
```

What will happen:
```
Preparing to resume from fine-tuned checkpoint: checkpoints/tuss/best_model.pt
  Found 10 existing prompts in checkpoint: ['airplane', 'background', 'bass', ...]
  
PROMPT VALIDATION
======================================================================
Checkpoint: checkpoints/tuss/best_model.pt
Prompts in checkpoint: ['airplane', 'background', 'bass', 'drums', ...]
Prompts in config: ['airplane', 'background', 'bird', 'train']

✓ Already trained (2): ['airplane', 'background']
+ NEW prompts (2): ['bird', 'train']

📌 Training strategy: INCREMENTAL (skip existing prompts)
   - Will train only NEW prompts: ['bird', 'train']
   - Will load existing prompts from checkpoint: ['airplane', 'background']
======================================================================

  Prompt 'airplane' exists in checkpoint – will be loaded
  Injected NEW prompt 'train' (init from 'sfx')
  Injected NEW prompt 'bird' (init from 'sfx')
  Prompt 'background' exists in checkpoint – will be loaded
Loading checkpoint weights (newly injected prompts will be preserved)...
  ✓ 2 new prompt(s) preserved: ['train', 'bird']

📋 Model will use 4 outputs: ['airplane', 'train', 'bird', 'background']
```

### 6. Result
After training, you'll have a model that can separate:
- **airplane** (retains learned weights from original checkpoint)
- **train** (newly trained)
- **bird** (newly trained)
- **background**

## To force retraining all prompts

If you want to RETRAIN existing prompts (not recommended unless you have more data):

### Option A: Modify the code
In `train.py`, call train with:
```python
train(config, timestamp=None, skip_existing_prompts=False)
```

### Option B: Start fresh
```yaml
training:
  resume_from: ""  # Don't resume
```

## Key Points

✅ **Existing prompts are preserved** - The 'airplane' prompt keeps its trained weights
✅ **New prompts are fresh** - 'train' and 'bird' start from 'sfx' initialization  
✅ **All prompts train together** - The model outputs all 4 sources simultaneously
✅ **Incremental learning** - No catastrophic forgetting if LR is low enough

## Troubleshooting

### All prompts already exist warning
```bash
❌ ERROR: No new prompts to train!
```
Solution: Add NEW prompts to `coi_prompts` that aren't in the checkpoint

### Checkpoint not found
```bash
resume_from path not found: checkpoints/tuss/best_model.pt
```
Solution: Check the path is relative to the train.py script location

### Dataset class mismatch
If your dataset has wrong coi_class indices, the model will learn the wrong associations.
Always verify:
```python
import pandas as pd
df = pd.read_csv("data/airplane_train_bird_data.csv")
print(df.groupby('coi_class')['filename'].count())
# Expected output:
# 0 (airplane): N files
# 1 (train): M files  
# 2 (bird): K files
```
