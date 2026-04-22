# 🔴 UPDATED CRITICAL ANALYSIS: Base Model vs Fine-Tuning

## Key Finding: Base Model Uses Variable-Prompt Training

### How the Base TUSS Model Handles Flexible Inference

The pretrained base model (`tuss.medium.2-4src`) was trained with **explicit prompt variability**:

**Evidence from `base/pretrained_models/tuss.medium.2-4src/hparams.yaml:156-160`:**
```yaml
num_srcs_and_weights:
  1: 0.0      # Never train with 1 source
  2: 0.333    # 33.3% of samples use 2 sources
  3: 0.333    # 33.3% of samples use 3 sources
  4: 0.333    # 33.3% of samples use 4 sources
```

**Evidence from `base/datasets/sampler.py:53-57`:**
```python
# For EACH batch, randomly sample the number of sources
num_src = random.choices(
    list(self.num_srcs_and_weights.keys()),
    weights=list(self.num_srcs_and_weights.values()),
)[0]
batch_indices = [f"{b}_{num_src}" for b in batch_indices]
```

**Evidence from `base/datasets/dataset.py:135-165`:**
```python
def _select_prompts(self, num_prompts, prompts=[], allow_duplication=True):
    for i in range(len(prompts), num_prompts):
        if i == 0:
            # First prompt selected by init_prob
            weights = self.prompt_init_prob
            prompts = random.choices(self.prompts, weights=weights, k=1)
        else:
            # Subsequent prompts selected by transition probabilities
            prev_prompt = prompts[-1]
            weights = list(self.prompt_metadata[prev_prompt]["next"].values())
            p = random.choices(self.prompts, weights=weights, k=1)[0]
            prompts.append(p)
    return prompts
```

---

## Base Model Training: Dynamic Prompt Combinations

**Example training samples:**
- Sample 1: `["speech", "sfx"]` (2 sources)
- Sample 2: `["drums", "bass", "vocals"]` (3 sources)
- Sample 3: `["speech", "sfx", "musicbg", "other"]` (4 sources)
- Sample 4: `["sfxbg", "speech"]` (2 sources)
- Sample 5: `["musicbg", "sfx", "speech"]` (3 sources)
- etc.

**What this achieves:**
- ✅ Cross-prompt attention learns to work with **variable numbers of prompts**
- ✅ Prompts learn to function **independently and in different combinations**
- ✅ Model generalizes to ANY subset of the 8 pretrained prompts at inference
- ✅ Can infer with `["speech"]`, `["speech", "sfx"]`, `["drums", "bass"]`, etc.

---

## Your Fine-Tuning: Fixed Prompt Configuration

**Your training (from `train.py:1944-1952`):**
```python
n_coi = len(config.model.coi_prompts)  # 2 (airplane, birds)
n_src = n_coi + 1  # 3 (airplane, birds, background)

all_prompts = config.model.coi_prompts + [config.model.bg_prompt]
# all_prompts = ["airplane", "birds", "background"]

prompts_template = [list(all_prompts)] * _MAX_BATCH
# ALWAYS uses all 3 prompts, never varies
```

**Every training sample:**
- Sample 1: `["airplane", "birds", "background"]` (3 sources - ALWAYS)
- Sample 2: `["airplane", "birds", "background"]` (3 sources - ALWAYS)
- Sample 3: `["airplane", "birds", "background"]` (3 sources - ALWAYS)
- ...

**What this achieves:**
- ❌ Cross-prompt attention ONLY sees fixed 3-prompt configuration
- ❌ "airplane" prompt NEVER learns to work without "birds" present
- ❌ "birds" prompt NEVER learns to work without "airplane" present
- ⚠️ Model may not generalize well to subset prompts at inference

---

## Why This Matters

### Base Model Prompt Flexibility
```python
# Base model can handle ALL of these at inference:
model(mix, [["speech"]])                          # 1 prompt
model(mix, [["speech", "sfx"]])                   # 2 prompts
model(mix, [["drums", "bass", "vocals"]])         # 3 prompts
model(mix, [["speech", "sfx", "musicbg", "other"]])  # 4 prompts

# Because it was trained with ALL these configurations!
```

### Your Fine-Tuned Model
```python
# Your model was ONLY trained with:
model(mix, [["airplane", "birds", "background"]])  # 3 prompts - ALWAYS

# Using subsets at inference is untested:
model(mix, [["airplane", "background"]])  # 2 prompts - may degrade
model(mix, [["birds", "background"]])     # 2 prompts - may degrade

# Because cross-prompt attention patterns are different
```

---

## 🔴 Root Cause: Cross-Prompt Attention Dependencies

**From `base/nets/tuss.py:172-174`:**
```python
# During training, ALL prompts are concatenated
batch = self._concatenate_prompt(batch, prompts)  # Shape: (B, C, T+n_src, F)
for block in self.cross_prompt_module:
    batch = block(batch)  # <--- Cross-attention across ALL prompts!
```

**Base Model:** Sees different prompt combinations → learns flexible attention patterns  
**Your Model:** Sees only one prompt combination → learns fixed attention patterns

---

## Solutions

### Option 1: ✅ Always Use Full Prompt Set at Inference (RECOMMENDED)
```python
# Safe: matches training configuration
prompts = [["airplane", "birds", "background"]]
outputs = model(mix, prompts)  # Shape: (1, 3, T)

# Extract individual classes:
airplane = outputs[0, 0, :]
birds = outputs[0, 1, :]
background = outputs[0, 2, :]
```

**Pros:** Guaranteed optimal performance (matches training)  
**Cons:** Always computes all 3 outputs even if you only need 1

---

### Option 2: ⚠️ Test Subset Prompts (EXPERIMENTAL)
```python
# Risky: different from training configuration
prompts = [["airplane", "background"]]
outputs = model(mix, prompts)  # Shape: (1, 2, T)
```

**Pros:** More efficient (only 2 outputs)  
**Cons:** Unknown performance degradation (needs empirical testing)

---

### Option 3: 🔧 Retrain with Variable Prompts (BEST LONG-TERM)

Modify your training to use variable prompt configurations like the base model:

```python
# Pseudocode for variable-prompt training
def create_prompts_for_sample():
    n_src = random.choice([2, 3], p=[0.5, 0.5])  # 2 or 3 sources
    
    if n_src == 2:
        # 50% airplane-only, 50% birds-only
        coi_prompt = random.choice(["airplane", "birds"])
        return [coi_prompt, "background"]
    else:
        # Both classes
        return ["airplane", "birds", "background"]
```

**Implementation in `train.py`:**
1. Add `variable_prompts: bool = False` to `TrainingConfig`
2. Modify `prompts_template` generation to vary per batch
3. Randomly drop one COI prompt 50% of the time during training

**Pros:**
- ✅ Model learns flexible cross-prompt attention
- ✅ Can use any subset at inference without degradation
- ✅ Matches base model's training philosophy

**Cons:**
- Requires code changes and retraining
- Training slightly more complex

---

## Recommendation

**For your current situation:**
1. ✅ **Continue training with current configuration** (it WILL work correctly)
2. ✅ **Always use full prompts at inference:** `["airplane", "birds", "background"]`
3. ✅ **Extract desired class from output channels** (efficient enough for real-time)

**For future improvements:**
1. 🔧 After proving concept with current approach, consider adding variable-prompt training
2. 📊 Benchmark subset-prompt inference to quantify degradation (if any)
3. 📈 Document findings for future multi-class extensions

---

## Bottom Line

**Base Model:** Trained with variable prompts → flexible inference  
**Your Model:** Trained with fixed prompts → optimal with fixed inference  

**This is NOT a bug** - it's an architectural design difference. Your training is correct, but you should match inference configuration to training configuration for best results.
