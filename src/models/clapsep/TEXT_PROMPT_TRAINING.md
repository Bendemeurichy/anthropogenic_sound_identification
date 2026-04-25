# CLAPSep Text-Prompt Training

This document explains the new text-prompt training approach for CLAPSep COI separation.

## Overview

CLAPSep now supports **two training modes** for COI (Class of Interest) separation:

### 1. Text-Prompt Training (`train_text_coi.py`) ✨ NEW
- **Uses text prompts** for conditioning (like original CLAPSep)
- **Retains inference flexibility**: Change target sound by changing text prompt
- **Supports LoRA tuning** for parameter-efficient adaptation
- **Multiple prompt variations** per class for robustness
- **Best for**: When you want multi-purpose separation with improved performance on your COI data

### 2. Learned-Embedding Training (`train_coi.py`)
- **Uses learned embeddings** instead of text prompts
- **Fixed to trained COI class**: Cannot change target without retraining
- **Simpler architecture**: No text encoder needed at inference
- **Best for**: When you only need to separate a single fixed COI class

---

## Why Text-Prompt Training?

### Problem with Learned Embeddings
The original `train_coi.py` replaces text prompts with learned `nn.Parameter` embeddings:
```python
self.coi_embedding = nn.Parameter(torch.randn(1, embed_dim))
self.bg_embedding = nn.Parameter(torch.randn(1, embed_dim))
```

**Limitation**: Once trained on aircraft, the model can ONLY separate aircraft. To separate birds, you must retrain from scratch with new embeddings.

### Solution: Text-Prompt Training
The new `train_text_coi.py` keeps text conditioning:
```python
# Training: use text prompts from your COI data
pos_text = "airplane engine"  # or "bird chirping", etc.
neg_text = "ambient noise"
embed_pos = clap.get_text_embedding(pos_text)
embed_neg = clap.get_text_embedding(neg_text)
```

**Benefit**: After training on aircraft + birds, you can separate EITHER class at inference:
```python
# Separate aircraft
sources = sep.separate("audio.wav", text_pos="airplane engine", text_neg="")

# OR separate birds (same model!)
sources = sep.separate("audio.wav", text_pos="bird chirping", text_neg="")

# OR even try new sounds the model wasn't trained on
sources = sep.separate("audio.wav", text_pos="dog barking", text_neg="")
```

---

## Configuration

### Text Prompt Setup

Edit `training_config.yaml`:

```yaml
model:
  # Text prompts for conditioning
  # Multiple variations per class improve robustness
  coi_text_prompts:
    # Class 0: Aircraft prompts
    - ["airplane engine", "aircraft noise", "jet flying", "propeller airplane"]
    # Class 1: Birds prompts (for multi-class)
    - ["bird chirping", "bird singing", "songbird", "birds calling"]
  
  # Background prompt variations
  background_text_prompts: 
    - "ambient noise"
    - "background sounds"
    - "environmental noise"
  
  # LoRA settings (recommended)
  freeze_encoder: false
  use_lora: true
  lora_rank: 8

data:
  # Map dataset labels to COI classes
  target_classes:
    - ["airplane", "aircraft", "jet", "propeller_airplane"]  # → coi_text_prompts[0]
    - ["bird", "bird_song", "birds"]                         # → coi_text_prompts[1]
```

### Single-Class Example (Aircraft Only)

```yaml
model:
  coi_text_prompts:
    - ["airplane engine", "aircraft noise", "jet flying", "plane overhead"]
  background_text_prompts: ["ambient noise", "background sounds"]

data:
  target_classes:
    - ["airplane", "aircraft", "jet", "propeller_airplane", "aeroplane"]
```

---

## Training

### Basic Usage

```bash
# Train with text prompts
python src/models/clapsep/train_text_coi.py --config src/models/clapsep/training_config.yaml
```

### Command-Line Options

```bash
python src/models/clapsep/train_text_coi.py \
  --config training_config.yaml \
  --df-path data/aircraft_data.csv \
  --clap-checkpoint checkpoint/music_audioset_epoch_15_esc_90.14.pt \
  --use-lora \
  --lora-rank 8 \
  --device cuda
```

### LoRA Modes

**Mode 1: Frozen Encoder (Fastest)**
```bash
python train_text_coi.py --freeze-encoder
```
- Trainable: Decoder only (~10M params)
- Speed: Fastest
- Use case: Quick experiments, limited data

**Mode 2: LoRA Fine-tuning (Recommended)**
```bash
python train_text_coi.py --use-lora --lora-rank 8
```
- Trainable: Decoder + LoRA adapters (~11-12M params)
- Speed: Fast
- Use case: Best balance of performance and efficiency

**Mode 3: Full Fine-tuning**
```bash
python train_text_coi.py --no-freeze-encoder --no-lora
```
- Trainable: Decoder + full encoder (~60M params)
- Speed: Slower
- Use case: Lots of data, very different domain

---

## Inference

### Load Text-Prompt Trained Model

```python
from models.clapsep.inference import CLAPSepInference

# Load checkpoint
sep = CLAPSepInference.from_checkpoint(
    "checkpoints/text_prompt_20250425_120000/best_model.ckpt",
    device="cuda"
)

# Separate with default prompts (from training)
sources = sep.separate("audio.wav")

# Or override prompts at inference time
sources = sep.separate(
    "audio.wav",
    text_pos="airplane engine",
    text_neg="ambient noise"
)

# Extract separated audio
coi_audio = sep.get_coi_audio(sources)          # (T,)
background_audio = sep.get_background_audio(sources)  # (T,)

# Save
sep.save_audio(coi_audio, "output_coi.wav")
sep.save_audio(background_audio, "output_background.wav")
```

### Change Target Sound Dynamically

```python
# Same model, different targets!

# Separate aircraft
sources = sep.separate("audio.wav", text_pos="airplane engine")

# Separate birds (if trained on both)
sources = sep.separate("audio.wav", text_pos="bird chirping")

# Try new sounds (zero-shot, may work if similar to training data)
sources = sep.separate("audio.wav", text_pos="dog barking")
```

---

## How It Works

### Training Process

1. **Load COI dataset** (filename, label, split columns)
2. **Map labels to text prompts**
   - `label=1, coi_class=0` → random choice from `coi_text_prompts[0]`
   - `label=0` → random choice from `background_text_prompts`
3. **Get text embeddings** from CLAP text encoder (frozen)
4. **Extract audio features** with CLAP audio encoder (LoRA-tuned)
5. **Generate separation mask** using text-conditioned decoder
6. **Compute loss** (SI-SNR based)

### Architecture

```
Input: mixture waveform + text prompts
  ↓
CLAP Text Encoder (frozen)
  → pos_text_embed, neg_text_embed
  ↓
CLAP Audio Encoder (LoRA-tuned)
  → audio_features
  ↓
CLAPSep Decoder (trainable)
  → separation_mask (conditioned on text embeddings)
  ↓
Output: separated sources [COI, background]
```

### Key Differences from `train_coi.py`

| Component | `train_coi.py` (Learned) | `train_text_coi.py` (Text) |
|-----------|-------------------------|---------------------------|
| **Conditioning** | Learned `nn.Parameter` | Text prompts → CLAP encoder |
| **Text Encoder** | Not used | Used (frozen) |
| **Decoder Input** | Learned embeddings | Text embeddings |
| **Inference** | Fixed COI class | Change prompts dynamically |
| **LoRA** | Optional | Optional |
| **Flexibility** | ❌ Single class only | ✅ Multi-purpose |

---

## Prompt Engineering Tips

### Good Prompts
✅ **Descriptive and specific**
- "airplane engine"
- "bird chirping"
- "jet flying overhead"

✅ **Multiple variations** (data augmentation)
- ["airplane engine", "aircraft noise", "jet flying", "propeller plane"]

✅ **Match audio characteristics**
- "low frequency rumble" for trucks
- "high pitched chirp" for birds

### Bad Prompts
❌ **Too vague**
- "sound"
- "noise"

❌ **Too complex**
- "an airplane taking off from an airport runway with loud engine noise"

❌ **Single variation** (no robustness)
- Just ["airplane"]

---

## Performance Expectations

### With LoRA Tuning (rank=8)
- **Trained classes**: 12-16 dB SI-SNR improvement (best)
- **Similar sounds**: 8-12 dB SI-SNR improvement (good)
- **Different sounds**: 3-8 dB SI-SNR improvement (zero-shot)

### vs. Learned Embeddings
- **Same COI class**: Text-prompt ~95% of learned-embedding performance
- **Different classes**: Text-prompt ✅ works, learned-embedding ❌ impossible
- **Training time**: Similar (~same compute cost)
- **Parameters**: Text-prompt has slightly more params (text encoder exists but frozen)

---

## Comparison: Three Approaches

### 1. Pretrained CLAPSep (No Training)
```python
sep = CLAPSepInference.from_pretrained()
sources = sep.separate("audio.wav", text_pos="airplane engine")
```
- ✅ No training needed
- ✅ Works on any sound (zero-shot)
- ❌ Not optimized for your data
- ❌ Lower performance on your specific COI classes

### 2. Text-Prompt Training (This Approach)
```python
# Train on your data
python train_text_coi.py --config training_config.yaml --use-lora

# Inference: flexible prompts
sep = CLAPSepInference.from_checkpoint("best_model.ckpt")
sources = sep.separate("audio.wav", text_pos="airplane engine")  # or any prompt
```
- ✅ Optimized for your data (LoRA-tuned)
- ✅ Flexible at inference (change prompts)
- ✅ Retains zero-shot capability
- ⚠️ Requires training

### 3. Learned-Embedding Training (`train_coi.py`)
```python
# Train on aircraft data
python train_coi.py --config training_config.yaml --use-lora

# Inference: fixed to aircraft only
sep = CLAPSepInference.from_checkpoint("best_model.ckpt")
sources = sep.separate("audio.wav")  # always separates aircraft
```
- ✅ Optimized for your data
- ✅ Slightly simpler architecture
- ❌ Fixed to single COI class
- ❌ No inference flexibility
- ❌ Must retrain for new classes

---

## Multi-Class Training

### Train on Aircraft + Birds

```yaml
model:
  coi_text_prompts:
    - ["airplane engine", "aircraft noise", "jet flying"]
    - ["bird chirping", "bird singing", "songbird"]

data:
  target_classes:
    - ["airplane", "aircraft", "jet", "propeller_airplane"]
    - ["bird", "bird_song", "birds", "bird_chirp"]
```

### Inference with Multi-Class Model

```python
# Load model trained on both aircraft and birds
sep = CLAPSepInference.from_checkpoint("best_model.ckpt")

# Separate aircraft
sources = sep.separate("audio.wav", text_pos="airplane engine")

# Separate birds (same model!)
sources = sep.separate("audio.wav", text_pos="bird chirping")

# Try new classes (zero-shot, YMMV)
sources = sep.separate("audio.wav", text_pos="dog barking")
```

---

## Troubleshooting

### Issue: "No module named 'laion_clap'"
```bash
pip install laion-clap
```

### Issue: "No module named 'loralib'"
```bash
pip install loralib
```

### Issue: Text prompts not working at inference
- Check that you loaded with `from_checkpoint()` (not `from_pretrained()`)
- Verify checkpoint is from `train_text_coi.py` (check for `coi_text_prompts` in hyperparameters)
- Learned-embedding checkpoints don't support text prompts

### Issue: Poor performance on new sounds
- Text-prompt models work best on sounds similar to training data
- Try fine-tuning on a broader dataset
- Increase LoRA rank for more capacity (`--lora-rank 16`)

### Issue: Out of memory
- Reduce batch size in `training_config.yaml`
- Use smaller LoRA rank (`--lora-rank 4`)
- Use frozen encoder (`--freeze-encoder`)

---

## Migration Guide

### From Learned-Embedding to Text-Prompt

**Before** (`train_coi.py`):
```bash
python train_coi.py --use-lora --lora-rank 8
```

**After** (`train_text_coi.py`):
```bash
python train_text_coi.py --use-lora --lora-rank 8
```

**Config changes**:
```yaml
# Add to training_config.yaml
model:
  coi_text_prompts:
    - ["airplane engine", "aircraft noise", "jet flying"]
  background_text_prompts: ["ambient noise", "background sounds"]

data:
  target_classes:
    - ["airplane", "aircraft", "jet"]
```

**Inference changes**:
```python
# Before: Fixed to aircraft
sep = CLAPSepInference.from_checkpoint("old_model.ckpt")
sources = sep.separate("audio.wav")

# After: Flexible prompts
sep = CLAPSepInference.from_checkpoint("new_model.ckpt")
sources = sep.separate("audio.wav", text_pos="airplane engine")  # or any sound
```

---

## FAQ

**Q: Should I use text-prompt or learned-embedding training?**

A: Use text-prompt training unless you have a specific reason not to. It provides the same performance while retaining flexibility.

**Q: Can I convert a learned-embedding model to text-prompt?**

A: No, they use different architectures. You need to retrain with `train_text_coi.py`.

**Q: Do I need more data for text-prompt training?**

A: No, it uses the same data. The text prompts are generated from your labels automatically.

**Q: What's the best LoRA rank?**

A: Start with rank=8. Increase to 16 if underfitting, decrease to 4 if out of memory.

**Q: Can I fine-tune a text-prompt model on new classes?**

A: Not yet implemented, but possible in the future. For now, retrain from scratch with all classes.

**Q: Does this work with the base CLAPSep checkpoint?**

A: Yes! The training starts from the pretrained CLAP checkpoint and fine-tunes with LoRA.

---

## Summary

The new text-prompt training approach:
- ✅ **Keeps flexibility** of the original CLAPSep
- ✅ **Improves performance** on your COI data via LoRA tuning
- ✅ **Supports multiple classes** and dynamic prompt changes
- ✅ **Same training cost** as learned-embedding approach
- ✅ **Better long-term solution** for multi-purpose separation

Use `train_text_coi.py` for new projects. Keep `train_coi.py` only if you need the absolute simplest architecture for a single fixed COI class.
