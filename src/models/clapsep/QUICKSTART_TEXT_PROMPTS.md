# Quick Start: Text-Prompt CLAPSep Training

Get started with text-prompt CLAPSep training in 3 steps!

## 1. Setup Configuration

Edit `src/models/clapsep/training_config.yaml`:

```yaml
data:
  df_path: "data/aircraft_data.csv"  # Your dataset CSV
  target_classes:
    - ["airplane", "aircraft", "jet", "propeller_airplane"]

model:
  coi_text_prompts:
    - ["airplane engine", "aircraft noise", "jet flying", "plane overhead"]
  background_text_prompts: ["ambient noise", "background sounds"]
  use_lora: true
  lora_rank: 8

training:
  batch_size: 16
  num_epochs: 150
  lr: 1e-4
```

## 2. Train

```bash
cd /home/bendm/Thesis/project/code

# Train with LoRA (recommended)
python src/models/clapsep/train_text_coi.py \
  --config src/models/clapsep/training_config.yaml \
  --use-lora \
  --lora-rank 8
```

## 3. Inference

```python
from models.clapsep.inference import CLAPSepInference

# Load your trained model
sep = CLAPSepInference.from_checkpoint(
    "src/models/clapsep/checkpoints/text_prompt_YYYYMMDD_HHMMSS/best_model.ckpt",
    device="cuda"
)

# Separate audio with text prompts
sources = sep.separate("audio.wav", text_pos="airplane engine", text_neg="")

# Save results
sep.save_audio(sep.get_coi_audio(sources), "output_airplane.wav")
sep.save_audio(sep.get_background_audio(sources), "output_background.wav")

# Try different prompts on the same model!
sources = sep.separate("audio.wav", text_pos="bird chirping", text_neg="")
```

---

## Key Benefits

✅ **Flexible**: Change target sound at inference by changing text prompt  
✅ **Efficient**: LoRA tuning adds only ~0.5-1M parameters  
✅ **Robust**: Multiple prompt variations improve generalization  
✅ **Multi-class**: Train once, separate multiple sound types  

---

## Command-Line Options

### Training Modes

**LoRA Fine-tuning** (Recommended):
```bash
python train_text_coi.py --use-lora --lora-rank 8
```
- ~11-12M trainable parameters
- Good performance, fast training

**Decoder-Only** (Fastest):
```bash
python train_text_coi.py --freeze-encoder
```
- ~10M trainable parameters  
- Use for quick experiments

**Full Fine-tuning** (Most Parameters):
```bash
python train_text_coi.py --no-freeze-encoder --no-lora
```
- ~60M trainable parameters
- Use when you have lots of data

### Override Config

```bash
python train_text_coi.py \
  --config training_config.yaml \
  --df-path data/my_custom_data.csv \
  --lora-rank 16 \
  --device cuda:0
```

---

## Multi-Class Training

### Setup (Aircraft + Birds)

```yaml
model:
  coi_text_prompts:
    - ["airplane engine", "aircraft noise", "jet flying"]
    - ["bird chirping", "bird singing", "songbird"]

data:
  target_classes:
    - ["airplane", "aircraft", "jet"]
    - ["bird", "bird_song", "birds"]
```

### Inference

```python
# Same model, different targets!
sep = CLAPSepInference.from_checkpoint("best_model.ckpt")

# Separate aircraft
sources = sep.separate("audio.wav", text_pos="airplane engine")

# Separate birds
sources = sep.separate("audio.wav", text_pos="bird chirping")
```

---

## Next Steps

- Read full documentation: `TEXT_PROMPT_TRAINING.md`
- Compare with learned embeddings: `LORA_IMPLEMENTATION.md`
- Check example configs: `training_config.yaml`

## Troubleshooting

**Missing dependencies:**
```bash
pip install laion-clap loralib
```

**Out of memory:**
- Reduce `batch_size` in config
- Use `--lora-rank 4` (smaller)
- Use `--freeze-encoder`

**Low performance:**
- Increase `lora_rank` to 16
- Train longer (`num_epochs: 200`)
- Add more prompt variations
