# CLAPSep (CLAP-based Source Separation)

Text-conditioned separation model. Two training modes available.

## Training modes

### Text-prompt training (`train_text_coi.py`)
- Uses CLAP text encoder for conditioning — change target sound at inference by changing the prompt
- Supports LoRA for parameter-efficient adaptation (~0.5-2M trainable params)
- Best for multi-purpose separation

### Learned-embedding training (`train_coi.py`)
- Uses learned embedding vectors (no text encoder)
- Fixed to one COI class per training run
- Slightly simpler architecture

## Quick start (text-prompt)

```bash
cd src/models/clapsep

# LoRA fine-tuning (recommended)
python train_text_coi.py --config training_config.yaml --use-lora --lora-rank 8

# Freeze encoder (fastest)
python train_text_coi.py --config training_config.yaml --freeze-encoder

# Full fine-tuning (most params)
python train_text_coi.py --config training_config.yaml --no-freeze-encoder --no-lora
```

## Config (text-prompt)

```yaml
model:
  coi_text_prompts:
    - ["airplane engine", "aircraft noise", "jet flying"]
  background_text_prompts: ["ambient noise", "background sounds"]
  freeze_encoder: false
  use_lora: true
  lora_rank: 8
data:
  target_classes:
    - ["airplane", "aircraft", "jet"]
```

## Multi-class text-prompt training

```yaml
model:
  coi_text_prompts:
    - ["airplane engine", "aircraft noise"]
    - ["bird chirping", "bird singing"]
data:
  target_classes:
    - ["airplane", "aircraft"]
    - ["bird", "birds"]
```

## Inference

```python
from models.clapsep.inference import CLAPSepInference

sep = CLAPSepInference.from_checkpoint("best_model.ckpt", device="cuda")

# Separate specific sound (can change prompt at inference!)
sources = sep.separate("audio.wav", text_pos="airplane engine")
sources = sep.separate("audio.wav", text_pos="bird chirping")  # same model

sep.save_audio(sep.get_coi_audio(sources), "output.wav")
```

## Training mode comparison

| Aspect | Text-prompt | Learned-embedding |
|--------|------------|-------------------|
| Inference flexibility | Change prompt at runtime | Fixed to trained class |
| Multi-class | Train once, separate multiple | One class per training run |
| Architecture | Text encoder (frozen) | `nn.Parameter` embeddings |
| New classes | Zero-shot possible on similar sounds | Must retrain |
