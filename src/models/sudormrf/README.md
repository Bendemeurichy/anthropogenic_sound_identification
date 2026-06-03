# SuDoRM-RF (Sub-band Decomposition of Multi-channel Room Impulse Response Filter)

Time-domain separation model fine-tuned for COI (Class of Interest) separation.

## Training

```bash
cd src/models/sudormrf
python train.py --config training_config.yaml
```

For hyperparameter tuning, use `configs/tuning/sudormrf_tuning.yaml` with `scripts/tuning/optuna_tune_models.py`.

## Config

See `training_config.yaml` for all model/data/training settings. Key sections:

- `model.separator` — Architecture (number of blocks, channels, basis functions)
- `data` — Dataset paths, target classes, augmentations, WebDataset settings
- `training` — Learning rate, epochs, checkpointing

## Inference

```python
from models.sudormrf.inference import SuDoRMRFInference

separator = SuDoRMRFInference.from_checkpoint("checkpoints/best_model.pt")
sources = separator.separate("audio.wav")
separator.save_audio(separator.get_coi_audio(sources), "output.wav")
```

Implements `BaseSeparator` interface (`src/models/base.py`).

## Files

| File | Purpose |
|------|---------|
| `train.py` | Main training script |
| `inference.py` | Inference wrapper |
| `finetune.py` | Fine-tuning from pretrained weights |
| `dataset.py` | PyTorch Dataset for CSV-based loading |
| `losses.py` | SI-SNR loss functions |
| `augmentations.py` | Time/frequency/mixing augmentations |
| `seperation_head.py` | SuDoRM-RF separation head |
| `config.py` | Config dataclasses |
| `utils.py` | Training helpers |
| `training_config.yaml` | Production training config |
| `finetune_config.yaml` | Fine-tuning config |
