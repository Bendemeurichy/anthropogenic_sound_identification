# PANN Fine-tuning Guide

## Overview

This implementation uses **PANN (Pretrained Audio Neural Networks)** CNN14 model pretrained on AudioSet for binary plane detection. The architecture and training strategy mirror the YAMNet implementation but leverage PANN's superior performance.

## Architecture

### CNN14 Backbone (Pretrained)
- **Input**: Raw waveform at 32kHz
- **Spectrogram Extraction**: Built-in STFT + Mel filterbank
- **Architecture**: 6 convolutional blocks (64→128→256→512→1024→2048 channels)
- **Temporal Pooling**: Attention-based pooling
- **Output**: 2048-dimensional embedding

### Classification Head (Trainable)
```
Embedding (2048-dim)
    ↓
Dense(512) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Dense(256) + ReLU + BatchNorm + Dropout(0.2)
    ↓
Dense(128) + ReLU + BatchNorm + Dropout(0.1)
    ↓
Dense(1) → Binary logit
```

## Key Differences from YAMNet

| Aspect | YAMNet | PANN CNN14 |
|--------|---------|------------|
| **Framework** | TensorFlow/Keras | PyTorch |
| **Sample Rate** | 16 kHz | 32 kHz |
| **Audio Duration** | 5 seconds | 10 seconds |
| **Embedding Dim** | 1024 | 2048 |
| **Pretrained On** | AudioSet (527 classes) | AudioSet (527 classes) |
| **mAP on AudioSet** | 0.317 (Google baseline) | 0.431 |
| **Batch Size** | 64 | 32 (due to larger embeddings) |

## Two-Phase Training Strategy

### Phase 1: Train Classifier Head Only (30 epochs)
- **Frozen**: CNN14 backbone (all 6 conv blocks + attention)
- **Trainable**: Classification head (~1M parameters)
- **Learning Rate**: 1e-3
- **Goal**: Learn good initial classifier weights from pretrained features

### Phase 2: Full Fine-tuning (20 epochs)
- **Trainable**: Entire model (CNN14 + classifier, ~80M parameters)
- **Learning Rate**: 1e-5 (100x lower than phase 1)
- **Goal**: Adapt PANN features specifically for plane detection

## Why Two-Phase Training Works

1. **Phase 1** quickly trains the classifier head using powerful pretrained features
2. **Phase 2** fine-tunes the entire network with a very low learning rate, adapting the feature extractor to your specific task without destroying the pretrained knowledge

## Pretrained Weights

The model automatically downloads pretrained CNN14 weights from Zenodo:
- **URL**: `https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth`
- **Size**: ~300 MB
- **Cache Location**: `~/.cache/pann/Cnn14_mAP=0.431.pth`
- **Training**: AudioSet full (1.9M samples, 527 classes)

## Data Requirements

### Expected DataFrame Format
```python
{
    'filename': str,      # Absolute path to audio file
    'start_time': float,  # Start time in seconds (can be NaN for full file)
    'end_time': float,    # End time in seconds (can be NaN for full file)
    'label': int,         # Binary label (0 or 1)
    'split': str,         # 'train', 'val', or 'test'
}
```

### Audio Processing
- **Sample Rate**: All audio resampled to 32kHz
- **Duration**: Clips are padded/cropped to 10 seconds
- **Long Annotations**: Automatically split into 10-second segments
- **Short Clips**: Padded with zeros to 10 seconds

## Training

### Basic Training
```bash
cd src/validation_functions/classification_models/plane_classifier_pann
python main.py
```

### Custom Configuration
```bash
python main.py \
    --checkpoint-dir ./my_checkpoints \
    --device cuda
```

### From Python
```python
from plane_classifier_pann.main import main

model = main()
```

## Inference

### Command Line
```bash
python inference.py \
    checkpoints/final_model.pth \
    audio1.wav audio2.wav \
    --threshold 0.5
```

### Python API
```python
from plane_classifier_pann import PlaneClassifierInference

# Load model
classifier = PlaneClassifierInference(
    checkpoint_path="checkpoints/final_model.pth",
    device='cuda'
)

# Predict single file
result = classifier.predict_file("audio.wav", threshold=0.5)
print(result['prediction'])   # 'plane' or 'no_plane'
print(result['confidence'])   # Probability score

# Predict batch
results = classifier.predict_batch(["audio1.wav", "audio2.wav"])
for r in results:
    print(f"{r['file']}: {r['prediction']} ({r['confidence']:.2f})")
```

## Loading a Trained Model

```python
from plane_classifier_pann import load_trained_model
import torch

# Load model
model = load_trained_model(
    checkpoint_path="checkpoints/final_model.pth",
    device='cuda'
)

# Use for inference
waveform = torch.randn(1, 320000)  # 1 sample, 10s at 32kHz
logits = model(waveform.cuda())
probs = torch.sigmoid(logits)
```

## Checkpoint Files

After training, you'll have:
- `best_model_phase1.pth` - Best model from Phase 1
- `best_model_phase2.pth` - Best model from Phase 2 (fine-tuned)
- `final_model.pth` - Final model after all training
- `history_phase1.json` - Phase 1 training metrics
- `history_phase2.json` - Phase 2 training metrics
- `optimal_threshold.json` - Optimal classification threshold (max F1)

## Metrics

### Primary Metric
- **PR-AUC** (Precision-Recall AUC): Best for imbalanced datasets

### Secondary Metrics
- **ROC-AUC**: Overall classification performance
- **Accuracy, Precision, Recall, F1**: At optimal threshold
- **Bootstrap CI**: Confidence intervals for validation PR-AUC

## Data Augmentation

Applied only during training:
1. **Time Stretching** (prob=0.5, range=0.8-1.2x)
2. **Additive Noise** (prob=0.5, stddev=0.005)
3. **Random Gain** (prob=0.5, range=0.7-1.3x)

## Hyperparameters

### Phase 1
- Epochs: 30
- Learning Rate: 1e-3
- Early Stopping Patience: 10
- LR Reduction Patience: 5

### Phase 2
- Epochs: 20
- Learning Rate: 1e-5
- Early Stopping Patience: 8
- LR Reduction Patience: 4

### Optimizer
- AdamW with weight decay 1e-5
- Gradient clipping: norm=1.0
- Betas: (0.9, 0.999)

## Technical Notes

### Why PANN Over YAMNet?

1. **Better Performance**: PANN achieves 0.431 mAP vs YAMNet's 0.317 on AudioSet
2. **Richer Features**: 2048-dim embeddings vs 1024-dim
3. **Better for Longer Audio**: Works well with 10-second clips
4. **Active Development**: PANN is more recent and actively maintained

### Memory Requirements

- **Training**: ~12 GB GPU memory (batch size 32)
- **Inference**: ~2 GB GPU memory
- **Reduce Batch Size** if OOM: Set `batch_size=16` or `8` in config

### Speed

- **Training**: ~3-7 days on single Tesla V100 GPU
- **Inference**: ~50ms per 10-second clip on GPU

## Troubleshooting

### Out of Memory
Reduce batch size in `config.py`:
```python
batch_size: int = 16  # or 8
```

### Slow Data Loading
Increase num_workers:
```python
num_workers: int = 8  # or more
```

### Model Not Improving
- Check class balance (should be reasonably balanced)
- Try different learning rates
- Ensure audio files are valid (use validation)
- Check data augmentation is working

## Citation

If you use this implementation, please cite:

```bibtex
@article{kong2020panns,
  title={Panns: Large-scale pretrained audio neural networks for audio pattern recognition},
  author={Kong, Qiuqiang and Cao, Yin and Iqbal, Turab and Wang, Yuxuan and Wang, Wenwu and Plumbley, Mark D},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={28},
  pages={2880--2894},
  year={2020},
  publisher={IEEE}
}
```

## References

1. [PANN GitHub Repository](https://github.com/qiuqiangkong/audioset_tagging_cnn)
2. [PANN Paper](https://arxiv.org/abs/1912.10211)
3. [Pretrained Weights (Zenodo)](https://zenodo.org/record/3987831)
4. [AudioSet Dataset](https://research.google.com/audioset/)
