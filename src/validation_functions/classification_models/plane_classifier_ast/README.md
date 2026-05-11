# PANN Plane Classifier

Binary plane/no-plane audio classifier using PANN CNN14 pretrained on AudioSet.

## Quick Start

### Training
```bash
cd /home/bendm/Thesis/project/code/src/validation_functions/classification_models/plane_classifier_pann
python main.py
```

### Inference
```bash
python inference.py checkpoints/final_model.pth audio.wav --threshold 0.5
```

## Features

✅ **Pretrained CNN14** - Automatically downloads weights from Zenodo (mAP=0.431 on AudioSet)  
✅ **Two-Phase Training** - Frozen backbone → Full fine-tuning  
✅ **32kHz Native** - Higher quality than YAMNet's 16kHz  
✅ **2048-dim Embeddings** - Richer features than YAMNet (1024-dim)  
✅ **Data Augmentation** - Time stretch, noise, gain  
✅ **Bootstrap CI** - Confidence intervals for validation metrics  
✅ **Abstracted Data Loading** - Shared PyTorch dataset for all models  

## Architecture

```
Raw Waveform (32kHz, 10 seconds)
    ↓
CNN14 Backbone (pretrained on AudioSet)
  • STFT + Mel Spectrogram
  • 6 ConvBlocks: 64→128→256→512→1024→2048
  • Attention Pooling
    ↓
2048-dim Embedding
    ↓
Classification Head
  • Dense(512) + ReLU + BN + Dropout(0.3)
  • Dense(256) + ReLU + BN + Dropout(0.2)
  • Dense(128) + ReLU + BN + Dropout(0.1)
  • Dense(1) → Binary Logit
    ↓
Sigmoid → Probability
```

## Files

```
plane_classifier_pann/
├── __init__.py           # Package initialization
├── config.py             # Training & model configuration
├── data_config.py        # Data loading configuration
├── model.py              # CNN14 + PlaneClassifierPANN
├── model_loader.py       # Load pretrained/trained models
├── dataset.py            # PyTorch Dataset & DataLoader
├── train.py              # Two-phase training pipeline
├── main.py               # Training entry point
├── inference.py          # Inference wrapper
├── FINETUNING.md         # Detailed documentation
└── README.md             # This file

src/common/ (shared utilities)
├── audio_dataset.py      # Generic audio classification dataset
└── audio_validation.py   # Audio file validation
```

## Requirements

All dependencies should already be installed in your `.venv`:
- `torch >= 2.0`
- `torchaudio >= 2.0`
- `torchlibrosa`
- `librosa`
- `pandas`
- `numpy`
- `scikit-learn`
- `tqdm`
- `requests`

## Training Pipeline

1. **Load metadata** from existing datasets
2. **Sample balanced data** (25% plane sounds)
3. **Validate audio files** (remove corrupted files)
4. **Create DataLoaders** (train/val/test)
5. **Load pretrained CNN14** (automatically downloads if needed)
6. **Phase 1**: Train classifier head (30 epochs, LR=1e-3)
7. **Phase 2**: Fine-tune entire model (20 epochs, LR=1e-5)
8. **Find optimal threshold** on validation set (max F1)
9. **Evaluate** on test set

## Configuration

Default settings in `config.py`:
- Sample Rate: 32000 Hz
- Audio Duration: 10 seconds
- Batch Size: 32
- Phase 1: 30 epochs, LR=1e-3
- Phase 2: 20 epochs, LR=1e-5
- Augmentation: Enabled (time stretch, noise, gain)
- Device: CUDA (auto-fallback to CPU)

## Python API

### Training
```python
from plane_classifier_pann.main import main
model = main()
```

### Inference
```python
from plane_classifier_pann import PlaneClassifierInference

classifier = PlaneClassifierInference("checkpoints/final_model.pth")
result = classifier.predict_file("audio.wav")

print(result['prediction'])   # 'plane' or 'no_plane'
print(result['confidence'])   # 0.0 to 1.0
```

### Load Trained Model
```python
from plane_classifier_pann import load_trained_model
import torch

model = load_trained_model("checkpoints/final_model.pth", device='cuda')
waveform = torch.randn(1, 320000).cuda()  # 10s at 32kHz
logits = model(waveform)
probs = torch.sigmoid(logits)
```

## Metrics

- **Primary**: PR-AUC (Precision-Recall AUC)
- **Secondary**: ROC-AUC, Accuracy, Precision, Recall, F1
- **Validation**: Bootstrap confidence intervals (1000 iterations)

## Performance

Expected on plane detection task:
- **Training Time**: 3-7 days on Tesla V100 GPU
- **Inference Speed**: ~50ms per 10-second clip (GPU)
- **Memory**: ~12 GB GPU for training, ~2 GB for inference

## Comparison with YAMNet

| Metric | YAMNet | PANN CNN14 |
|--------|---------|------------|
| mAP on AudioSet | 0.317 | **0.431** ✓ |
| Embedding Dim | 1024 | **2048** ✓ |
| Sample Rate | 16 kHz | **32 kHz** ✓ |
| Audio Duration | 5s | **10s** ✓ |
| Framework | TensorFlow | **PyTorch** ✓ |

## Citation

```bibtex
@article{kong2020panns,
  title={Panns: Large-scale pretrained audio neural networks for audio pattern recognition},
  author={Kong, Qiuqiang and Cao, Yin and Iqbal, Turab and Wang, Yuxuan and Wang, Wenwu and Plumbley, Mark D},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={28},
  pages={2880--2894},
  year={2020}
}
```

## See Also

- [FINETUNING.md](FINETUNING.md) - Detailed training guide
- [PANN GitHub](https://github.com/qiuqiangkong/audioset_tagging_cnn)
- [Pretrained Weights](https://zenodo.org/record/3987831)
- `plane_clasifier/` - YAMNet-based implementation for comparison
